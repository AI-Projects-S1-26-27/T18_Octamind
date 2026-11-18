"""Shared embedding service and vector store for knowledge graphs (Role 3).

Provides the project-wide embedding service required by docs/INTERFACE.md:
  embed(text: str) -> tuple[float, ...]

Computes:
1. Entity embeddings phi(entity): embeddings of canonical entity names, saved
   as data/graph/<dataset>/entity_embeddings.json for Role 5's heuristic h(n).
   Covers every entity in Role 2's entities.jsonl *and* every edge endpoint that
   Role 2 did not extract (typically oracle-graph gold entities).
2. Edge embeddings phi(e): embeddings of 'head relation tail', saved into
   Edge.embedding for calculating edge relevance rel(e, q) = cos(phi(e), phi(q)).

Includes:
- Gemini Embedding model (``gemini-embedding-2``, the project's fixed model).
- Deterministic offline mock embedder for tests and local development ONLY: its
  vectors are hash noise, so similarities carry no meaning. Runs in mock mode are
  flagged in the manifest and print a warning.
- Atomic per-text disk caching in data/cache/embeddings/ for zero redundant API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import yaml

from src.graph.extract import (
    QuotaExhausted,
    RateLimiter,
    is_daily_quota_error,
    load_dotenv,
)
from src.graph.oracle import read_edges, write_edges
from src.traversal.types import Edge

DEFAULT_CONFIG = "configs/graph.yaml"
DEFAULT_MODEL = "gemini-embedding-2"


class TeeOutput:
    """Mirror terminal output to a persistent log and terminal."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.file = path.open("a", encoding="utf-8")

    def write(self, text: str) -> int:
        written = self.terminal.write(text)
        self.file.write(text)
        self.file.flush()
        return written

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def open_run_log(dataset: Optional[str], mode: Optional[str]) -> tuple[Path, TeeOutput]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = dataset or "all"
    log_path = Path("logs") / f"embeddings_{label}_{mode or 'configured'}_{stamp}.log"
    return log_path, TeeOutput(log_path)


class EmbeddingCache:
    """Atomic disk cache storing text embeddings keyed by sha256(model + text)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _key(self, model: str, text: str) -> str:
        payload = f"{model}\x00{text.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> Optional[tuple[float, ...]]:
        path = self.dir / f"{self._key(model, text)}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return tuple(float(x) for x in raw.get("embedding", ()))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def put(self, model: str, text: str, vector: tuple[float, ...]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self._key(model, text)}.json"
        payload = {"model": model, "text": text, "embedding": list(vector)}
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def mock_embed(text: str, dim: int = 64) -> tuple[float, ...]:
    """Deterministic unit-normalized embedding for offline testing.

    Uses a deterministic PRNG seeded by the SHA-256 of the normalized text. These are
    NOT semantic: paraphrases are as dissimilar as unrelated strings.
    """
    clean = text.strip().lower()
    if not clean:
        return tuple(0.0 for _ in range(dim))

    # Seed PRNG with integer hash of text
    seed_int = int(hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_int)

    # Generate pseudo-random vector
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return tuple(0.0 for _ in range(dim))

    return tuple(x / norm for x in vec)


def make_gemini_client():
    """google-genai client; reads GEMINI_API_KEY from the environment or the git-ignored .env."""
    load_dotenv()
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini embedding requires `pip install google-genai`") from exc
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY (in .env, never committed) is required for gemini embedding mode; "
                           "use --mode mock only for offline tests.")
    return genai.Client(http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1),
    ))


def gemini_embed_batch(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    client=None,
    max_attempts: int = 5,
) -> list[tuple[float, ...]]:
    """Embed a list of texts using google-genai client.

    Always returns exactly one vector per text, in order; a response with a different
    number of vectors raises instead of silently shifting the alignment.
    """
    if not texts:
        return []
    client = client or make_gemini_client()

    try:
        from google.genai import types
        contents_payload = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
    except Exception:
        contents_payload = texts

    for attempt in range(max_attempts):
        try:
            response = client.models.embed_content(model=model, contents=contents_payload)
            results = [tuple(float(v) for v in item.values) for item in response.embeddings]
        except Exception as exc:
            message = str(exc)
            code = getattr(exc, "code", None)
            if code == 429 and is_daily_quota_error(message):
                raise QuotaExhausted(message[:400]) from exc
            transient = code in (429, 500, 502, 503, 504) or bool(
                re.search(r"timeout|connect|reset|closed|unavailable|resource exhausted", message, re.I)
            )
            if attempt == max_attempts - 1:
                raise
            if not transient:
                raise

            delay = float(min(8.0, 2.0 ** attempt) + random.random())
            if code == 429 or "resource exhausted" in message.lower():
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", message) or re.search(r"'retryDelay':\s*'(\d+)s'", message)
                if m:
                    delay = float(m.group(1)) + 2.0
                else:
                    delay = max(delay, 60.0)

            print(
                f"  embedding retry {attempt + 1}/{max_attempts}: "
                f"{code if code is not None else 'transient error'} (sleeping {delay:.1f}s)",
                flush=True,
            )
            time.sleep(delay)
            continue
        if len(results) != len(texts):
            raise RuntimeError(
                f"embedding API returned {len(results)} vectors for {len(texts)} texts"
            )
        return results

    raise RuntimeError("unreachable")


class EmbeddingService:
    """Project-wide embedding service providing embed() and embed_many() with caching."""

    def __init__(
        self,
        mode: str = "mock",
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path = "data/cache/embeddings",
        dimension: int = 64,
        batch_size: int = 100,
        use_cache: Optional[bool] = None,
        requests_per_minute: Optional[float] = None,
        requests_per_day: Optional[float] = None,
    ):
        if mode not in ("mock", "gemini"):
            raise ValueError(f"unknown embedding mode {mode!r}; choose 'mock' or 'gemini'")
        self.mode = mode
        self.model = model if mode != "mock" else f"mock-{dimension}"
        self.cache = EmbeddingCache(cache_dir)
        self.dimension = dimension
        self.batch_size = batch_size
        self.use_cache = use_cache if use_cache is not None else True
        self.limiter = RateLimiter(requests_per_minute, requests_per_day)
        self._client = None

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed a single string."""
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed a batch of strings, utilizing the disk cache to minimize calls.

        Returns one vector per input text, in order. Never silently drops or shifts
        entries: if a vector is missing it raises.
        """
        if not texts:
            return []

        results: list[Optional[tuple[float, ...]]] = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        # Check cache first
        for i, t in enumerate(texts):
            cached = self.cache.get(self.model, t) if self.use_cache else None
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(t)

        cached_count = len(texts) - len(missing_texts)
        if len(missing_texts):
            batch_count = math.ceil(len(missing_texts) / self.batch_size)
            print(
                f"  embedding progress: {cached_count}/{len(texts)} reused from cache, "
                f"{len(missing_texts)} to fetch in {batch_count} {self.mode} batches "
                f"(batch size {self.batch_size})",
                flush=True,
            )
        elif self.use_cache:
            print(
                f"  embedding progress: {cached_count}/{len(texts)} reused from cache",
                flush=True,
            )

        # Fetch missing in batches
        for start_idx in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start_idx : start_idx + self.batch_size]
            batch_number = start_idx // self.batch_size + 1
            batch_count = math.ceil(len(missing_texts) / self.batch_size)
            batch_done_before = start_idx
            batch_done_after = min(start_idx + len(batch), len(missing_texts))
            print(
                f"  embedding batch {batch_number}/{batch_count}: "
                f"{batch_done_before}/{len(missing_texts)} fetched, "
                f"{len(missing_texts) - batch_done_before} left; "
                f"processing {len(batch)} texts",
                flush=True,
            )
            if self.mode == "mock":
                batch_vectors = [mock_embed(t, dim=self.dimension) for t in batch]
            else:
                if self._client is None:
                    self._client = make_gemini_client()
                self.limiter.wait()
                try:
                    batch_vectors = gemini_embed_batch(
                        batch, model=self.model, client=self._client
                    )
                except RuntimeError as exc:
                    if "embedding API returned" not in str(exc):
                        raise
                    print(
                        f"  batch mismatch for {len(batch)} texts; "
                        "using individual fallback",
                        flush=True,
                    )
                    batch_vectors = []
                    for fallback_index, text in enumerate(batch, start=1):
                        cached = self.cache.get(self.model, text) if self.use_cache else None
                        if cached is not None:
                            batch_vectors.append(cached)
                            continue
                        self.limiter.wait()
                        individual = gemini_embed_batch(
                            [text],
                            model=self.model,
                            client=self._client,
                            max_attempts=3,
                        )
                        if len(individual) != 1:
                            raise RuntimeError(
                                f"individual embedding returned {len(individual)} vectors"
                            )
                        batch_vectors.append(individual[0])
                        if self.use_cache:
                            self.cache.put(self.model, text, individual[0])
                        if fallback_index % 100 == 0 or fallback_index == len(batch):
                            print(
                                f"  individual fallback: {fallback_index}/{len(batch)} texts",
                                flush=True,
                            )
            if len(batch_vectors) != len(batch):
                raise RuntimeError(f"embedder returned {len(batch_vectors)} vectors for {len(batch)} texts")

            # Store in cache and results
            for sub_idx, (vec, t) in enumerate(zip(batch_vectors, batch)):
                orig_idx = missing_indices[start_idx + sub_idx]
                results[orig_idx] = vec
                if self.use_cache:
                    self.cache.put(self.model, t, vec)

            remaining = len(missing_texts) - batch_done_after
            print(
                f"  embedding batch {batch_number}/{batch_count} complete: "
                f"{batch_done_after}/{len(missing_texts)} fetched, "
                f"{remaining} left",
                flush=True,
            )

        if any(r is None for r in results):
            raise RuntimeError("embedding failed for some texts")
        return results  # type: ignore[return-value]


def service_from_config(cfg: dict, mode_override: Optional[str] = None) -> EmbeddingService:
    emb_cfg = cfg.get("embedding", {})
    return EmbeddingService(
        mode=mode_override or emb_cfg.get("mode", "gemini"),
        model=emb_cfg.get("model", DEFAULT_MODEL),
        cache_dir=emb_cfg.get("cache_dir", "data/cache/embeddings"),
        dimension=int(emb_cfg.get("dimension", 64)),
        batch_size=int(emb_cfg.get("batch_size", 100)),
        requests_per_minute=emb_cfg.get("requests_per_minute"),
        requests_per_day=emb_cfg.get("requests_per_day"),
    )


_SERVICES: dict = {}


def get_service(config_path: str | Path = DEFAULT_CONFIG, mode_override: Optional[str] = None) -> EmbeddingService:
    """Process-wide service built from configs/graph.yaml (one instance per config + mode)."""
    key = (str(config_path), mode_override)
    if key not in _SERVICES:
        with open(config_path, encoding="utf-8") as f:
            _SERVICES[key] = service_from_config(yaml.safe_load(f) or {}, mode_override)
    return _SERVICES[key]


def embed(text: str) -> tuple[float, ...]:
    """The shared ``embed(text)`` of docs/INTERFACE.md, using configs/graph.yaml."""
    return get_service().embed(text)


def write_entity_embeddings(embeddings: dict[str, tuple[float, ...]], path: str | Path) -> None:
    """Write entity embeddings mapping to a JSON file (atomically)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialized = {k: list(v) for k, v in embeddings.items()}
    tmp = p.with_name(f"{p.name}.{os.getpid()}.part")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serialized, f)
        tmp.replace(p)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_entity_embeddings(path: str | Path) -> dict[str, tuple[float, ...]]:
    """Read entity embeddings mapping from a JSON file."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: tuple(float(x) for x in v) for k, v in raw.items()}


def entity_name_from_id(eid: str) -> str:
    """Readable name for an entity id that has no Role 2 record: ``ent:new_york_city`` -> ``new york city``."""
    return eid.removeprefix("ent:").replace("_", " ").strip() or eid


def edge_text(edge: Edge, names: Optional[dict] = None) -> str:
    """Text embedded for phi(e): ``head relation tail``, with readable entity names when
    ``names`` is given (``Barack Obama was born in Honolulu``) instead of raw ids
    (``ent:barack_obama was born in ent:honolulu``)."""
    if not names:
        return edge.text
    head = names.get(edge.head) or entity_name_from_id(edge.head)
    tail = names.get(edge.tail) or entity_name_from_id(edge.tail)
    return f"{head} {edge.relation} {tail}"


def populate_edge_embeddings(edges: list[Edge], service: EmbeddingService, names: Optional[dict] = None) -> list[Edge]:
    """Compute and attach phi(e) = embed('head relation tail') for each edge."""
    vectors = service.embed_many([edge_text(e, names) for e in edges])
    if len(vectors) != len(edges):
        raise RuntimeError(f"got {len(vectors)} vectors for {len(edges)} edges")
    return [
        Edge(
            head=edge.head,
            relation=edge.relation,
            tail=edge.tail,
            confidence=edge.confidence,
            support=edge.support,
            provenance=edge.provenance,
            embedding=vec,
        )
        for edge, vec in zip(edges, vectors)
    ]


def load_entity_names(entities_file: str | Path) -> dict[str, str]:
    """``{entity id: name}`` from Role 2's entities.jsonl (empty if the file is missing)."""
    p = Path(entities_file)
    names: dict[str, str] = {}
    if not p.exists():
        return names
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                names[it["id"]] = it.get("name", it["id"])
    return names


def embed_entities(names: dict[str, str], service: EmbeddingService) -> dict[str, tuple[float, ...]]:
    """Embed ``{id: name}``; one vector per id."""
    ids = list(names)
    vectors = service.embed_many([names[i] for i in ids])
    return dict(zip(ids, vectors))


def embed_dataset_entities(
    entities_file: str | Path,
    service: EmbeddingService,
    extra_ids: Iterable[str] = (),
) -> dict[str, tuple[float, ...]]:
    """Embed all canonical entities from entities.jsonl, plus ``extra_ids`` (edge endpoints
    Role 2 never extracted) under a name derived from the id."""
    names = load_entity_names(entities_file)
    for eid in extra_ids:
        names.setdefault(eid, entity_name_from_id(eid))
    return embed_entities(names, service)


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_filter: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> dict:
    """Embed entities and edges for configured datasets."""
    log_path, tee = open_run_log(dataset_filter, mode_override)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    print(f"[embeddings] log -> {log_path}", flush=True)
    print(f"[embeddings] starting at {datetime.now().isoformat(timespec='seconds')}", flush=True)

    try:
        return _run(config_path, dataset_filter, mode_override)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.write(f"[embeddings] log retained -> {log_path}\n")
        tee.close()


def _run(
    config_path: str | Path,
    dataset_filter: Optional[str],
    mode_override: Optional[str],
) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    emb_cfg = cfg.get("embedding", {})
    service = service_from_config(cfg, mode_override)
    use_names = bool(emb_cfg.get("use_entity_names", True))

    extraction_dir = Path(cfg.get("extraction_dir", "data/graph/extraction"))
    out_dir = Path(cfg.get("out_dir", "data/graph"))
    datasets = [dataset_filter] if dataset_filter else cfg.get("datasets", ["musique", "twiki", "hotpotqa"])
    auto_name = cfg.get("automatic_edges_file", "edges.jsonl")

    if service.is_mock:
        print("WARNING: mock embedding mode. These vectors are hash noise with no semantic meaning; "
              "use them for tests only, never for experiments.")

    manifest = {"mode": service.mode, "model": service.model, "mock": service.is_mock,
                "use_entity_names": use_names, "datasets": {}}

    for ds in datasets:
        ds_graph_dir = out_dir / ds
        entities_path = extraction_dir / ds / "entities.jsonl"
        oracle_path = ds_graph_dir / "oracle_edges.jsonl"
        auto_path = ds_graph_dir / auto_name

        oracle_edges = read_edges(oracle_path) if oracle_path.exists() else None
        auto_edges = read_edges(auto_path) if auto_path.exists() else None
        stats: dict = {}

        # 1. Embed canonical entities + every edge endpoint Role 2 did not extract
        names = load_entity_names(entities_path)
        if names or oracle_edges or auto_edges:
            extracted = len(names)
            for edges in (oracle_edges, auto_edges):
                for e in edges or ():
                    for eid in (e.head, e.tail):
                        names.setdefault(eid, entity_name_from_id(eid))
            out_emb_path = ds_graph_dir / "entity_embeddings.json"
            entity_embs = read_entity_embeddings(out_emb_path) if out_emb_path.exists() else {}
            missing_names = {
                eid: name for eid, name in names.items()
                if eid not in entity_embs
            }
            if out_emb_path.exists():
                print(
                    f"[{ds}] Reusing entity_embeddings.json: {len(entity_embs)} stored, "
                    f"{len(missing_names)} missing",
                    flush=True,
                )
            if missing_names:
                print(
                    f"[{ds}] Embedding {len(missing_names)} missing entities "
                    f"({len(names) - extracted} edge endpoints not in entities.jsonl)...",
                    flush=True,
                )
                new_embs = embed_entities(missing_names, service)
                entity_embs.update(new_embs)
                write_entity_embeddings(entity_embs, out_emb_path)
            elif not out_emb_path.exists():
                print(f"[{ds}] Embedding {len(names)} entities...", flush=True)
                entity_embs = embed_entities(names, service)
                write_entity_embeddings(entity_embs, out_emb_path)
            stats["entities_embedded"] = len(entity_embs)
            stats["endpoint_entities_added"] = len(names) - extracted
            stats["entity_embeddings_file"] = str(out_emb_path)
            print(f"[{ds}] Saved {len(entity_embs)} entity embeddings -> {out_emb_path}")

        # 2. Oracle and automatic edges
        for label, edges, path in (("oracle", oracle_edges, oracle_path), ("auto", auto_edges, auto_path)):
            if edges is None:
                continue
            missing = [edge for edge in edges if len(edge.embedding) == 0]
            reused = len(edges) - len(missing)
            print(
                f"[{ds}] {label} edge embeddings: {reused} reused, "
                f"{len(missing)} missing",
                flush=True,
            )
            if missing:
                print(f"[{ds}] Embedding {len(missing)} missing {label} edges...", flush=True)
                embedded_missing = populate_edge_embeddings(
                    missing, service, names if use_names else None
                )
                embedded_by_key = {
                    (edge.head, edge.relation, edge.tail, edge.support, edge.provenance): edge.embedding
                    for edge in embedded_missing
                }
                edges = [
                    edge if len(edge.embedding) else
                    Edge(
                        head=edge.head,
                        relation=edge.relation,
                        tail=edge.tail,
                        confidence=edge.confidence,
                        support=edge.support,
                        provenance=edge.provenance,
                        embedding=embedded_by_key[
                            (edge.head, edge.relation, edge.tail, edge.support, edge.provenance)
                        ],
                    )
                    for edge in edges
                ]
                write_edges(edges, path)
            stats[f"{label}_edges_embedded"] = len(edges)
            stats[f"{label}_edges_reused"] = reused
            print(f"[{ds}] Updated {len(edges)} {label} edges with vectors.", flush=True)

        manifest["datasets"][ds] = stats

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_embeddings.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared Embedding Service & Vector Store")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--dataset", default=None, help="Dataset to process (musique, twiki, hotpotqa)")
    parser.add_argument("--mode", default=None, choices=["mock", "gemini"], help="Embedding mode ('mock' or 'gemini')")
    args = parser.parse_args()
    run(args.config, args.dataset, args.mode)

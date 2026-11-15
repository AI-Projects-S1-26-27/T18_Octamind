"""Shared embedding service and vector store for knowledge graphs (Role 3).

Provides the project-wide embedding service required by docs/INTERFACE.md:
  embed(text: str) -> tuple[float, ...]

Computes:
1. Entity embeddings phi(entity): embeddings of canonical entity names, saved
   as data/graph/<dataset>/entity_embeddings.json for Role 5's heuristic h(n).
2. Edge embeddings phi(e): embeddings of 'head relation tail', saved into
   Edge.embedding for calculating edge relevance rel(e, q) = cos(phi(e), phi(q)).

Includes:
- Gemini Embedding model (text-embedding-004 / Gemini Embedding 2).
- Deterministic offline mock embedder for tests and local development.
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
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from src.graph.oracle import read_edges, write_edges
from src.traversal.types import Edge

DEFAULT_CONFIG = "configs/graph.yaml"


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

    Uses a deterministic PRNG seeded by the SHA-256 of the normalized text.
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


def gemini_embed_batch(
    texts: list[str],
    model: str = "text-embedding-004",
    client=None,
    max_attempts: int = 5,
) -> list[tuple[float, ...]]:
    """Embed a list of texts using google-genai client."""
    if not texts:
        return []

    if client is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Gemini embedding requires `pip install google-genai`") from exc

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable required for gemini embedding mode.")
        client = genai.Client(api_key=api_key)

    for attempt in range(max_attempts):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
            )
            results = []
            for item in response.embeddings:
                results.append(tuple(float(v) for v in item.values))
            return results
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)

    return []


class EmbeddingService:
    """Project-wide embedding service providing embed() and embed_many() with caching."""

    def __init__(
        self,
        mode: str = "mock",
        model: str = "text-embedding-004",
        cache_dir: str | Path = "data/cache/embeddings",
        dimension: int = 64,
        batch_size: int = 100,
        use_cache: Optional[bool] = None,
    ):
        self.mode = mode
        self.model = model if mode != "mock" else f"mock-{dimension}"
        self.cache = EmbeddingCache(cache_dir)
        self.dimension = dimension
        self.batch_size = batch_size
        self.use_cache = use_cache if use_cache is not None else True
        self._client = None

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed a single string."""
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed a batch of strings, utilizing the disk cache to minimize calls."""
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

        # If all were cached, return immediately
        if not missing_texts:
            return [r for r in results if r is not None]

        # Fetch missing in batches
        for start_idx in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start_idx : start_idx + self.batch_size]
            if self.mode == "mock":
                batch_vectors = [mock_embed(t, dim=self.dimension) for t in batch]
            else:
                batch_vectors = gemini_embed_batch(batch, model=self.model, client=self._client)

            # Store in cache and results
            for sub_idx, (vec, t) in enumerate(zip(batch_vectors, batch)):
                orig_idx = missing_indices[start_idx + sub_idx]
                results[orig_idx] = vec
                if self.use_cache:
                    self.cache.put(self.model, t, vec)

        return [r for r in results if r is not None]


def write_entity_embeddings(embeddings: dict[str, tuple[float, ...]], path: str | Path) -> None:
    """Write entity embeddings mapping to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialized = {k: list(v) for k, v in embeddings.items()}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(serialized, f)


def read_entity_embeddings(path: str | Path) -> dict[str, tuple[float, ...]]:
    """Read entity embeddings mapping from a JSON file."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: tuple(float(x) for x in v) for k, v in raw.items()}


def populate_edge_embeddings(edges: list[Edge], service: EmbeddingService) -> list[Edge]:
    """Compute and attach phi(e) = embed('head relation tail') for each edge."""
    edge_texts = [e.text for e in edges]
    vectors = service.embed_many(edge_texts)
    updated = []
    for edge, vec in zip(edges, vectors):
        updated.append(
            Edge(
                head=edge.head,
                relation=edge.relation,
                tail=edge.tail,
                confidence=edge.confidence,
                support=edge.support,
                provenance=edge.provenance,
                embedding=vec,
            )
        )
    return updated


def embed_dataset_entities(
    entities_file: str | Path,
    service: EmbeddingService,
) -> dict[str, tuple[float, ...]]:
    """Embed all canonical entities from entities.jsonl."""
    p = Path(entities_file)
    if not p.exists():
        return {}

    entity_ids = []
    names = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                entity_ids.append(it["id"])
                names.append(it.get("name", it["id"]))

    vectors = service.embed_many(names)
    return dict(zip(entity_ids, vectors))


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_filter: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> dict:
    """Embed entities and edges for configured datasets."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    emb_cfg = cfg.get("embedding", {})
    mode = mode_override or emb_cfg.get("mode", "mock")
    model = emb_cfg.get("model", "text-embedding-004")
    cache_dir = emb_cfg.get("cache_dir", "data/cache/embeddings")
    dimension = int(emb_cfg.get("dimension", 64))
    batch_size = int(emb_cfg.get("batch_size", 100))

    service = EmbeddingService(
        mode=mode,
        model=model,
        cache_dir=cache_dir,
        dimension=dimension,
        batch_size=batch_size,
    )

    extraction_dir = Path(cfg.get("extraction_dir", "data/graph/extraction"))
    out_dir = Path(cfg.get("out_dir", "data/graph"))
    datasets = [dataset_filter] if dataset_filter else cfg.get("datasets", ["musique", "twiki", "hotpotqa"])

    manifest = {"mode": mode, "model": service.model, "datasets": {}}

    for ds in datasets:
        ds_graph_dir = out_dir / ds
        entities_path = extraction_dir / ds / "entities.jsonl"
        oracle_path = ds_graph_dir / "oracle_edges.jsonl"
        auto_path = ds_graph_dir / "edges.jsonl"

        stats = {}

        # 1. Embed canonical entities
        if entities_path.exists():
            print(f"[{ds}] Embedding canonical entities from {entities_path}...")
            entity_embs = embed_dataset_entities(entities_path, service)
            out_emb_path = ds_graph_dir / "entity_embeddings.json"
            write_entity_embeddings(entity_embs, out_emb_path)
            stats["entities_embedded"] = len(entity_embs)
            stats["entity_embeddings_file"] = str(out_emb_path)
            print(f"[{ds}] Saved {len(entity_embs)} entity embeddings -> {out_emb_path}")

        # 2. Embed Oracle edges (if present)
        if oracle_path.exists():
            print(f"[{ds}] Embedding Oracle edges from {oracle_path}...")
            oracle_edges = read_edges(oracle_path)
            oracle_edges = populate_edge_embeddings(oracle_edges, service)
            write_edges(oracle_edges, oracle_path)
            stats["oracle_edges_embedded"] = len(oracle_edges)
            print(f"[{ds}] Updated {len(oracle_edges)} Oracle edges with vectors.")

        # 3. Embed Automatic edges (if present)
        if auto_path.exists():
            print(f"[{ds}] Embedding Automatic edges from {auto_path}...")
            auto_edges = read_edges(auto_path)
            auto_edges = populate_edge_embeddings(auto_edges, service)
            write_edges(auto_edges, auto_path)
            stats["auto_edges_embedded"] = len(auto_edges)
            print(f"[{ds}] Updated {len(auto_edges)} Automatic edges with vectors.")

        manifest["datasets"][ds] = stats

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

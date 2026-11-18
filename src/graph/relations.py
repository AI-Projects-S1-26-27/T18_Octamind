"""Automatic relationship discovery and graph construction (Role 3).

Discovers directed relationships between entities extracted by Role 2 from
source paragraphs, producing the automatic Semantic Knowledge Network (System S6
in the proposal).

Two extractors share one interface:
  extract(paragraph: Paragraph, mentions: list[LinkedMention]) -> list[RawRelation]

- RuleRelationExtractor: dependency-free baseline extracting connective verb
  phrases and prepositions between co-occurring entities in a sentence, or
  pairing an entity with the paragraph title entity.
- LLMRelationExtractor: prompts Gemini (temperature 0) to extract factual
  relationships between known entity IDs in the paragraph, cached per paragraph
  in data/cache/relations/ for zero-cost reproducibility.

LLM runs follow the same rules as Role 2's extraction (docs/EXTRACTION.md):
  * only a *valid* model answer is cached (an empty list is a valid answer; an API
    failure or unparseable answer is not, so re-running retries exactly those);
  * everything cached is already validated and clamped, so a re-run gives the
    same graph as the first run;
  * a run is complete when ``failed_paragraphs`` and ``invalid_responses`` are 0
    in ``data/graph/manifest_llm.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from data.loaders.base import Paragraph, QuestionRecord, read_question_records
from src.graph.entities import split_sentences
from src.graph.extract import QuotaExhausted, gemini_generate
from src.traversal.types import Edge

DEFAULT_CONFIG = "configs/graph.yaml"
PROMPT_VERSION = 2   # v2: numbered sentences + per-relation sentence index

# Punctuation to trim from relation phrases
_EDGE_PUNCT = re.compile(r"^[,\s;:()\-–—'\"\[\]]+|[,\s;:()\-–—'\"\[\]]+$")
_LEADING_PRONOUNS = re.compile(r"^(he|she|it|they|who|which)\s+", re.IGNORECASE)
_AUX_VERBS = re.compile(r"^(is|was|are|were)\s+(a|an|the)\s+", re.IGNORECASE)
_TRAILING_ARTICLES = re.compile(r"\s+(a|an|the)$", re.IGNORECASE)
_UNINFORMATIVE_WORDS = {"and", "or", "but", "also", "then", "of", "in", "to", "for", "with", "on", "at"}


@dataclass(frozen=True)
class LinkedMention:
    """A mention linked to a canonical entity_id."""
    text: str
    type: str
    paragraph_id: str
    sentence_index: int
    start: int
    end: int
    entity_id: str
    source: str = "rule"
    confidence: float = 1.0
    paragraph_sha1: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "LinkedMention":
        return cls(
            text=str(raw["text"]),
            type=str(raw.get("type", "CONCEPT")),
            paragraph_id=str(raw["paragraph_id"]),
            sentence_index=int(raw["sentence_index"]),
            start=int(raw["start"]),
            end=int(raw["end"]),
            entity_id=str(raw.get("entity_id", "")),
            source=str(raw.get("source", "rule")),
            confidence=float(raw.get("confidence", 1.0)),
            paragraph_sha1=str(raw.get("paragraph_sha1", "")),
        )


@dataclass(frozen=True)
class RawRelation:
    """One extracted relation occurrence before corpus-wide aggregation."""
    head: str
    relation: str
    tail: str
    confidence: float
    paragraph_id: str
    sentence_index: int


def clean_relation_label(text: str) -> str:
    """Clean connective text into a canonical relation label.

    E.g.: ' was married to the actress ' -> 'was married to'
          ', who directed '             -> 'directed'
          ' is located in the city of '  -> 'is located in the city of'
    """
    text = text.strip()
    text = _EDGE_PUNCT.sub("", text)
    text = _LEADING_PRONOUNS.sub("", text)
    text = _AUX_VERBS.sub(r"\1 ", text)
    text = _TRAILING_ARTICLES.sub("", text)
    cleaned = " ".join(text.split()).strip().lower()
    return cleaned


class RuleRelationExtractor:
    """Dependency-free relation extractor using sentence co-occurrence and connective text."""
    name = "rule"

    def __init__(self, max_tokens: int = 15, include_title: bool = True,
                 pair_confidence: float = 0.6, title_confidence: float = 0.5):
        self.max_tokens = max_tokens
        self.include_title = include_title
        self.pair_confidence = pair_confidence      # two entities in one sentence
        self.title_confidence = title_confidence    # entity paired with the paragraph title

    def extract(self, paragraph: Paragraph, mentions: list[LinkedMention]) -> list[RawRelation]:
        relations = []
        title_mention = next((m for m in mentions if m.sentence_index == -1), None)

        # Group mentions by sentence
        by_sentence = defaultdict(list)
        for m in mentions:
            if m.sentence_index >= 0:
                by_sentence[m.sentence_index].append(m)

        sentence_spans = split_sentences(paragraph.text)

        for s_idx, (s_start, s_end) in enumerate(sentence_spans):
            sent_ms = by_sentence.get(s_idx, [])
            # Sort mentions by their start offset in the paragraph
            sent_ms.sort(key=lambda m: m.start)

            # Case A: Two or more entities in the same sentence
            if len(sent_ms) >= 2:
                for i in range(len(sent_ms) - 1):
                    m1, m2 = sent_ms[i], sent_ms[i + 1]
                    if m1.entity_id == m2.entity_id:
                        continue
                    if m2.start >= m1.end:
                        between_text = paragraph.text[m1.end:m2.start]
                        rel_label = clean_relation_label(between_text)
                        # Check word count and ensure it is not purely empty or stop words
                        words = rel_label.split()
                        if 1 <= len(words) <= self.max_tokens and set(words) - _UNINFORMATIVE_WORDS:
                            relations.append(
                                RawRelation(
                                    head=m1.entity_id,
                                    relation=rel_label,
                                    tail=m2.entity_id,
                                    confidence=self.pair_confidence,
                                    paragraph_id=paragraph.id,
                                    sentence_index=s_idx,
                                )
                            )

            # Case B: Exactly one entity in the sentence -> pair with paragraph title
            elif len(sent_ms) == 1 and self.include_title and title_mention:
                m = sent_ms[0]
                if m.entity_id != title_mention.entity_id:
                    # Text before the entity in this sentence
                    before_text = paragraph.text[s_start:m.start]
                    rel_label = clean_relation_label(before_text)
                    words = rel_label.split()
                    if 1 <= len(words) <= self.max_tokens and set(words) - _UNINFORMATIVE_WORDS:
                        relations.append(
                            RawRelation(
                                head=title_mention.entity_id,
                                relation=rel_label,
                                tail=m.entity_id,
                                confidence=self.title_confidence,
                                paragraph_id=paragraph.id,
                                sentence_index=s_idx,
                            )
                        )

        return relations


# --------------------------------------------------------------------------- #
# LLM Extractor & Cache
# --------------------------------------------------------------------------- #
RELATION_PROMPT = """You extract factual relationships between entities from a text passage to build a knowledge graph.

Paragraph (title: {title}), one sentence per line, numbered:
{sentences}

Entities mentioned in this paragraph:
{entities}

Extract factual relationships between these entities that are stated in the text.
Return ONLY a valid JSON list. Each item is an object:
  {{"head": "<entity_id>", "relation": "<short relation label, e.g. director, birthplace, member_of>", "tail": "<entity_id>", "confidence": <float between 0.5 and 1.0>, "sentence": <number of the sentence that states it>}}

Rules:
- "head" and "tail" MUST be chosen from the provided entity IDs above.
- Do not invent relations that are not supported by the text.
- If no relations exist between these entities, return an empty list []."""

MIN_LLM_CONFIDENCE = 0.5


class InvalidResponse(ValueError):
    """The model answered, but not with a JSON list of relations."""


class RelationResponseCache:
    """Per-paragraph disk cache for validated LLM relation extractions.

    The key covers the model, prompt version, paragraph text *and* the entity ids
    offered to the model, so a Role 2 re-run that changes the entities never reuses
    relations that point at stale ids.
    """

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _key(self, model: str, paragraph_id: str, text: str, entities_key: str = "") -> str:
        payload = f"{model}\x00{PROMPT_VERSION}\x00{paragraph_id}\x00{text}\x00{entities_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, paragraph_id: str, text: str, entities_key: str = "") -> Optional[list]:
        path = self.dir / f"{self._key(model, paragraph_id, text, entities_key)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("items")
        except (json.JSONDecodeError, OSError):
            return None   # unreadable entry: treat as a miss and re-ask the model

    def put(self, model: str, paragraph_id: str, text: str, items: list, entities_key: str = "") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self._key(model, paragraph_id, text, entities_key)}.json"
        payload = {"model": model, "prompt_version": PROMPT_VERSION, "paragraph_id": paragraph_id, "items": items}
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def parse_relation_items(raw_response: str) -> list:
    """Parse the model answer into a list of raw items; raises InvalidResponse otherwise."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_response or "").strip())
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidResponse(f"not valid JSON: {exc}") from exc
    if isinstance(items, dict):
        items = items.get("relations") or items.get("items") or []
    if not isinstance(items, list):
        raise InvalidResponse("expected a JSON list of relations")
    return items


def validate_relation_items(items: list, valid_entity_ids: set, n_sentences: int) -> list:
    """Keep only well-formed items and normalise them: known entity ids, no self-loop,
    cleaned label, confidence clamped to [0.5, 1], sentence index inside the paragraph.

    Anything malformed is skipped, never raised, and the result is what gets cached,
    so a cached re-run is identical to the first run.
    """
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        head, tail, rel = it.get("head"), it.get("tail"), it.get("relation")
        if not (isinstance(head, str) and isinstance(tail, str) and rel):
            continue
        if head not in valid_entity_ids or tail not in valid_entity_ids or head == tail:
            continue
        label = clean_relation_label(str(rel))
        if not label:
            continue
        try:
            conf = float(it.get("confidence", 0.9))
        except (TypeError, ValueError):
            conf = 0.9
        if not math.isfinite(conf):
            conf = 0.9
        conf = max(MIN_LLM_CONFIDENCE, min(1.0, conf))
        try:
            sent = int(it.get("sentence", 0))
        except (TypeError, ValueError):
            sent = 0
        if not 0 <= sent < max(n_sentences, 1):
            sent = 0
        out.append({"head": head, "relation": label, "tail": tail, "confidence": conf, "sentence": sent})
    return out


class LLMRelationExtractor:
    """LLM-based relation extractor targeting Gemini Flash-Lite with local caching.

    ``generate`` is a plain ``prompt -> str`` callable (``extract.gemini_generate`` in
    real runs, a fake in tests). Counters after a run: ``calls``, ``cache_hits``,
    ``failed_paragraphs`` (API errors), ``invalid_responses`` (unusable answers).
    """
    name = "llm"

    def __init__(
        self,
        generate: Callable[[str], str],
        model: str,
        cache_dir: str | Path,
    ):
        self.generate = generate
        self.model = model
        self.cache = RelationResponseCache(cache_dir)
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.invalid_responses = 0
        self.failed_paragraphs: list = []
        self.quota_exhausted = False

    @staticmethod
    def _entities_key(entity_ids: Iterable[str]) -> str:
        return hashlib.sha256("\x00".join(sorted(entity_ids)).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _to_raw(paragraph: Paragraph, items: list) -> list[RawRelation]:
        return [
            RawRelation(head=i["head"], relation=i["relation"], tail=i["tail"], confidence=i["confidence"],
                        paragraph_id=paragraph.id, sentence_index=i.get("sentence", 0))
            for i in items
        ]

    def extract(self, paragraph: Paragraph, mentions: list[LinkedMention]) -> list[RawRelation]:
        # Collect distinct entities in this paragraph
        entities = {}
        for m in mentions:
            entities[m.entity_id] = m.text
        ekey = self._entities_key(entities)

        cached = self.cache.get(self.model, paragraph.id, paragraph.text, ekey)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return self._to_raw(paragraph, cached)

        if len(entities) < 2:
            self.cache.put(self.model, paragraph.id, paragraph.text, [], ekey)
            return []

        spans = split_sentences(paragraph.text)
        sentences = "\n".join(f"[{i}] {paragraph.text[s:e]}" for i, (s, e) in enumerate(spans))
        ent_str = "\n".join(f"- {eid}: {name}" for eid, name in sorted(entities.items()))
        prompt = RELATION_PROMPT.format(title=paragraph.title, sentences=sentences, entities=ent_str)

        with self._lock:
            self.calls += 1
        try:
            raw_response = self.generate(prompt)
        except QuotaExhausted:
            with self._lock:
                self.quota_exhausted = True
                self.failed_paragraphs.append(paragraph.id)
            return []                       # not cached: a later run retries it
        except Exception:  # noqa: BLE001 - API/network failure, retried on the next run
            with self._lock:
                self.failed_paragraphs.append(paragraph.id)
            return []                       # not cached

        try:
            items = parse_relation_items(raw_response)
        except InvalidResponse:
            with self._lock:
                self.invalid_responses += 1
            return []                       # not cached

        valid = validate_relation_items(items, set(entities), len(spans))
        self.cache.put(self.model, paragraph.id, paragraph.text, valid, ekey)   # [] is a valid answer
        return self._to_raw(paragraph, valid)

    def reset_counters(self) -> None:
        with self._lock:
            self.calls = self.cache_hits = self.invalid_responses = 0
            self.failed_paragraphs = []
            self.quota_exhausted = False

    def summary(self) -> dict:
        return {
            "model": self.model,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "failed_paragraphs": list(self.failed_paragraphs),
            "invalid_responses": self.invalid_responses,
            "quota_exhausted": self.quota_exhausted,
        }


def aggregate_relations(raw_relations: Iterable[RawRelation]) -> list[Edge]:
    """Aggregate raw relation triples across paragraphs into unique Edge objects."""
    aggregated: dict[tuple[str, str, str], dict] = {}

    for r in raw_relations:
        key = (r.head, r.relation, r.tail)
        if key not in aggregated:
            aggregated[key] = {
                "head": r.head,
                "relation": r.relation,
                "tail": r.tail,
                "confidence": r.confidence,
                "support_sentences": {(r.paragraph_id, r.sentence_index)},
                "provenance": {r.paragraph_id},
            }
        else:
            # Aggregate confidence: take maximum confidence observed
            aggregated[key]["confidence"] = max(aggregated[key]["confidence"], r.confidence)
            aggregated[key]["support_sentences"].add((r.paragraph_id, r.sentence_index))
            aggregated[key]["provenance"].add(r.paragraph_id)

    result = []
    for item in sorted(aggregated.values(), key=lambda x: (x["head"], x["relation"], x["tail"])):
        result.append(
            Edge(
                head=item["head"],
                relation=item["relation"],
                tail=item["tail"],
                confidence=round(item["confidence"], 4),
                support=len(item["support_sentences"]),
                provenance=tuple(sorted(item["provenance"])),
                embedding=(),
            )
        )

    return result


def load_dataset_mentions(mentions_path: str | Path) -> dict[str, list[LinkedMention]]:
    """Load mentions.jsonl mapped by paragraph_id."""
    mentions_by_pid = defaultdict(list)
    p = Path(mentions_path)
    if not p.exists():
        return mentions_by_pid
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                m = LinkedMention.from_dict(json.loads(line))
                if m.entity_id:
                    mentions_by_pid[m.paragraph_id].append(m)
    return mentions_by_pid


def build_automatic_graph(
    records: list[QuestionRecord],
    mentions_by_pid: dict[str, list[LinkedMention]],
    extractor,
    min_confidence: float = 0.0,
    workers: int = 1,
) -> list[Edge]:
    """Extract relations across all unique paragraphs in records and aggregate into edges.

    Relations below ``min_confidence`` are dropped before aggregation. ``workers`` > 1
    runs paragraphs in parallel (useful for the LLM extractor); the result does not
    depend on it because aggregation sorts its output.
    """
    # Collect unique paragraphs
    unique_paragraphs = {}
    for r in records:
        for p in r.paragraphs:
            unique_paragraphs.setdefault((p.id, p.text), p)

    tasks = [(p, mentions_by_pid[p.id]) for p in unique_paragraphs.values() if mentions_by_pid.get(p.id)]
    print(f"  starting relationship extraction: {len(tasks)} paragraphs, {workers} workers", flush=True)
    if not tasks:
        print("  no paragraphs have mentions; check extraction output", flush=True)
        return []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            found_lists = []
            for index, found in enumerate(pool.map(lambda t: extractor.extract(*t), tasks), start=1):
                found_lists.append(found)
                if index % 10 == 0 or index == len(tasks):
                    print(f"  {index}/{len(tasks)} paragraphs", flush=True)
    else:
        found_lists = []
        for index, task in enumerate(tasks, start=1):
            found_lists.append(extractor.extract(*task))
            if index % 10 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} paragraphs", flush=True)

    raw_relations = [r for found in found_lists for r in found if r.confidence >= min_confidence]
    return aggregate_relations(raw_relations)


def build_extractor(cfg: dict, mode: str):
    """Build the extractor for ``mode`` ('rule' or 'llm') from configs/graph.yaml."""
    if mode == "rule":
        rel_cfg = cfg.get("relation", {})
        return RuleRelationExtractor(
            max_tokens=rel_cfg.get("max_tokens_between_entities", 15),
            include_title=rel_cfg.get("include_title_relations", True),
            pair_confidence=float(rel_cfg.get("pair_confidence", 0.6)),
            title_confidence=float(rel_cfg.get("title_confidence", 0.5)),
        )
    if mode == "llm":
        llm = cfg.get("llm") or {}
        model = llm["model"]
        generate = gemini_generate(
            model, float(llm.get("temperature", 0.0)),
            thinking_level=llm.get("thinking_level"),
            requests_per_minute=llm.get("requests_per_minute"),
            max_attempts=int(llm.get("max_attempts", 8)),
        )
        return LLMRelationExtractor(generate, model=model, cache_dir=llm.get("cache_dir", "data/cache/relations"))
    raise ValueError(f"unknown extractor mode {mode!r}; choose 'rule' or 'llm'")


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_filter: Optional[str] = None,
    mode: Optional[str] = None,
    out_name: Optional[str] = None,
) -> dict:
    """Run automatic relationship discovery and graph construction."""
    print(f"[relations] starting (config={config_path})", flush=True)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    extraction_dir = Path(cfg.get("extraction_dir", "data/graph/extraction"))
    out_dir = Path(cfg.get("out_dir", "data/graph"))
    datasets = [dataset_filter] if dataset_filter else cfg.get("datasets", ["musique", "twiki", "hotpotqa"])
    out_name = out_name or cfg.get("automatic_edges_file", "edges.jsonl")

    extractor_mode = mode or cfg.get("extractor", "rule")
    extractor = build_extractor(cfg, extractor_mode)
    rel_cfg = cfg.get("relation", {})
    min_confidence = float(rel_cfg.get("min_confidence", 0.0))
    workers = int((cfg.get("llm") or {}).get("workers", 1)) if extractor_mode == "llm" else 1
    print(f"[relations] mode={extractor_mode}, datasets={','.join(datasets)}, workers={workers}", flush=True)

    manifest = {"config": cfg, "mode": extractor_mode, "datasets": {}}

    from src.graph.oracle import write_edges

    for ds in datasets:
        in_path = processed_dir / f"{ds}_dev.jsonl"
        mentions_path = extraction_dir / ds / "mentions.jsonl"
        print(f"[{ds}] loading records and mentions", flush=True)
        if not in_path.exists() or not mentions_path.exists():
            print(f"[{ds}] missing input files ({in_path} or {mentions_path}).", flush=True)
            continue

        records = read_question_records(in_path)
        mentions_by_pid = load_dataset_mentions(mentions_path)

        if isinstance(extractor, LLMRelationExtractor):
            extractor.reset_counters()
        edges = build_automatic_graph(records, mentions_by_pid, extractor, min_confidence, workers)

        out_path = out_dir / ds / out_name
        write_edges(edges, out_path)

        entities = {e.head for e in edges} | {e.tail for e in edges}
        stats = {
            "questions": len(records),
            "edges": len(edges),
            "unique_entities": len(entities),
            "avg_support": round(sum(e.support for e in edges) / max(len(edges), 1), 2),
            "out_file": str(out_path),
        }
        if isinstance(extractor, LLMRelationExtractor):
            stats["llm"] = extractor.summary()
            stats["complete"] = not stats["llm"]["failed_paragraphs"] and not stats["llm"]["invalid_responses"]
            if not stats["complete"]:
                print(f"[{ds}] WARNING: incomplete run ({len(stats['llm']['failed_paragraphs'])} failed paragraphs, "
                      f"{stats['llm']['invalid_responses']} invalid answers); re-run the same command to resume from the cache.")
        manifest["datasets"][ds] = stats
        print(f"[{ds}] Automatic Graph ({extractor.name}): {len(edges)} edges, {len(entities)} entities -> {out_path}")

    # Write manifest summary
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{extractor_mode}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[relations] wrote manifest -> {manifest_path}", flush=True)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Relationship Discovery & Graph Construction")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--dataset", default=None, help="Dataset to process (musique, twiki, hotpotqa)")
    parser.add_argument("--mode", default=None, choices=["rule", "llm"],
                        help="Extractor mode (default: `extractor` in the config)")
    parser.add_argument("--out-name", default=None, help="Edge file name inside data/graph/<dataset>/ (default edges.jsonl)")
    args = parser.parse_args()
    run(args.config, args.dataset, args.mode, args.out_name)

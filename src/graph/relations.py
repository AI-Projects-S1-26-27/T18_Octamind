"""Automatic relationship discovery and graph construction (Role 3).

Discovers directed relationships between entities extracted by Role 2 from
source paragraphs, producing the automatic Semantic Knowledge Network (System S6
in the proposal).

Two extractors share one interface:
  extract(paragraph: Paragraph, mentions: list[Mention]) -> list[RawRelation]

- RuleRelationExtractor: dependency-free baseline extracting connective verb
  phrases and prepositions between co-occurring entities in a sentence, or
  pairing an entity with the paragraph title entity.
- LLMRelationExtractor: prompts Gemini (temperature 0) to extract factual
  relationships between known entity IDs in the paragraph, cached per paragraph
  in data/cache/relations/ for zero-cost reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from data.loaders.base import Paragraph, QuestionRecord, read_question_records
from src.graph.entities import Mention, split_sentences
from src.traversal.types import Edge

DEFAULT_CONFIG = "configs/graph.yaml"
PROMPT_VERSION = 1

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

    def __init__(self, max_tokens: int = 15, include_title: bool = True):
        self.max_tokens = max_tokens
        self.include_title = include_title

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
            sent_text = paragraph.text[s_start:s_end]
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
                                    confidence=0.6,
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
                                confidence=0.5,
                                paragraph_id=paragraph.id,
                                sentence_index=s_idx,
                            )
                        )

        return relations


# --------------------------------------------------------------------------- #
# LLM Extractor & Cache
# --------------------------------------------------------------------------- #
RELATION_PROMPT = """You extract factual relationships between entities from a text passage to build a knowledge graph.

Paragraph (title: {title}):
{text}

Entities mentioned in this paragraph:
{entities}

Extract factual relationships between these entities that are stated in the text.
Return ONLY a valid JSON list. Each item is an object:
  {{"head": "<entity_id>", "relation": "<short relation label, e.g. director, birthplace, member_of>", "tail": "<entity_id>", "confidence": <float between 0.5 and 1.0>}}

Rules:
- "head" and "tail" MUST be chosen from the provided entity IDs above.
- Do not invent relations that are not supported by the text.
- If no relations exist between these entities, return an empty list []."""


class RelationResponseCache:
    """Per-paragraph disk cache for LLM relation extractions."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _key(self, model: str, paragraph_id: str, text: str) -> str:
        payload = f"{model}\x00{PROMPT_VERSION}\x00{paragraph_id}\x00{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, paragraph_id: str, text: str) -> Optional[list]:
        path = self.dir / f"{self._key(model, paragraph_id, text)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("items")

    def put(self, model: str, paragraph_id: str, text: str, items: list) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self._key(model, paragraph_id, text)}.json"
        payload = {"model": model, "prompt_version": PROMPT_VERSION, "paragraph_id": paragraph_id, "items": items}
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class LLMRelationExtractor:
    """LLM-based relation extractor targeting Gemini Flash-Lite with local caching."""
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

    def extract(self, paragraph: Paragraph, mentions: list[LinkedMention]) -> list[RawRelation]:
        cached = self.cache.get(self.model, paragraph.id, paragraph.text)
        if cached is not None:
            return [RawRelation(paragraph_id=paragraph.id, sentence_index=0, **item) for item in cached]

        # Collect distinct entities in this paragraph
        entities = {}
        for m in mentions:
            entities[m.entity_id] = m.text

        if len(entities) < 2:
            self.cache.put(self.model, paragraph.id, paragraph.text, [])
            return []

        ent_str = "\n".join(f"- {eid}: {name}" for eid, name in sorted(entities.items()))
        prompt = RELATION_PROMPT.format(title=paragraph.title, text=paragraph.text, entities=ent_str)

        try:
            raw_response = self.generate(prompt)
            # Strip markdown fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_response or "").strip())
            items = json.loads(cleaned)
            if not isinstance(items, list):
                items = items.get("relations") or items.get("items") or []
        except Exception:
            items = []

        valid_eids = set(entities)
        results = []
        cache_items = []
        for it in items:
            h = it.get("head")
            r = it.get("relation")
            t = it.get("tail")
            c = float(it.get("confidence", 0.9))
            if h in valid_eids and t in valid_eids and h != t and r:
                r_clean = clean_relation_label(str(r))
                if r_clean:
                    results.append(
                        RawRelation(
                            head=h,
                            relation=r_clean,
                            tail=t,
                            confidence=max(0.5, min(1.0, c)),
                            paragraph_id=paragraph.id,
                            sentence_index=0,
                        )
                    )
                    cache_items.append({"head": h, "relation": r_clean, "tail": t, "confidence": c})

        self.cache.put(self.model, paragraph.id, paragraph.text, cache_items)
        return results


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
    mentions_by_pid: dict[str, list[Mention]],
    extractor,
) -> list[Edge]:
    """Extract relations across all unique paragraphs in records and aggregate into edges."""
    # Collect unique paragraphs
    unique_paragraphs = {}
    for r in records:
        for p in r.paragraphs:
            unique_paragraphs.setdefault((p.id, p.text), p)

    paragraphs = list(unique_paragraphs.values())
    raw_relations = []

    for p in paragraphs:
        ms = mentions_by_pid.get(p.id, [])
        if ms:
            found = extractor.extract(p, ms)
            raw_relations.extend(found)

    return aggregate_relations(raw_relations)


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_filter: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """Run automatic relationship discovery and graph construction."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    extraction_dir = Path(cfg.get("extraction_dir", "data/graph/extraction"))
    out_dir = Path(cfg.get("out_dir", "data/graph"))
    datasets = [dataset_filter] if dataset_filter else cfg.get("datasets", ["musique", "twiki", "hotpotqa"])

    extractor_mode = mode or cfg.get("extractor", "rule")
    if extractor_mode == "rule":
        rel_cfg = cfg.get("relation", {})
        extractor = RuleRelationExtractor(
            max_tokens=rel_cfg.get("max_tokens_between_entities", 15),
            include_title=rel_cfg.get("include_title_relations", True),
        )
    else:
        raise NotImplementedError(f"Extractor mode '{extractor_mode}' requires API integration.")

    manifest = {"config": cfg, "mode": extractor_mode, "datasets": {}}

    from src.graph.oracle import write_edges

    for ds in datasets:
        in_path = processed_dir / f"{ds}_dev.jsonl"
        mentions_path = extraction_dir / ds / "mentions.jsonl"
        if not in_path.exists() or not mentions_path.exists():
            print(f"Skipping {ds}: missing input files ({in_path} or {mentions_path}).")
            continue

        records = read_question_records(in_path)
        mentions_by_pid = load_dataset_mentions(mentions_path)

        edges = build_automatic_graph(records, mentions_by_pid, extractor)

        out_path = out_dir / ds / "edges.jsonl"
        write_edges(edges, out_path)

        entities = {e.head for e in edges} | {e.tail for e in edges}
        stats = {
            "questions": len(records),
            "edges": len(edges),
            "unique_entities": len(entities),
            "avg_support": round(sum(e.support for e in edges) / max(len(edges), 1), 2),
            "out_file": str(out_path),
        }
        manifest["datasets"][ds] = stats
        print(f"[{ds}] Automatic Graph ({extractor.name}): {len(edges)} edges, {len(entities)} entities -> {out_path}")

    # Write manifest summary
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{extractor_mode}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Relationship Discovery & Graph Construction")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--dataset", default=None, help="Dataset to process (musique, twiki, hotpotqa)")
    parser.add_argument("--mode", default="rule", choices=["rule", "llm"], help="Extractor mode")
    args = parser.parse_args()
    run(args.config, args.dataset, args.mode)

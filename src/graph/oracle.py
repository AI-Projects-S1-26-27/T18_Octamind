"""Oracle graph construction from gold reasoning chains (Role 3).

Produces the oracle graph (System S7 in the proposal) by translating gold
reasoning chains (MuSiQue and 2Wiki) into canonical Edge objects conforming to
docs/INTERFACE.md and src/traversal/types.py.

This provides the upper-bound benchmark for RQ4 (separating graph-construction
errors from traversal errors) and directly unblocks Role 5 traversal testing.

Gold step formats handled
-------------------------
* 2Wiki: explicit ``head`` / ``relation`` / ``tail``.
* MuSiQue, three forms of ``GoldStep.relation``:
    - ``"CIMI-FM >> licensed to broadcast"``  entity >> relation (first hops)
    - ``"#1 >> country of citizenship"``      earlier answer >> relation
    - ``"when did the carnival take place in #1"``  free-text sub-question; every
      ``#N`` is replaced by the answer of step N, and every referenced step gets
      its own edge into this step's tail (merge steps depend on several steps).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import yaml

from data.loaders.base import QuestionRecord, read_question_records
from src.graph.entities import entity_id, normalize_name
from src.traversal.types import Edge

DEFAULT_CONFIG = "configs/graph.yaml"
_ARROW_PATTERN = re.compile(r"^.*?>>\s*", re.DOTALL)          # any "<left> >> " prefix
_PLACEHOLDER_PATTERN = re.compile(r"#(\d+)")
_PLACEHOLDER_ONLY = re.compile(r"^#\d+$")
_PID_HASH_SUFFIX = re.compile(r"#[0-9a-f]{8}(?:-\d+)?$")      # added by Role 1 for ambiguous titles


def clean_relation(raw_relation: str) -> str:
    """Normalize a relation string: drop an ``<entity or #N> >> `` prefix and excess whitespace."""
    cleaned = _ARROW_PATTERN.sub("", raw_relation, count=1).strip()
    return cleaned if cleaned else raw_relation.strip()


def title_from_pid(pid: str) -> str:
    """Title part of a paragraph id ``<dataset>::<title>[#<hash>]``.

    Only Role 1's trailing ``#<8 hex>[-n]`` suffix is removed, so titles that
    themselves contain ``#`` (``C# (programming language)``) stay intact.
    """
    title = pid.split("::", 1)[1] if "::" in pid else pid
    return _PID_HASH_SUFFIX.sub("", title)


def _substitute_placeholders(text: str, step_tails: dict[str, str]) -> str:
    return _PLACEHOLDER_PATTERN.sub(lambda m: step_tails.get(m.group(0), m.group(0)), text)


def resolve_step_endpoints(
    step,
    step_tails: dict[str, str],
    record: QuestionRecord,
) -> tuple[list[str], str, Optional[str], tuple[str, ...]]:
    """Resolve ``(head_names, relation, tail_name, provenance)`` for one GoldStep.

    ``head_names`` is a list: a MuSiQue merge step ("who is older, #1 or #2")
    depends on several earlier answers, and each one gets an edge to the tail.
    It is empty when no head can be found (the step is then skipped).
    """
    tail = step.tail.strip() if step.tail else None
    explicit_head = step.head.strip() if step.head else None
    raw = (step.relation or "").strip()

    heads: list[str] = []
    if explicit_head:                                         # 2Wiki
        heads = [explicit_head]
        rel = clean_relation(raw)
    elif ">>" in raw:                                         # "X >> rel" or "#N >> rel"
        left = raw.split(">>", 1)[0].strip()
        rel = clean_relation(raw)
        if _PLACEHOLDER_PATTERN.search(left):
            marks = list(dict.fromkeys(m.group(0) for m in _PLACEHOLDER_PATTERN.finditer(left)))
            heads = [step_tails[p] for p in marks if p in step_tails]
        elif left:
            heads = [left]
    else:                                                     # free-text sub-question
        marks = list(dict.fromkeys(m.group(0) for m in _PLACEHOLDER_PATTERN.finditer(raw)))
        heads = [step_tails[p] for p in marks if p in step_tails]
        rel = _substitute_placeholders(raw, step_tails).strip().rstrip("?").strip()

    if not heads and step.support_paragraph_ids:
        # Last resort: the subject (title) of the first supporting paragraph.
        pid = step.support_paragraph_ids[0]
        titles = {p.id: p.title for p in record.paragraphs}
        heads = [titles.get(pid) or title_from_pid(pid)]

    heads = [h.strip() for h in dict.fromkeys(heads) if h and h.strip()]
    return heads, rel, tail, step_provenance(step, record, heads, tail)


def step_provenance(step, record: QuestionRecord, heads: list[str], tail: Optional[str]) -> tuple[str, ...]:
    """Paragraph ids supporting a gold step.

    Uses the step's own support ids when the dataset has them (MuSiQue). 2Wiki has
    none, so take the gold paragraphs whose title is the head or the tail; only if
    no title matches fall back to all gold paragraphs of the question.
    """
    if step.support_paragraph_ids:
        return tuple(dict.fromkeys(step.support_paragraph_ids))
    names = {normalize_name(n) for n in [*heads, tail] if n}
    gold = tuple(dict.fromkeys(record.gold_evidence.paragraph_ids))
    titles = {p.id: normalize_name(p.title) for p in record.paragraphs}
    matched = tuple(pid for pid in gold if titles.get(pid) in names)
    return matched or gold


def extract_record_oracle_edges(record: QuestionRecord, stats: Optional[Counter] = None) -> list[Edge]:
    """Extract gold edges from a single QuestionRecord.

    ``stats`` (optional) counts ``steps``, ``edges`` and the reasons steps were
    skipped (``skip_no_head``, ``skip_no_tail``, ``skip_no_relation``,
    ``skip_self_loop``), so a run shows how much of the gold chain survived.
    """
    stats = stats if stats is not None else Counter()
    edges: list[Edge] = []
    step_tails: dict[str, str] = {}

    for step in record.gold_chain:
        stats["steps"] += 1
        heads, rel, tail_name, pids = resolve_step_endpoints(step, step_tails, record)

        # Remember this step's tail for later "#N" references.
        if tail_name:
            step_tails[f"#{step.step}"] = tail_name

        if not tail_name:
            stats["skip_no_tail"] += 1
            continue
        if not heads:
            stats["skip_no_head"] += 1
            continue
        if not rel:
            stats["skip_no_relation"] += 1
            continue

        t_id = entity_id(normalize_name(tail_name))
        for head_name in heads:
            h_id = entity_id(normalize_name(head_name))
            if not h_id or not t_id or h_id == t_id:          # empty slug or self-loop
                stats["skip_self_loop"] += 1
                continue
            edges.append(Edge(head=h_id, relation=rel, tail=t_id, confidence=1.0, support=1,
                              provenance=pids, embedding=()))
            stats["edges"] += 1

    return edges


def build_oracle_graph(records: Iterable[QuestionRecord], stats: Optional[Counter] = None) -> list[Edge]:
    """Aggregate gold edges across records into a deduplicated list of unique Edge objects.

    When the same (head, relation, tail) appears across multiple questions/paragraphs,
    support is incremented and provenance paragraph IDs are unioned.
    """
    aggregated: dict[tuple[str, str, str], dict] = {}

    for record in records:
        for edge in extract_record_oracle_edges(record, stats):
            key = (edge.head, edge.relation, edge.tail)
            if key not in aggregated:
                aggregated[key] = {
                    "head": edge.head,
                    "relation": edge.relation,
                    "tail": edge.tail,
                    "confidence": edge.confidence,
                    "support": 1,
                    "provenance": set(edge.provenance),
                }
            else:
                aggregated[key]["support"] += 1
                aggregated[key]["provenance"].update(edge.provenance)

    result = []
    for item in sorted(aggregated.values(), key=lambda x: (x["head"], x["relation"], x["tail"])):
        result.append(
            Edge(
                head=item["head"],
                relation=item["relation"],
                tail=item["tail"],
                confidence=item["confidence"],
                support=item["support"],
                provenance=tuple(sorted(item["provenance"])),
                embedding=(),
            )
        )

    return result


def edge_to_dict(edge: Edge) -> dict:
    """Serialize Edge to dict for JSON Lines persistence."""
    return {
        "head": edge.head,
        "relation": edge.relation,
        "tail": edge.tail,
        "confidence": edge.confidence,
        "support": edge.support,
        "provenance": list(edge.provenance),
        "embedding": list(edge.embedding),
    }


def edge_from_dict(raw: dict) -> Edge:
    """Deserialize Edge from dict."""
    return Edge(
        head=str(raw["head"]),
        relation=str(raw["relation"]),
        tail=str(raw["tail"]),
        confidence=float(raw.get("confidence", 1.0)),
        support=int(raw.get("support", 1)),
        provenance=tuple(raw.get("provenance") or ()),
        embedding=tuple(raw.get("embedding") or ()),
    )


def write_edges(edges: list[Edge], path: str | Path) -> None:
    """Write list of edges to a JSON Lines file, atomically.

    The data goes to a temporary file that replaces ``path`` only once complete, so a
    crash (or a caller handing in a truncated list) never leaves a half-written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.part")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for edge in edges:
                f.write(json.dumps(edge_to_dict(edge), ensure_ascii=False) + "\n")
        tmp.replace(p)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_edges(path: str | Path) -> list[Edge]:
    """Read edges from a JSON Lines file."""
    p = Path(path)
    if not p.exists():
        return []
    edges = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                edges.append(edge_from_dict(json.loads(line)))
    return edges


def run(config_path: str | Path = DEFAULT_CONFIG, dataset_filter: Optional[str] = None) -> dict:
    """Run oracle graph construction according to config."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    out_dir = Path(cfg.get("out_dir", "data/graph"))
    datasets = [dataset_filter] if dataset_filter else cfg.get("datasets", ["musique", "twiki", "hotpotqa"])

    manifest = {"config": cfg, "datasets": {}}

    for ds in datasets:
        in_path = processed_dir / f"{ds}_dev.jsonl"
        if not in_path.exists():
            print(f"Skipping {ds}: {in_path} does not exist.")
            continue

        records = read_question_records(in_path)
        if not any(r.gold_chain for r in records):
            # e.g. HotpotQA: no gold reasoning chain, so there is no oracle graph (S7).
            print(f"Skipping {ds}: its records have no gold reasoning chain (no oracle graph).")
            continue

        stats_counter: Counter = Counter()
        edges = build_oracle_graph(records, stats_counter)

        out_path = out_dir / ds / "oracle_edges.jsonl"
        write_edges(edges, out_path)

        entities = {e.head for e in edges} | {e.tail for e in edges}
        stats = {
            "questions": len(records),
            "gold_steps": stats_counter["steps"],
            "gold_steps_skipped": {k: v for k, v in sorted(stats_counter.items()) if k.startswith("skip_")},
            "edges": len(edges),
            "unique_entities": len(entities),
            "avg_support": round(sum(e.support for e in edges) / max(len(edges), 1), 2),
            "out_file": str(out_path),
        }
        manifest["datasets"][ds] = stats
        print(f"[{ds}] Oracle Graph: {len(edges)} edges, {len(entities)} entities from "
              f"{stats['gold_steps']} gold steps (skipped: {stats['gold_steps_skipped'] or 'none'}) -> {out_path}")

    # Write manifest summary
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "oracle_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct Oracle Graphs from gold reasoning chains")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--dataset", default=None, help="Specific dataset to build (musique, twiki, hotpotqa)")
    args = parser.parse_args()
    run(args.config, args.dataset)

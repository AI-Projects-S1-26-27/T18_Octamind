"""Oracle graph construction from gold reasoning chains (Role 3).

Produces the oracle graph (System S7 in the proposal) by translating gold
reasoning chains (MuSiQue and 2Wiki) into canonical Edge objects conforming to
docs/INTERFACE.md and src/traversal/types.py.

This provides the upper-bound benchmark for RQ4 (separating graph-construction
errors from traversal errors) and directly unblocks Role 5 traversal testing.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import yaml

from data.loaders.base import QuestionRecord, read_question_records
from src.graph.entities import entity_id, normalize_name
from src.traversal.types import Edge

DEFAULT_CONFIG = "configs/graph.yaml"
_ARROW_PATTERN = re.compile(r"^#\d+\s*>>\s*")
_PLACEHOLDER_PATTERN = re.compile(r"#\d+")


def clean_relation(raw_relation: str) -> str:
    """Normalize relation string: strip step prefixes like '#1 >> ' and excess whitespace."""
    cleaned = _ARROW_PATTERN.sub("", raw_relation).strip()
    return cleaned if cleaned else raw_relation.strip()


def resolve_step_endpoints(
    step,
    step_tails: dict[str, str],
    record: QuestionRecord,
) -> tuple[Optional[str], str, Optional[str], tuple[str, ...]]:
    """Resolve (head_name, relation, tail_name, provenance) for one GoldStep.

    In 2Wiki, head and tail are explicitly provided.
    In MuSiQue:
      - Head may be None. If so, a '#N' placeholder in the relation string is
        resolved to the tail of step N.
      - If still None (typically step 1), the title of the first supporting
        paragraph serves as the head.
    """
    tail = step.tail.strip() if step.tail else None
    head = step.head.strip() if step.head else None

    # Resolve MuSiQue placeholders or paragraph title for head
    if not head:
        placeholders = _PLACEHOLDER_PATTERN.findall(step.relation)
        if placeholders and placeholders[0] in step_tails:
            head = step_tails[placeholders[0]]
        elif ">>" in step.relation:
            left = step.relation.split(">>", 1)[0].strip()
            if not _PLACEHOLDER_PATTERN.match(left):
                head = left
        if not head and step.support_paragraph_ids:
            # Fallback to the subject of the supporting paragraph
            pid = step.support_paragraph_ids[0]
            # Paragraph ID format: <dataset>::<title>[#<hash>]
            title = pid.split("::", 1)[1].split("#", 1)[0]
            head = title

    rel = clean_relation(step.relation)
    pids = tuple(dict.fromkeys(step.support_paragraph_ids or record.gold_evidence.paragraph_ids))
    return head, rel, tail, pids


def extract_record_oracle_edges(record: QuestionRecord) -> list[Edge]:
    """Extract gold edges from a single QuestionRecord."""
    edges = []
    step_tails = {}

    for step in record.gold_chain:
        head_name, rel, tail_name, pids = resolve_step_endpoints(step, step_tails, record)

        # Store this step's tail for subsequent step resolution (#1, #2, etc.)
        if tail_name:
            step_tails[f"#{step.step}"] = tail_name

        if not head_name or not tail_name or not rel:
            continue

        h_id = entity_id(normalize_name(head_name))
        t_id = entity_id(normalize_name(tail_name))

        # Ignore empty slugs or self-loops
        if not h_id or not t_id or h_id == t_id:
            continue

        edges.append(
            Edge(
                head=h_id,
                relation=rel,
                tail=t_id,
                confidence=1.0,
                support=1,
                provenance=pids,
                embedding=(),
            )
        )

    return edges


def build_oracle_graph(records: Iterable[QuestionRecord]) -> list[Edge]:
    """Aggregate gold edges across records into a deduplicated list of unique Edge objects.

    When the same (head, relation, tail) appears across multiple questions/paragraphs,
    support is incremented and provenance paragraph IDs are unioned.
    """
    aggregated: dict[tuple[str, str, str], dict] = {}

    for record in records:
        for edge in extract_record_oracle_edges(record):
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
    """Write list of edges to a JSON Lines file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge_to_dict(edge), ensure_ascii=False) + "\n")


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
        edges = build_oracle_graph(records)

        out_path = out_dir / ds / "oracle_edges.jsonl"
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
        print(f"[{ds}] Oracle Graph: {len(edges)} edges, {len(entities)} entities -> {out_path}")

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


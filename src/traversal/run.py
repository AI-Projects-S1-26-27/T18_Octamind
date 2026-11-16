"""Run Role 5 traversal + context selection over sampled questions.

    python -m src.traversal.run --dataset musique --kind oracle            # S7
    python -m src.traversal.run --dataset musique --kind auto --W 1.5 --beam-k 5 --budget 1000   # S6
    python -m src.traversal.run --dataset musique --kind auto --limit 20 --tag smoke

Inputs  : data/graph/<dataset>/ (Role 3), data/query/<dataset>_dev.jsonl (Role 4, seeds + q_emb),
          data/processed/<dataset>_dev.jsonl (Role 1, gold paragraphs).
Outputs : results/traversal/<dataset>_<kind>_<tag>.jsonl   one line per question (selected edges + metrics)
          results/traversal/<dataset>_<kind>_<tag>_summary.json   means overall and per hop level

Every setting is a CLI flag or a line in configs/traversal.yaml, so each ablation is one command.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from .search import traverse
from .select import select_context
from .types import Config, Graph


def simple_token_count(text: str) -> int:
    """PLACEHOLDER tokenizer (words + punctuation marks) until the team's shared `count_tokens` exists.
    All systems must use the same tokenizer, so replace this with the shared one (see INTERFACE.md)."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _readable(entity_id: str) -> str:
    return entity_id[4:].replace("_", " ") if entity_id.startswith("ent:") else entity_id


def edge_text_readable(edge) -> str:
    """Edge as it would appear in a prompt: readable entity names instead of `ent:` ids."""
    return f"{_readable(edge.head)} {edge.relation} {_readable(edge.tail)}"


def evidence_metrics(selected: list, gold_ids) -> dict:
    """Paragraph-level evidence recall / precision (Section 9): Ev(H) = provenance of the selected edges."""
    gold = set(gold_ids)
    found = {p for e in selected for p in e.provenance}
    hit = len(found & gold)
    return {"evidence_recall": hit / len(gold) if gold else None,
            "evidence_precision": hit / len(found) if found else 0.0,
            "evidence_paragraphs": len(found)}


def run_question(graph: Graph, seeds: list, q_emb, gold_ids, cfg: Config,
                 token_counter: Callable[[str], int] = simple_token_count,
                 edge_text_fn: Callable = edge_text_readable,
                 path_embed_many_fn: Optional[Callable[[list[str]], list]] = None) -> dict:
    """Run traversal and selection for one question.

    ``path_embed_many_fn`` is the shared embedding service's ``embed_many`` in research mode
    (``path_relevance_mode: text``).  It is optional for the API-free mean-edge test mode.
    """
    t0 = time.perf_counter()
    result = traverse(graph, list(seeds), q_emb, cfg, embed_many_fn=path_embed_many_fn)
    selected = select_context(result.paths, q_emb, token_counter, cfg, edge_text_fn)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    out = {
        "n_seeds": len(seeds), "n_candidate_paths": len(result.paths), "n_selected_edges": len(selected),
        "context_tokens": sum(token_counter(edge_text_fn(e)) for e in selected),
        "nodes_expanded": result.nodes_expanded, "peak_frontier": result.peak_frontier,
        "initial_frontier": result.initial_frontier,
        "nodes_pruned_by_beam": result.nodes_pruned_by_beam,
        "paths_pruned_by_relevance": result.paths_pruned_by_relevance,
        "hit_expansion_cap": result.hit_expansion_cap,
        "missing_entity_embeddings": result.missing_entity_embeddings,
        "latency_ms": round(latency_ms, 2),
        "selected": [{"head": e.head, "relation": e.relation, "tail": e.tail,
                      "confidence": e.confidence, "support": e.support,
                      "provenance": list(e.provenance)} for e in selected],
    }
    out.update(evidence_metrics(selected, gold_ids))
    return out


def _mean(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list) -> dict:
    keys = ["evidence_recall", "evidence_precision", "context_tokens", "n_selected_edges",
            "nodes_expanded", "peak_frontier", "initial_frontier", "nodes_pruned_by_beam",
            "paths_pruned_by_relevance", "missing_entity_embeddings", "latency_ms"]
    def block(rs):
        d = {"questions": len(rs)}
        d.update({k: _mean([r.get(k) for r in rs]) for k in keys})
        d["share_with_no_edges"] = _mean([1.0 if r["n_selected_edges"] == 0 else 0.0 for r in rs])
        d["share_hit_cap"] = _mean([1.0 if r["hit_expansion_cap"] else 0.0 for r in rs])
        return d
    summary = {"all": block(rows)}
    for hop in sorted({r["hop_level"] for r in rows}):
        summary[f"hop_{hop}"] = block([r for r in rows if r["hop_level"] == hop])
    return summary


def _read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_dataset(dataset: str, kind: str, cfg: Config, *, tag: str = "run", limit: Optional[int] = None,
                graph_dir: str = "data/graph", query_dir: str = "data/query",
                processed_dir: str = "data/processed", out_dir: str = "results/traversal",
                token_counter: Callable[[str], int] = simple_token_count,
                edge_text_fn: Callable = edge_text_readable,
                tokenizer_name: Optional[str] = None,
                path_embed_many_fn: Optional[Callable[[list[str]], list]] = None) -> dict:
    from data.loaders.base import read_question_records   # imported here so the module imports without data
    from src.graph.graph_io import graph_report, load_graph

    cfg.validate()
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    # The final research configuration embeds the full path text with the same shared model
    # used for entities, edges and questions.  Tests/offline callers can inject a deterministic
    # embedder or use path_relevance_mode="mean_edge" explicitly.
    if cfg.path_relevance_mode == "text" and path_embed_many_fn is None:
        from src.graph.embeddings import get_service
        path_embed_many_fn = get_service().embed_many

    graph = load_graph(dataset, kind=kind, graph_dir=graph_dir)
    report = graph_report(graph)
    print(f"[{dataset}/{kind}] graph: {report}", flush=True)
    if report["missing_edge_embeddings"] or report["missing_entity_embeddings"]:
        print("WARNING: graph has missing embeddings; results will be unreliable (see docs/GRAPH.md)", flush=True)
    queries = {q["question_id"]: q for q in _read_jsonl(Path(query_dir) / f"{dataset}_dev.jsonl")}
    records = read_question_records(Path(processed_dir) / f"{dataset}_dev.jsonl")
    if limit is not None:
        records = records[:limit]

    rows = []
    for i, rec in enumerate(records, 1):
        q = queries.get(rec.question_id)
        if q is None:
            print(f"skip {rec.question_id}: no query analysis", flush=True)
            continue
        row = run_question(
            graph, q["seeds"], q["q_emb"], rec.gold_evidence.paragraph_ids, cfg,
            token_counter=token_counter, edge_text_fn=edge_text_fn,
            path_embed_many_fn=path_embed_many_fn,
        )
        row.update({"question_id": rec.question_id, "dataset": dataset, "hop_level": rec.hop_level, "kind": kind})
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(records)} questions", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{kind}_{tag}"
    with (out / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tokenizer_label = tokenizer_name or getattr(token_counter, "__name__", token_counter.__class__.__name__)
    summary = {
        "dataset": dataset, "kind": kind, "tag": tag, "config": asdict(cfg),
        "tokenizer": tokenizer_label,
        "tokenizer_is_placeholder": token_counter is simple_token_count,
        "path_relevance_embedding": cfg.path_relevance_mode,
        "graph_report": report,
        **summarize(rows),
    }
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["all"], indent=2), flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="musique", choices=["musique", "twiki", "hotpotqa"])
    ap.add_argument("--kind", default="auto", choices=["auto", "oracle"])
    ap.add_argument("--config", default="configs/traversal.yaml")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--W", type=float); ap.add_argument("--beam-k", type=int)
    ap.add_argument("--beam-margin", type=float); ap.add_argument("--budget", type=int)
    ap.add_argument("--max-depth", type=int); ap.add_argument("--epsilon", type=float)
    ap.add_argument("--no-g", action="store_true", help="drop g: greedy best-first")
    ap.add_argument("--no-beam", action="store_true", help="beam_k = None (unbounded frontier)")
    ap.add_argument("--path-relevance-mode", choices=["text", "mean_edge"],
                    help="embed complete paths (research mode) or use the offline mean-edge approximation")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.W is not None: cfg.W = args.W
    if args.beam_k is not None: cfg.beam_k = args.beam_k
    if args.beam_margin is not None: cfg.beam_margin = args.beam_margin
    if args.budget is not None: cfg.budget_B = args.budget
    if args.max_depth is not None: cfg.max_depth = args.max_depth
    if args.epsilon is not None: cfg.epsilon = args.epsilon
    if args.no_g: cfg.use_g = False
    if args.no_beam: cfg.beam_k = None
    if args.path_relevance_mode is not None: cfg.path_relevance_mode = args.path_relevance_mode
    cfg.validate()
    run_dataset(args.dataset, args.kind, cfg, tag=args.tag, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

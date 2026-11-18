"""Run Role 5 retrieval followed by Role 6 evidence serialization and optional generation.

Examples:
  python -m src.context.run --dataset musique --kind auto --limit 5 --tag context_smoke --path-relevance-mode mean_edge
  python -m src.context.run --dataset musique --kind auto --answer-mode gemini --tag context_gemini
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from data.loaders.base import read_question_records
from src.common.tokenization import RegexTokenCounter
from src.context.answerer import GeminiAnswerer
from src.context.prompt import build_qa_prompt, parse_answer_response
from src.context.serializer import ContextSerializer, CorpusIndex
from src.graph.embeddings import load_entity_names
from src.graph.graph_io import graph_report, load_graph
from src.traversal.search import traverse
from src.traversal.select import fact_key, select_context
from src.traversal.types import Config


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_manifest_mode(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("mode") or payload.get("embedding_mode")

def _validate_embedding_modes(dataset: str, query_dir: str | Path, graph_dir: str | Path,
                              path_relevance_mode: str) -> str | None:
    query_mode = _read_manifest_mode(Path(query_dir) / f"{dataset}_manifest.json")
    graph_mode = _read_manifest_mode(Path(graph_dir) / "manifest_embeddings.json")
    if query_mode and graph_mode and query_mode != graph_mode:
        raise ValueError(
            f"Role 4 query embeddings use mode={query_mode}, but Role 3 graph embeddings "
            f"use mode={graph_mode}; regenerate both with the same embedding model"
        )
    mode = query_mode or graph_mode
    if path_relevance_mode == "text" and mode == "mock":
        raise ValueError(
            "path_relevance_mode='text' requires real, mutually consistent embeddings; "
            "mock vectors are for plumbing tests only. Use mean_edge only for offline diagnostics."
        )
    return mode


def run_dataset(
    dataset: str,
    kind: str,
    cfg: Config,
    *,
    tag: str = "context",
    limit: Optional[int] = None,
    answer_mode: str = "none",
    answer_model: str = "gemini-3.5-flash-lite",
    processed_dir: str = "data/processed",
    query_dir: str = "data/query",
    graph_dir: str = "data/graph",
    extraction_dir: str = "data/graph/extraction",
    out_dir: str = "results/context",
    max_sources_per_edge: int = 2,
    split_manifest: str | None = None,
    split_part: str | None = None,
    answerer=None,
    requests_per_minute: float = 1000,
) -> dict:
    cfg.validate()
    embedding_mode = _validate_embedding_modes(dataset, query_dir, graph_dir, cfg.path_relevance_mode)
    if answer_mode not in {"none", "gemini"}:
        raise ValueError("answer_mode must be 'none' or 'gemini'")
    all_records = read_question_records(Path(processed_dir) / f"{dataset}_dev.jsonl")
    # Context provenance must resolve against the full sampled corpus even when a split or
    # limit restricts the questions being evaluated.
    from src.common.splits import filter_records_by_split
    records = filter_records_by_split(all_records, split_manifest=split_manifest,
                                      split_part=split_part, limit=limit)
    queries = {row["question_id"]: row for row in read_jsonl(Path(query_dir) / f"{dataset}_dev.jsonl")}
    corpus = CorpusIndex(all_records)
    names = load_entity_names(Path(extraction_dir) / dataset / "entities.jsonl")
    token_counter = RegexTokenCounter()
    serializer = ContextSerializer(corpus, names, max_sources_per_edge=max_sources_per_edge,
                                   token_counter=token_counter)
    graph = load_graph(dataset, kind=kind, graph_dir=graph_dir)
    report = graph_report(graph)
    if report["missing_edge_embeddings"] or report["missing_entity_embeddings"] or report["bad_embedding_dims"]:
        raise ValueError(f"graph embedding validation failed: {report}")

    path_embed_many_fn = None
    if cfg.path_relevance_mode == "text":
        from src.graph.embeddings import get_service
        path_embed_many_fn = get_service().embed_many

    if answer_mode == "gemini" and answerer is None:
        answerer = GeminiAnswerer(answer_model, temperature=0.0,
                                 requests_per_minute=requests_per_minute)
    if answer_mode == "none":
        answerer = None
    output_rows = []
    graph_entities = graph.entities()
    for index, record in enumerate(records, 1):
        question_started = time.perf_counter()
        query = queries.get(record.question_id)
        if query is None:
            raise KeyError(f"no Role 4 query record for {record.question_id}")
        seeds = list(query.get("seeds", []))
        search_started = time.perf_counter()
        result = traverse(
            graph, seeds, query.get("q_emb", []), cfg,
            embed_many_fn=path_embed_many_fn,
        )
        traversal_latency_ms = (time.perf_counter() - search_started) * 1000.0
        selection_started = time.perf_counter()
        selected = select_context(
            result.paths, query.get("q_emb", []), token_counter, cfg,
            edge_text_fn=serializer.edge_block,
        )
        context = serializer.serialize_edges(selected)
        selection_and_serialization_ms = (time.perf_counter() - selection_started) * 1000.0
        seed_degree = {seed: len(graph.out_edges(seed)) for seed in seeds}
        gold_pid_set = set(record.gold_evidence.paragraph_ids)
        seed_facts = {
            fact_key(edge): edge
            for seed in seeds for edge in graph.out_edges(seed)
        }
        seed_distractor_edges = sum(
            1 for edge in seed_facts.values()
            if not (set(edge.provenance) & gold_pid_set)
        )
        prompt = build_qa_prompt(record.question, context)
        row = {
            "question_id": record.question_id,
            "dataset": dataset,
            "hop_level": record.hop_level,
            "kind": kind,
            "system": "S7" if kind == "oracle" else "S6",
            "question": record.question,
            "answer_gold": record.answer,
            "gold_paragraph_ids": list(record.gold_evidence.paragraph_ids),
            "answer_aliases": list(record.metadata.get("answer_aliases") or []),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_seed_entities_in_graph": sum(seed in graph_entities for seed in seeds),
            "n_seed_entities_with_edges": sum(degree > 0 for degree in seed_degree.values()),
            "seed_out_degree": sum(seed_degree.values()),
            "seed_distractor_edge_count": seed_distractor_edges,
            "max_sources_per_edge": max_sources_per_edge,
            "selected_edges": [
                {"head": edge.head, "relation": edge.relation, "tail": edge.tail,
                 "confidence": edge.confidence, "support": edge.support,
                 "provenance": list(edge.provenance)}
                for edge in selected
            ],
            "retrieved_paragraph_ids": serializer.paragraph_ids(selected),
            "edge_provenance_paragraph_ids": serializer.all_provenance_ids(selected),
            "context": context,
            "context_tokens": token_counter(context),
            "prompt_tokens_shared": token_counter(prompt),
            "prompt": prompt,
            "n_candidate_paths": len(result.paths),
            "n_selected_edges": len(selected),
            "nodes_expanded": result.nodes_expanded,
            "peak_frontier": result.peak_frontier,
            "nodes_pruned_by_beam": result.nodes_pruned_by_beam,
            "paths_pruned_by_relevance": result.paths_pruned_by_relevance,
            "hit_expansion_cap": result.hit_expansion_cap,
            "tokenizer": token_counter.name,
            "answer_mode": answer_mode,
        }
        answer_latency_ms = 0.0
        if answerer is not None:
            answer_started = time.perf_counter()
            generated = answerer.generate(prompt)
            answer_latency_ms = (time.perf_counter() - answer_started) * 1000.0
            parsed = parse_answer_response(generated.response_text)
            allowed_ids = set(row["retrieved_paragraph_ids"])
            cited = [pid for pid in parsed["supporting_paragraph_ids"] if pid in allowed_ids]
            row.update({
                "answer": parsed["answer"],
                "supporting_paragraph_ids": cited,
                "evidence_sufficiency": parsed["evidence_sufficiency"],
                "answer_model": generated.model,
                "prompt_tokens_model_reported": generated.prompt_tokens,
                "output_tokens_model_reported": generated.output_tokens,
            })
        row["traversal_latency_ms"] = round(traversal_latency_ms, 3)
        row["selection_and_serialization_latency_ms"] = round(selection_and_serialization_ms, 3)
        row["retrieval_latency_ms"] = round(traversal_latency_ms + selection_and_serialization_ms, 3)
        row["answer_latency_ms"] = round(answer_latency_ms, 3) if answerer is not None else None
        row["latency_ms"] = round((time.perf_counter() - question_started) * 1000.0, 3)
        output_rows.append(row)
        if index % 50 == 0:
            print(f"[{dataset}/{kind}] prepared {index}/{len(records)} questions", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{kind}_{tag}"
    jsonl_path = out / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": dataset, "kind": kind, "tag": tag, "answer_mode": answer_mode,
        "answer_model": answer_model if answer_mode == "gemini" else None,
        "questions": len(output_rows),
        "mean_context_tokens": (sum(r["context_tokens"] for r in output_rows) / len(output_rows)) if output_rows else 0,
        "mean_selected_edges": (sum(r["n_selected_edges"] for r in output_rows) / len(output_rows)) if output_rows else 0,
        "mean_retrieval_latency_ms": (sum(r["retrieval_latency_ms"] for r in output_rows) / len(output_rows)) if output_rows else 0,
        "mean_answer_latency_ms": (sum((r["answer_latency_ms"] or 0.0) for r in output_rows) / len(output_rows)) if output_rows else 0,
        "mean_latency_ms": (sum(r["latency_ms"] for r in output_rows) / len(output_rows)) if output_rows else 0,
        "share_with_no_edges": (sum(not r["selected_edges"] for r in output_rows) / len(output_rows)) if output_rows else 0,
        "graph_report": report, "traversal_config": asdict(cfg),
        "tokenizer": token_counter.name,
        "embedding_mode": embedding_mode,
        "embedding_mode_is_mock": embedding_mode == "mock",
        "max_sources_per_edge": max_sources_per_edge,
        "token_count_is_model_native": False,
        "output": str(jsonl_path),
    }
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["musique", "twiki", "hotpotqa"], default="musique")
    parser.add_argument("--kind", choices=["auto", "oracle"], default="auto")
    parser.add_argument("--config", default="configs/traversal.yaml")
    parser.add_argument("--tag", default="context")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-manifest", default=None, help="shared hop-stratified tuning/test manifest")
    parser.add_argument("--split-part", choices=["tune", "test"], default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--query-dir", default="data/query")
    parser.add_argument("--graph-dir", default="data/graph")
    parser.add_argument("--extraction-dir", default="data/graph/extraction")
    parser.add_argument("--out-dir", default="results/context")
    parser.add_argument("--answer-mode", choices=["none", "gemini"], default="none")
    parser.add_argument("--answer-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--requests-per-minute", type=float)
    parser.add_argument("--max-sources-per-edge", type=int, default=2)
    parser.add_argument("--path-relevance-mode", choices=["text", "mean_edge"])
    parser.add_argument("--epsilon", type=float)
    args = parser.parse_args(argv)
    context_cfg = yaml.safe_load(Path("configs/context.yaml").read_text(encoding="utf-8")) or {}
    cfg = Config.from_yaml(args.config)
    if args.path_relevance_mode is not None:
        cfg.path_relevance_mode = args.path_relevance_mode
    if args.epsilon is not None:
        cfg.epsilon = args.epsilon
    run_dataset(args.dataset, args.kind, cfg, tag=args.tag, limit=args.limit,
                answer_mode=args.answer_mode, answer_model=args.answer_model,
                max_sources_per_edge=args.max_sources_per_edge,
                split_manifest=args.split_manifest, split_part=args.split_part,
                processed_dir=args.processed_dir, query_dir=args.query_dir,
                graph_dir=args.graph_dir, extraction_dir=args.extraction_dir,
                out_dir=args.out_dir,
                requests_per_minute=args.requests_per_minute or float(
                    context_cfg.get("requests_per_minute", 1000)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

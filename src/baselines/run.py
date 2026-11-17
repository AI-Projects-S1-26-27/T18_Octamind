"""Run S1 full-context, S2 vector RAG and S3 naive k-hop baselines.

Examples:
  python -m src.baselines.run --dataset musique --limit 5 --embedding-mode mock --tag offline
  python -m src.baselines.run --dataset musique --embedding-mode gemini --tag dev
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

from data.loaders.base import read_question_records
from src.baselines.retrievers import (
    FullContextRetriever, NaiveKHopRetriever, ParagraphCorpus, VectorRAGRetriever, make_chunks,
)
from src.common.tokenization import RegexTokenCounter
from src.graph.embeddings import get_service
from src.graph.graph_io import graph_report, load_graph


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _evidence_metrics(retrieved: list[str], gold: list[str]) -> dict:
    retrieved_set, gold_set = set(retrieved), set(gold)
    hits = len(retrieved_set & gold_set)
    return {
        "evidence_recall": hits / len(gold_set) if gold_set else None,
        "evidence_precision": hits / len(retrieved_set) if retrieved_set else 0.0,
        "evidence_paragraphs": len(retrieved_set),
    }


def run_dataset(
    dataset: str = "musique",
    *,
    config_path: str = "configs/baselines.yaml",
    embedding_mode: str | None = None,
    graph_kind: str = "auto",
    limit: int | None = None,
    split_manifest: str | None = None,
    split_part: str | None = None,
    tag: str = "run",
    processed_dir: str = "data/processed",
    query_dir: str = "data/query",
    graph_dir: str = "data/graph",
    out_dir: str = "results/baselines",
) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    mode = embedding_mode or cfg.get("embedding_mode", "gemini")
    if mode not in {"mock", "gemini"}:
        raise ValueError("embedding_mode must be mock or gemini")
    all_records = read_question_records(Path(processed_dir) / f"{dataset}_dev.jsonl")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    from src.common.splits import filter_records_by_split
    records = filter_records_by_split(all_records, split_manifest=split_manifest,
                                      split_part=split_part, limit=limit)
    queries = {q["question_id"]: q for q in _read_jsonl(Path(query_dir) / f"{dataset}_dev.jsonl")}
    query_manifest_path = Path(query_dir) / f"{dataset}_manifest.json"
    if query_manifest_path.exists():
        query_mode = json.loads(query_manifest_path.read_text(encoding="utf-8")).get("embedding_mode")
        if query_mode and query_mode != mode:
            raise ValueError(f"query embeddings use mode={query_mode}, requested baseline mode={mode}; regenerate matching embeddings")
    embedding_manifest_path = Path(graph_dir) / "manifest_embeddings.json"
    if embedding_manifest_path.exists():
        graph_mode = json.loads(embedding_manifest_path.read_text(encoding="utf-8")).get("mode")
        if graph_mode and graph_mode != mode:
            raise ValueError(f"graph embeddings use mode={graph_mode}, requested baseline mode={mode}; rebuild graph embeddings in matching mode")
    corpus = ParagraphCorpus(all_records)
    token_counter = RegexTokenCounter()
    service = get_service(mode_override=mode)
    graph = load_graph(dataset, kind=graph_kind, graph_dir=graph_dir)
    report = graph_report(graph)
    if report["missing_entity_embeddings"] or report["missing_edge_embeddings"] or report["bad_embedding_dims"]:
        raise ValueError(f"graph embedding validation failed: {report}")

    rows = []
    full = FullContextRetriever(token_counter)
    khop = NaiveKHopRetriever(graph, corpus, token_counter=token_counter)
    # Index/embed each configured chunk-size once for the entire run, not once per question.
    vector_retrievers = {}
    chunk_index_build_ms = {}
    for chunk_size in cfg.get("chunk_sizes", [256]):
        index_started = time.perf_counter()
        chunks = make_chunks(corpus, int(chunk_size), int(cfg.get("chunk_overlap", 32)))
        vector_retrievers[int(chunk_size)] = VectorRAGRetriever(chunks, service.embed_many, token_counter)
        chunk_index_build_ms[str(int(chunk_size))] = round((time.perf_counter() - index_started) * 1000.0, 3)
    for index, record in enumerate(records, 1):
        query = queries.get(record.question_id)
        if query is None:
            raise KeyError(f"no query analysis for {record.question_id}")
        q_emb = query.get("q_emb", [])
        gold = list(record.gold_evidence.paragraph_ids)

        # S1: every paragraph in the question's dataset-provided candidate context.
        result = full.retrieve(record)
        rows.append(_as_row(record, result, gold, mode))

        # S2: all configured chunk-size and k combinations share one embedding model.
        for chunk_size, vector in vector_retrievers.items():
            rank_start = time.perf_counter()
            ranked_chunks = vector.rank(q_emb)
            rank_latency_ms = (time.perf_counter() - rank_start) * 1000.0
            for k in cfg.get("vector_k", [5]):
                result = vector.retrieve_ranked(ranked_chunks, k=int(k))
                # Include ranking cost in per-query retrieval latency; the offline chunk-index
                # build remains a one-time cost and is reported separately in the run summary.
                result.latency_ms += rank_latency_ms
                result.parameters["chunk_size"] = int(chunk_size)
                rows.append(_as_row(record, result, gold, mode))

        # S3: unranked depth-limited BFS; budget sweep is shared with graph retrieval.
        for hops in cfg.get("naive_hops", [2]):
            for budget in cfg.get("context_budgets", [1000]):
                result = khop.retrieve(query.get("seeds", []), max_hops=int(hops), budget=int(budget))
                rows.append(_as_row(record, result, gold, mode))

        if index % 50 == 0:
            print(f"[{dataset}] baseline retrieval {index}/{len(records)} questions", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{tag}"
    jsonl_path = out / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_system = defaultdict(list)
    by_variant = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
        params = json.dumps(row.get("parameters", {}), sort_keys=True, separators=(",", ":"))
        by_variant[f"{row['system']}|{params}"].append(row)
    summary = {
        "dataset": dataset, "tag": tag, "embedding_mode": mode,
        "embedding_mode_is_mock": mode == "mock",
        "questions": len(records), "rows": len(rows), "graph_kind_for_s3": graph_kind,
        "graph_report": report,
        "chunk_index_build_ms": chunk_index_build_ms,
        "chunk_index_build_is_offline_cost": True,
        "systems": {
            system: {
                "runs": len(group),
                "mean_evidence_recall": _mean([r["evidence_recall"] for r in group]),
                "mean_evidence_precision": _mean([r["evidence_precision"] for r in group]),
                "mean_context_tokens": _mean([r["context_tokens"] for r in group]),
                "mean_latency_ms": _mean([r["retrieval_latency_ms"] for r in group]),
            } for system, group in sorted(by_system.items())
        },
        "variants": {
            variant: {
                "runs": len(group),
                "mean_evidence_recall": _mean([r["evidence_recall"] for r in group]),
                "mean_evidence_precision": _mean([r["evidence_precision"] for r in group]),
                "mean_context_tokens": _mean([r["context_tokens"] for r in group]),
                "mean_latency_ms": _mean([r["retrieval_latency_ms"] for r in group]),
            } for variant, group in sorted(by_variant.items())
        },
        "output": str(jsonl_path),
        "tokenizer": token_counter.name,
        "note": "S4 HippoRAG/LightRAG and optional S5 IRCoT are not included in this local S1-S3 runner.",
    }
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _as_row(record, result, gold, mode):
    retrieved = result.retrieved_paragraph_ids
    metrics = _evidence_metrics(retrieved, gold)
    return {
        "question_id": record.question_id, "dataset": record.dataset,
        "hop_level": record.hop_level, "question": record.question,
        "answer_gold": record.answer, "answer_aliases": list(record.metadata.get("answer_aliases") or []),
        "gold_paragraph_ids": list(gold),
        "system": result.system, "parameters": result.parameters,
        "embedding_mode": mode, "retrieval_items": [
            {"item_id": item.item_id, "text": item.text,
             "paragraph_ids": list(item.paragraph_ids), "score": item.score, "title": item.title}
            for item in result.items
        ],
        "retrieved_paragraph_ids": retrieved, "context": result.context,
        "context_tokens": result.context_tokens,
        "retrieval_latency_ms": round(result.latency_ms, 3), **metrics,
    }


def _mean(values):
    vals = [value for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["musique", "twiki", "hotpotqa"], default="musique")
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--embedding-mode", choices=["mock", "gemini"])
    parser.add_argument("--graph-kind", choices=["auto", "oracle"], default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-manifest", default=None, help="shared hop-stratified tuning/test manifest")
    parser.add_argument("--split-part", choices=["tune", "test"], default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--query-dir", default="data/query")
    parser.add_argument("--graph-dir", default="data/graph")
    parser.add_argument("--out-dir", default="results/baselines")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args(argv)
    run_dataset(args.dataset, config_path=args.config, embedding_mode=args.embedding_mode,
                graph_kind=args.graph_kind, limit=args.limit, split_manifest=args.split_manifest,
                split_part=args.split_part, tag=args.tag, processed_dir=args.processed_dir,
                query_dir=args.query_dir, graph_dir=args.graph_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

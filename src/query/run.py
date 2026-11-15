"""Run Role 4 query analysis over sampled questions.

Examples::

    python -m src.query.run --dataset musique --mode mock
    python -m src.query.run --mode gemini

The default output is ``data/query/<dataset>_dev.jsonl`` plus one
``data/query/<dataset>_manifest.json``. Mock embeddings are for local tests
only; use Gemini embeddings for experiment data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from data.loaders.base import read_question_records
from src.graph.embeddings import get_service

from .analyze import analyze_question
from .io import load_entities, write_jsonl


def _analysis_record(record, analysis) -> dict:
    return {
        "question_id": record.question_id,
        "dataset": record.dataset,
        "seeds": list(analysis.seeds),
        "q_emb": list(analysis.q_emb),
        "matches": [
            {
                "text": match.text,
                "normalized": match.normalized,
                "entity_ids": list(match.entity_ids),
                "start": match.start,
                "end": match.end,
                "ambiguous": match.ambiguous,
            }
            for match in analysis.matches
        ],
    }


def _manifest(dataset: str, records: list, analyses: list, mode: str) -> dict:
    seed_counts = [len(analysis.seeds) for analysis in analyses]
    ambiguous_matches = sum(len(analysis.ambiguous_matches) for analysis in analyses)
    ambiguous_candidates = sum(
        len(match.entity_ids)
        for analysis in analyses
        for match in analysis.ambiguous_matches
    )
    return {
        "dataset": dataset,
        "embedding_mode": mode,
        "questions": len(records),
        "questions_with_seeds": sum(bool(analysis.seeds) for analysis in analyses),
        "questions_without_seeds": sum(not analysis.seeds for analysis in analyses),
        "total_seeds": sum(seed_counts),
        "average_seed_count": sum(seed_counts) / len(seed_counts) if seed_counts else 0.0,
        "max_seed_count": max(seed_counts, default=0),
        "ambiguous_questions": sum(bool(analysis.ambiguous_matches) for analysis in analyses),
        "ambiguous_matches": ambiguous_matches,
        "ambiguous_candidates": ambiguous_candidates,
    }


def run_dataset(
    dataset: str,
    *,
    processed_dir: str | Path = "data/processed",
    extraction_dir: str | Path = "data/graph/extraction",
    out_dir: str | Path = "data/query",
    mode: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """Analyze one sampled dataset and return its diagnostics manifest."""
    processed_path = Path(processed_dir) / f"{dataset}_dev.jsonl"
    records = read_question_records(processed_path)
    if limit is not None:
        records = records[:limit]
    entities = load_entities(Path(extraction_dir) / dataset / "entities.jsonl")
    service = get_service(mode_override=mode)
    analyses = [
        analyze_question(record.question, entities, service.embed)
        for record in records
    ]

    output_dir = Path(out_dir)
    write_jsonl(
        (_analysis_record(record, analysis) for record, analysis in zip(records, analyses)),
        output_dir / f"{dataset}_dev.jsonl",
    )
    manifest = _manifest(dataset, records, analyses, service.mode)
    (output_dir / f"{dataset}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{dataset}] {manifest['questions']} questions; "
        f"{manifest['questions_with_seeds']} with seeds; "
        f"{manifest['ambiguous_questions']} ambiguous -> {output_dir}",
        flush=True,
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="dataset; repeatable (default: all sampled datasets)")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--extraction-dir", default="data/graph/extraction")
    parser.add_argument("--out-dir", default="data/query")
    parser.add_argument("--mode", choices=["mock", "gemini"], default=None)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    datasets = args.dataset or ["musique", "twiki", "hotpotqa"]
    for dataset in datasets:
        run_dataset(
            dataset,
            processed_dir=args.processed_dir,
            extraction_dir=args.extraction_dir,
            out_dir=args.out_dir,
            mode=args.mode,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic, hop-stratified tuning/test split for the sampled dev questions."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from data.loaders.base import read_question_records


def make_split(records, *, tune_fraction: float = 0.2, seed: int = 0) -> dict:
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be between 0 and 1")
    by_hop = defaultdict(list)
    for record in records:
        by_hop[record.hop_level].append(record.question_id)
    tune_ids, test_ids = [], []
    for hop in sorted(by_hop):
        ids = sorted(by_hop[hop])
        random.Random(f"{seed}:{hop}").shuffle(ids)
        n_tune = max(1, round(len(ids) * tune_fraction)) if len(ids) > 1 else 0
        tune_ids.extend(ids[:n_tune])
        test_ids.extend(ids[n_tune:])
    return {
        "seed": seed, "tune_fraction": tune_fraction,
        "tune_ids": sorted(tune_ids), "test_ids": sorted(test_ids),
        "counts_by_hop": {
            str(hop): {"total": len(ids), "tune": sum(1 for qid in tune_ids if qid in ids),
                       "test": sum(1 for qid in test_ids if qid in ids)}
            for hop, ids in sorted(by_hop.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["musique", "twiki", "hotpotqa"], default="musique")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tune-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    records = read_question_records(Path(args.processed_dir) / f"{args.dataset}_dev.jsonl")
    result = make_split(records, tune_fraction=args.tune_fraction, seed=args.seed)
    output = Path(args.out or f"results/evaluation/{args.dataset}_split.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "counts_by_hop": result["counts_by_hop"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

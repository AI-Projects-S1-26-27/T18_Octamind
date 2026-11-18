"""Evaluate retrieval/answer JSONL files and report proposal-aligned metrics.

Example:
  python -m src.evaluation.run --dataset musique --inputs results/context/musique_auto_context.jsonl results/baselines/musique_dev.jsonl --tag dev
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from data.loaders.base import read_question_records
from src.evaluation.metrics import paired_bootstrap_ci, score_prediction_row
from src.graph.oracle import extract_record_oracle_edges
from src.traversal.select import fact_key


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selected_fact_keys(row: dict) -> set[tuple]:
    keys = set()
    # Role 5-only rows use `selected`; Role 6 common rows use `selected_edges`.
    for raw in row.get("selected_edges", row.get("selected", [])):
        head, relation, tail = raw.get("head"), raw.get("relation"), raw.get("tail")
        if not (head and relation and tail):
            continue
        if relation.startswith("inverse of "):
            keys.add((tail, relation.removeprefix("inverse of "), head))
        else:
            keys.add((head, relation, tail))
    return keys


def _path_recall(row: dict, record) -> float | None:
    if not record.gold_chain:
        return None
    gold = {(edge.head, edge.relation, edge.tail) for edge in extract_record_oracle_edges(record)}
    if not gold:
        return None
    selected = _selected_fact_keys(row)
    return len(gold & selected) / len(gold)


def _variant_name(row: dict) -> str:
    system = row.get("system", row.get("kind", "unknown"))
    params = row.get("parameters") or {}
    # `expanded_edges` is a per-question diagnostic, not a configuration parameter.
    # Including it would split one S3 configuration into many one-row variants and
    # invalidate paired evaluation. Keep only the knobs that define the retriever.
    if system == "S3_naive_khop":
        params = {key: params[key] for key in ("max_hops", "budget") if key in params}
    if not params:
        config = row.get("traversal_config") or {}
        params = {k: config.get(k) for k in ("alpha", "beta", "eta", "gamma", "W", "beam_k", "budget_B") if k in config}
    return system if not params else f"{system} | " + json.dumps(params, sort_keys=True, separators=(",", ":"))


def _mean(values: list) -> float | None:
    vals = [value for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate_rows(
    rows: list[dict],
    records: list,
    *,
    vector_reference_variant: str | None = None,
    expected_question_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    record_by_id = {record.question_id: record for record in records}
    scored = []
    groups = defaultdict(list)
    for source in rows:
        qid = source.get("question_id")
        record = record_by_id.get(qid)
        if record is None:
            raise KeyError(f"result question_id not present in processed data: {qid}")
        row = dict(source)
        row.setdefault("answer_gold", record.answer)
        row.setdefault("gold_paragraph_ids", list(record.gold_evidence.paragraph_ids))
        row.setdefault("answer_aliases", record.metadata.get("answer_aliases", []))
        metrics = score_prediction_row(row)
        path_recall = _path_recall(row, record)
        metrics["path_recall"] = path_recall
        row.update(metrics)
        row["variant"] = _variant_name(row)
        scored.append(row)
        groups[row["variant"]].append(row)

    # A paired comparison needs exactly one result per question per variant.
    for variant, group in groups.items():
        ids = [row["question_id"] for row in group]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate question_id rows for variant {variant!r}; merge or disambiguate runs first")
        if expected_question_ids is not None and set(ids) != expected_question_ids:
            missing = sorted(expected_question_ids - set(ids))
            extra = sorted(set(ids) - expected_question_ids)
            raise ValueError(
                f"variant {variant!r} does not cover the requested split; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    summary = {"variants": {}, "questions": len({r["question_id"] for r in scored})}
    metric_names = ["exact_match", "token_f1", "evidence_recall", "evidence_precision",
                    "evidence_f1", "path_recall", "context_tokens", "retrieval_latency_ms",
                    "latency_ms"]
    for variant, group in sorted(groups.items()):
        stats = {"runs": len(group), "unique_questions": len({r["question_id"] for r in group})}
        for metric in metric_names:
            vals = [r.get(metric) for r in group if r.get(metric) is not None]
            stats[f"mean_{metric}"] = sum(vals) / len(vals) if vals else None
        stats["failure_counts"] = {
            failure: sum(r["failure_type"] == failure for r in group)
            for failure in sorted({r["failure_type"] for r in group})
        }
        stats["failure_stage_counts"] = {
            stage: sum(r.get("failure_stage") == stage for r in group)
            for stage in sorted({r.get("failure_stage", "unknown") for r in group})
        }
        stats["tokenizer_labels"] = sorted({r.get("tokenizer") for r in group if r.get("tokenizer")})
        distractor_rows = [r for r in group if isinstance(r.get("seed_distractor_edge_count"), (int, float))]
        if distractor_rows:
            counts = sorted(r["seed_distractor_edge_count"] for r in distractor_rows)
            threshold = counts[min(len(counts) - 1, int(0.75 * (len(counts) - 1)))]
            high = [r for r in distractor_rows if r["seed_distractor_edge_count"] >= threshold]
            low = [r for r in distractor_rows if r["seed_distractor_edge_count"] < threshold]
            stats["distractor_analysis"] = {
                "definition": "top-quartile seed-outgoing edges with no provenance overlap with gold evidence",
                "threshold": threshold,
                "high_distractor_questions": len(high),
                "high_distractor_mean_token_f1": _mean([r.get("token_f1") for r in high]),
                "high_distractor_mean_evidence_recall": _mean([r.get("evidence_recall") for r in high]),
                "other_questions": len(low),
                "other_mean_token_f1": _mean([r.get("token_f1") for r in low]),
                "other_mean_evidence_recall": _mean([r.get("evidence_recall") for r in low]),
            }
        # Store no more than 50 examples for each hop/variant, matching the proposal's
        # bounded error-analysis sample while retaining full scored rows in the JSONL.
        stats["failure_examples_by_hop"] = {}
        for hop in sorted({r.get("hop_level") for r in group if r.get("failure_type") != "success"}):
            failures = [r for r in group if r.get("hop_level") == hop and r.get("failure_type") != "success"]
            stats["failure_examples_by_hop"][str(hop)] = [
                {key: r.get(key) for key in (
                    "question_id", "failure_type", "failure_stage", "evidence_recall",
                    "evidence_precision", "exact_match", "token_f1", "n_seeds",
                    "n_seed_entities_in_graph", "n_seed_entities_with_edges",
                    "n_candidate_paths", "n_selected_edges", "seed_distractor_edge_count",
                )}
                for r in failures[:50]
            ]
        summary["variants"][variant] = stats

    # The tuning split selects the vector-RAG configuration. If a reference variant is
    # supplied, never re-select it on the held-out test rows.
    vector_variants = [
        (name, stats) for name, stats in summary["variants"].items()
        if name.startswith("S2_vector_rag") and stats.get("mean_token_f1") is not None
    ]
    if vector_reference_variant is None and vector_variants:
        reference_name, _ = max(vector_variants, key=lambda pair: pair[1]["mean_token_f1"])
    else:
        reference_name = vector_reference_variant

    if reference_name is not None:
        if reference_name not in groups:
            raise ValueError(
                f"tuned vector-RAG variant {reference_name!r} is absent from these rows; "
                "ensure the same baseline sweep was run for tuning and test questions"
            )
        reference_group = groups[reference_name]
        reference = {r["question_id"]: r for r in reference_group}
        comparisons = {}
        for variant, group in groups.items():
            if variant == reference_name:
                continue
            paired = [(r, reference[r["question_id"]]) for r in group if r["question_id"] in reference]
            pair_f1 = [(a.get("token_f1"), b.get("token_f1")) for a, b in paired
                       if a.get("token_f1") is not None and b.get("token_f1") is not None]
            if len(pair_f1) >= 1:
                comparisons[variant] = {
                    "reference_variant": reference_name,
                    "token_f1_paired_bootstrap": paired_bootstrap_ci(
                        [x for x, _ in pair_f1], [y for _, y in pair_f1], iterations=2000, seed=0
                    ),
                }
        summary["selected_vector_reference_variant"] = reference_name
        summary["paired_comparisons_vs_selected_vector_rag"] = comparisons
        summary["context_reduction_at_matched_accuracy"] = _matched_context_reduction(
            groups, reference_name
        )
    else:
        summary["selected_vector_reference_variant"] = None
        summary["paired_comparisons_vs_selected_vector_rag"] = {}
        summary["context_reduction_at_matched_accuracy"] = None
        summary["note"] = (
            "No answer-bearing vector-RAG rows were supplied; paired answer-accuracy comparisons "
            "and matched-accuracy context reduction were skipped."
        )
    return scored, summary


def _select_s6_variant_on_tuning(tune_variants: dict, target_f1: float) -> dict | None:
    """Select the S6 operating point on tuning data only.

    Prefer the lowest-token S6 configuration meeting the tuned vector-RAG F1 target.
    If none reaches the target, choose the highest-F1 S6 configuration (then lower tokens)
    and explicitly mark the target as unreached on tuning data.
    """
    candidates = []
    for name, stats in tune_variants.items():
        if not name.startswith("S6"):
            continue
        f1 = stats.get("mean_token_f1")
        tokens = stats.get("mean_context_tokens")
        if f1 is None or tokens is None:
            continue
        candidates.append((name, float(f1), float(tokens)))
    if not candidates:
        return None
    meeting = [item for item in candidates if item[1] >= target_f1]
    if meeting:
        name, f1, tokens = min(meeting, key=lambda item: (item[2], -item[1], item[0]))
        reached = True
    else:
        name, f1, tokens = min(candidates, key=lambda item: (-item[1], item[2], item[0]))
        reached = False
    return {
        "variant": name,
        "target_tuning_f1": float(target_f1),
        "tuning_mean_f1": f1,
        "tuning_mean_context_tokens": tokens,
        "reached_target_on_tuning": reached,
        "selection_rule": "minimum mean context tokens among S6 variants meeting tuned S2 F1; otherwise maximum S6 tuning F1",
    }


def _heldout_context_reduction(test_variants: dict, vector_reference: str,
                               s6_selection: dict | None) -> dict | None:
    """Evaluate the tuning-selected S6 operating point on test IDs without retuning."""
    if not s6_selection:
        return None
    vector = test_variants.get(vector_reference)
    s6 = test_variants.get(s6_selection["variant"])
    if not vector or not s6:
        return None
    vector_tokens = vector.get("mean_context_tokens")
    vector_f1 = vector.get("mean_token_f1")
    s6_tokens = s6.get("mean_context_tokens")
    s6_f1 = s6.get("mean_token_f1")
    # The configuration is selected only on tuning IDs. For the held-out matched-
    # accuracy calculation, a* is the test F1 of the vector-RAG configuration that
    # was selected on tuning IDs. This evaluates the locked S6 setting against the
    # locked S2 reference; it never re-selects either system on test data.
    target_test_f1 = vector_f1
    reached = (
        target_test_f1 is not None and s6_f1 is not None
        and s6_f1 >= target_test_f1
    )
    reduction = None
    if reached and vector_tokens is not None and vector_tokens > 0 and s6_tokens is not None:
        reduction = 1.0 - float(s6_tokens) / float(vector_tokens)
    return {
        "target_tuning_f1_used_to_choose_s6": s6_selection["target_tuning_f1"],
        "target_test_f1_from_locked_vector_reference": target_test_f1,
        "vector_reference_variant": vector_reference,
        "vector_test_mean_f1": vector_f1,
        "vector_test_mean_context_tokens": vector_tokens,
        "s6_variant_selected_on_tuning": s6_selection["variant"],
        "s6_test_mean_f1": s6_f1,
        "s6_test_mean_context_tokens": s6_tokens,
        "s6_reached_vector_reference_accuracy_on_test": reached,
        "context_reduction": reduction,
        "tuning_selection": s6_selection,
        "note": "S2 configuration and S6 operating point were selected on tune IDs. On held-out test IDs, context reduction is calculated only if the locked S6 setting reaches the locked S2 reference's test F1. No configuration is re-selected on test data.",
    }


def _matched_context_reduction(groups: dict[str, list[dict]], reference_name: str) -> dict | None:
    """Compute S6 context reduction at the selected vector-RAG test F1.

    The vector-RAG configuration is selected on the tuning split; on the held-out split,
    its test mean F1 defines a*. The method uses the smallest mean S6 context that reaches
    that same test F1.
    """
    reference = groups.get(reference_name, [])
    ref_f1 = [r["token_f1"] for r in reference if r.get("token_f1") is not None]
    ref_tokens = [r["context_tokens"] for r in reference if r.get("context_tokens") is not None]
    if not ref_f1 or not ref_tokens:
        return None
    target = sum(ref_f1) / len(ref_f1)
    vector_tokens = sum(ref_tokens) / len(ref_tokens)
    candidates = []
    for name, rows in groups.items():
        if not name.startswith("S6"):
            continue
        f1 = [r["token_f1"] for r in rows if r.get("token_f1") is not None]
        tokens = [r["context_tokens"] for r in rows if r.get("context_tokens") is not None]
        if f1 and tokens and sum(f1) / len(f1) >= target:
            candidates.append((sum(tokens) / len(tokens), name, sum(f1) / len(f1)))
    best = min(candidates, default=None)
    return {
        "target_test_f1": target,
        "vector_reference_variant": reference_name,
        "vector_tokens_at_target": vector_tokens,
        "s6_variant_at_target": best[1] if best else None,
        "s6_tokens_at_target": best[0] if best else None,
        "context_reduction": 1.0 - best[0] / vector_tokens if best and vector_tokens > 0 else None,
        "s6_best_f1_if_target_unreached": max(
            (sum(r["token_f1"] for r in rows if r.get("token_f1") is not None) /
             len([r for r in rows if r.get("token_f1") is not None])
             for name, rows in groups.items() if name.startswith("S6") and
             any(r.get("token_f1") is not None for r in rows)),
            default=None,
        ),
    }


def run_evaluation(dataset: str, input_files: list[str], *, tag: str = "evaluation",
                   processed_dir: str = "data/processed", out_dir: str = "results/evaluation",
                   split_manifest: str | None = None) -> dict:
    records = read_question_records(Path(processed_dir) / f"{dataset}_dev.jsonl")
    all_rows = []
    for path in input_files:
        all_rows.extend(_read_jsonl(path))
    split_info = None
    if split_manifest:
        split_info = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        tune_ids, test_ids = set(split_info.get("tune_ids", [])), set(split_info.get("test_ids", []))
        if len(tune_ids) != len(split_info.get("tune_ids", [])) or len(test_ids) != len(split_info.get("test_ids", [])):
            raise ValueError("split manifest contains duplicate IDs")
        overlap = tune_ids & test_ids
        if overlap:
            raise ValueError(f"tuning/test split overlaps on {len(overlap)} question IDs")
        known_ids = {record.question_id for record in records}
        unknown_ids = (tune_ids | test_ids) - known_ids
        missing_ids = known_ids - (tune_ids | test_ids)
        if unknown_ids:
            raise ValueError(f"split manifest contains unknown question IDs: {sorted(unknown_ids)[:5]}")
        if missing_ids:
            raise ValueError(f"split manifest omits processed question IDs: {sorted(missing_ids)[:5]}")
        tune_rows = [row for row in all_rows if row.get("question_id") in tune_ids]
        test_rows = [row for row in all_rows if row.get("question_id") in test_ids]
        if not tune_rows or not test_rows:
            raise ValueError("split manifest produced an empty tuning or test result set")
        _, tune_summary = evaluate_rows(tune_rows, records, expected_question_ids=tune_ids)
        reference_name = tune_summary.get("selected_vector_reference_variant")
        if reference_name is None:
            raise ValueError("tuning split has no answer-bearing S2 vector-RAG rows; cannot select baseline")
        target_f1 = tune_summary["variants"][reference_name]["mean_token_f1"]
        s6_selection = _select_s6_variant_on_tuning(tune_summary["variants"], target_f1)
        scored, summary = evaluate_rows(test_rows, records, vector_reference_variant=reference_name,
                                        expected_question_ids=test_ids)
        # The S6 budget/configuration must be selected on tune IDs as well. Do not pick
        # whichever S6 variant happens to look best on the held-out test set.
        summary["context_reduction_at_matched_accuracy"] = _heldout_context_reduction(
            summary["variants"], reference_name, s6_selection
        )
        summary["split"] = {
            "manifest": split_manifest,
            "tune_questions": len(tune_ids),
            "test_questions": len(test_ids),
            "tuning_vector_reference_variant": reference_name,
            "tuning_selected_variant_f1": target_f1,
            "tuning_selected_s6_variant": s6_selection,
            "seed": split_info.get("seed"),
        }
    else:
        scored, summary = evaluate_rows(all_rows, records)
        summary["evaluation_warning"] = (
            "No --split-manifest supplied: vector-RAG selection and reporting use the same rows. "
            "Treat this as exploratory, not held-out headline evaluation."
        )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{tag}"
    jsonl_path = output / f"{stem}_scored.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["dataset"] = dataset
    summary["inputs"] = input_files
    summary_path = output / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output / f"{stem}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "runs", "unique_questions", "mean_exact_match", "mean_token_f1",
                  "mean_evidence_recall", "mean_evidence_precision", "mean_evidence_f1",
                  "mean_path_recall", "mean_context_tokens", "mean_retrieval_latency_ms", "mean_latency_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, stats in summary["variants"].items():
            writer.writerow({"variant": variant, **{key: stats.get(key) for key in fields[1:]}})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["musique", "twiki", "hotpotqa"], default="musique")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--tag", default="evaluation")
    parser.add_argument("--split-manifest", default=None, help="hop-stratified tune/test JSON manifest")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results/evaluation")
    args = parser.parse_args(argv)
    run_evaluation(args.dataset, args.inputs, tag=args.tag, processed_dir=args.processed_dir,
                   out_dir=args.out_dir, split_manifest=args.split_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

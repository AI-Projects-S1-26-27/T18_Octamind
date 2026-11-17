"""Proposal-aligned answer, evidence and statistical evaluation metrics."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, remove punctuation/articles, collapse spaces."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(token for token in text.split() if token not in _ARTICLES)


def _aliases(gold: str | Iterable[str]) -> list[str]:
    if isinstance(gold, str):
        return [gold]
    return [str(value) for value in gold]


def exact_match(prediction: str, gold: str | Iterable[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(candidate) for candidate in _aliases(gold)))


def token_f1(prediction: str, gold: str | Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return float(any(not normalize_answer(candidate) for candidate in _aliases(gold)))
    best = 0.0
    for candidate in _aliases(gold):
        gold_tokens = normalize_answer(candidate).split()
        if not gold_tokens:
            score = 0.0
        else:
            overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
            if not overlap:
                score = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def evidence_scores(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> dict:
    retrieved, gold = set(retrieved_ids), set(gold_ids)
    hits = len(retrieved & gold)
    return {
        "evidence_recall": hits / len(gold) if gold else None,
        "evidence_precision": hits / len(retrieved) if retrieved else 0.0,
        "evidence_f1": (2 * hits / (len(gold) + len(retrieved))) if (gold or retrieved) else 0.0,
        "evidence_hits": hits,
        "retrieved_paragraph_count": len(retrieved),
        "gold_paragraph_count": len(gold),
        "extraneous_context_fraction": 1.0 - (hits / len(retrieved)) if retrieved else 1.0,
    }


def score_prediction_row(row: dict) -> dict:
    """Add answer and evidence metrics to one common-format result row."""
    gold_answer = row.get("answer_gold", "")
    aliases = row.get("answer_aliases") or []
    accepted = [gold_answer, *aliases]
    prediction = row.get("answer")
    out = evidence_scores(row.get("retrieved_paragraph_ids", []), row.get("gold_paragraph_ids", []))
    if prediction is not None:
        out["exact_match"] = exact_match(prediction, accepted)
        out["token_f1"] = token_f1(prediction, accepted)
        out["failure_type"] = (
            "retrieval_failure" if (out["evidence_recall"] is not None and out["evidence_recall"] < 1.0)
            else "generation_failure" if out["exact_match"] < 1.0
            else "success"
        )
    else:
        out["exact_match"] = None
        out["token_f1"] = None
        out["failure_type"] = "answer_not_generated"

    # Stage labels are diagnostic heuristics, not proof of root cause. Fine-grained
    # stage labels are available for Role 6 / S6/S7 rows; baseline rows get broader labels.
    if out["failure_type"] == "success":
        stage = "success"
    elif out["failure_type"] == "answer_not_generated":
        stage = "generation_not_run_or_failed"
    elif row.get("n_seeds", None) == 0:
        stage = "query_understanding"
    elif row.get("n_seed_entities_in_graph") == 0:
        stage = "entity_linking"
    elif row.get("n_seed_entities_with_edges") == 0:
        stage = "relationship_extraction"
    elif row.get("n_candidate_paths") == 0:
        stage = "traversal"
    elif row.get("n_selected_edges") == 0:
        stage = "selection"
    elif row.get("max_sources_per_edge") == 0:
        stage = "source_free_context_ablation"
    elif not row.get("retrieved_paragraph_ids"):
        stage = "context_construction"
    elif out["failure_type"] == "retrieval_failure":
        stage = "graph_or_retrieval_coverage"
    else:
        stage = "answer_generation"
    out["failure_stage"] = stage
    return out


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI for mean(A - B); inputs must be matched question-level values."""
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap inputs must have equal length")
    if not values_a:
        raise ValueError("paired bootstrap requires at least one pair")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("bootstrap values must be finite")
    diffs = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(iterations, len(diffs)))
    means = diffs[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_pairs": len(diffs), "mean_difference_a_minus_b": float(diffs.mean()),
        "confidence": confidence,
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "iterations": iterations, "seed": seed,
    }


def context_reduction_at_matched_accuracy(vector_rows: list[dict], method_rows: list[dict]) -> dict:
    """Compute context reduction using per-variant mean F1 and context size.

    Each row is one question-level result. Rows should carry a stable ``variant`` field
    (or ``system`` plus ``parameters``); compare configuration-level means, not individual
    question F1 values. For strict held-out evaluation, select the vector variant on the
    tuning split first and pass only its test rows as ``vector_rows``.
    """
    import json
    from collections import defaultdict

    def aggregate(rows):
        groups = defaultdict(list)
        for row in rows:
            name = row.get("variant")
            if not name:
                system = row.get("system", "unknown")
                params = row.get("parameters") or {}
                name = system if not params else f"{system}|{json.dumps(params, sort_keys=True)}"
            groups[name].append(row)
        out = []
        for name, group in groups.items():
            f1 = [r["token_f1"] for r in group if r.get("token_f1") is not None]
            tokens = [r["context_tokens"] for r in group if r.get("context_tokens") is not None]
            if f1 and tokens:
                out.append({"variant": name, "mean_f1": sum(f1) / len(f1),
                            "mean_tokens": sum(tokens) / len(tokens)})
        return out

    vector = aggregate(vector_rows)
    method = aggregate(method_rows)
    if not vector or not method:
        return {"target_f1": None, "context_reduction": None, "method_best_f1": None}
    target = max(item["mean_f1"] for item in vector)
    vector_at_target = [item["mean_tokens"] for item in vector if item["mean_f1"] >= target]
    method_at_target = [item["mean_tokens"] for item in method if item["mean_f1"] >= target]
    min_vector = min(vector_at_target) if vector_at_target else None
    min_method = min(method_at_target) if method_at_target else None
    return {
        "target_f1": target,
        "vector_variant_at_target": min(vector, key=lambda item: item["mean_tokens"] if item["mean_f1"] >= target else float("inf"))["variant"],
        "vector_tokens_at_target": min_vector,
        "method_variant_at_target": min(method, key=lambda item: item["mean_tokens"] if item["mean_f1"] >= target else float("inf"))["variant"] if method_at_target else None,
        "method_tokens_at_target": min_method,
        "context_reduction": 1.0 - min_method / min_vector if min_method is not None and min_vector else None,
        "method_best_f1": max(item["mean_f1"] for item in method),
    }

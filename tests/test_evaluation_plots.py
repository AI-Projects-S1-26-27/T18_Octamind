import json

import pytest

from src.evaluation.plots import create_plots


def test_create_evaluation_plots(tmp_path):
    pytest.importorskip("matplotlib")
    summary = tmp_path / "toy_summary.json"
    summary.write_text(json.dumps({
        "variants": {
            "S1_full_context": {"mean_token_f1": 0.5, "mean_exact_match": 0.4,
                                "mean_evidence_recall": 0.9, "mean_evidence_precision": 0.3,
                                "mean_context_tokens": 1000, "mean_latency_ms": 10},
            "S6": {"mean_token_f1": 0.6, "mean_exact_match": 0.5,
                   "mean_evidence_recall": 0.8, "mean_evidence_precision": 0.7,
                   "mean_context_tokens": 300, "mean_latency_ms": 20},
        }
    }), encoding="utf-8")
    paths = create_plots(summary)
    assert len(paths) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)

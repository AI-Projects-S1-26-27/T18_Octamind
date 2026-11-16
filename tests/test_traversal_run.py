"""Tests for the traversal runner's pure parts (no data files needed)."""
from src.traversal.run import edge_text_readable, evidence_metrics, run_question, summarize
from src.traversal.toy_graph import GOLD_CHAIN, Q_EMB, build_toy_graph
from src.traversal.types import Config, Edge


def test_evidence_metrics():
    e1 = Edge("a", "r", "b", .9, 1, ("p1", "p2"), ())
    e2 = Edge("b", "r", "c", .9, 1, ("p2", "p9"), ())
    m = evidence_metrics([e1, e2], ["p1", "p2", "p3"])
    assert m["evidence_recall"] == 2 / 3 and m["evidence_precision"] == 2 / 3
    assert evidence_metrics([], ["p1"])["evidence_precision"] == 0.0
    assert evidence_metrics([e1], [])["evidence_recall"] is None


def test_readable_edge_text():
    assert edge_text_readable(Edge("ent:new_york", "located in", "ent:usa", 1, 1)) == "new york located in usa"


def test_run_question_on_toy_graph():
    cfg = Config(budget_B=40)
    row = run_question(build_toy_graph(), ["Priya"], Q_EMB, ["p1", "p2", "p3"], cfg)
    assert [s["relation"] for s in row["selected"]] == list(GOLD_CHAIN)
    assert row["evidence_recall"] == 1.0 and row["evidence_precision"] == 1.0
    assert row["context_tokens"] <= cfg.budget_B and row["nodes_expanded"] > 0


def test_run_question_no_seeds_gives_empty_context():
    row = run_question(build_toy_graph(), [], Q_EMB, ["p1"], Config())
    assert row["n_selected_edges"] == 0 and row["evidence_recall"] == 0.0


def test_summarize_groups_by_hop():
    rows = [{"hop_level": 2, "evidence_recall": 1.0, "evidence_precision": 0.5, "context_tokens": 10,
             "n_selected_edges": 3, "nodes_expanded": 5, "peak_frontier": 2, "latency_ms": 1.0, "hit_expansion_cap": False},
            {"hop_level": 3, "evidence_recall": 0.0, "evidence_precision": 0.0, "context_tokens": 0,
             "n_selected_edges": 0, "nodes_expanded": 1, "peak_frontier": 1, "latency_ms": 1.0, "hit_expansion_cap": False}]
    s = summarize(rows)
    assert s["all"]["questions"] == 2 and s["hop_2"]["evidence_recall"] == 1.0 and s["hop_3"]["share_with_no_edges"] == 1.0


def test_run_question_uses_injected_full_path_embedder():
    seen = []

    def embed_many(texts):
        seen.extend(texts)
        return [Q_EMB for _ in texts]

    row = run_question(
        build_toy_graph(), ["Priya"], Q_EMB, ["p1", "p2", "p3"],
        Config(path_relevance_mode="text"), path_embed_many_fn=embed_many,
    )
    assert seen and row["n_candidate_paths"] > 0
    assert row["paths_pruned_by_relevance"] == 0
    assert row["initial_frontier"] == 1

import pytest

from data.loaders.base import GoldEvidence, Paragraph, QuestionRecord
from src.evaluation.ablation import ablation_configs
from src.evaluation.error_injection import inject_edge_deletions, inject_entity_merges, inject_tail_corruption
from src.evaluation.metrics import (
    context_reduction_at_matched_accuracy, evidence_scores, exact_match, paired_bootstrap_ci, token_f1,
)
from src.evaluation.run import (
    _heldout_context_reduction, _select_s6_variant_on_tuning, evaluate_rows,
)
from src.traversal.types import Edge, Graph


def _record():
    return QuestionRecord("q1", "musique", 2, "Where?", "London",
                          [Paragraph("p1", "P1", "Person was born in London.")],
                          GoldEvidence(("p1",)),
                          metadata={"answer_aliases": ["London, UK"]})


def test_answer_and_evidence_metrics():
    assert exact_match("The London!", ["London"]) == 1.0
    assert token_f1("London city", "London") > 0.5
    m = evidence_scores(["p1", "p2"], ["p1"])
    assert m["evidence_recall"] == 1.0
    assert m["evidence_precision"] == 0.5


def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap_ci([1, 0, 1], [0, 0, 1], iterations=1000, seed=42)
    second = paired_bootstrap_ci([1, 0, 1], [0, 0, 1], iterations=1000, seed=42)
    assert first == second
    assert first["mean_difference_a_minus_b"] == pytest.approx(1 / 3)


def test_context_reduction_at_matched_accuracy():
    vector = [{"variant": "S2|k1", "token_f1": 0.7, "context_tokens": 100},
              {"variant": "S2|k2", "token_f1": 0.8, "context_tokens": 200}]
    proposed = [{"variant": "S6|default", "token_f1": 0.8, "context_tokens": 100}]
    result = context_reduction_at_matched_accuracy(vector, proposed)
    assert result["target_f1"] == 0.8
    assert result["context_reduction"] == 0.5


def test_ablation_configs_cover_weights_search_beam_and_budgets():
    base = {"alpha": 1, "beta": 0.5, "eta": 0.25, "gamma": 0.1,
            "W": 1.5, "use_g": True, "beam_k": 5, "budget_B": 1000}
    configs = ablation_configs(base)
    assert configs["without_alpha"]["alpha"] == 0
    assert configs["uniform_cost"]["W"] == 0
    assert configs["greedy_best_first"]["use_g"] is False
    assert configs["beam_10"]["beam_k"] == 10
    assert configs["budget_4000"]["budget_B"] == 4000


def test_error_injection_helpers_preserve_graph_contract():
    edges = [Edge("A", "r", "B", 0.9, 2, ("p1",), (1.0, 0.0)),
             Edge("B", "s", "C", 0.8, 1, ("p2",), (0.0, 1.0))]
    graph = Graph(edges, {"A": (1.0, 0.0), "B": (0.0, 1.0), "C": (1.0, 1.0)})
    deleted = inject_edge_deletions(graph, 0.5, seed=1)
    assert len([e for e in deleted.edges if not e.relation.startswith("inverse of ")]) == 1
    with pytest.raises(ValueError, match="embed_text"):
        inject_tail_corruption(graph, 0.5, seed=1)
    corrupted = inject_tail_corruption(graph, 0.5, seed=1, embed_text=lambda text: (0.5, 0.5))
    assert all(e.embedding for e in corrupted.edges)
    merged = inject_entity_merges(graph, 0.5, seed=1, embed_text=lambda text: (0.5, 0.5))
    assert isinstance(merged, Graph)


def test_evaluate_rows_adds_metrics():
    record = _record()
    row = {"question_id": "q1", "system": "S6", "answer": "London",
           "retrieved_paragraph_ids": ["p1"], "gold_paragraph_ids": ["p1"],
           "answer_gold": "London", "selected_edges": [],
           "seed_distractor_edge_count": 3}
    scored, summary = evaluate_rows([row], [record])
    assert scored[0]["exact_match"] == 1.0
    assert scored[0]["evidence_recall"] == 1.0
    assert summary["questions"] == 1
    distractor = summary["variants"][list(summary["variants"])[0]]["distractor_analysis"]
    assert distractor["high_distractor_questions"] == 1
    assert distractor["other_questions"] == 0
    assert distractor["other_mean_token_f1"] is None


def test_write_corrupted_graphs_creates_isolated_loadable_graph(tmp_path):
    import json
    from src.evaluation.error_injection import write_corrupted_graphs
    from src.graph.embeddings import write_entity_embeddings
    from src.graph.graph_io import graph_report, load_graph
    from src.graph.oracle import write_edges

    graph_dir = tmp_path / "graph"
    dataset_dir = graph_dir / "toy"
    edges = [Edge("A", "r", "B", 0.9, 2, ("p1",), (1.0, 0.0)),
             Edge("B", "s", "C", 0.8, 1, ("p2",), (0.0, 1.0))]
    write_edges(edges, dataset_dir / "edges.jsonl")
    write_entity_embeddings({"A": (1.0, 0.0), "B": (0.0, 1.0), "C": (1.0, 1.0)},
                            dataset_dir / "entity_embeddings.json")
    (graph_dir / "manifest_embeddings.json").write_text(json.dumps({"mode": "mock"}))
    outputs = write_corrupted_graphs(
        "toy", corruptions=("deletions",), rates=(0.5,), graph_dir=str(graph_dir),
        out_root=str(tmp_path / "perturbed"), embedding_mode="mock", seed=3,
    )
    assert len(outputs) == 1
    loaded = load_graph("toy", "auto", graph_dir=outputs[0])
    assert graph_report(loaded)["missing_edge_embeddings"] == 0
    manifest = json.loads((tmp_path / "perturbed/deletions_0.50/manifest_corruption.json").read_text())
    assert manifest["research_valid_embeddings"] is False


def test_s6_operating_point_is_selected_on_tuning_and_evaluated_without_test_retuning():
    tune_variants = {
        "S2_vector_rag|k1": {"mean_token_f1": 0.70, "mean_context_tokens": 100},
        "S2_vector_rag|k5": {"mean_token_f1": 0.80, "mean_context_tokens": 300},
        "S6|budget250": {"mean_token_f1": 0.75, "mean_context_tokens": 120},
        "S6|budget500": {"mean_token_f1": 0.81, "mean_context_tokens": 220},
    }
    selected = _select_s6_variant_on_tuning(tune_variants, target_f1=0.80)
    assert selected["variant"] == "S6|budget500"
    assert selected["reached_target_on_tuning"] is True

    # On test, another S6 configuration looks better, but it must not replace the
    # tuning-selected configuration in the matched-context calculation.
    test_variants = {
        "S2_vector_rag|k5": {"mean_token_f1": 0.76, "mean_context_tokens": 320},
        "S6|budget250": {"mean_token_f1": 0.90, "mean_context_tokens": 120},
        "S6|budget500": {"mean_token_f1": 0.75, "mean_context_tokens": 210},
    }
    result = _heldout_context_reduction(test_variants, "S2_vector_rag|k5", selected)
    assert result["s6_variant_selected_on_tuning"] == "S6|budget500"
    assert result["target_test_f1_from_locked_vector_reference"] == 0.76
    assert result["s6_reached_vector_reference_accuracy_on_test"] is False
    assert result["context_reduction"] is None


def test_error_injection_batches_reembedding_for_changed_triples():
    edges = [
        Edge("A", "r", "B", 0.9, 2, ("p1",), (1.0, 0.0)),
        Edge("B", "s", "C", 0.8, 1, ("p2",), (0.0, 1.0)),
        Edge("C", "t", "D", 0.7, 1, ("p3",), (1.0, 1.0)),
    ]
    graph = Graph(edges, {node: (1.0, 0.0) for node in "ABCD"})
    calls = []

    def embed_many(texts):
        calls.append(list(texts))
        return [(0.25, 0.75) for _ in texts]

    corrupted = inject_tail_corruption(
        graph, 1.0, seed=3, embed_many_texts=embed_many
    )
    assert len(calls) == 1
    assert len(calls[0]) == len(edges)
    assert all(edge.embedding == (0.25, 0.75) for edge in corrupted.edges)


def test_s3_variant_identity_ignores_per_question_expansion_count():
    from src.evaluation.run import _variant_name

    first = _variant_name({
        "system": "S3_naive_khop",
        "parameters": {"max_hops": 2, "budget": 1000, "expanded_edges": 12},
    })
    second = _variant_name({
        "system": "S3_naive_khop",
        "parameters": {"max_hops": 2, "budget": 1000, "expanded_edges": 47},
    })
    assert first == second

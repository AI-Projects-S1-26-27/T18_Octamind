"""Tests for traversal (Role 5) on the Figure 1 toy graph (hand-made 2-D embeddings, no API needed)."""
import math

from src.traversal.frontier import Frontier
from src.traversal.scoring import Scorer, cosine, edge_cost, path_relevance, semantic_distance
from src.traversal.search import traverse
from src.traversal.select import fact_key, select_context
from src.traversal.toy_graph import GOLD_CHAIN, Q_EMB, build_toy_graph
from src.traversal.types import Config, Edge, Graph, Path, SearchNode

WORDS = lambda t: len(t.split())


def rels(path):
    return tuple(e.relation for e in path.edges)


# ---- building blocks ----------------------------------------------------------------
def test_cosine():
    assert math.isclose(cosine((1, 0), (1, 0)), 1.0)
    assert math.isclose(cosine((1, 0), (0, 1)), 0.0, abs_tol=1e-12)
    assert cosine((0, 0), (1, 0)) == 0.0 and cosine((), (1, 0)) == 0.0


def test_out_edges_and_entities():
    g = build_toy_graph()
    assert {e.relation for e in g.out_edges("Priya")} == {"works_in", "joined_on", "sits_in"}
    assert "PyTorch" in g.entities() and g.max_support == 3


def test_search_node_path_follows_parents():
    g = build_toy_graph()
    e1, e2 = g.out_edges("Priya")[0], g.out_edges("DeptA")[0]
    n2 = SearchNode(e2.tail, SearchNode(e1.tail, SearchNode("Priya"), e1, 0.2, 1), e2, 0.5, 2)
    assert n2.path() == [e1, e2]


def test_frontier_orders_by_f_and_beam_prunes():
    fr = Frontier()
    for i, f in enumerate([3.0, 1.0, 2.0, 5.0]):
        fr.push(SearchNode(str(i)), f)
    fr.keep_best(k=2)
    assert len(fr) == 2 and fr.pop()[1] == 1.0 and fr.pop()[1] == 2.0


def test_frontier_margin():
    fr = Frontier()
    for i, f in enumerate([1.0, 1.2, 4.0]):
        fr.push(SearchNode(str(i)), f)
    fr.keep_best(margin=0.5)
    assert len(fr) == 2


# ---- scoring ------------------------------------------------------------------------
def test_relevant_edges_cost_less_than_distractors():
    g, cfg = build_toy_graph(), Config()
    by_rel = {e.relation: edge_cost(e, Q_EMB, 1, g.max_support, cfg) for e in g.edges}
    assert all(c >= 0 for c in by_rel.values())
    for good in GOLD_CHAIN:
        for bad in ("joined_on", "sits_in", "near", "funded_by"):
            assert by_rel[good] < by_rel[bad]


def test_first_edge_cost_by_hand():
    # works_in: rel = cos((.9,.1),(1,0)) = .99388; 1*(1-rel) + .5*(1-.95) + .25*(1-3/3) + .1*1
    g = build_toy_graph()
    e = g.out_edges("Priya")[0]
    assert math.isclose(edge_cost(e, Q_EMB, 1, 3, Config()), 0.1312, abs_tol=1e-3)


def test_fast_scorer_matches_plain_functions():
    g, cfg = build_toy_graph(), Config()
    sc = Scorer(g, Q_EMB, cfg)
    for e in g.edges:
        assert math.isclose(sc.edge_cost(e, 2), edge_cost(e, Q_EMB, 2, g.max_support, cfg), abs_tol=1e-5)
    for ent, vec in g.entity_embeddings.items():
        assert math.isclose(sc.h(ent), semantic_distance(vec, Q_EMB), abs_tol=1e-5)


def test_path_relevance_prefers_gold_chain():
    g = build_toy_graph()
    gold = [g.out_edges("Priya")[0], g.out_edges("DeptA")[0], g.out_edges("ProjectX")[0]]
    junk = [g.out_edges("Priya")[1]]
    assert path_relevance(gold, Q_EMB) > path_relevance(junk, Q_EMB)


# ---- traversal ----------------------------------------------------------------------
def test_traverse_finds_gold_chain_first():
    res = traverse(build_toy_graph(), ["Priya"], Q_EMB, Config(beam_k=5))
    best = res.paths[0]
    assert rels(best) == GOLD_CHAIN and best.edges[-1].tail == "PyTorch"
    assert math.isclose(best.f, 0.919, abs_tol=2e-3)                 # TRACE.md: g = .131+.363+.425, h = 0
    assert len(best.edge_costs) == 3 and math.isclose(sum(best.edge_costs), best.f, abs_tol=1e-6)
    assert res.nodes_expanded > 0 and res.peak_frontier > 0


def test_search_strategies_all_find_gold_chain():
    g = build_toy_graph()
    for cfg in (Config(W=0.0, beam_k=None),               # uniform-cost
                Config(W=1.0, beam_k=None),               # A*
                Config(W=1.5, beam_k=None),               # weighted A*
                Config(use_g=False, W=1.0, beam_k=None)): # greedy best-first
        assert rels(traverse(g, ["Priya"], Q_EMB, cfg).paths[0]) == GOLD_CHAIN


def test_beam_of_two_prunes_candidates():
    g = build_toy_graph()
    wide = traverse(g, ["Priya"], Q_EMB, Config(beam_k=None))
    narrow = traverse(g, ["Priya"], Q_EMB, Config(beam_k=2))
    assert len(narrow.paths) < len(wide.paths) and narrow.peak_frontier >= 2
    assert rels(narrow.paths[0]) == GOLD_CHAIN


def test_unknown_and_duplicate_seeds():
    g = build_toy_graph()
    res = traverse(g, ["Nobody", "Priya", "Priya"], Q_EMB, Config())
    assert rels(res.paths[0]) == GOLD_CHAIN
    assert traverse(g, ["Nobody"], Q_EMB, Config()).paths == []
    assert traverse(g, [], Q_EMB, Config()).paths == []


def test_paths_never_revisit_an_entity():
    e = Edge
    g = Graph([e("A", "r", "B", .9, 1, (), (1.0, 0.0)), e("B", "r", "A", .9, 1, (), (1.0, 0.0)),
               e("B", "r", "C", .9, 1, (), (1.0, 0.0))], {"A": (1, 0), "B": (1, 0), "C": (1, 0)})
    res = traverse(g, ["A"], (1.0, 0.0), Config(max_depth=6))
    for p in res.paths:
        seen = [p.edges[0].head] + [x.tail for x in p.edges]
        assert len(seen) == len(set(seen))


def test_expansion_cap_is_reported():
    res = traverse(build_toy_graph(), ["Priya"], Q_EMB, Config(beam_k=None, max_expansions=2))
    assert res.hit_expansion_cap and res.nodes_expanded == 2


def test_missing_entity_embedding_is_counted_not_fatal():
    g = build_toy_graph()
    del g.entity_embeddings["Cafeteria"]
    res = traverse(g, ["Priya"], Q_EMB, Config(beam_k=None))
    assert res.missing_entity_embeddings > 0 and rels(res.paths[0]) == GOLD_CHAIN


# ---- context selection --------------------------------------------------------------
def test_select_context_keeps_gold_and_drops_distractors_under_budget():
    cfg = Config(budget_B=12)
    res = traverse(build_toy_graph(), ["Priya"], Q_EMB, cfg)
    H = select_context(res.paths, Q_EMB, WORDS, cfg)
    names = {e.relation for e in H}
    assert set(GOLD_CHAIN) <= names and not names & {"joined_on", "sits_in", "near"}
    assert sum(WORDS(e.text) for e in H) <= cfg.budget_B


def test_budget_is_a_hard_cap():
    cfg = Config(budget_B=6, epsilon=-10.0)             # negative epsilon: only the budget stops us
    res = traverse(build_toy_graph(), ["Priya"], Q_EMB, cfg)
    H = select_context(res.paths, Q_EMB, WORDS, cfg)
    assert sum(WORDS(e.text) for e in H) <= 6


def test_epsilon_stops_selection_early():
    cfg = Config(budget_B=1000, epsilon=100.0)          # nothing has utility >= 100
    res = traverse(build_toy_graph(), ["Priya"], Q_EMB, cfg)
    assert select_context(res.paths, Q_EMB, WORDS, cfg) == []


def test_inverse_edge_counts_as_the_same_fact():
    fwd = Edge("A", "likes", "B", .9, 1, ("p",), (1.0, 0.0))
    inv = Edge("B", "inverse of likes", "A", .9, 1, ("p",), (1.0, 0.0))
    assert fact_key(fwd) == fact_key(inv)
    cfg = Config(budget_B=100)
    paths = [Path([fwd], [0.1], 0.1), Path([inv], [0.1], 0.1)]
    assert len(select_context(paths, (1.0, 0.0), WORDS, cfg)) == 1


def test_selected_edges_are_not_duplicated_across_paths():
    g = build_toy_graph()
    res = traverse(g, ["Priya"], Q_EMB, Config(budget_B=1000, epsilon=-10.0))
    H = select_context(res.paths, Q_EMB, WORDS, Config(budget_B=1000, epsilon=-10.0))
    assert len(H) == len({fact_key(e) for e in H})


# ---- additional correctness/regression checks --------------------------------------
def test_initial_seed_frontier_obeys_beam_and_is_reported():
    edges = [
        Edge("A", "ra", "A1", 1.0, 1, ("p1",), (1.0, 0.0)),
        Edge("B", "rb", "B1", 1.0, 1, ("p2",), (1.0, 0.0)),
        Edge("C", "rc", "C1", 1.0, 1, ("p3",), (1.0, 0.0)),
    ]
    entities = {"A": (1.0, 0.0), "B": (0.0, 1.0), "C": (-1.0, 0.0),
                "A1": (1.0, 0.0), "B1": (1.0, 0.0), "C1": (1.0, 0.0)}
    result = traverse(Graph(edges, entities), ["A", "B", "C"], (1.0, 0.0),
                      Config(W=1.0, beam_k=1, max_depth=1))
    assert result.initial_frontier == 1
    assert result.peak_frontier == 3  # includes all generated seed roots before beam pruning
    assert result.nodes_pruned_by_beam == 2
    assert [edge.relation for edge in result.paths[0].edges] == ["ra"]


def test_missing_entity_embeddings_count_unique_entities_not_visits():
    edges = [
        Edge("A", "to_x", "X", 1.0, 1, (), (1.0, 0.0)),
        Edge("B", "also_to_x", "X", 1.0, 1, (), (1.0, 0.0)),
        Edge("X", "to_y", "Y", 1.0, 1, (), (1.0, 0.0)),
    ]
    entities = {"A": (1.0, 0.0), "B": (1.0, 0.0), "Y": (1.0, 0.0)}
    result = traverse(Graph(edges, entities), ["A", "B"], (1.0, 0.0),
                      Config(W=1.0, beam_k=None, max_depth=2))
    assert result.missing_entity_embeddings == 1


def test_missing_edge_embedding_does_not_crash_or_prune_path():
    edges = [
        Edge("A", "r1", "B", 1.0, 1, (), (1.0, 0.0)),
        Edge("B", "r2", "C", 1.0, 1, (), ()),
    ]
    graph = Graph(edges, {"A": (1.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 0.0)})
    result = traverse(graph, ["A"], (1.0, 0.0), Config(max_depth=2, tau=0.9))
    assert len(result.paths) == 1
    assert result.paths[0].rho is None
    assert result.paths_pruned_by_relevance == 0


def test_text_path_relevance_uses_batch_embedding_callback():
    seen = []

    def embed_many(texts):
        seen.extend(texts)
        return [Q_EMB for _ in texts]

    result = traverse(build_toy_graph(), ["Priya"], Q_EMB,
                      Config(path_relevance_mode="text", tau=0.9),
                      embed_many_fn=embed_many)
    assert seen and all(" ; " in text or len(text.split()) >= 3 for text in seen)
    assert result.paths and all(math.isclose(path.rho, 1.0) for path in result.paths)


def test_text_path_relevance_requires_embedding_callback():
    import pytest
    with pytest.raises(ValueError, match="requires embed_fn or embed_many_fn"):
        traverse(build_toy_graph(), ["Priya"], Q_EMB,
                 Config(path_relevance_mode="text"))


def test_cosine_rejects_dimension_mismatch_and_non_finite_vectors():
    import pytest
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine((1.0, 0.0), (1.0,))
    with pytest.raises(ValueError, match="NaN or infinite"):
        cosine((float("nan"), 0.0), (1.0, 0.0))


def test_config_rejects_invalid_search_settings():
    import pytest
    with pytest.raises(ValueError, match="beam_k"):
        traverse(build_toy_graph(), ["Priya"], Q_EMB, Config(beam_k=0))
    with pytest.raises(ValueError, match="W must be non-negative"):
        traverse(build_toy_graph(), ["Priya"], Q_EMB, Config(W=-1.0))


def test_select_context_rejects_misaligned_path_costs_and_bad_tokenizer():
    import pytest
    edge = Edge("A", "r", "B", 1.0, 1, (), (1.0, 0.0))
    cfg = Config(budget_B=100)
    with pytest.raises(ValueError, match="one edge_cost per edge"):
        select_context([Path([edge], [], 0.0)], Q_EMB, WORDS, cfg)
    with pytest.raises(ValueError, match="non-negative integer"):
        select_context([Path([edge], [0.1], 0.0)], Q_EMB, lambda _: -1, cfg)


def test_uniform_cost_does_not_consult_or_count_entity_heuristics():
    graph = build_toy_graph()
    graph.entity_embeddings.clear()
    result = traverse(graph, ["Priya"], Q_EMB, Config(W=0.0, beam_k=None))
    assert result.paths
    assert result.missing_entity_embeddings == 0


def test_edge_cost_rejects_invalid_confidence():
    import pytest
    edge = Edge("A", "r", "B", 1.5, 1, (), (1.0, 0.0))
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        edge_cost(edge, Q_EMB, 1, 1, Config())


def test_repository_research_config_uses_full_text_path_relevance():
    cfg = Config.from_yaml("configs/traversal.yaml")
    assert cfg.path_relevance_mode == "text"

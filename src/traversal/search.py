"""Query-aware weighted A* traversal with beam search (Modules 2 and 3).

    f(n) = g(n) + W*h(n)
    W = 0 -> uniform-cost ordering; use_g=False -> greedy best-first;
    W = 1 -> A*-style ordering; W > 1 -> weighted A* (satisficing).

A path is a candidate when its end node has no unvisited outgoing edge or reaches max_depth.
The goal is intentionally a candidate path rather than a claimed answer entity: traversal is not
provided with an answer-type schema.  Role 6 receives the selected evidence edges, not a final
answer, and the project evaluates retrieval separately from answer generation.
"""
from __future__ import annotations

from typing import Callable, Optional

from .frontier import Frontier
from .scoring import Scorer, cosine, path_relevance, path_text
from .types import Config, Graph, Path, SearchNode, TraversalResult


def make_seed_nodes(seeds: list) -> list:
    """Create one root search node per input seed (deduplication happens in ``traverse``)."""
    return [SearchNode(state=s) for s in seeds]


def traverse(
    graph: Graph,
    seeds: list,
    q_emb,
    cfg: Config,
    embed_fn: Optional[Callable] = None,
    embed_many_fn: Optional[Callable] = None,
) -> TraversalResult:
    """Run best-first search and return candidate paths, ranked by terminal-node f-score.

    ``embed_many_fn`` is used only when ``path_relevance_mode == 'text'``.  It receives unique,
    ordered path strings and must return one embedding per string.  Batch embedding keeps the
    proposal's phi(P) definition while allowing the shared embedding service to cache and batch
    requests.  ``embed_fn`` is retained as a single-text integration option.  The ``mean_edge``
    mode is the explicit offline/test approximation.
    """
    cfg.validate()
    if cfg.path_relevance_mode == "text" and embed_fn is None and embed_many_fn is None:
        raise ValueError(
            "path_relevance_mode='text' requires embed_fn or embed_many_fn; "
            "use the shared embedding service or select mean_edge for offline tests"
        )

    scorer = Scorer(graph, q_emb, cfg)
    frontier = Frontier()

    def f_score(node: SearchNode) -> float:
        g_value = node.path_cost if cfg.use_g else 0.0
        # Do not access h when W=0: uniform-cost search has no heuristic dependency, and its
        # missing-embedding diagnostic should not count vectors it never needed.
        h_value = scorer.h(node.state) if cfg.W != 0 else 0.0
        return g_value + cfg.W * h_value

    seen_seeds = set()
    for seed_node in make_seed_nodes(seeds):
        if seed_node.state in seen_seeds or not graph.out_edges(seed_node.state):
            continue  # duplicate seed or entity with no traversable fact
        seen_seeds.add(seed_node.state)
        frontier.push(seed_node, f_score(seed_node))

    # Apply the beam to initial seeds too. Record the frontier before pruning as the peak: that
    # captures the temporary successor set as well as the retained frontier width.
    peak_frontier = len(frontier)
    nodes_pruned_by_beam = frontier.keep_best(cfg.beam_k, cfg.beam_margin)
    initial_frontier = len(frontier)

    candidates: list[Path] = []
    seen_paths = set()
    nodes_expanded = 0
    hit_expansion_cap = False

    while len(frontier):
        if nodes_expanded >= cfg.max_expansions:
            hit_expansion_cap = True
            break

        node, node_f = frontier.pop()
        nodes_expanded += 1
        # The no-revisit rule is path-local, not global: alternate paths to an entity remain
        # available, but a path cannot cycle back through an entity it already contains.
        on_path = {node.state} | {edge.head for edge in node.path()}
        children = [edge for edge in graph.out_edges(node.state) if edge.tail not in on_path]

        if node.depth >= cfg.max_depth or not children:
            if node.action is not None:  # a bare seed is not an evidence path
                edges = node.path()
                key = tuple(id(edge) for edge in edges)
                if key not in seen_paths:
                    seen_paths.add(key)
                    costs_reversed = []
                    cursor = node
                    while cursor.action is not None:
                        costs_reversed.append(cursor.path_cost - cursor.parent.path_cost)
                        cursor = cursor.parent
                    candidates.append(Path(edges, list(reversed(costs_reversed)), node_f))
            continue

        for edge in children:
            child = SearchNode(
                state=edge.tail,
                parent=node,
                action=edge,
                path_cost=node.path_cost + scorer.edge_cost(edge, node.depth + 1),
                depth=node.depth + 1,
            )
            frontier.push(child, f_score(child))

        peak_frontier = max(peak_frontier, len(frontier))
        nodes_pruned_by_beam += frontier.keep_best(cfg.beam_k, cfg.beam_margin)

    # Query-aware path relevance.  The research configuration embeds the complete path text;
    # the mean-edge fallback is retained only for API-free tests and local debugging.
    paths_pruned_by_relevance = 0
    if cfg.path_relevance_mode == "text" and candidates:
        path_texts = [path_text(candidate.edges) for candidate in candidates]
        if embed_many_fn is not None:
            unique_texts = list(dict.fromkeys(path_texts))
            vectors = embed_many_fn(unique_texts)
            if len(vectors) != len(unique_texts):
                raise RuntimeError(
                    f"path embedder returned {len(vectors)} vectors for {len(unique_texts)} unique paths"
                )
            vector_by_text = dict(zip(unique_texts, vectors))
            for candidate, text in zip(candidates, path_texts):
                candidate.rho = cosine(vector_by_text[text], q_emb)
        else:
            for candidate in candidates:
                candidate.rho = path_relevance(candidate.edges, q_emb, embed_fn)
    elif cfg.path_relevance_mode == "mean_edge":
        for candidate in candidates:
            candidate.rho = path_relevance(candidate.edges, q_emb)

    kept = []
    for candidate in candidates:
        # Missing edge vectors do not count as a negative relevance score; retain the path and
        # let graph_report expose the construction problem rather than pruning on missing data.
        if candidate.rho is not None and candidate.rho < cfg.tau:
            paths_pruned_by_relevance += 1
            continue
        kept.append(candidate)

    kept.sort(key=lambda candidate: candidate.f)
    return TraversalResult(
        paths=kept,
        nodes_expanded=nodes_expanded,
        peak_frontier=peak_frontier,
        missing_entity_embeddings=scorer.missing_entity_embeddings,
        hit_expansion_cap=hit_expansion_cap,
        initial_frontier=initial_frontier,
        nodes_pruned_by_beam=nodes_pruned_by_beam,
        paths_pruned_by_relevance=paths_pruned_by_relevance,
    )

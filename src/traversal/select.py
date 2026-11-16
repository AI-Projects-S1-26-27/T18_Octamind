"""Context selection under the proposal's dynamic token budget (Section 6)."""
from __future__ import annotations

from numbers import Integral
from typing import Callable, Optional

from .types import Config, Edge

INVERSE_PREFIX = "inverse of "  # same contract as src/graph/graph_io.py


def fact_key(edge: Edge) -> tuple:
    """Identity of an underlying fact; its generated inverse shares this key."""
    if edge.relation.startswith(INVERSE_PREFIX):
        return (edge.tail, edge.relation[len(INVERSE_PREFIX):], edge.head)
    return (edge.head, edge.relation, edge.tail)


def select_context(
    paths: list,
    q_emb,
    token_counter: Callable[[str], int],
    cfg: Config,
    edge_text_fn: Optional[Callable[[Edge], str]] = None,
) -> list[Edge]:
    """Greedily select candidate-path edges by marginal utility per token.

    For each path, only facts not already selected contribute utility and token cost.  The
    selection process respects path edge order, deduplicates an edge and its inverse, and never
    exceeds ``budget_B``.  It is a greedy heuristic for the proposal's budgeted objective, not an
    exact knapsack optimizer.  ``q_emb`` remains in the signature for Role 6 compatibility;
    edge costs were computed during traversal.

    ``edge_text_fn`` must return the representation that the downstream prompt serializer will
    actually count.  The current runner's default is explicitly an estimate until the team plugs
    in its shared tokenizer and Role 6 serializer.
    """
    cfg.validate()
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    text_of = edge_text_fn or (lambda edge: edge.text)

    selected: list[Edge] = []
    taken: set[tuple] = set()
    used_tokens = 0
    remaining = list(paths)

    # Validate the path contract once rather than silently truncating with zip().
    import math

    for path in remaining:
        if len(path.edges) != len(path.edge_costs):
            raise ValueError("each candidate Path must have one edge_cost per edge")
        if any(not math.isfinite(cost) or cost < 0 for cost in path.edge_costs):
            raise ValueError("candidate edge costs must be finite and non-negative")

    token_cache: dict[tuple, int] = {}

    def edge_tokens(edge: Edge) -> int:
        key = (id(edge), text_of(edge))
        if not isinstance(key[1], str):
            raise TypeError("edge_text_fn must return a string")
        if key not in token_cache:
            count = token_counter(key[1])
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError("token_counter must return a non-negative integer")
            token_cache[key] = int(count)
        return token_cache[key]

    while remaining:
        best = None
        for path in remaining:
            new_edges = [
                (edge, cost) for edge, cost in zip(path.edges, path.edge_costs)
                if fact_key(edge) not in taken
            ]
            if not new_edges:
                continue

            token_cost = sum(edge_tokens(edge) for edge, _ in new_edges)
            if used_tokens + token_cost > cfg.budget_B:
                continue  # this candidate cannot fit; a shorter/cheaper path may still fit

            marginal_utility = sum(1.0 - cost for _, cost in new_edges)
            utility_per_token = marginal_utility / max(token_cost, 1)
            if best is None or utility_per_token > best[0]:
                best = (utility_per_token, marginal_utility, token_cost, path, new_edges)

        if best is None:
            break

        _, marginal_utility, token_cost, path, new_edges = best
        if marginal_utility < cfg.epsilon:
            break

        for edge, _ in new_edges:
            fact = fact_key(edge)
            if fact not in taken:  # protects against duplicate facts inside a malformed path
                taken.add(fact)
                selected.append(edge)
        used_tokens += token_cost
        remaining.remove(path)

    return selected

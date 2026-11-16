"""Shared data types for traversal (Role 5).

The Edge fields are the agreed Role 3 -> Role 5 interface; do not rename them without
coordinating with the graph-construction owner.  Search/result diagnostics are deliberately
additive so Role 6 can continue consuming ``TraversalResult.paths`` and selected ``Edge``
objects without an interface change.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

Vector = tuple  # tuple[float, ...]; vectors come from the project's fixed embedding model


@dataclass(frozen=True)
class Edge:
    """One relationship e = (head, relation, tail) in the Semantic Knowledge Network."""
    head: str                     # canonical entity id
    relation: str                 # relation label, e.g. "works_in"
    tail: str                     # canonical entity id
    confidence: float             # c(e) in [0, 1], extractor confidence
    support: int                  # number of source sentences supporting e (>= 1)
    provenance: tuple = ()        # source paragraph ids, used for evidence recall/precision
    embedding: Vector = ()        # phi(e): embedding of the text "head relation tail"

    @property
    def text(self) -> str:
        return f"{self.head} {self.relation} {self.tail}"


class Graph:
    """Directed multigraph with adjacency lists and one embedding per entity."""

    def __init__(self, edges: list, entity_embeddings: Optional[dict] = None):
        self.edges = list(edges)
        self.entity_embeddings = entity_embeddings or {}
        self._out: dict = {}
        for e in self.edges:
            self._out.setdefault(e.head, []).append(e)
        self.max_support: int = max((e.support for e in self.edges), default=1)
        # Unit-length numpy copies, filled lazily by scoring.Scorer and shared across queries.
        # Keys are ("e", id(edge)) or ("n", entity id).
        self.vector_cache: dict = {}

    def out_edges(self, entity: str) -> list:
        return self._out.get(entity, [])

    def entities(self) -> set:
        return {e.head for e in self.edges} | {e.tail for e in self.edges}


@dataclass
class SearchNode:
    """Module 2 search node: state, parent, action, path cost, and depth."""
    state: str
    parent: Optional["SearchNode"] = None
    action: Optional[Edge] = None     # edge that led here (None for a seed node)
    path_cost: float = 0.0            # g(n)
    depth: int = 0

    def path(self) -> list:
        """Edges from the seed to this node, in order."""
        edges, node = [], self
        while node is not None and node.action is not None:
            edges.append(node.action)
            node = node.parent
        return list(reversed(edges))


@dataclass
class Path:
    """A candidate path, edge costs, end-node f-score, and optional path relevance rho(P)."""
    edges: list
    edge_costs: list
    f: float
    rho: Optional[float] = None


@dataclass
class TraversalResult:
    paths: list                         # candidate Paths, best first (lowest f)
    nodes_expanded: int = 0             # number of nodes popped from the frontier
    peak_frontier: int = 0
    missing_entity_embeddings: int = 0  # unique entities without a usable phi(n)
    hit_expansion_cap: bool = False
    initial_frontier: int = 0            # seed nodes remaining after initial beam pruning
    nodes_pruned_by_beam: int = 0
    paths_pruned_by_relevance: int = 0


@dataclass
class Config:
    alpha: float = 1.0
    beta: float = 0.5
    eta: float = 0.25
    gamma: float = 0.1
    W: float = 1.5
    use_g: bool = True
    beam_k: Optional[int] = 5
    beam_margin: Optional[float] = None
    max_depth: int = 4
    tau: float = 0.0
    budget_B: int = 1000
    epsilon: float = 0.0
    max_expansions: int = 2000
    seed: int = 0
    # Production experiments use phi(P) from the shared embedding model.  The mean-edge mode
    # remains available for deterministic, API-free unit tests and offline debugging.
    path_relevance_mode: str = "mean_edge"

    def validate(self) -> "Config":
        """Validate mutable configuration, including values changed by CLI overrides."""
        for name in ("alpha", "beta", "eta", "gamma", "W", "epsilon"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("alpha", "beta", "eta", "gamma", "W"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.beam_k is not None and (isinstance(self.beam_k, bool) or not isinstance(self.beam_k, int) or self.beam_k < 1):
            raise ValueError("beam_k must be a positive integer or None")
        if self.beam_margin is not None:
            if (not isinstance(self.beam_margin, (int, float))
                    or not math.isfinite(self.beam_margin) or self.beam_margin < 0):
                raise ValueError("beam_margin must be finite and non-negative or None")
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if isinstance(self.budget_B, bool) or not isinstance(self.budget_B, int) or self.budget_B < 0:
            raise ValueError("budget_B must be a non-negative integer")
        if isinstance(self.max_expansions, bool) or not isinstance(self.max_expansions, int) or self.max_expansions < 1:
            raise ValueError("max_expansions must be a positive integer")
        if not isinstance(self.tau, (int, float)) or not math.isfinite(self.tau) or not -1.0 <= self.tau <= 1.0:
            raise ValueError("tau must be a finite cosine threshold in [-1, 1]")
        if self.path_relevance_mode not in {"text", "mean_edge"}:
            raise ValueError("path_relevance_mode must be 'text' or 'mean_edge'")
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown traversal config key(s): {', '.join(sorted(unknown))}")
        return cls(**raw).validate()

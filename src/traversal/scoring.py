"""Query-aware scoring signals from Section 6 of the project proposal.

    rel(e, q)   = cos(phi(e), phi(q))
    cost(e | q) = alpha*(1-rel) + beta*(1-c(e)) + eta*(1-s(e)) + gamma*depth(e)
    s(e)        = support(e) / max_support_in_graph
    h(n)        = 1 - cos(phi(end(n)), phi(q))
    rho(P)      = cos(phi(P), phi(q))

``cost_from_parts`` is the single implementation of the edge-cost equation.  ``Scorer`` uses
cached, unit-length float32 vectors for repeated graph queries; the plain functions remain
available as a readable mathematical reference and for tests.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .types import Config, Edge, Graph


def _array(vector, *, name: str) -> np.ndarray:
    """Convert a vector to a finite, flat float64 array with a useful error message."""
    if vector is None:
        return np.asarray([], dtype=np.float64)
    try:
        result = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric embedding") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {result.shape}")
    if result.size and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def cosine(a, b) -> float:
    """Cosine similarity; empty/zero vectors yield 0.0, mismatched dimensions are rejected."""
    av = _array(a, name="left vector")
    bv = _array(b, name="right vector")
    if av.size == 0 or bv.size == 0:
        return 0.0
    if av.size != bv.size:
        raise ValueError(f"embedding dimensions differ: {av.size} != {bv.size}")
    na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    # Clip floating-point overshoot so downstream costs remain non-negative.
    return float(np.clip(np.dot(av, bv) / (na * nb), -1.0, 1.0))


def strength(edge: Edge, max_support: int) -> float:
    """s(e) in [0,1]: support count normalized by the largest graph support count."""
    if edge.support < 1:
        raise ValueError(f"edge support must be >= 1 for {edge.text!r}")
    if max_support <= 0:
        return 0.0
    return min(1.0, edge.support / max_support)


def cost_from_parts(rel: float, edge: Edge, depth: int, max_support: int, cfg: Config) -> float:
    """The proposal's cost(e|q) equation, given rel(e,q)."""
    rel = float(np.clip(rel, -1.0, 1.0))
    confidence = float(edge.confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"edge confidence must be in [0, 1] for {edge.text!r}")
    return (cfg.alpha * (1.0 - rel)
            + cfg.beta * (1.0 - confidence)
            + cfg.eta * (1.0 - strength(edge, max_support))
            + cfg.gamma * depth)


def edge_cost(edge: Edge, q_emb, depth: int, max_support: int, cfg: Config) -> float:
    """cost(e|q), with depth 1 for an edge leaving a seed."""
    return cost_from_parts(cosine(edge.embedding, q_emb), edge, depth, max_support, cfg)


def semantic_distance(entity_emb, q_emb) -> float:
    """h(n) = 1 - cosine(phi(n), phi(q)); range [0, 2]."""
    return 1.0 - cosine(entity_emb, q_emb)


def path_text(path_edges: list[Edge]) -> str:
    """Canonical readable text for phi(P), preserving the ordered relationship chain."""
    def readable(entity_id: str) -> str:
        return entity_id[4:].replace("_", " ") if entity_id.startswith("ent:") else entity_id
    return " ; ".join(f"{readable(e.head)} {e.relation} {readable(e.tail)}" for e in path_edges)


def path_relevance(path_edges: list[Edge], q_emb, embed_fn: Optional[Callable] = None) -> Optional[float]:
    """rho(P) = cos(phi(P), phi(q)).

    When ``embed_fn`` is supplied, it embeds the full ordered path text as specified in the
    proposal.  Without it, the mean of edge embeddings is an explicitly approximate, API-free
    fallback for tests and offline debugging.  If any edge vector is missing, returns None so
    traversal does not incorrectly prune a path on unavailable evidence.
    """
    if not path_edges:
        return None
    if embed_fn is not None:
        return cosine(embed_fn(path_text(path_edges)), q_emb)
    vectors = [_array(edge.embedding, name=f"edge embedding for {edge.text}") for edge in path_edges]
    if any(vector.size == 0 for vector in vectors):
        return None
    dimensions = {vector.size for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("edge embeddings along a path have inconsistent dimensions")
    mean = np.mean(np.stack(vectors), axis=0)
    return cosine(mean, q_emb)


def _unit(vector, *, name: str, expected_dim: Optional[int] = None) -> Optional[np.ndarray]:
    """Return a cached-friendly unit float32 vector; empty/zero vectors are unavailable."""
    raw = _array(vector, name=name)
    if raw.size == 0:
        return None
    if expected_dim is not None and raw.size != expected_dim:
        raise ValueError(f"{name} has dimension {raw.size}; expected {expected_dim}")
    arr = raw.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm):
        raise ValueError(f"{name} has a non-finite norm")
    return None if norm == 0.0 else arr / norm


class Scorer:
    """Per-question cached scorer; counts each missing entity embedding only once."""

    def __init__(self, graph: Graph, q_emb, cfg: Config):
        self.graph, self.cfg = graph, cfg
        self.q = _unit(q_emb, name="query embedding")
        if self.q is None:
            raise ValueError("query embedding is empty or has zero norm")
        self.embedding_dim = int(self.q.size)
        self.max_support = graph.max_support
        self._cache = graph.vector_cache
        self.missing_entity_ids: set[str] = set()

    @property
    def missing_entity_embeddings(self) -> int:
        """Number of distinct entities with missing or zero-norm embeddings encountered."""
        return len(self.missing_entity_ids)

    def _edge_vec(self, edge: Edge) -> Optional[np.ndarray]:
        key = ("e", id(edge))
        if key not in self._cache:
            self._cache[key] = _unit(edge.embedding, name=f"edge embedding for {edge.text}",
                                     expected_dim=self.embedding_dim)
        return self._cache[key]

    def rel(self, edge: Edge) -> float:
        vector = self._edge_vec(edge)
        if vector is None:
            return 0.0
        return float(np.clip(np.dot(vector, self.q), -1.0, 1.0))

    def edge_cost(self, edge: Edge, depth: int) -> float:
        return cost_from_parts(self.rel(edge), edge, depth, self.max_support, self.cfg)

    def h(self, entity: str) -> float:
        key = ("n", entity)
        if key not in self._cache:
            raw = self.graph.entity_embeddings.get(entity)
            self._cache[key] = _unit(raw, name=f"entity embedding for {entity}",
                                     expected_dim=self.embedding_dim) if raw is not None else None
        vector = self._cache[key]
        if vector is None:
            self.missing_entity_ids.add(entity)
            return 1.0  # missing semantic evidence: maximally uninformative heuristic value
        return 1.0 - float(np.clip(np.dot(vector, self.q), -1.0, 1.0))

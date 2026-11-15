"""Load a built graph for traversal (Role 3 -> Role 5) and check it is consistent.

    from src.graph.graph_io import load_graph, graph_report
    g = load_graph("musique", kind="oracle")      # or kind="auto"
    print(graph_report(g))

Files read (written by ``python -m src.graph.oracle`` / ``relations`` / ``embeddings``):

    data/graph/<dataset>/oracle_edges.jsonl      oracle graph (S7)
    data/graph/<dataset>/edges.jsonl             automatic graph (S6)
    data/graph/<dataset>/entity_embeddings.json  phi(entity) for h(n)

Inverse edges
-------------
Traversal expands ``Graph.out_edges(entity)`` only, and an extracted edge points the
way the sentence reads ("Inception was directed by Nolan" gives Inception -> Nolan;
"Nolan directed Inception" gives Nolan -> Inception). A question seed on the *object*
would have no outgoing edge. ``add_inverse_edges`` therefore adds, for every edge
``a --r--> b``, an edge ``b --"inverse of r"--> a`` that carries the same confidence,
support, provenance and embedding (it is the same fact, so the same phi(e)). It is
applied here at load time, not stored, so files stay half the size. Role 5 can tell
inverse edges apart with :func:`is_inverse` (for example to avoid counting a fact
twice in the context).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from src.graph.embeddings import read_entity_embeddings
from src.graph.oracle import read_edges
from src.traversal.types import Edge, Graph

INVERSE_PREFIX = "inverse of "
EDGE_FILES = {"oracle": "oracle_edges.jsonl", "auto": "edges.jsonl"}


def is_inverse(edge: Edge) -> bool:
    return edge.relation.startswith(INVERSE_PREFIX)


def inverse_edge(edge: Edge) -> Edge:
    """The same fact read from the other end."""
    return Edge(
        head=edge.tail,
        relation=INVERSE_PREFIX + edge.relation,
        tail=edge.head,
        confidence=edge.confidence,
        support=edge.support,
        provenance=edge.provenance,
        embedding=edge.embedding,
    )


def add_inverse_edges(edges: Iterable[Edge]) -> list[Edge]:
    """``edges`` plus one inverse edge per edge (existing inverse edges are never doubled)."""
    edges = list(edges)
    seen = {(e.head, e.relation, e.tail) for e in edges}
    extra = []
    for e in edges:
        if is_inverse(e):
            continue
        inv = inverse_edge(e)
        key = (inv.head, inv.relation, inv.tail)
        if key not in seen:
            seen.add(key)
            extra.append(inv)
    return edges + extra


def load_graph(
    dataset: str,
    kind: str = "auto",
    graph_dir: str | Path = "data/graph",
    add_inverse: bool = True,
    edges_file: Optional[str] = None,
) -> Graph:
    """Build the traversal ``Graph`` for ``dataset``.

    ``kind`` is ``"oracle"`` (S7) or ``"auto"`` (S6). Raises FileNotFoundError when the
    edge file has not been built yet. Use :func:`graph_report` to check the result.
    """
    if kind not in EDGE_FILES:
        raise ValueError(f"kind must be one of {sorted(EDGE_FILES)}, got {kind!r}")
    base = Path(graph_dir) / dataset
    path = base / (edges_file or EDGE_FILES[kind])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it first (see docs/GRAPH.md)")
    edges = read_edges(path)
    if add_inverse:
        edges = add_inverse_edges(edges)
    return Graph(edges, read_entity_embeddings(base / "entity_embeddings.json"))


def graph_report(graph: Graph) -> dict:
    """Consistency summary of a loaded graph. Anything but zeros in the ``missing_*`` /
    ``bad_*`` fields will break scoring (``cos`` of an empty vector) and needs a rebuild."""
    entities = graph.entities()
    dims = {len(e.embedding) for e in graph.edges if e.embedding}
    dims |= {len(v) for v in graph.entity_embeddings.values()}
    missing_entities = sorted(e for e in entities if e not in graph.entity_embeddings)
    return {
        "edges": len(graph.edges),
        "inverse_edges": sum(1 for e in graph.edges if is_inverse(e)),
        "entities": len(entities),
        "missing_edge_embeddings": sum(1 for e in graph.edges if not e.embedding),
        "missing_entity_embeddings": len(missing_entities),
        "missing_entity_examples": missing_entities[:5],
        "embedding_dims": sorted(dims),
        "bad_embedding_dims": len(dims) > 1,
    }

"""Controlled graph corruption helpers for the proposal's RQ4 robustness tests.

Corrupted edge vectors are recomputed by the supplied embed_text callback. This avoids
silently retaining embeddings for a different triple.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Mapping

from src.graph.graph_io import add_inverse_edges, is_inverse
from src.traversal.types import Edge, Graph


def _base_edges(graph: Graph) -> list[Edge]:
    return [edge for edge in graph.edges if not is_inverse(edge)]


def _validate_rate(rate: float) -> None:
    if not isinstance(rate, (int, float)) or not 0.0 <= rate <= 1.0:
        raise ValueError("corruption rate must be in [0, 1]")


def inject_edge_deletions(graph: Graph, rate: float, *, seed: int = 0) -> Graph:
    """Randomly delete a fraction of unique automatic-graph facts, then rebuild inverses."""
    _validate_rate(rate)
    edges = _base_edges(graph)
    count = round(len(edges) * rate)
    rng = random.Random(seed)
    removed = set(rng.sample(range(len(edges)), min(count, len(edges))))
    kept = [edge for index, edge in enumerate(edges) if index not in removed]
    return Graph(add_inverse_edges(kept), dict(graph.entity_embeddings))


def _embed_pending_texts(
    texts: list[str],
    *,
    embed_text: Callable[[str], tuple] | None,
    embed_many_texts: Callable[[list[str]], list[tuple]] | None,
) -> list[tuple]:
    if not texts:
        return []
    if embed_many_texts is not None:
        vectors = list(embed_many_texts(texts))
    elif embed_text is not None:
        vectors = [tuple(embed_text(text)) for text in texts]
    else:
        raise ValueError("an embedding callback is required to recompute changed triples")
    if len(vectors) != len(texts):
        raise ValueError("embedding callback returned a different number of vectors than changed triples")
    if not all(vector and all(isinstance(x, (int, float)) for x in vector) for vector in vectors):
        raise ValueError("embedding callback returned an empty or non-numeric vector")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("embedding callback returned inconsistent vector dimensions")
    return [tuple(float(x) for x in vector) for vector in vectors]


def inject_tail_corruption(
    graph: Graph,
    rate: float,
    *,
    seed: int = 0,
    embed_text: Callable[[str], tuple] | None = None,
    embed_many_texts: Callable[[list[str]], list[tuple]] | None = None,
    entity_names: Mapping[str, str] | None = None,
) -> Graph:
    """Replace tails on a fraction of facts and re-embed changed triples.

    Prefer ``embed_many_texts`` so provider embeddings are computed in batches. The
    single-text callback remains supported for small unit tests and injected services.
    """
    _validate_rate(rate)
    if rate and embed_text is None and embed_many_texts is None:
        raise ValueError("embed_text or embed_many_texts is required to recompute vectors for corrupted triples")
    edges = _base_edges(graph)
    entities = sorted(graph.entities())
    if len(entities) < 2:
        return Graph(add_inverse_edges(edges), dict(graph.entity_embeddings))
    rng = random.Random(seed)
    count = round(len(edges) * rate)
    changed = set(rng.sample(range(len(edges)), min(count, len(edges))))
    names = dict(entity_names or {})
    out: list[Edge] = []
    pending: list[tuple[int, str]] = []
    for index, edge in enumerate(edges):
        if index not in changed:
            out.append(edge)
            continue
        candidates = [entity for entity in entities if entity not in {edge.head, edge.tail}]
        if not candidates:
            out.append(edge)
            continue
        new_tail = rng.choice(candidates)
        head_text = names.get(edge.head, edge.head.removeprefix("ent:").replace("_", " "))
        tail_text = names.get(new_tail, new_tail.removeprefix("ent:").replace("_", " "))
        text = f"{head_text} {edge.relation} {tail_text}"
        out.append(Edge(edge.head, edge.relation, new_tail, edge.confidence, edge.support,
                        edge.provenance, edge.embedding))
        pending.append((len(out) - 1, text))
    vectors = _embed_pending_texts(
        [text for _, text in pending], embed_text=embed_text, embed_many_texts=embed_many_texts
    )
    for (position, _), vector in zip(pending, vectors):
        edge = out[position]
        out[position] = Edge(edge.head, edge.relation, edge.tail, edge.confidence,
                             edge.support, edge.provenance, vector)
    return Graph(add_inverse_edges(out), dict(graph.entity_embeddings))


def inject_entity_merges(
    graph: Graph,
    rate: float,
    *,
    seed: int = 0,
    embed_text: Callable[[str], tuple] | None = None,
    embed_many_texts: Callable[[list[str]], list[tuple]] | None = None,
    entity_names: Mapping[str, str] | None = None,
) -> Graph:
    """Merge a fraction of entities into other IDs and re-embed changed triples."""
    _validate_rate(rate)
    if rate and embed_text is None and embed_many_texts is None:
        raise ValueError("embed_text or embed_many_texts is required to recompute vectors after entity merges")
    edges = _base_edges(graph)
    entities = sorted(graph.entities())
    rng = random.Random(seed)
    n_merge = min(len(entities) // 2, round(len(entities) * rate))
    shuffled = list(entities)
    rng.shuffle(shuffled)
    mapping = {shuffled[i]: shuffled[i + 1] for i in range(0, n_merge * 2, 2)}
    names = dict(entity_names or {})
    out: list[Edge] = []
    pending: list[tuple[int, str]] = []
    for edge in edges:
        head = mapping.get(edge.head, edge.head)
        tail = mapping.get(edge.tail, edge.tail)
        if head == tail:
            continue
        embedding = edge.embedding
        if head != edge.head or tail != edge.tail:
            hname = names.get(head, head.removeprefix("ent:").replace("_", " "))
            tname = names.get(tail, tail.removeprefix("ent:").replace("_", " "))
            text = f"{hname} {edge.relation} {tname}"
            out.append(Edge(head, edge.relation, tail, edge.confidence, edge.support,
                            edge.provenance, embedding))
            pending.append((len(out) - 1, text))
        else:
            out.append(edge)
    vectors = _embed_pending_texts(
        [text for _, text in pending], embed_text=embed_text, embed_many_texts=embed_many_texts
    )
    for (position, _), vector in zip(pending, vectors):
        edge = out[position]
        out[position] = Edge(edge.head, edge.relation, edge.tail, edge.confidence,
                             edge.support, edge.provenance, vector)
    merged_embeddings = dict(graph.entity_embeddings)
    for source in mapping:
        merged_embeddings.pop(source, None)
    return Graph(add_inverse_edges(out), merged_embeddings)


def write_corrupted_graphs(
    dataset: str,
    *,
    corruptions: tuple[str, ...] = ("deletions", "tail", "merge"),
    rates: tuple[float, ...] = (0.1, 0.2, 0.3),
    seed: int = 0,
    graph_dir: str = "data/graph",
    out_root: str = "data/graph_perturbed",
    embedding_mode: str = "gemini",
) -> list[str]:
    """Generate isolated RQ4 graph copies for Role 5 to consume with ``--graph-dir``.

    Edge deletions are topology-only. Tail corruption and entity merging recompute every
    changed edge embedding with the requested shared embedding service. Original graphs are
    never overwritten. Mock mode is permitted only for plumbing checks and is labelled.
    """
    import json
    from pathlib import Path

    from src.graph.embeddings import get_service, write_entity_embeddings
    from src.graph.graph_io import graph_report, load_graph
    from src.graph.oracle import write_edges

    allowed = {"deletions", "tail", "merge"}
    if not corruptions or set(corruptions) - allowed:
        raise ValueError(f"corruptions must be a non-empty subset of {sorted(allowed)}")
    for rate in rates:
        _validate_rate(rate)
        if rate == 0.0:
            raise ValueError("corruption rates must be greater than zero")
    source_manifest = Path(graph_dir) / "manifest_embeddings.json"
    if source_manifest.exists():
        recorded_mode = json.loads(source_manifest.read_text(encoding="utf-8")).get("mode")
        if recorded_mode and recorded_mode != embedding_mode:
            raise ValueError(
                f"source graph embeddings use mode={recorded_mode}, requested mode={embedding_mode}; "
                "rebuild source embeddings consistently first"
            )
    graph = load_graph(dataset, kind="auto", graph_dir=graph_dir)
    report = graph_report(graph)
    if report["missing_edge_embeddings"] or report["missing_entity_embeddings"] or report["bad_embedding_dims"]:
        raise ValueError(f"source graph embedding validation failed: {report}")
    service = get_service(mode_override=embedding_mode)
    if embedding_mode == "mock":
        print("WARNING: generating error-injection graphs with mock vectors; plumbing only, not research evidence")
    names_path = Path(graph_dir) / "extraction" / dataset / "entities.jsonl"
    names: dict[str, str] = {}
    if names_path.exists():
        for line in names_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                names[row["id"]] = row.get("name", row["id"])

    written = []
    for corruption in corruptions:
        for rate in rates:
            if corruption == "deletions":
                corrupted = inject_edge_deletions(graph, rate, seed=seed)
            elif corruption == "tail":
                corrupted = inject_tail_corruption(
                    graph, rate, seed=seed, embed_many_texts=service.embed_many, entity_names=names
                )
            else:
                corrupted = inject_entity_merges(
                    graph, rate, seed=seed, embed_many_texts=service.embed_many, entity_names=names
                )
            label = f"{corruption}_{rate:.2f}"
            target_graph_dir = Path(out_root) / label / dataset
            base_edges = [edge for edge in corrupted.edges if not edge.relation.startswith("inverse of ")]
            write_edges(base_edges, target_graph_dir / "edges.jsonl")
            write_entity_embeddings(corrupted.entity_embeddings, target_graph_dir / "entity_embeddings.json")
            manifest = {
                "dataset": dataset, "source_kind": "auto", "corruption": corruption,
                "rate": rate, "seed": seed, "embedding_mode": service.mode,
                "embedding_model": service.model, "research_valid_embeddings": not service.is_mock,
                "source_graph_dir": graph_dir, "source_base_edges": len(_base_edges(graph)),
                "output_base_edges": len(base_edges), "graph_report": graph_report(corrupted),
            }
            manifest_path = Path(out_root) / label / "manifest_corruption.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            written.append(str(Path(out_root) / label))
            print(f"[{dataset}] wrote {corruption} rate={rate:.2f} -> {written[-1]}", flush=True)
    return written


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["musique", "twiki"], default="musique")
    parser.add_argument("--corruptions", nargs="+", choices=["deletions", "tail", "merge"],
                        default=["deletions", "tail", "merge"])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-dir", default="data/graph")
    parser.add_argument("--out-root", default="data/graph_perturbed")
    parser.add_argument("--embedding-mode", choices=["mock", "gemini"], default="gemini")
    args = parser.parse_args(argv)
    write_corrupted_graphs(
        args.dataset, corruptions=tuple(args.corruptions), rates=tuple(args.rates),
        seed=args.seed, graph_dir=args.graph_dir, out_root=args.out_root,
        embedding_mode=args.embedding_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

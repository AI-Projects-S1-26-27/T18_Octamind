"""Entity linking: group mentions into canonical entities (Role 2).

Step 1 (always): mentions with the same normalized name are one entity
("The Beatles", "Beatles", "beatles." -> ``ent:beatles``).
Step 2 (optional): merge entities whose name embeddings are near-duplicates
("NYC" / "New York City"), using the shared ``embed()`` service once it exists.
Both steps are configured in configs/extraction.yaml.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from .entities import DEFAULT_TYPE, Entity, Mention, entity_id, normalize_name


def _majority_type(types: list) -> str:
    """Most common type; a specific type beats the CONCEPT fallback."""
    counts = Counter(types)
    specific = [(n, t) for t, n in counts.items() if t != DEFAULT_TYPE]
    if specific:
        return max(specific, key=lambda nt: (nt[0], nt[1]))[1]
    return DEFAULT_TYPE


def link_mentions(mentions: list) -> tuple:
    """Return ``(entities, norm_to_id)``; ``norm_to_id`` maps a normalized name to its entity id."""
    groups = defaultdict(list)
    for m in mentions:
        norm = m.norm
        if norm:
            groups[norm].append(m)

    entities, norm_to_id, used_ids = [], {}, set()
    for norm in sorted(groups):
        group = groups[norm]
        surface = Counter(m.text for m in group)
        name = max(surface, key=lambda s: (surface[s], s))   # ties broken deterministically
        eid = entity_id(norm)
        if eid in used_ids:            # two different norms slugged to the same id
            suffix = 2
            while f"{eid}-{suffix}" in used_ids:
                suffix += 1
            eid = f"{eid}-{suffix}"
        used_ids.add(eid)
        norm_to_id[norm] = eid
        entities.append(Entity(
            id=eid,
            name=name,
            type=_majority_type([m.type for m in group]),
            aliases=tuple(sorted(s for s in surface if s != name)),
            paragraph_ids=tuple(dict.fromkeys(m.paragraph_id for m in group)),
            mention_count=len(group),
        ))
    return entities, norm_to_id


def merge_by_embedding(entities: list, embed: Callable[[str], tuple], threshold: float,
                       block: int = 1024) -> tuple:
    """Merge entities whose name embeddings have cosine >= ``threshold``.

    Only entities of the same type merge (CONCEPT may merge with anything, since it
    means "untyped"). Returns ``(merged_entities, remap)`` where ``remap`` maps every
    old id to its surviving id. The survivor is the entity with more mentions.
    """
    if len(entities) < 2:
        return list(entities), {e.id: e.id for e in entities}
    vectors = np.array([embed(e.name) for e in entities], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

    parent = list(range(len(entities)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # compare in row blocks so a 30k-entity corpus does not need a 30k x 30k matrix
    for lo in range(0, len(entities), block):
        sims = vectors[lo:lo + block] @ vectors.T
        for r, c in zip(*np.nonzero(sims >= threshold)):
            i, j = lo + int(r), int(c)
            if j <= i:
                continue
            ti, tj = entities[i].type, entities[j].type
            if ti != tj and DEFAULT_TYPE not in (ti, tj):
                continue
            parent[find(j)] = find(i)

    clusters = defaultdict(list)
    for i in range(len(entities)):
        clusters[find(i)].append(i)

    merged, remap = [], {}
    for indices in clusters.values():
        indices.sort(key=lambda i: (-entities[i].mention_count, entities[i].id))
        members = [entities[i] for i in indices]
        head = members[0]
        names = {e.name for e in members} | {a for e in members for a in e.aliases}
        merged.append(Entity(
            id=head.id,
            name=head.name,
            type=_majority_type([e.type for e in members for _ in range(e.mention_count)]),
            aliases=tuple(sorted(names - {head.name})),
            paragraph_ids=tuple(dict.fromkeys(p for e in members for p in e.paragraph_ids)),
            mention_count=sum(e.mention_count for e in members),
            embedding=tuple(float(x) for x in vectors[indices[0]]),
        ))
        for e in members:
            remap[e.id] = head.id
    merged.sort(key=lambda e: e.id)
    return merged, remap


def name_index(entities: list) -> dict:
    """Normalized name or alias -> entity ids that carry it (used by query linking and eval)."""
    index = defaultdict(set)
    for e in entities:
        for surface in (e.name, *e.aliases):
            norm = normalize_name(surface)
            if norm:
                index[norm].add(e.id)
    return index

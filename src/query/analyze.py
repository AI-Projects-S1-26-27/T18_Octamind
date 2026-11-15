"""Deterministic question understanding for Role 4."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from src.graph.entities import Entity, normalize_name
from src.graph.linking import name_index

from .types import EntityMatch, QueryAnalysis


def _normalized_question(question: str) -> str:
    """Normalize question whitespace while retaining internal punctuation."""
    return " ".join(normalize_name(question).split())


def _rank_candidates(surface: str, entity_ids: Iterable[str], entities: dict[str, Entity]) -> tuple[str, ...]:
    """Apply the initial transparent ranking without capping candidates."""
    return tuple(
        sorted(
            entity_ids,
            key=lambda entity_id: (
                normalize_name(entities[entity_id].name) != surface,
                -entities[entity_id].mention_count,
                entity_id,
            ),
        )
    )


def _find_matches(question: str, entities: Iterable[Entity]) -> tuple[EntityMatch, ...]:
    normalized_question = _normalized_question(question)
    if not normalized_question:
        return ()

    entity_list = list(entities)
    entity_by_id = {entity.id: entity for entity in entity_list}
    index = name_index(entity_list)
    candidates = []
    for surface, entity_ids in index.items():
        if not surface:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")
        for match in pattern.finditer(normalized_question):
            candidates.append(
                (
                    match.start(),
                    -(match.end() - match.start()),
                    surface,
                    _rank_candidates(surface, entity_ids, entity_by_id),
                    match.end(),
                )
            )

    # Prefer the longest match at each position and discard overlapping
    # shorter matches. Ambiguous aliases retain every matching entity; the
    # current ranking is intentionally simple and can be replaced later.
    selected = []
    for start, negative_length, surface, entity_ids, end in sorted(candidates):
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _ in selected):
            continue
        selected.append((start, end, surface))

    matches = []
    seen_matches = set()
    for start, end, surface in sorted(selected):
        entity_ids = _rank_candidates(surface, index[surface], entity_by_id)
        match_key = (surface, entity_ids)
        if match_key in seen_matches:
            continue
        seen_matches.add(match_key)
        matches.append(
            EntityMatch(
                text=normalized_question[start:end],
                normalized=surface,
                entity_ids=entity_ids,
                start=start,
                end=end,
            )
        )
    return tuple(matches)


def analyze_question(
    question: str,
    entities: Iterable[Entity],
    embed_fn: Callable[[str], Iterable[float]],
) -> QueryAnalysis:
    """Return all deterministic seed candidates and the question embedding."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")

    matches = _find_matches(question, entities)
    # Preserve mention order while deduplicating candidates.
    seen = set()
    ordered_seeds = []
    for match in matches:
        for entity_id in match.entity_ids:
            if entity_id not in seen:
                seen.add(entity_id)
                ordered_seeds.append(entity_id)

    vector = tuple(float(value) for value in embed_fn(question))
    return QueryAnalysis(
        question=question,
        seeds=tuple(ordered_seeds),
        q_emb=vector,
        matches=matches,
    )

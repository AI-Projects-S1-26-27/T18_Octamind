"""Seed-quality metrics for Role 4 diagnostics."""
from __future__ import annotations

import re
from collections.abc import Iterable

from src.graph.entities import Entity, normalize_name
from src.graph.eval_extraction import gold_names

from .types import QueryAnalysis


def _contains(needle: str, haystack: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _question_gold_names(record) -> set[str]:
    question = normalize_name(record.question)
    return {
        name
        for name in gold_names(record)
        if _contains(name, question)
    }


def seed_quality(analysis: QueryAnalysis, record, entities: dict[str, Entity]) -> dict:
    """Measure exact seed coverage of gold names explicitly present in a question.

    Gold entities that are not written in the question are excluded from recall:
    Role 4 cannot be expected to seed an entity it was never given as a mention.
    """
    expected = _question_gold_names(record)
    seed_names = {
        normalize_name(entities[entity_id].name)
        for entity_id in analysis.seeds
        if entity_id in entities
    }
    hits = expected & seed_names
    return {
        "question_gold_names": len(expected),
        "question_gold_hits": len(hits),
        "question_gold_recall": len(hits) / len(expected) if expected else None,
        "seed_count": len(analysis.seeds),
        "seed_names": len(seed_names),
        "non_gold_seed_names": len(seed_names - expected) if expected else len(seed_names),
    }


def summarize_quality(
    analyses: Iterable[QueryAnalysis],
    records: Iterable,
    entities: Iterable[Entity],
) -> dict:
    """Aggregate seed-quality metrics over aligned query analyses and records."""
    entity_map = {entity.id: entity for entity in entities}
    rows = [seed_quality(analysis, record, entity_map) for analysis, record in zip(analyses, records)]
    expected = sum(row["question_gold_names"] for row in rows)
    hits = sum(row["question_gold_hits"] for row in rows)
    return {
        "questions_evaluated": len(rows),
        "question_gold_names": expected,
        "question_gold_hits": hits,
        "question_gold_recall": hits / expected if expected else None,
        "average_seed_count": (
            sum(row["seed_count"] for row in rows) / len(rows) if rows else 0.0
        ),
        "non_gold_seed_names": sum(row["non_gold_seed_names"] for row in rows),
    }

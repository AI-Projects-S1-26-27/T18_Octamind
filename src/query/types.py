"""Types produced by Role 4 query understanding."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityMatch:
    """One normalized-name match in a question.

    ``start`` and ``end`` refer to the normalized question string, not the
    original question, because normalization can remove punctuation.
    """

    text: str
    normalized: str
    entity_ids: tuple[str, ...]
    start: int
    end: int

    @property
    def ambiguous(self) -> bool:
        return len(self.entity_ids) > 1


@dataclass(frozen=True)
class QueryAnalysis:
    """The query-side inputs and diagnostics for Role 5."""

    question: str
    seeds: tuple[str, ...]
    q_emb: tuple[float, ...]
    matches: tuple[EntityMatch, ...] = ()

    @property
    def ambiguous_matches(self) -> tuple[EntityMatch, ...]:
        return tuple(match for match in self.matches if match.ambiguous)

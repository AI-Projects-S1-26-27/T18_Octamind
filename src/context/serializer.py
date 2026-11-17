"""Serialize Role 5's selected subgraph into evidence-grounded context (Role 6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from data.loaders.base import Paragraph, QuestionRecord
from src.common.tokenization import RegexTokenCounter
from src.graph.embeddings import entity_name_from_id
from src.graph.graph_io import is_inverse
from src.traversal.select import fact_key
from src.traversal.types import Edge


@dataclass(frozen=True)
class EvidenceSource:
    paragraph_id: str
    title: str
    text: str


class CorpusIndex:
    """Dataset-wide paragraph index keyed by the globally unique Role 1 id."""

    def __init__(self, records: Iterable[QuestionRecord]):
        self.paragraphs: dict[str, Paragraph] = {}
        for record in records:
            for paragraph in record.paragraphs:
                old = self.paragraphs.get(paragraph.id)
                if old is not None and (old.title != paragraph.title or old.text != paragraph.text):
                    raise ValueError(f"paragraph id collision with different content: {paragraph.id}")
                self.paragraphs[paragraph.id] = paragraph

    def get(self, paragraph_id: str) -> Paragraph | None:
        return self.paragraphs.get(paragraph_id)


class ContextSerializer:
    """Render selected edges with readable triples and source-paragraph evidence.

    The same ``edge_block`` function is passed into Role 5's selector so its budget
    counts the representation that Role 6 actually serializes.
    """

    def __init__(
        self,
        corpus: CorpusIndex,
        entity_names: Mapping[str, str] | None = None,
        *,
        max_sources_per_edge: int = 2,
        token_counter=None,
    ):
        if max_sources_per_edge < 0:
            raise ValueError("max_sources_per_edge must be non-negative")
        self.corpus = corpus
        self.entity_names = dict(entity_names or {})
        self.max_sources_per_edge = max_sources_per_edge
        self.token_counter = token_counter or RegexTokenCounter()

    def _name(self, entity_id: str) -> str:
        return self.entity_names.get(entity_id) or entity_name_from_id(entity_id)

    @staticmethod
    def _natural_direction(edge: Edge) -> tuple[str, str, str]:
        if is_inverse(edge):
            relation = edge.relation.removeprefix("inverse of ")
            return edge.tail, relation, edge.head
        return edge.head, edge.relation, edge.tail

    def sources_for(self, edge: Edge) -> list[EvidenceSource]:
        if self.max_sources_per_edge == 0:
            return []
        sources = []
        for pid in edge.provenance:
            paragraph = self.corpus.get(pid)
            if paragraph is not None:
                sources.append(EvidenceSource(pid, paragraph.title, paragraph.text))
            if len(sources) >= self.max_sources_per_edge:
                break
        return sources

    def edge_block(self, edge: Edge) -> str:
        """The exact per-edge block used for context selection and rendering."""
        head, relation, tail = self._natural_direction(edge)
        lines = [f"FACT: {self._name(head)} --{relation}--> {self._name(tail)}"]
        sources = self.sources_for(edge)
        if sources:
            for source in sources:
                lines.append(f"SOURCE [{source.paragraph_id}] {source.title}: {source.text}")
        elif self.max_sources_per_edge > 0:
            lines.append("SOURCE: unavailable in the indexed corpus")
        return "\n".join(lines)

    def serialize_edges(self, edges: Iterable[Edge]) -> str:
        """Serialize selected facts once each, normalizing inverse edges."""
        blocks = []
        seen = set()
        for edge in edges:
            key = fact_key(edge)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(self.edge_block(edge))
        return "\n\n".join(blocks)

    def paragraph_ids(self, edges: Iterable[Edge]) -> list[str]:
        """Paragraph IDs actually rendered as source text in the serialized context."""
        return list(dict.fromkeys(
            source.paragraph_id for edge in edges for source in self.sources_for(edge)
        ))

    @staticmethod
    def all_provenance_ids(edges: Iterable[Edge]) -> list[str]:
        """All edge provenance IDs, including source paragraphs omitted by the display cap."""
        return list(dict.fromkeys(pid for edge in edges for pid in edge.provenance))

    def context_tokens(self, edges: Iterable[Edge]) -> int:
        return int(self.token_counter(self.serialize_edges(edges)))

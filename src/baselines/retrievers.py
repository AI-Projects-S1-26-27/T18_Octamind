"""S1-S3 retrieval baselines specified in the approved proposal.

S1: full question context. S2: vector retrieval over a shared corpus of paragraph chunks.
S3: unranked depth-limited BFS expansion over the same graph used by S6.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from data.loaders.base import Paragraph, QuestionRecord
from src.common.tokenization import RegexTokenCounter
from src.graph.embeddings import entity_name_from_id
from src.traversal.scoring import cosine
from src.traversal.select import fact_key
from src.traversal.types import Edge, Graph

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    return text


@dataclass(frozen=True)
class RetrievalItem:
    item_id: str
    text: str
    paragraph_ids: tuple[str, ...]
    score: float | None = None
    title: str = ""


@dataclass
class RetrievalResult:
    system: str
    items: list[RetrievalItem]
    context: str
    retrieved_paragraph_ids: list[str]
    context_tokens: int
    latency_ms: float
    parameters: dict


class ParagraphCorpus:
    """Build a deterministic dataset-wide pool with globally unique paragraph IDs."""
    def __init__(self, records: Iterable[QuestionRecord]):
        self.paragraphs: dict[str, Paragraph] = {}
        for record in records:
            for paragraph in record.paragraphs:
                old = self.paragraphs.get(paragraph.id)
                if old and (old.text != paragraph.text or old.title != paragraph.title):
                    raise ValueError(f"paragraph ID collision: {paragraph.id}")
                self.paragraphs[paragraph.id] = paragraph

    def get(self, paragraph_id: str) -> Paragraph | None:
        return self.paragraphs.get(paragraph_id)

    def all(self) -> list[Paragraph]:
        return [self.paragraphs[key] for key in sorted(self.paragraphs)]


def make_chunks(corpus: ParagraphCorpus, chunk_size: int = 256, overlap: int = 32) -> list[RetrievalItem]:
    """Split pooled paragraphs into wordpunct-token chunks with paragraph provenance."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    chunks = []
    stride = chunk_size - overlap
    for paragraph in corpus.all():
        tokens = _TOKEN_RE.findall(paragraph.text)
        if not tokens:
            continue
        starts = list(range(0, len(tokens), stride))
        for chunk_no, start in enumerate(starts):
            piece = tokens[start:start + chunk_size]
            if not piece:
                continue
            text = _detokenize(piece)
            chunks.append(RetrievalItem(
                item_id=f"{paragraph.id}::chunk{chunk_no}",
                text=text,
                paragraph_ids=(paragraph.id,),
                title=paragraph.title,
            ))
            if start + chunk_size >= len(tokens):
                break
    return chunks


def _render_items(items: Iterable[RetrievalItem]) -> str:
    return "\n\n".join(
        f"SOURCE [{pid}] {item.title}: {item.text}"
        for item in items for pid in item.paragraph_ids
    )


def _paragraph_items(paragraphs: Iterable[Paragraph]) -> list[RetrievalItem]:
    return [RetrievalItem(p.id, p.text, (p.id,), title=p.title) for p in paragraphs]


class FullContextRetriever:
    """S1: all paragraphs supplied with a question, including dataset distractors."""
    def __init__(self, token_counter: Callable[[str], int] | None = None):
        self.token_counter = token_counter or RegexTokenCounter()

    def retrieve(self, record: QuestionRecord) -> RetrievalResult:
        import time
        start = time.perf_counter()
        items = _paragraph_items(record.paragraphs)
        context = _render_items(items)
        return RetrievalResult(
            system="S1_full_context", items=items, context=context,
            retrieved_paragraph_ids=list(dict.fromkeys(pid for item in items for pid in item.paragraph_ids)),
            context_tokens=self.token_counter(context), latency_ms=(time.perf_counter() - start) * 1000,
            parameters={"mode": "all_question_paragraphs"},
        )


class VectorRAGRetriever:
    """S2: cosine retrieval over embeddings of chunks from the pooled corpus."""
    def __init__(
        self,
        chunks: list[RetrievalItem],
        embed_many: Callable[[list[str]], list],
        token_counter: Callable[[str], int] | None = None,
    ):
        import numpy as np
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("vector RAG requires at least one non-empty corpus chunk")
        self.embed_many = embed_many
        self.token_counter = token_counter or RegexTokenCounter()
        vectors = embed_many([item.text for item in self.chunks])
        if len(vectors) != len(self.chunks):
            raise ValueError("embedding service returned a different number of chunk vectors")
        self.matrix = np.asarray(vectors, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(self.chunks):
            raise ValueError("chunk embeddings must form a 2D matrix aligned with the chunk list")
        if self.matrix.shape[1] == 0 or not np.isfinite(self.matrix).all():
            raise ValueError("chunk embeddings must have a positive dimension and finite values")
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.unit_matrix = np.divide(self.matrix, norms, out=np.zeros_like(self.matrix), where=norms > 0)

    def rank(self, q_emb) -> list[tuple[float, int, RetrievalItem]]:
        """Score and sort the corpus once; use the result for an entire k sweep."""
        import numpy as np
        query = np.asarray(q_emb, dtype=float)
        if query.ndim != 1 or query.shape[0] != self.unit_matrix.shape[1]:
            raise ValueError("query embedding dimension does not match chunk embeddings")
        if not np.isfinite(query).all():
            raise ValueError("query embedding must contain only finite values")
        norm = float(np.linalg.norm(query))
        query = query / norm if norm > 0 else np.zeros_like(query)
        scores = self.unit_matrix @ query
        order = np.argsort(-scores, kind="stable")
        return [(float(scores[idx]), int(idx), self.chunks[int(idx)]) for idx in order]

    def retrieve_ranked(self, ranked: list[tuple[float, int, RetrievalItem]], *, k: int = 5) -> RetrievalResult:
        import time
        if k < 1:
            raise ValueError("k must be positive")
        start = time.perf_counter()
        chosen = ranked[:k]
        items = [RetrievalItem(item.item_id, item.text, item.paragraph_ids, score, item.title)
                 for score, _, item in chosen]
        context = _render_items(items)
        pids = list(dict.fromkeys(pid for item in items for pid in item.paragraph_ids))
        return RetrievalResult(
            system="S2_vector_rag", items=items, context=context,
            retrieved_paragraph_ids=pids, context_tokens=self.token_counter(context),
            latency_ms=(time.perf_counter() - start) * 1000,
            parameters={"k": k, "chunk_count": len(self.chunks)},
        )

    def retrieve(self, q_emb, *, k: int = 5) -> RetrievalResult:
        return self.retrieve_ranked(self.rank(q_emb), k=k)


class NaiveKHopRetriever:
    """S3: breadth-first k-hop subgraph expansion without query-aware edge scoring."""
    def __init__(
        self,
        graph: Graph,
        corpus: ParagraphCorpus,
        *,
        token_counter: Callable[[str], int] | None = None,
    ):
        self.graph = graph
        self.corpus = corpus
        self.token_counter = token_counter or RegexTokenCounter()

    def retrieve(self, seeds: Iterable[str], *, max_hops: int = 2, budget: int = 1000) -> RetrievalResult:
        import time
        if max_hops < 1 or budget < 0:
            raise ValueError("max_hops must be >= 1 and budget must be >= 0")
        start = time.perf_counter()
        queue = deque((seed, 0) for seed in dict.fromkeys(seeds))
        visited_at_depth: dict[str, int] = {}
        selected_edges: list[Edge] = []
        seen_facts = set()
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            previous_depth = visited_at_depth.get(node)
            if previous_depth is not None and previous_depth <= depth:
                continue
            visited_at_depth[node] = depth
            for edge in self.graph.out_edges(node):
                key = fact_key(edge)
                if key not in seen_facts:
                    seen_facts.add(key)
                    selected_edges.append(edge)
                queue.append((edge.tail, depth + 1))

        # Expand provenance to source paragraphs, preserving BFS discovery order.
        ordered_pids = list(dict.fromkeys(pid for edge in selected_edges for pid in edge.provenance))
        items = []
        used = 0
        for pid in ordered_pids:
            paragraph = self.corpus.get(pid)
            if paragraph is None:
                continue
            item = RetrievalItem(paragraph.id, paragraph.text, (paragraph.id,), title=paragraph.title)
            block = _render_items([item])
            count = self.token_counter(block)
            addition = count if not items else count + 1
            if used + addition > budget:
                continue
            items.append(item)
            used += addition
        context = _render_items(items)
        return RetrievalResult(
            system="S3_naive_khop", items=items, context=context,
            retrieved_paragraph_ids=[item.item_id for item in items],
            context_tokens=self.token_counter(context), latency_ms=(time.perf_counter() - start) * 1000,
            parameters={"max_hops": max_hops, "budget": budget, "expanded_edges": len(selected_edges)},
        )

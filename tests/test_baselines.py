from data.loaders.base import GoldEvidence, Paragraph, QuestionRecord
from src.baselines.retrievers import (
    FullContextRetriever, NaiveKHopRetriever, ParagraphCorpus, VectorRAGRetriever, make_chunks,
)
from src.traversal.types import Edge, Graph


def _records():
    return [
        QuestionRecord("q1", "musique", 2, "Where?", "London",
                       [Paragraph("p1", "Person", "Person was born in London."),
                        Paragraph("p2", "London", "London is a city.")],
                       GoldEvidence(("p1",))),
        QuestionRecord("q2", "musique", 2, "What?", "Paris",
                       [Paragraph("p3", "France", "Paris is in France.")],
                       GoldEvidence(("p3",))),
    ]


def test_full_context_returns_all_question_paragraphs():
    result = FullContextRetriever().retrieve(_records()[0])
    assert result.system == "S1_full_context"
    assert result.retrieved_paragraph_ids == ["p1", "p2"]


def test_make_chunks_preserves_paragraph_provenance():
    corpus = ParagraphCorpus(_records())
    chunks = make_chunks(corpus, chunk_size=5, overlap=1)
    assert chunks
    assert all(len(item.paragraph_ids) == 1 for item in chunks)
    assert {pid for item in chunks for pid in item.paragraph_ids} == {"p1", "p2", "p3"}


def test_vector_rag_ranks_and_retrieves_top_k_once():
    chunks = make_chunks(ParagraphCorpus(_records()), chunk_size=100, overlap=0)
    # Toy 2D vectors: p1 chunk is aligned, other chunks are orthogonal/opposite.
    vectors = [(1.0, 0.0) if "born" in item.text else (0.0, 1.0) for item in chunks]
    retriever = VectorRAGRetriever(chunks, lambda texts: vectors)
    ranked = retriever.rank((1.0, 0.0))
    result = retriever.retrieve_ranked(ranked, k=1)
    assert result.system == "S2_vector_rag"
    assert result.items[0].score == 1.0
    assert result.context_tokens > 0


def test_naive_khop_bfs_collects_provenance_under_budget():
    records = _records()
    corpus = ParagraphCorpus(records)
    edges = [
        Edge("A", "born in", "B", 0.9, 1, ("p1",), (1.0, 0.0)),
        Edge("B", "located in", "C", 0.8, 1, ("p2",), (0.0, 1.0)),
    ]
    retriever = NaiveKHopRetriever(Graph(edges), corpus)
    result = retriever.retrieve(["A"], max_hops=2, budget=1000)
    assert result.system == "S3_naive_khop"
    assert result.retrieved_paragraph_ids == ["p1", "p2"]
    assert result.context_tokens <= 1000

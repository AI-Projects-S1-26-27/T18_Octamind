"""Tests for Role 3 shared embedding service and vector store.

Tests mock embedding math (unit norm, determinism), disk caching, entity
embedding persistence, and edge vector population.
"""
import math
import tempfile
from pathlib import Path

import pytest

from src.graph.embeddings import (
    EmbeddingCache,
    EmbeddingService,
    mock_embed,
    populate_edge_embeddings,
    read_entity_embeddings,
    write_entity_embeddings,
)
from src.traversal.scoring import cosine
from src.traversal.types import Edge


def test_mock_embed_properties():
    v1 = mock_embed("Priya", dim=64)
    v2 = mock_embed("Priya", dim=64)
    v_diff = mock_embed("Acme Corp", dim=64)

    assert len(v1) == 64
    # Determinism
    assert v1 == v2
    # Unit norm
    norm = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)
    # Cosine self-similarity is 1.0
    assert math.isclose(cosine(v1, v2), 1.0, rel_tol=1e-5)
    # Different strings are not identical
    assert cosine(v1, v_diff) < 0.99


def test_mock_embed_empty_string():
    v = mock_embed("", dim=32)
    assert len(v) == 32
    assert all(x == 0.0 for x in v)


def test_embedding_cache_put_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(tmpdir)
        assert cache.get("model-a", "hello") is None

        vec = (0.5, 0.5, 0.5, 0.5)
        cache.put("model-a", "hello", vec)

        restored = cache.get("model-a", "hello")
        assert restored is not None
        assert math.isclose(restored[0], 0.5)
        # Case insensitive key
        assert cache.get("model-a", "HELLO") == restored
        # Different model returns None
        assert cache.get("model-b", "hello") is None


def test_embedding_service_batch_and_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        service = EmbeddingService(
            mode="mock",
            model="test-model",
            cache_dir=tmpdir,
            dimension=32,
            batch_size=2,
        )
        texts = ["apple", "banana", "cherry"]
        vectors = service.embed_many(texts)

        assert len(vectors) == 3
        assert len(vectors[0]) == 32

        # Verify disk cache exists
        for t in texts:
            assert service.cache.get("mock-32", t) is not None

        # Re-running uses cache
        cached_vectors = service.embed_many(texts)
        assert cached_vectors == vectors


def test_write_and_read_entity_embeddings():
    embs = {
        "ent:apple": (0.1, 0.2, 0.3),
        "ent:banana": (0.4, 0.5, 0.6),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "entity_embeddings.json"
        write_entity_embeddings(embs, path)
        restored = read_entity_embeddings(path)

        assert set(restored) == {"ent:apple", "ent:banana"}
        assert math.isclose(restored["ent:apple"][0], 0.1)
        assert math.isclose(restored["ent:banana"][2], 0.6)


def test_populate_edge_embeddings():
    edges = [
        Edge(head="ent:priya", relation="works_in", tail="ent:depta", confidence=1.0, support=1),
        Edge(head="ent:depta", relation="develops", tail="ent:projx", confidence=0.9, support=2),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        service = EmbeddingService(mode="mock", cache_dir=tmpdir, dimension=16)
        populated = populate_edge_embeddings(edges, service)

        assert len(populated) == 2
        assert len(populated[0].embedding) == 16
        assert len(populated[1].embedding) == 16
        # Edge fields remain intact
        assert populated[0].head == "ent:priya"
        assert populated[0].relation == "works_in"
        assert populated[0].tail == "ent:depta"
        assert populated[0].confidence == 1.0
        assert populated[0].support == 1


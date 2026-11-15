"""Tests for Role 3 automatic relationship discovery and graph construction.

Tests connective cleaning, rule-based extraction, LLM extraction with caching,
and multi-sentence triple aggregation using synthetic fixtures.
"""
import json
import tempfile
from pathlib import Path

import pytest

from data.loaders.base import Paragraph
from src.graph.relations import (
    LinkedMention,
    LLMRelationExtractor,
    RawRelation,
    RelationResponseCache,
    RuleRelationExtractor,
    aggregate_relations,
    clean_relation_label,
)
from src.traversal.types import Edge, Graph


def test_clean_relation_label():
    assert clean_relation_label(" was married to the actress ") == "was married to the actress"
    assert clean_relation_label(", who directed ") == "directed"
    assert clean_relation_label(" is located in the city of ") == "is located in the city of"
    assert clean_relation_label(" (born in) ") == "born in"
    assert clean_relation_label("he was a French director") == "was french director"


def test_rule_extractor_sentence_cooccurrence():
    text = "John Smith works in the AI Department of Acme Corporation."
    p = Paragraph(id="toy::1", title="John Smith", text=text)

    # Two mentions in sentence 0
    # "John Smith" (0-10) and "Acme Corporation" (41-57)
    m1 = LinkedMention("John Smith", "PERSON", p.id, 0, 0, 10, "ent:john_smith")
    m2 = LinkedMention("Acme Corporation", "ORGANIZATION", p.id, 0, 41, 57, "ent:acme_corporation")
    title_m = LinkedMention("John Smith", "PERSON", p.id, -1, -1, -1, "ent:john_smith")

    extractor = RuleRelationExtractor(max_tokens=15, include_title=True)
    relations = extractor.extract(p, [title_m, m1, m2])

    assert len(relations) == 1
    rel = relations[0]
    assert rel.head == "ent:john_smith"
    assert rel.tail == "ent:acme_corporation"
    assert rel.relation == "works in the ai department of"
    assert rel.confidence == 0.6
    assert rel.paragraph_id == p.id


def test_rule_extractor_pairs_single_mention_with_title():
    text = "Edmond T. Gréville was born in 1906. He died in Nice."
    p = Paragraph(id="toy::2", title="Edmond T. Gréville", text=text)

    title_m = LinkedMention("Edmond T. Gréville", "PERSON", p.id, -1, -1, -1, "ent:edmond_t_greville")
    # In sentence 1: "Nice" is the only mention (start offset in paragraph: 48, end: 52)
    m_nice = LinkedMention("Nice", "LOCATION", p.id, 1, 48, 52, "ent:nice")

    extractor = RuleRelationExtractor(max_tokens=15, include_title=True)
    relations = extractor.extract(p, [title_m, m_nice])

    assert len(relations) == 1
    rel = relations[0]
    assert rel.head == "ent:edmond_t_greville"
    assert rel.relation == "died in"
    assert rel.tail == "ent:nice"
    assert rel.confidence == 0.5


def test_llm_extractor_with_mock_and_cache():
    text = "Secret Lives was directed by Edmond T. Gréville."
    p = Paragraph(id="toy::3", title="Secret Lives", text=text)

    m1 = LinkedMention("Secret Lives", "WORK", p.id, 0, 0, 12, "ent:secret_lives")
    m2 = LinkedMention("Edmond T. Gréville", "PERSON", p.id, 0, 29, 47, "ent:edmond_t_greville")

    mock_response = json.dumps([
        {"head": "ent:secret_lives", "relation": "director", "tail": "ent:edmond_t_greville", "confidence": 0.95},
        {"head": "ent:invalid", "relation": "unknown", "tail": "ent:edmond_t_greville", "confidence": 0.5},  # should be filtered
    ])

    calls = 0
    def mock_generate(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = LLMRelationExtractor(mock_generate, "mock-model", tmpdir)

        # First call triggers mock_generate
        rels1 = extractor.extract(p, [m1, m2])
        assert calls == 1
        assert len(rels1) == 1
        assert rels1[0].head == "ent:secret_lives"
        assert rels1[0].relation == "director"
        assert rels1[0].tail == "ent:edmond_t_greville"
        assert rels1[0].confidence == 0.95

        # Second call hits cache; mock_generate NOT called
        rels2 = extractor.extract(p, [m1, m2])
        assert calls == 1
        assert len(rels2) == 1
        assert rels2[0] == rels1[0]


def test_aggregate_relations_combines_support_and_provenance():
    raw = [
        RawRelation("ent:a", "works_in", "ent:b", 0.6, "p1", 0),
        RawRelation("ent:a", "works_in", "ent:b", 0.8, "p2", 1),
        RawRelation("ent:b", "located_in", "ent:c", 0.7, "p1", 2),
    ]

    edges = aggregate_relations(raw)
    assert len(edges) == 2

    e_ab = next(e for e in edges if e.head == "ent:a")
    assert e_ab.tail == "ent:b"
    assert e_ab.confidence == 0.8  # maximum confidence
    assert e_ab.support == 2        # 2 distinct sentences
    assert set(e_ab.provenance) == {"p1", "p2"}

    e_bc = next(e for e in edges if e.head == "ent:b")
    assert e_bc.tail == "ent:c"
    assert e_bc.support == 1
    assert e_bc.provenance == ("p1",)


def test_automatic_edges_form_valid_graph():
    raw = [
        RawRelation("ent:a", "step1", "ent:b", 0.6, "p1", 0),
        RawRelation("ent:b", "step2", "ent:c", 0.6, "p1", 1),
    ]
    edges = aggregate_relations(raw)
    graph = Graph(edges)

    assert graph.entities() == {"ent:a", "ent:b", "ent:c"}
    assert len(graph.out_edges("ent:a")) == 1
    assert graph.out_edges("ent:a")[0].tail == "ent:b"
    assert graph.out_edges("ent:b")[0].tail == "ent:c"

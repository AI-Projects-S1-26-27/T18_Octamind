import json

import pytest

from data.loaders.base import GoldEvidence, Paragraph, QuestionRecord
from src.context.prompt import build_qa_prompt, parse_answer_response
from src.context.serializer import ContextSerializer, CorpusIndex
from src.traversal.types import Edge


def _record():
    return QuestionRecord(
        question_id="q1", dataset="musique", hop_level=2,
        question="Where was the person born?", answer="London",
        paragraphs=[Paragraph("musique::p1", "P1", "The person was born in London.")],
        gold_evidence=GoldEvidence(paragraph_ids=("musique::p1",)),
    )


def test_context_serializer_includes_readable_fact_and_source():
    record = _record()
    serializer = ContextSerializer(CorpusIndex([record]), {"ent:person": "Person", "ent:london": "London"})
    edge = Edge("ent:person", "was born in", "ent:london", 0.9, 1, ("musique::p1",), ())
    block = serializer.edge_block(edge)
    assert "FACT: Person --was born in--> London" in block
    assert "SOURCE [musique::p1] P1" in block
    assert "born in London" in block
    assert serializer.paragraph_ids([edge]) == ["musique::p1"]
    assert serializer.all_provenance_ids([edge]) == ["musique::p1"]


def test_context_serializer_normalizes_inverse_and_deduplicates():
    record = _record()
    serializer = ContextSerializer(CorpusIndex([record]), {"ent:person": "Person", "ent:london": "London"})
    edge = Edge("ent:person", "was born in", "ent:london", 0.9, 1, ("musique::p1",), ())
    inverse = Edge("ent:london", "inverse of was born in", "ent:person", 0.9, 1, ("musique::p1",), ())
    text = serializer.serialize_edges([inverse, edge])
    assert text.count("FACT:") == 1
    assert "Person --was born in--> London" in text


def test_source_free_ablation_serializes_only_triple():
    serializer = ContextSerializer(CorpusIndex([_record()]), max_sources_per_edge=0)
    edge = Edge("ent:person", "was born in", "ent:london", 0.9, 1, ("musique::p1",), ())
    assert "FACT:" in serializer.edge_block(edge)
    assert "SOURCE" not in serializer.edge_block(edge)
    assert serializer.paragraph_ids([edge]) == []
    assert serializer.all_provenance_ids([edge]) == ["musique::p1"]


def test_corpus_index_rejects_conflicting_duplicate_ids():
    r1 = _record()
    r2 = _record()
    r2.paragraphs = [Paragraph("musique::p1", "P1", "different text")]
    with pytest.raises(ValueError, match="collision"):
        CorpusIndex([r1, r2])


def test_prompt_and_json_response_parser():
    prompt = build_qa_prompt("Where?", "SOURCE [p1] London")
    assert "Use only facts" in prompt
    parsed = parse_answer_response('```json\n{"answer":"London","supporting_paragraph_ids":["p1"],"evidence_sufficiency":"sufficient"}\n```')
    assert parsed["answer"] == "London"
    assert parsed["supporting_paragraph_ids"] == ["p1"]
    parsed = parse_answer_response('Here is the answer:\n{"answer":"London","supporting_paragraph_ids":["p1"],"evidence_sufficiency":"sufficient"}')
    assert parsed["answer"] == "London"
    parsed = parse_answer_response('"London"')
    assert parsed["answer"] == "London"
    parsed = parse_answer_response('[{"answer":"London","supporting_paragraph_ids":["p1"],"evidence_sufficiency":"sufficient"}]')
    assert parsed["answer"] == "London"
    with pytest.raises(ValueError, match="valid JSON"):
        parse_answer_response("not json")

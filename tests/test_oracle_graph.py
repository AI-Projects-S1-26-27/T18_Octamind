"""Tests for Role 3 Oracle Graph construction.

Uses synthetic records to test gold chain extraction, placeholder resolution,
relation cleaning, edge aggregation, and serialization.
"""
from pathlib import Path
import tempfile

import pytest

from data.loaders.base import GoldEvidence, GoldStep, Paragraph, QuestionRecord
from src.graph.oracle import (
    clean_relation,
    extract_record_oracle_edges,
    build_oracle_graph,
    edge_to_dict,
    edge_from_dict,
    read_edges,
    write_edges,
)
from src.traversal.types import Edge, Graph


def make_synthetic_record(
    qid: str,
    steps: list[dict],
    gold_pids: list[str] = (),
) -> QuestionRecord:
    gold_steps = [
        GoldStep(
            step=i + 1,
            relation=s["relation"],
            head=s.get("head"),
            tail=s.get("tail"),
            support_paragraph_ids=tuple(s.get("support_pids", ())),
        )
        for i, s in enumerate(steps)
    ]
    return QuestionRecord(
        question_id=qid,
        dataset="test_ds",
        hop_level=len(steps),
        question="Synthetic question?",
        answer="Final Answer",
        paragraphs=[Paragraph(id="test::p1", title="Alpha", text="Sample text.")],
        gold_evidence=GoldEvidence(paragraph_ids=tuple(gold_pids)),
        gold_chain=gold_steps,
    )


def test_clean_relation():
    assert clean_relation("#1 >> director") == "director"
    assert clean_relation("#2 >> place of death") == "place of death"
    assert clean_relation("who was born in?") == "who was born in?"
    assert clean_relation("   director   ") == "director"


def test_extract_twiki_style_oracle_edges():
    record = make_synthetic_record(
        qid="twiki_1",
        steps=[
            {"head": "Secret Lives", "relation": "director", "tail": "Edmond T. Gréville", "support_pids": ["twiki::p1"]},
            {"head": "Edmond T. Gréville", "relation": "place of death", "tail": "Nice", "support_pids": ["twiki::p2"]},
        ],
    )
    edges = extract_record_oracle_edges(record)
    assert len(edges) == 2
    assert edges[0].head == "ent:secret_lives"
    assert edges[0].relation == "director"
    assert edges[0].tail == "ent:edmond_t_greville"
    assert edges[0].confidence == 1.0
    assert edges[0].provenance == ("twiki::p1",)

    assert edges[1].head == "ent:edmond_t_greville"
    assert edges[1].relation == "place of death"
    assert edges[1].tail == "ent:nice"


def test_extract_musique_style_with_placeholders():
    record = make_synthetic_record(
        qid="musique_1",
        steps=[
            {"head": None, "relation": "Who wrote Hamlet?", "tail": "William Shakespeare", "support_pids": ["musique::Hamlet#1a2b3c4d"]},
            {"head": None, "relation": "#1 >> birthplace", "tail": "Stratford-upon-Avon", "support_pids": ["musique::Shakespeare_Bio"]},
        ],
    )
    edges = extract_record_oracle_edges(record)
    assert len(edges) == 2
    # Step 1 head inferred from paragraph title 'Hamlet'
    assert edges[0].head == "ent:hamlet"
    assert edges[0].tail == "ent:william_shakespeare"

    # Step 2 head resolved from step 1 tail via '#1'
    assert edges[1].head == "ent:william_shakespeare"
    assert edges[1].relation == "birthplace"
    assert edges[1].tail == "ent:stratford_upon_avon"


def test_build_oracle_graph_aggregates_support_and_provenance():
    r1 = make_synthetic_record(
        qid="q1",
        steps=[{"head": "Priya", "relation": "works_in", "tail": "DeptA", "support_pids": ["p1"]}],
    )
    r2 = make_synthetic_record(
        qid="q2",
        steps=[{"head": "Priya", "relation": "works_in", "tail": "DeptA", "support_pids": ["p2"]}],
    )
    edges = build_oracle_graph([r1, r2])
    assert len(edges) == 1
    edge = edges[0]
    assert edge.head == "ent:priya"
    assert edge.tail == "ent:depta"
    assert edge.support == 2
    assert set(edge.provenance) == {"p1", "p2"}


def test_edge_roundtrip_serialization():
    original = Edge(
        head="ent:alice",
        relation="founded",
        tail="ent:wonderland",
        confidence=0.95,
        support=3,
        provenance=("p1", "p2"),
        embedding=(),
    )
    raw = edge_to_dict(original)
    restored = edge_from_dict(raw)
    assert restored.head == original.head
    assert restored.relation == original.relation
    assert restored.tail == original.tail
    assert restored.confidence == original.confidence
    assert restored.support == original.support
    assert restored.provenance == original.provenance


def test_write_and_read_edges():
    edges = [
        Edge(head="ent:a", relation="r1", tail="ent:b", confidence=1.0, support=1, provenance=("p1",)),
        Edge(head="ent:b", relation="r2", tail="ent:c", confidence=1.0, support=2, provenance=("p2", "p3")),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "edges.jsonl"
        write_edges(edges, path)
        loaded = read_edges(path)
        assert len(loaded) == 2
        assert loaded[0] == edges[0]
        assert loaded[1] == edges[1]


def test_oracle_edges_load_into_graph():
    edges = [
        Edge(head="ent:a", relation="r1", tail="ent:b", confidence=1.0, support=1),
        Edge(head="ent:b", relation="r2", tail="ent:c", confidence=1.0, support=1),
    ]
    graph = Graph(edges)
    assert "ent:a" in graph.entities()
    assert "ent:b" in graph.entities()
    assert "ent:c" in graph.entities()
    assert len(graph.out_edges("ent:a")) == 1
    assert graph.out_edges("ent:a")[0].tail == "ent:b"


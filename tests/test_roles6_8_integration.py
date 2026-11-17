"""End-to-end Role 6/7/8 smoke tests using real repo contracts and no network calls."""
from __future__ import annotations

import json

import yaml

from data.loaders.base import GoldEvidence, Paragraph, QuestionRecord
from src.baselines.run import run_dataset as run_baselines
from src.common.tokenization import RegexTokenCounter
from src.context.answer_run import answer_file
from src.context.answerer import CallableAnswerer
from src.context.run import run_dataset as run_context
from src.evaluation.run import run_evaluation
from src.evaluation.splits import make_split
from src.graph.embeddings import EmbeddingService, mock_embed, write_entity_embeddings
from src.graph.oracle import write_edges
from src.traversal.types import Config, Edge


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _project(tmp_path):
    record = QuestionRecord(
        question_id="q1", dataset="toy", hop_level=2,
        question="Where was the person born?", answer="London",
        paragraphs=[
            Paragraph("toy::p1", "Person", "The person was born in London."),
            Paragraph("toy::p2", "London", "London is a city in England."),
        ],
        gold_evidence=GoldEvidence(paragraph_ids=("toy::p1",)),
        metadata={"answer_aliases": ["London, UK"]},
    )
    processed = tmp_path / "data" / "processed"
    _write_jsonl(processed / "toy_dev.jsonl", [record.to_dict()])

    q_emb = mock_embed(record.question)
    edge_text = "Person was born in London"
    edge = Edge(
        "ent:person", "was born in", "ent:london", 0.95, 2,
        ("toy::p1",), mock_embed(edge_text),
    )
    graph_base = tmp_path / "data" / "graph"
    graph_dataset = graph_base / "toy"
    write_edges([edge], graph_dataset / "edges.jsonl")
    write_entity_embeddings({
        "ent:person": mock_embed("Person"),
        "ent:london": mock_embed("London"),
    }, graph_dataset / "entity_embeddings.json")
    (graph_base / "manifest_embeddings.json").write_text(
        json.dumps({"mode": "mock", "model": "mock-64"}), encoding="utf-8"
    )

    extraction = tmp_path / "data" / "graph" / "extraction" / "toy"
    _write_jsonl(extraction / "entities.jsonl", [
        {"id": "ent:person", "name": "Person"},
        {"id": "ent:london", "name": "London"},
    ])
    query_dir = tmp_path / "data" / "query"
    _write_jsonl(query_dir / "toy_dev.jsonl", [{
        "question_id": "q1", "dataset": "toy", "seeds": ["ent:person"],
        "q_emb": list(q_emb), "matches": [],
    }])
    (query_dir / "toy_manifest.json").write_text(
        json.dumps({"embedding_mode": "mock", "questions": 1}), encoding="utf-8"
    )
    return record, tmp_path


def test_role6_context_and_answer_pipeline_end_to_end(tmp_path):
    record, root = _project(tmp_path)
    answerer = CallableAnswerer(lambda prompt: json.dumps({
        "answer": "London",
        "supporting_paragraph_ids": ["toy::p1", "not-in-context"],
        "evidence_sufficiency": "sufficient",
    }))
    cfg = Config(W=0.0, tau=-1.0, epsilon=-10.0, budget_B=1000, path_relevance_mode="mean_edge")
    summary = run_context(
        "toy", "auto", cfg, tag="test", answer_mode="gemini", answerer=answerer,
        processed_dir=str(root / "data" / "processed"),
        query_dir=str(root / "data" / "query"),
        graph_dir=str(root / "data" / "graph"),
        extraction_dir=str(root / "data" / "graph" / "extraction"),
        out_dir=str(root / "results" / "context"),
    )
    assert summary["questions"] == 1
    row = json.loads((root / "results/context/toy_auto_test.jsonl").read_text().strip())
    assert row["answer"] == "London"
    assert row["supporting_paragraph_ids"] == ["toy::p1"]
    assert "FACT:" in row["context"]
    assert row["retrieved_paragraph_ids"] == ["toy::p1"]
    assert row["n_candidate_paths"] >= 1
    assert row["prompt_tokens_shared"] > row["context_tokens"]
    assert row["prompt_tokens_model_reported"] is None  # injected answerer has no provider usage


def test_role7_s1_s2_s3_runner_end_to_end(tmp_path, monkeypatch):
    record, root = _project(tmp_path)
    config = root / "baselines.yaml"
    config.write_text(yaml.safe_dump({
        "chunk_sizes": [32], "chunk_overlap": 4, "vector_k": [1, 2],
        "naive_hops": [1], "context_budgets": [100, 200], "embedding_mode": "mock",
    }), encoding="utf-8")
    service = EmbeddingService(mode="mock", dimension=64, cache_dir=root / "cache")
    monkeypatch.setattr("src.baselines.run.get_service", lambda mode_override=None: service)
    summary = run_baselines(
        "toy", config_path=str(config), embedding_mode="mock", limit=1, tag="test",
        processed_dir=str(root / "data" / "processed"),
        query_dir=str(root / "data" / "query"),
        graph_dir=str(root / "data" / "graph"),
        out_dir=str(root / "results" / "baselines"),
    )
    rows = [json.loads(line) for line in (root / "results/baselines/toy_test.jsonl").read_text().splitlines()]
    assert {row["system"] for row in rows} == {"S1_full_context", "S2_vector_rag", "S3_naive_khop"}
    assert summary["questions"] == 1
    assert all(row["context_tokens"] >= 0 for row in rows)
    assert all(row["retrieval_latency_ms"] >= 0 for row in rows)


def test_role7_answer_runner_uses_shared_prompt_and_answerer(tmp_path):
    source = tmp_path / "retrieval.jsonl"
    output = tmp_path / "answers.jsonl"
    _write_jsonl(source, [{
        "question_id": "q1", "question": "Where?", "context": "SOURCE [p1] London",
        "retrieved_paragraph_ids": ["p1"], "system": "S1_full_context",
    }])
    answerer = CallableAnswerer(lambda prompt: json.dumps({
        "answer": "London", "supporting_paragraph_ids": ["p1", "p2"],
        "evidence_sufficiency": "sufficient",
    }))
    rows = answer_file(str(source), str(output), answerer=answerer)
    assert rows[0]["answer"] == "London"
    assert rows[0]["supporting_paragraph_ids"] == ["p1"]
    assert rows[0]["prompt_tokens_shared"] > 0
    assert rows[0]["answer_latency_ms"] >= 0
    assert rows[0]["latency_ms"] >= rows[0]["answer_latency_ms"]
    assert output.exists()


def test_role8_split_aware_evaluation_selects_s2_on_tune_only(tmp_path):
    records = [
        QuestionRecord(f"q{i}", "toy", 2, "Where?", "London",
                       [Paragraph(f"p{i}", "London", "London is here.")],
                       GoldEvidence((f"p{i}",)))
        for i in range(10)
    ]
    processed = tmp_path / "processed"
    _write_jsonl(processed / "toy_dev.jsonl", [record.to_dict() for record in records])
    split = make_split(records, tune_fraction=0.2, seed=7)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    rows = []
    for record in records:
        # Variant A is best on the tuning set; test rows retain the same exact variant.
        for system, params, score, tokens in [
            ("S2_vector_rag", {"chunk_size": 128, "k": 3}, 1.0 if record.question_id in split["tune_ids"] else 0.5, 100),
            ("S2_vector_rag", {"chunk_size": 256, "k": 5}, 0.0 if record.question_id in split["tune_ids"] else 1.0, 80),
            ("S6", {}, 0.7, 60),
        ]:
            rows.append({
                "question_id": record.question_id, "system": system, "parameters": params,
                "answer": "London" if score > 0 else "Paris",
                "retrieved_paragraph_ids": [f"p{record.question_id[1:]}"] if score > 0 else [],
                "gold_paragraph_ids": [f"p{record.question_id[1:]}"] ,
                "context_tokens": tokens,
            })
    inputs = tmp_path / "all_rows.jsonl"
    _write_jsonl(inputs, rows)
    summary = run_evaluation(
        "toy", [str(inputs)], tag="heldout", processed_dir=str(processed),
        out_dir=str(tmp_path / "eval"), split_manifest=str(split_path),
    )
    assert summary["split"]["tuning_vector_reference_variant"].endswith('"k":3}')
    assert summary["questions"] == len(split["test_ids"])
    assert summary["selected_vector_reference_variant"] == summary["split"]["tuning_vector_reference_variant"]


def test_shared_split_filter_requires_manifest_and_part(tmp_path):
    import pytest
    from src.common.splits import filter_records_by_split

    record, _ = _project(tmp_path)
    with pytest.raises(ValueError, match="both split_manifest and split_part"):
        filter_records_by_split([record], split_manifest="missing.json")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"tune_ids": ["q1"], "test_ids": []}), encoding="utf-8")
    assert filter_records_by_split([record], split_manifest=split_path, split_part="tune") == [record]

"""End-to-end check of `src.traversal.run.run_dataset` on a tiny synthetic dataset written to a temp dir,
using the real file formats of Roles 1, 3 and 4 (no network, no real data)."""
import json

from src.graph.embeddings import write_entity_embeddings
from src.graph.oracle import write_edges
from src.traversal.run import run_dataset
from src.traversal.toy_graph import Q_EMB, build_toy_graph
from src.traversal.types import Config


def test_run_dataset_end_to_end(tmp_path):
    toy = build_toy_graph()
    gdir = tmp_path / "graph" / "toy"
    write_edges(toy.edges, gdir / "edges.jsonl")
    write_entity_embeddings(toy.entity_embeddings, gdir / "entity_embeddings.json")

    rec = {"question_id": "q1", "dataset": "toy", "hop_level": 3, "question": "what tech?", "answer": "PyTorch",
           "paragraphs": [{"id": "p1", "title": "t", "text": "x"}],
           "gold_evidence": {"paragraph_ids": ["p1", "p2", "p3"], "sentence_facts": []}, "gold_chain": [], "metadata": {}}
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "toy_dev.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "query").mkdir()
    (tmp_path / "query" / "toy_dev.jsonl").write_text(
        json.dumps({"question_id": "q1", "dataset": "toy", "seeds": ["Priya"], "q_emb": list(Q_EMB), "matches": []}) + "\n")

    summary = run_dataset("toy", "auto", Config(budget_B=40), tag="t", graph_dir=str(tmp_path / "graph"),
                          query_dir=str(tmp_path / "query"), processed_dir=str(tmp_path / "processed"),
                          out_dir=str(tmp_path / "out"))
    assert summary["all"]["questions"] == 1
    assert summary["hop_3"]["evidence_recall"] == 1.0
    lines = (tmp_path / "out" / "toy_auto_t.jsonl").read_text().strip().splitlines()
    row = json.loads(lines[0])
    assert row["question_id"] == "q1" and row["n_selected_edges"] >= 3
    assert (tmp_path / "out" / "toy_auto_t_summary.json").exists()

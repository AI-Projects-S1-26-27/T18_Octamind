import json
import tempfile
from pathlib import Path

from src.graph.entities import Entity
from src.query import analyze_question
from src.query.run import run_dataset


def entity(entity_id, name, aliases=()):
    return Entity(id=entity_id, name=name, type="CONCEPT", aliases=aliases)


def test_analyze_question_returns_all_candidates_for_ambiguous_alias():
    entities = [
        entity("ent:mercury-planet", "Mercury", aliases=("Planet Mercury",)),
        entity("ent:mercury-element", "Mercury", aliases=("Element Mercury",)),
    ]

    analysis = analyze_question("What is Mercury made of?", entities, lambda _: (1, 2, 3))

    assert analysis.seeds == ("ent:mercury-element", "ent:mercury-planet")
    assert len(analysis.ambiguous_matches) == 1
    assert analysis.q_emb == (1.0, 2.0, 3.0)


def test_analyze_question_ranks_canonical_name_before_alias_without_capping():
    entities = [
        entity("ent:nyc-alias", "New York", aliases=("NYC",)),
        entity("ent:nyc-canonical", "NYC"),
    ]
    entities[0].mention_count = 100

    analysis = analyze_question("Where is NYC?", entities, lambda _: ())

    assert analysis.seeds == ("ent:nyc-canonical", "ent:nyc-alias")
    assert len(analysis.seeds) == 2


def test_analyze_question_prefers_longest_non_overlapping_match():
    entities = [
        entity("ent:york", "York"),
        entity("ent:new_york_city", "New York City"),
    ]

    analysis = analyze_question("Who lives in New York City?", entities, lambda _: (0.5,))

    assert analysis.seeds == ("ent:new_york_city",)
    assert [match.text for match in analysis.matches] == ["new york city"]


def test_analyze_question_matches_aliases_once_and_preserves_question_order():
    entities = [
        entity("ent:nyc", "New York City", aliases=("NYC",)),
        entity("ent:paris", "Paris"),
    ]

    analysis = analyze_question("Paris is farther from NYC than Paris.", entities, lambda _: ())

    assert analysis.seeds == ("ent:paris", "ent:nyc")
    assert [match.normalized for match in analysis.matches] == ["paris", "nyc"]


def test_analyze_question_has_no_seeds_when_nothing_matches():
    calls = []

    def embed(question):
        calls.append(question)
        return (0.0, 1.0)

    analysis = analyze_question("Who is unknown?", [], embed)

    assert analysis.seeds == ()
    assert analysis.matches == ()
    assert analysis.q_emb == (0.0, 1.0)
    assert calls == ["Who is unknown?"]


def test_run_dataset_writes_query_records_and_diagnostics():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed = root / "processed"
        extraction = root / "extraction" / "demo"
        output = root / "query"
        processed.mkdir()
        extraction.mkdir(parents=True)

        record = {
            "question_id": "demo-1",
            "dataset": "demo",
            "hop_level": 2,
            "question": "Where is Paris?",
            "answer": "France",
            "paragraphs": [],
            "gold_evidence": {"paragraph_ids": [], "sentence_facts": []},
            "gold_chain": [],
            "metadata": {},
        }
        (processed / "demo_dev.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        (extraction / "entities.jsonl").write_text(
            json.dumps(entity("ent:paris", "Paris").to_dict()) + "\n",
            encoding="utf-8",
        )

        manifest = run_dataset(
            "demo",
            processed_dir=processed,
            extraction_dir=root / "extraction",
            out_dir=output,
            mode="mock",
        )

        result = json.loads((output / "demo_dev.jsonl").read_text(encoding="utf-8"))
        assert result["seeds"] == ["ent:paris"]
        assert result["q_emb"]
        assert manifest["questions"] == 1
        assert manifest["questions_with_seeds"] == 1
        assert manifest["questions_without_seeds"] == 0

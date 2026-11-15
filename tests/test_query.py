from src.graph.entities import Entity
from src.query import analyze_question


def entity(entity_id, name, aliases=()):
    return Entity(id=entity_id, name=name, type="CONCEPT", aliases=aliases)


def test_analyze_question_returns_all_candidates_for_ambiguous_alias():
    entities = [
        entity("ent:mercury-planet", "Mercury"),
        entity("ent:mercury-element", "Mercury"),
    ]

    analysis = analyze_question("What is Mercury made of?", entities, lambda _: (1, 2, 3))

    assert analysis.seeds == ("ent:mercury-element", "ent:mercury-planet")
    assert len(analysis.ambiguous_matches) == 1
    assert analysis.q_emb == (1.0, 2.0, 3.0)


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

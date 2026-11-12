"""Tests for Role 2 entity extraction and linking.

Synthetic paragraphs and a fake LLM only: no network, no API key, no datasets.
"""
import json

import pytest
import yaml

from data.loaders.base import GoldEvidence, GoldStep, Paragraph, QuestionRecord, write_jsonl
from src.graph.entities import Mention, entity_id, normalize_name, split_sentences, text_sha1
from src.graph.eval_extraction import gold_entity_recall, gold_names
from src.graph.extract import (LLMExtractor, QuotaExhausted, RuleExtractor, is_daily_quota_error,
                               load_dotenv, parse_response)
from src.graph.linking import link_mentions, merge_by_embedding, name_index
from src.graph.run_extraction import ambiguous_ids, extract_dataset, run, unique_paragraphs

TEXT = ("John Smith works in the AI Department of Acme Corporation. "
        "He joined in March 2023. The department develops ML Systems, which use Python.")
PARA = Paragraph(id="toy::John Smith", title="John Smith", text=TEXT)


# ---- normalization and sentences ---------------------------------------------------
@pytest.mark.parametrize("a, b", [
    ("The Beatles", "beatles."),
    ("Pelé", "Pele"),
    ("New  York", "new york"),
])
def test_normalize_name_matches_variants(a, b):
    assert normalize_name(a) == normalize_name(b)


def test_normalize_keeps_short_article_names():
    assert normalize_name("The") == "the"
    assert normalize_name("A") == "a"


def test_entity_id_is_slug():
    assert entity_id("new york city") == "ent:new_york_city"
    assert entity_id("!!!") == "ent:unnamed"


def test_split_sentences_spans_and_initials():
    text = "J. R. R. Tolkien wrote books. He lived in Oxford. Dr. Who is fiction."
    sentences = [text[s:e] for s, e in split_sentences(text)]
    assert sentences == ["J. R. R. Tolkien wrote books.", "He lived in Oxford.", "Dr. Who is fiction."]


# ---- rule extractor ----------------------------------------------------------------
def test_rule_extractor_finds_names_dates_and_title():
    mentions = RuleExtractor().extract(PARA)
    texts = {m.text for m in mentions}
    assert {"John Smith", "AI Department of Acme Corporation", "March 2023", "ML Systems", "Python"} <= texts
    assert "He" not in texts and "March" not in texts
    title = [m for m in mentions if m.sentence_index == -1]
    assert len(title) == 1 and title[0].text == "John Smith"
    for m in mentions:
        if m.start >= 0:
            assert TEXT[m.start:m.end] == m.text      # offsets point into the paragraph
    assert next(m for m in mentions if m.text == "March 2023").type == "DATE"


def test_rule_extractor_drops_sentence_initial_article():
    p = Paragraph(id="toy::x", title="", text="The Beatles were a band from Liverpool.")
    texts = {m.text for m in RuleExtractor().extract(p)}
    assert "Beatles" in texts and "Liverpool" in texts


def test_rule_extractor_name_boundaries():
    text = ('It was published by Square Enix for the PlayStation 2. In the United States, '
            'Tetsuya Nomura said "It is great." Leonardo da Vinci and the University of the Arts differ.')
    mentions = RuleExtractor().extract(Paragraph(id="toy::y", title="", text=text))
    texts = {m.text for m in mentions}
    assert {"Square Enix", "PlayStation 2", "United States", "Tetsuya Nomura",
            "Leonardo da Vinci", "University of the Arts"} <= texts
    assert "It" not in texts
    assert max(m.sentence_index for m in mentions) == 2   # split after the closing quote


def test_rule_extractor_names_next_to_dates_and_brackets():
    # real MuSiQue cases where the name used to be swallowed by the following date
    text = ("It was the final career victory for Indy legend Mario Andretti (1993). "
            "Parent company Warner Music Group Founded March 19, 1958. "
            "Plague occurred in Venice 22 times. Mike Viola (of the Candy Butchers) sang. "
            "Warner Bros. Records was founded. Her feast day is May 4. It may rain.")
    mentions = RuleExtractor().extract(Paragraph(id="toy::z", title="", text=text))
    texts = {m.text for m in mentions}
    assert {"Mario Andretti", "1993", "Warner Music Group Founded", "March 19, 1958",
            "Venice", "Mike Viola", "Candy Butchers", "Warner Bros. Records", "May 4"} <= texts
    assert {m.text for m in mentions if m.type == "DATE"} == {"1993", "March 19, 1958", "May 4"}
    assert not any("(" in t for t in texts)


# ---- LLM extractor (fake model) ----------------------------------------------------
def fake_llm(calls):
    def generate(prompt):
        calls.append(prompt)
        return "```json\n" + json.dumps([
            {"text": "John Smith", "type": "person", "sentence": 0},
            {"text": "Acme Corporation", "type": "ORGANIZATION", "sentence": 0},
            {"text": "Python", "type": "LANGUAGE", "sentence": 2},     # unknown type -> CONCEPT
            {"text": "Berlin", "type": "LOCATION", "sentence": 1},     # not in sentence -> dropped
            {"text": "John Smith", "type": "PERSON", "sentence": 0},   # duplicate span
        ]) + "\n```"
    return generate


def test_llm_extractor_parses_validates_and_caches(tmp_path):
    calls = []
    extractor = LLMExtractor(fake_llm(calls), model="fake", cache_dir=tmp_path)
    mentions = extractor.extract(PARA)
    title, spans = mentions[0], mentions[1:]
    # title mention takes the type the model gave the same name in the text
    assert (title.text, title.type, title.sentence_index) == ("John Smith", "PERSON", -1)
    assert [(m.text, m.type) for m in spans] == [
        ("John Smith", "PERSON"), ("Acme Corporation", "ORGANIZATION"), ("Python", "CONCEPT")]
    assert extractor.dropped == 1 and extractor.calls == 1
    for m in spans:
        assert TEXT[m.start:m.end] == m.text

    again = LLMExtractor(fake_llm(calls), model="fake", cache_dir=tmp_path).extract(PARA)
    assert again == mentions and len(calls) == 1          # second run served from the cache


P2 = Paragraph(id="toy::Initech", title="Initech", text="Initech is based in Austin. Jane Doe founded it.")


def test_llm_batch_splits_by_paragraph_and_caches_per_paragraph(tmp_path):
    prompts = []

    def batched(prompt):
        prompts.append(prompt)
        return json.dumps([
            {"paragraph": 0, "sentence": 0, "text": "Acme Corporation", "type": "ORGANIZATION"},
            {"paragraph": 1, "sentence": 0, "text": "Austin", "type": "LOCATION"},
            {"paragraph": 1, "sentence": 1, "text": "Jane Doe", "type": "PERSON"},
            {"paragraph": 7, "sentence": 0, "text": "Nowhere", "type": "LOCATION"},   # bad index
        ])

    extractor = LLMExtractor(batched, "fake", tmp_path)
    first, second = extractor.extract_many([PARA, P2])
    assert len(prompts) == 1 and "Paragraph 1 (title: Initech)" in prompts[0]
    assert {m.text for m in first[1:]} == {"Acme Corporation"}
    assert {(m.text, m.type) for m in second[1:]} == {("Austin", "LOCATION"), ("Jane Doe", "PERSON")}
    assert extractor.dropped == 1
    # each paragraph is cached on its own: a later single-paragraph request hits the cache
    single = LLMExtractor(lambda _: pytest.fail("should be cached"), "fake", tmp_path)
    assert single.extract(P2) == second


def test_llm_batch_does_not_cache_skipped_paragraph(tmp_path):
    only_first = lambda _: json.dumps([{"paragraph": 0, "sentence": 0, "text": "Acme Corporation",
                                         "type": "ORGANIZATION"}])
    extractor = LLMExtractor(only_first, "fake", tmp_path)
    extractor.extract_many([PARA, P2])
    assert extractor.missing == 1
    calls = []
    retry = LLMExtractor(fake_llm(calls), "fake", tmp_path)
    retry.extract_many([PARA, P2])
    assert len(calls) == 1 and "Paragraph 0 (title: Initech)" in calls[0]   # only P2 is re-sent


def test_llm_invalid_response_is_not_cached(tmp_path):
    extractor = LLMExtractor(lambda _: "not json", "fake", tmp_path)
    assert [m.sentence_index for m in extractor.extract(PARA)] == [-1]   # only the title
    assert extractor.invalid == 1 and not list(tmp_path.glob("*.json"))


def test_quota_exhausted_stops_run(tmp_path, monkeypatch):
    from src.graph import run_extraction

    def broke(_):
        raise QuotaExhausted("per day limit")

    processed = tmp_path / "processed"
    write_jsonl([record([])], processed / "toy_dev.jsonl")
    config = tmp_path / "extraction.yaml"
    config.write_text(yaml.safe_dump({"processed_dir": str(processed), "out_dir": str(tmp_path / "out"),
                                      "datasets": ["toy"], "extractor": "llm",
                                      "llm": {"model": "fake", "batch_size": 4, "workers": 2}}), encoding="utf-8")
    monkeypatch.setattr(run_extraction, "build_extractor",
                        lambda cfg: LLMExtractor(broke, "fake", tmp_path / "cache"))
    assert run(str(config)) == {}
    assert not (tmp_path / "out" / "toy").exists()      # nothing half-written


def test_daily_quota_detection():
    assert is_daily_quota_error("Quota exceeded for metric: generate_content_free_tier_requests ... "
                                "Generate content requests per day limit exceeded")
    assert not is_daily_quota_error("429 Resource exhausted: requests per minute")


def test_llm_extractor_repeated_name_in_one_sentence(tmp_path):
    p = Paragraph(id="toy::r", title="", text="Paris loves Paris.")
    item = {"text": "Paris", "type": "LOCATION", "sentence": 0}
    mentions = LLMExtractor(lambda _: json.dumps([item, item, item]), "fake", tmp_path).extract(p)
    assert [(mm.start, mm.end) for mm in mentions] == [(0, 5), (12, 17)]


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport OCTA_A="1"\nOCTA_B=2\n', encoding="utf-8")
    monkeypatch.setenv("OCTA_B", "keep")
    monkeypatch.delenv("OCTA_A", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["OCTA_A"] == "1" and os.environ["OCTA_B"] == "keep"
    monkeypatch.delenv("OCTA_A")


def test_parse_response_tolerates_bad_output():
    assert parse_response("not json") == []
    assert parse_response('{"entities": [{"text": "x"}]}') == [{"text": "x"}]


# ---- linking -----------------------------------------------------------------------
def m(text, pid="p1", etype="CONCEPT"):
    return Mention(text, etype, pid, 0, 0, len(text))


def test_link_mentions_groups_by_normalized_name():
    entities, norm_to_id = link_mentions([
        m("The Beatles", "p1"), m("Beatles", "p2", "ORGANIZATION"), m("Beatles", "p2"), m("Liverpool")])
    by_id = {e.id: e for e in entities}
    beatles = by_id["ent:beatles"]
    assert beatles.name == "Beatles" and beatles.aliases == ("The Beatles",)
    assert beatles.type == "ORGANIZATION"           # specific type beats CONCEPT
    assert beatles.paragraph_ids == ("p1", "p2") and beatles.mention_count == 3
    assert norm_to_id["liverpool"] == "ent:liverpool"


def test_link_mentions_resolves_id_collisions():
    entities, _ = link_mentions([m("U.S."), m("U S")])
    assert sorted(e.id for e in entities) == ["ent:u_s", "ent:u_s-2"]


def test_merge_by_embedding_respects_type():
    vectors = {"NYC": (1.0, 0.0), "New York City": (0.99, 0.01), "New York (song)": (1.0, 0.0)}
    entities, _ = link_mentions([
        m("NYC", etype="LOCATION"), m("New York City", etype="LOCATION"),
        m("New York City", etype="LOCATION"), m("New York (song)", etype="WORK")])
    merged, remap = merge_by_embedding(entities, lambda s: vectors[s], threshold=0.95)
    assert remap["ent:nyc"] == remap["ent:new_york_city"] == "ent:new_york_city"
    assert remap["ent:new_york_song"] == "ent:new_york_song"
    city = next(e for e in merged if e.id == "ent:new_york_city")
    assert "NYC" in city.aliases and city.mention_count == 3
    assert name_index(merged)["nyc"] == {"ent:new_york_city"}


# ---- evaluation and runner ---------------------------------------------------------
def record(chain, answer="Python", dataset="toy"):
    return QuestionRecord(
        question_id="q1", dataset=dataset, hop_level=len(chain), question="Which tech?",
        answer=answer, paragraphs=[PARA], gold_evidence=GoldEvidence(paragraph_ids=(PARA.id,)),
        gold_chain=chain)


def test_gold_names_skip_placeholders():
    chain = [GoldStep(1, "works in", head="John Smith", tail="AI Department"),
             GoldStep(2, "who uses #1", tail="#1")]
    assert gold_names(record(chain)) == ["john smith", "ai department", "python"]
    assert gold_names(record([], answer="Yes")) == []


def test_gold_entity_recall_strict_and_lenient():
    chain = [GoldStep(1, "works in", head="John Smith", tail="AI Department"),
             GoldStep(2, "develops", tail="ML Systems")]
    rec = record(chain)
    _, mentions, _, _ = extract_dataset([rec], RuleExtractor(), log_every=0)
    result = gold_entity_recall([rec], mentions)["all"]
    # strict: john smith, ml systems, python; "ai department" only inside a longer name
    assert result["gold_names"] == 4
    assert result["strict_recall"] == 0.75 and result["lenient_recall"] == 1.0


def test_run_end_to_end(tmp_path):
    processed = tmp_path / "processed"
    write_jsonl([record([GoldStep(1, "r", head="John Smith", tail="Python")], dataset="toy")],
                processed / "toy_dev.jsonl")
    cfg = {"processed_dir": str(processed), "out_dir": str(tmp_path / "out"),
           "datasets": ["toy", "missing"], "extractor": "rule"}
    config = tmp_path / "extraction.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    report = run(str(config))
    assert set(report) == {"toy"}                        # missing dataset is skipped
    out = tmp_path / "out" / "toy"
    entities = [json.loads(l) for l in (out / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    mentions = [json.loads(l) for l in (out / "mentions.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {e["id"] for e in entities}
    assert "ent:john_smith" in ids and all(mm["entity_id"] in ids for mm in mentions)
    assert report["toy"]["gold_entity_recall"]["all"]["strict_recall"] == 1.0


def test_unique_paragraphs_dedupes_across_questions():
    r1, r2 = record([]), record([])
    assert unique_paragraphs([r1, r2]) == [PARA]


def test_same_id_different_text_is_not_dropped():
    # MuSiQue reuses "<dataset>::<title>" for different paragraphs of one article
    other = Paragraph(id=PARA.id, title=PARA.title, text="Jane Doe founded Initech in Austin.")
    r1 = record([GoldStep(1, "r", tail="Python")])
    r2 = QuestionRecord(question_id="q2", dataset="toy", hop_level=1, question="Who founded it?",
                        answer="Jane Doe", paragraphs=[other], gold_evidence=GoldEvidence((other.id,)))
    paragraphs, mentions, _, _ = extract_dataset([r1, r2], RuleExtractor(), log_every=0)
    assert len(paragraphs) == 2 and ambiguous_ids(paragraphs) == 1
    jane = next(mm for mm in mentions if mm.text == "Jane Doe")
    assert jane.paragraph_sha1 == text_sha1(other.text)
    assert other.text[jane.start:jane.end] == "Jane Doe"
    # each question only sees names from its own text variant
    assert gold_entity_recall([r2], mentions)["all"]["strict_recall"] == 1.0
    assert gold_entity_recall([r1], [mm for mm in mentions if mm.text == "Jane Doe"])["all"]["strict_recall"] == 0.0

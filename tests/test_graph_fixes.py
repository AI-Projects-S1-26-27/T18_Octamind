"""Regression tests for the Role 3 review fixes (oracle labels, LLM cache rules,
embedding alignment, inverse edges, loader). One test per bug that was found."""
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import yaml

from data.loaders.base import GoldEvidence, GoldStep, Paragraph, QuestionRecord, write_jsonl
from src.graph import embeddings, oracle, relations
from src.graph.extract import QuotaExhausted
from src.graph.graph_io import (
    INVERSE_PREFIX,
    add_inverse_edges,
    graph_report,
    is_inverse,
    load_graph,
)
from src.graph.relations import (
    LLMRelationExtractor,
    LinkedMention,
    RuleRelationExtractor,
    aggregate_relations,
    build_automatic_graph,
    build_extractor,
)
from src.traversal.types import Edge, Graph


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_record(steps, paragraphs=(), gold_pids=(), dataset="musique", qid="q"):
    chain = [
        GoldStep(step=i + 1, relation=s["relation"], head=s.get("head"), tail=s.get("tail"),
                 support_paragraph_ids=tuple(s.get("pids", ())))
        for i, s in enumerate(steps)
    ]
    return QuestionRecord(
        question_id=qid, dataset=dataset, hop_level=len(steps), question="?", answer="a",
        paragraphs=[Paragraph(id=pid, title=title, text=f"{title} text.") for pid, title in paragraphs],
        gold_evidence=GoldEvidence(paragraph_ids=tuple(gold_pids)), gold_chain=chain,
    )


P = Paragraph(id="toy::3", title="Secret Lives", text="Secret Lives was directed by Edmond T. Gréville.")
M = [
    LinkedMention("Secret Lives", "WORK", P.id, 0, 0, 12, "ent:secret_lives"),
    LinkedMention("Edmond T. Gréville", "PERSON", P.id, 0, 29, 47, "ent:edmond_t_greville"),
]


def llm_json(conf=0.9, sentence=None, head="ent:secret_lives", tail="ent:edmond_t_greville", rel="director"):
    item = {"head": head, "relation": rel, "tail": tail, "confidence": conf}
    if sentence is not None:
        item["sentence"] = sentence
    return json.dumps([item])


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
def test_entity_arrow_prefix_is_removed_from_relation():
    assert oracle.clean_relation("CIMI-FM >> licensed to broadcast") == "licensed to broadcast"
    assert oracle.clean_relation("#1 >> country of citizenship") == "country of citizenship"
    assert oracle.clean_relation("director") == "director"


def test_free_text_step_substitutes_placeholder():
    rec = make_record([
        {"relation": "CIMI-FM >> licensed to broadcast", "tail": "Quebec City", "pids": ["m::a"]},
        {"relation": "when did the first large winter carnival take place in #1 ", "tail": "1894", "pids": ["m::b"]},
    ])
    edges = oracle.extract_record_oracle_edges(rec)
    assert [(e.head, e.relation, e.tail) for e in edges] == [
        ("ent:cimi_fm", "licensed to broadcast", "ent:quebec_city"),
        ("ent:quebec_city", "when did the first large winter carnival take place in Quebec City", "ent:1894"),
    ]


def test_merge_step_links_every_dependency():
    rec = make_record([
        {"relation": "Film A >> director", "tail": "Alice", "pids": ["p1"]},
        {"relation": "Film B >> director", "tail": "Bob", "pids": ["p2"]},
        {"relation": "who is older, #1 or #2", "tail": "Carol", "pids": ["p3"]},
    ])
    edges = oracle.extract_record_oracle_edges(rec)
    into_carol = {e.head for e in edges if e.tail == "ent:carol"}
    assert into_carol == {"ent:alice", "ent:bob"}


def test_title_with_hash_is_not_truncated():
    assert oracle.title_from_pid("musique::C# (programming language)#1a2b3c4d") == "C# (programming language)"
    assert oracle.title_from_pid("musique::Hamlet#1a2b3c4d-2") == "Hamlet"
    assert oracle.title_from_pid("musique::Plain Title") == "Plain Title"
    rec = make_record([{"relation": "Which language did Anders create?", "tail": "C Sharp",
                        "pids": ["musique::C# (programming language)#1a2b3c4d"]}])
    (edge,) = oracle.extract_record_oracle_edges(rec)
    assert edge.head == "ent:c_programming_language"


def test_provenance_prefers_paragraphs_matching_head_or_tail():
    rec = make_record(
        [{"head": "Film A", "relation": "director", "tail": "Person B"}],     # 2Wiki: no support ids
        paragraphs=[("t::1", "Film A"), ("t::2", "Person B"), ("t::3", "Unrelated")],
        gold_pids=["t::1", "t::2", "t::3"], dataset="twiki",
    )
    (edge,) = oracle.extract_record_oracle_edges(rec)
    assert edge.provenance == ("t::1", "t::2")


def test_provenance_falls_back_to_gold_paragraphs_when_no_title_matches():
    rec = make_record(
        [{"head": "X", "relation": "r", "tail": "Y"}],
        paragraphs=[("t::1", "Foo"), ("t::2", "Bar")], gold_pids=["t::1", "t::2"], dataset="twiki",
    )
    (edge,) = oracle.extract_record_oracle_edges(rec)
    assert edge.provenance == ("t::1", "t::2")


def test_skipped_steps_are_counted():
    rec = make_record([
        {"relation": "A >> r", "tail": None},                  # no tail
        {"relation": "what is #9", "tail": "Z"},               # placeholder never defined, no support -> no head
        {"head": "Same", "relation": "r", "tail": "same"},     # self loop
        {"head": "H", "relation": "r", "tail": "T"},           # fine
    ])
    stats = Counter()
    edges = oracle.extract_record_oracle_edges(rec, stats)
    assert len(edges) == 1
    assert stats["steps"] == 4 and stats["edges"] == 1
    assert stats["skip_no_tail"] == 1 and stats["skip_no_head"] == 1 and stats["skip_self_loop"] == 1


def test_write_edges_is_atomic(tmp_path):
    path = tmp_path / "edges.jsonl"
    good = [Edge("ent:a", "r", "ent:b", 1.0, 1)]
    oracle.write_edges(good, path)
    with pytest.raises(AttributeError):
        oracle.write_edges([Edge("ent:c", "r", "ent:d", 1.0, 1), object()], path)   # fails halfway
    assert oracle.read_edges(path) == good                                           # original untouched
    assert list(tmp_path.glob("*.part")) == []                                       # no leftovers


# --------------------------------------------------------------------------- #
# relations
# --------------------------------------------------------------------------- #
def test_failed_llm_call_is_not_cached_and_is_retried(tmp_path):
    def boom(_):
        raise RuntimeError("503 high demand")

    ex = LLMRelationExtractor(boom, "m", tmp_path)
    assert ex.extract(P, M) == []
    assert ex.failed_paragraphs == [P.id]

    ex2 = LLMRelationExtractor(lambda _: llm_json(), "m", tmp_path)
    assert len(ex2.extract(P, M)) == 1                      # not stuck on the earlier failure


def test_unparseable_answer_is_not_cached(tmp_path):
    ex = LLMRelationExtractor(lambda _: "not json at all", "m", tmp_path)
    assert ex.extract(P, M) == []
    assert ex.invalid_responses == 1
    ex2 = LLMRelationExtractor(lambda _: llm_json(), "m", tmp_path)
    assert len(ex2.extract(P, M)) == 1


def test_empty_list_is_a_valid_cached_answer(tmp_path):
    calls = []
    ex = LLMRelationExtractor(lambda p: calls.append(p) or "[]", "m", tmp_path)
    assert ex.extract(P, M) == [] and ex.extract(P, M) == []
    assert len(calls) == 1 and ex.cache_hits == 1


def test_quota_exhaustion_is_flagged_and_not_cached(tmp_path):
    def quota(_):
        raise QuotaExhausted("per day")

    ex = LLMRelationExtractor(quota, "m", tmp_path)
    assert ex.extract(P, M) == []
    assert ex.quota_exhausted and ex.failed_paragraphs == [P.id]
    assert len(LLMRelationExtractor(lambda _: llm_json(), "m", tmp_path).extract(P, M)) == 1


@pytest.mark.parametrize("raw,expected", [(0.3, 0.5), (1.7, 1.0), ("high", 0.9), (None, 0.9), (0.8, 0.8)])
def test_confidence_is_clamped_identically_on_first_run_and_from_cache(tmp_path, raw, expected):
    ex = LLMRelationExtractor(lambda _: llm_json(conf=raw), "m", tmp_path)
    fresh = ex.extract(P, M)[0].confidence
    cached = ex.extract(P, M)[0].confidence
    assert fresh == cached == expected


def test_malformed_items_are_skipped_not_fatal(tmp_path):
    payload = json.dumps(["just a string", 5, {"head": ["x"], "relation": "r", "tail": "ent:secret_lives"},
                          {"head": "ent:secret_lives", "relation": "director", "tail": "ent:edmond_t_greville"}])
    rels = LLMRelationExtractor(lambda _: payload, "m", tmp_path).extract(P, M)
    assert [(r.head, r.relation, r.tail) for r in rels] == [("ent:secret_lives", "director", "ent:edmond_t_greville")]


def test_sentence_index_from_model_gives_sentence_level_support(tmp_path):
    p = Paragraph(id="toy::4", title="Secret Lives",
                  text="Secret Lives was directed by Edmond T. Gréville. Gréville also directed Secret Lives.")
    answer = json.dumps([
        {"head": "ent:secret_lives", "relation": "director", "tail": "ent:edmond_t_greville", "confidence": 0.9, "sentence": 0},
        {"head": "ent:secret_lives", "relation": "director", "tail": "ent:edmond_t_greville", "confidence": 0.9, "sentence": 1},
        {"head": "ent:secret_lives", "relation": "director", "tail": "ent:edmond_t_greville", "confidence": 0.9, "sentence": 99},
    ])
    ms = [LinkedMention("Secret Lives", "WORK", p.id, 0, 0, 12, "ent:secret_lives"),
          LinkedMention("Edmond T. Gréville", "PERSON", p.id, 0, 29, 47, "ent:edmond_t_greville")]
    rels = LLMRelationExtractor(lambda _: answer, "m", tmp_path).extract(p, ms)
    (edge,) = aggregate_relations(rels)
    assert edge.support == 2                                 # sentences 0 and 1; the out-of-range 99 maps to 0


def test_cache_is_not_reused_when_the_entity_set_changes(tmp_path):
    calls = []
    ex = LLMRelationExtractor(lambda p: calls.append(p) or "[]", "m", tmp_path)
    ex.extract(P, M)
    ex.extract(P, M[:1] + [LinkedMention("Nice", "LOCATION", P.id, 0, 0, 4, "ent:nice")])   # Role 2 changed its entities
    assert len(calls) == 2


def test_unknown_extractor_mode_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown extractor mode 'oracle'"):
        build_extractor({}, "oracle")


def test_rule_confidences_come_from_config():
    ex = build_extractor({"relation": {"pair_confidence": 0.7, "title_confidence": 0.55}}, "rule")
    assert (ex.pair_confidence, ex.title_confidence) == (0.7, 0.55)


def _toy_corpus():
    text = "Alpha Corp was founded by Beta Person. It operates in Gamma City."
    rec = make_record([], paragraphs=[("t::alpha", "Alpha Corp")])
    rec.paragraphs = [Paragraph(id="t::alpha", title="Alpha Corp", text=text)]
    ms = [
        LinkedMention("Alpha Corp", "ORGANIZATION", "t::alpha", -1, -1, -1, "ent:alpha_corp"),
        LinkedMention("Alpha Corp", "ORGANIZATION", "t::alpha", 0, 0, 10, "ent:alpha_corp"),
        LinkedMention("Beta Person", "PERSON", "t::alpha", 0, text.index("Beta"), text.index("Beta") + 11, "ent:beta_person"),
        LinkedMention("Gamma City", "LOCATION", "t::alpha", 1, text.index("Gamma"), text.index("Gamma") + 10, "ent:gamma_city"),
    ]
    return rec, {"t::alpha": ms}


def test_min_confidence_filters_weak_relations():
    rec, ms = _toy_corpus()
    ex = RuleRelationExtractor()
    all_edges = build_automatic_graph([rec], ms, ex, min_confidence=0.0)
    strong = build_automatic_graph([rec], ms, ex, min_confidence=0.55)
    assert len(all_edges) == 2 and {e.confidence for e in all_edges} == {0.6, 0.5}
    assert [e.confidence for e in strong] == [0.6]


def test_parallel_build_matches_sequential(tmp_path):
    rec, ms = _toy_corpus()
    seq = build_automatic_graph([rec], ms, RuleRelationExtractor(), workers=1)
    par = build_automatic_graph([rec], ms, RuleRelationExtractor(), workers=4)
    assert seq == par


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def test_embed_many_never_silently_drops_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "make_gemini_client", lambda: object())
    monkeypatch.setattr(embeddings, "gemini_embed_batch",
                        lambda texts, model=None, client=None, **k: [(1.0, 0.0)] * (len(texts) - 1))
    svc = embeddings.EmbeddingService(mode="gemini", model="m", cache_dir=tmp_path, batch_size=100)
    with pytest.raises(RuntimeError, match="vectors for"):
        svc.embed_many(["a", "b", "c"])


def test_gemini_embed_batch_rejects_wrong_vector_count():
    class Item:
        values = [0.1, 0.2]

    class Client:
        class models:
            @staticmethod
            def embed_content(model, contents):
                class R:
                    embeddings = [Item()]
                return R()

    vectors = embeddings.gemini_embed_batch(["a"], client=Client(), max_attempts=1)
    assert vectors == [(0.1, 0.2)]


def test_gemini_embed_batch_rejects_one_vector_for_multiple_texts():
    class Item:
        values = [0.1, 0.2]

    class Client:
        class models:
            @staticmethod
            def embed_content(model, contents):
                class R:
                    embeddings = [Item()]
                if isinstance(contents, list) and len(contents) > 1:
                    R.embeddings = [Item()]
                return R()

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        embeddings.gemini_embed_batch(["a", "b"], client=Client(), max_attempts=1)


def test_gemini_embed_batch_stops_after_bounded_503(monkeypatch):
    class ServerError(Exception):
        code = 503

        def __str__(self):
            return "503 UNAVAILABLE: service unavailable"

    class Client:
        class models:
            @staticmethod
            def embed_content(model, contents):
                raise ServerError()

    monkeypatch.setattr(embeddings.time, "sleep", lambda _: None)
    with pytest.raises(ServerError):
        embeddings.gemini_embed_batch(["a"], client=Client(), max_attempts=3)


def test_gemini_embed_batch_treats_daily_429_as_quota(monkeypatch):
    class RateError(Exception):
        code = 429

        def __str__(self):
            return "429 RESOURCE_EXHAUSTED: quota reached per day"

    class Client:
        class models:
            @staticmethod
            def embed_content(model, contents):
                raise RateError()

    monkeypatch.setattr(embeddings.time, "sleep", lambda _: None)
    with pytest.raises(QuotaExhausted):
        embeddings.gemini_embed_batch(["a"], client=Client(), max_attempts=5)


def test_embedding_service_uses_individual_fallback(tmp_path, monkeypatch):
    service = embeddings.EmbeddingService(
        mode="gemini",
        model="m",
        cache_dir=tmp_path,
        batch_size=2,
    )
    service._client = object()
    calls = []

    def fake_batch(texts, model=None, client=None, max_attempts=5):
        calls.append((list(texts), max_attempts))
        if len(texts) == 2:
            raise RuntimeError("embedding API returned 1 vectors for 2 texts")
        return [(float(ord(texts[0])), 0.0)]

    monkeypatch.setattr(embeddings, "gemini_embed_batch", fake_batch)
    vectors = service.embed_many(["a", "b"])

    assert vectors == [(97.0, 0.0), (98.0, 0.0)]
    assert calls == [(["a", "b"], 5), (["a"], 3), (["b"], 3)]


def test_embedding_service_shows_individual_fallback_progress(tmp_path, monkeypatch, capsys):
    service = embeddings.EmbeddingService(
        mode="gemini",
        model="m",
        cache_dir=tmp_path,
        batch_size=2,
    )
    service._client = object()

    def fake_batch(texts, model=None, client=None, max_attempts=5):
        if len(texts) == 2:
            raise RuntimeError("embedding API returned 1 vectors for 2 texts")
        return [(1.0, 0.0)]

    monkeypatch.setattr(embeddings, "gemini_embed_batch", fake_batch)
    service.embed_many(["a", "b"])

    output = capsys.readouterr().out
    assert "batch mismatch" in output
    assert "individual fallback: 2/2 texts" in output


def test_entity_embeddings_file_is_reused(tmp_path):
    path = tmp_path / "entity_embeddings.json"
    embeddings.write_entity_embeddings({"ent:a": (0.1, 0.2)}, path)
    assert embeddings.read_entity_embeddings(path) == {"ent:a": (0.1, 0.2)}


def test_unknown_embedding_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        embeddings.EmbeddingService(mode="text-embedding-004", cache_dir=tmp_path)


def test_edge_text_uses_entity_names_when_given(tmp_path):
    e = Edge("ent:barack_obama", "was born in", "ent:honolulu", 0.6, 1)
    assert embeddings.edge_text(e) == "ent:barack_obama was born in ent:honolulu"
    assert embeddings.edge_text(e, {"ent:barack_obama": "Barack Obama"}) == "Barack Obama was born in honolulu"
    svc = embeddings.EmbeddingService(mode="mock", cache_dir=tmp_path, dimension=16)
    (out,) = embeddings.populate_edge_embeddings([e], svc, {"ent:barack_obama": "Barack Obama", "ent:honolulu": "Honolulu"})
    assert out.embedding == svc.embed("Barack Obama was born in Honolulu")


def test_extra_endpoint_entities_get_embeddings(tmp_path):
    ents = tmp_path / "entities.jsonl"
    ents.write_text(json.dumps({"id": "ent:cimi_fm", "name": "CIMI-FM"}) + "\n", encoding="utf-8")
    svc = embeddings.EmbeddingService(mode="mock", cache_dir=tmp_path / "c", dimension=8)
    out = embeddings.embed_dataset_entities(ents, svc, extra_ids=["ent:cimi_fm", "ent:quebec_city"])
    assert set(out) == {"ent:cimi_fm", "ent:quebec_city"}
    assert out["ent:quebec_city"] == svc.embed("quebec city")


def test_write_entity_embeddings_leaves_no_partial_file(tmp_path):
    path = tmp_path / "e.json"
    embeddings.write_entity_embeddings({"ent:a": (0.1, 0.2)}, path)
    assert embeddings.read_entity_embeddings(path) == {"ent:a": (0.1, 0.2)}
    assert list(tmp_path.glob("*.part")) == []


# --------------------------------------------------------------------------- #
# inverse edges + loader
# --------------------------------------------------------------------------- #
def test_inverse_edges_let_a_seed_on_the_object_expand():
    fwd = Edge("ent:inception", "was directed by", "ent:christopher_nolan", 0.6, 2, ("p1",), (1.0, 0.0))
    assert Graph([fwd]).out_edges("ent:christopher_nolan") == []
    g = Graph(add_inverse_edges([fwd]))
    (inv,) = g.out_edges("ent:christopher_nolan")
    assert inv.tail == "ent:inception" and inv.relation == INVERSE_PREFIX + "was directed by"
    assert (inv.confidence, inv.support, inv.provenance, inv.embedding) == (0.6, 2, ("p1",), (1.0, 0.0))
    assert is_inverse(inv) and not is_inverse(fwd)


def test_add_inverse_edges_is_idempotent_and_never_doubles():
    edges = [Edge("ent:a", "r", "ent:b", 1.0, 1)]
    once = add_inverse_edges(edges)
    assert len(once) == 2
    assert len(add_inverse_edges(once)) == 2


def test_load_graph_round_trip_and_report(tmp_path):
    base = tmp_path / "toy"
    oracle.write_edges([Edge("ent:a", "r", "ent:b", 1.0, 1, ("p",), (0.1, 0.2))], base / "oracle_edges.jsonl")
    embeddings.write_entity_embeddings({"ent:a": (0.1, 0.2)}, base / "entity_embeddings.json")   # ent:b is missing
    g = load_graph("toy", kind="oracle", graph_dir=tmp_path)
    assert len(g.out_edges("ent:b")) == 1                                     # inverse added
    rep = graph_report(g)
    assert rep["edges"] == 2 and rep["inverse_edges"] == 1
    assert rep["missing_entity_embeddings"] == 1 and rep["missing_entity_examples"] == ["ent:b"]
    assert rep["missing_edge_embeddings"] == 0 and not rep["bad_embedding_dims"]
    assert len(load_graph("toy", kind="oracle", graph_dir=tmp_path, add_inverse=False).edges) == 1


def test_load_graph_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph("nope", kind="auto", graph_dir=tmp_path)
    with pytest.raises(ValueError):
        load_graph("nope", kind="weird", graph_dir=tmp_path)


# --------------------------------------------------------------------------- #
# end to end: oracle -> relations -> embeddings -> load_graph
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end_produces_complete_graphs(tmp_path):
    text = "Alpha Corp was founded by Beta Person. It operates in Gamma City."
    rec = QuestionRecord(
        question_id="q1", dataset="musique", hop_level=2, question="?", answer="Gamma City",
        paragraphs=[Paragraph(id="musique::Alpha Corp", title="Alpha Corp", text=text)],
        gold_evidence=GoldEvidence(paragraph_ids=("musique::Alpha Corp",)),
        gold_chain=[
            GoldStep(1, "Alpha Corp >> founded by", "Beta Person", None, ("musique::Alpha Corp",)),
            GoldStep(2, "#1 >> birthplace", "Delta Town", None, ("musique::Alpha Corp",)),   # Delta Town: never extracted
        ],
    )
    hotpot = QuestionRecord("h1", "hotpotqa", 2, "?", "a", rec.paragraphs, GoldEvidence(), [])
    processed = tmp_path / "processed"
    write_jsonl([rec], processed / "musique_dev.jsonl")
    write_jsonl([hotpot], processed / "hotpotqa_dev.jsonl")

    ext = tmp_path / "extraction" / "musique"
    ext.mkdir(parents=True)
    pid = "musique::Alpha Corp"
    mention = lambda t, ty, s, a, b, e: {"text": t, "type": ty, "paragraph_id": pid, "sentence_index": s,
                                         "start": a, "end": b, "entity_id": e}
    b0, g0 = text.index("Beta"), text.index("Gamma")
    (ext / "mentions.jsonl").write_text("\n".join(json.dumps(m) for m in [
        mention("Alpha Corp", "ORGANIZATION", -1, -1, -1, "ent:alpha_corp"),
        mention("Alpha Corp", "ORGANIZATION", 0, 0, 10, "ent:alpha_corp"),
        mention("Beta Person", "PERSON", 0, b0, b0 + 11, "ent:beta_person"),
        mention("Gamma City", "LOCATION", 1, g0, g0 + 10, "ent:gamma_city"),
    ]), encoding="utf-8")
    (ext / "entities.jsonl").write_text("\n".join(json.dumps({"id": i, "name": n}) for i, n in [
        ("ent:alpha_corp", "Alpha Corp"), ("ent:beta_person", "Beta Person"), ("ent:gamma_city", "Gamma City")]),
        encoding="utf-8")

    cfg = tmp_path / "graph.yaml"
    cfg.write_text(yaml.safe_dump({
        "processed_dir": str(processed), "extraction_dir": str(tmp_path / "extraction"), "out_dir": str(tmp_path / "graph"),
        "datasets": ["musique", "hotpotqa"], "extractor": "rule",
        "relation": {"min_confidence": 0.5},
        "embedding": {"mode": "mock", "dimension": 8, "cache_dir": str(tmp_path / "cache")},
    }), encoding="utf-8")

    om = oracle.run(cfg)
    assert set(om["datasets"]) == {"musique"}                                    # hotpotqa skipped: no gold chain
    assert not (tmp_path / "graph" / "hotpotqa" / "oracle_edges.jsonl").exists()
    assert om["datasets"]["musique"]["gold_steps"] == 2 and om["datasets"]["musique"]["edges"] == 2

    relations.run(cfg, dataset_filter="musique")
    em = embeddings.run(cfg, dataset_filter="musique")
    assert em["mock"] is True
    assert em["datasets"]["musique"]["endpoint_entities_added"] == 1             # ent:delta_town

    for kind in ("oracle", "auto"):
        g = load_graph("musique", kind=kind, graph_dir=tmp_path / "graph")
        rep = graph_report(g)
        assert rep["missing_edge_embeddings"] == 0 and rep["missing_entity_embeddings"] == 0, (kind, rep)
        assert rep["embedding_dims"] == [8] and rep["inverse_edges"] == rep["edges"] // 2
        assert all(math.isclose(sum(x * x for x in e.embedding), 1.0, rel_tol=1e-5) for e in g.edges)
    assert len(load_graph("musique", "oracle", tmp_path / "graph").out_edges("ent:delta_town")) == 1

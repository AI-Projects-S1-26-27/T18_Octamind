# Entity and concept extraction (Role 2)

Owner: Shantanu Pandey. Code: `src/graph/` (`entities.py`, `extract.py`, `linking.py`,
`eval_extraction.py`, `run_extraction.py`). Config: `configs/extraction.yaml`.
Output format: `docs/INTERFACE.md`, section "Entities (Role 2 to Role 3)".

## What it does

For every distinct paragraph of the sampled dev sets (Role 1, `data/processed/`):

1. split the paragraph into sentences (`split_sentences`);
2. extract typed mentions of entities and concepts:
   - **LLM extractor (main):** Gemini, temperature 0, 10 paragraphs per request, JSON output.
     Every item is checked against the sentence text: a span the model invented is dropped, so
     offsets are always exact;
   - **rule extractor (baseline):** paragraph title, capitalized name spans, dates; no API,
     untyped apart from DATE;
3. add a title mention per paragraph (the paragraph's subject, for "He was born ..." sentences);
4. link mentions into canonical entities by normalized name (`The Beatles` = `Beatles` =
   `beatles.`), with the majority type;
5. score extraction against the gold chains (see Results).

## How to run

```bash
pip install -r requirements.txt
python -m data.sampling                      # Role 1: data/processed/*_dev.jsonl
python -m src.graph.run_extraction           # LLM extractor (needs GEMINI_API_KEY in .env)
python -m src.graph.run_extraction --extractor rule --out-dir data/graph/extraction_rule
python -m src.graph.run_extraction --dataset musique --limit 20   # quick check
```

LLM answers are cached per paragraph in `data/cache/extraction/`, so:

- re-running costs no API calls and gives byte-identical output;
- an interrupted run, failed batches (503 "high demand") or a daily-quota stop are resumed by
  running the same command again; only missing paragraphs are sent;
- changing the model or the prompt (`PROMPT_VERSION`) starts a fresh cache.

The manifest (`data/graph/extraction/<dataset>/manifest.json`) records counts, entity types,
API calls, failed/invalid/skipped paragraphs and gold-entity recall. A run is complete when
`failed_paragraphs` is empty and `llm_invalid_responses` and `llm_missing_paragraphs` are 0.

## Model choice and quota

Pinned: `gemini-3.5-flash-lite` (2026-11-12). On the free API tier `gemini-2.5-flash` is closed to
new users and the Flash models allow only 20 requests per day, which cannot cover ~10.5k
paragraphs; flash-lite has a usable free quota. Batching 10 paragraphs per request brings the
full run to ~1,050 requests. With billing enabled, raise `requests_per_minute` and `workers` in
the config (or switch to a Flash model, which needs a fresh cache).

## Results (sampled dev sets, seed 0)

Gold-entity recall: share of gold-chain heads/tails and final answers (yes/no answers excluded)
that appear as an extracted mention in one of the question's paragraphs. **Strict** = same
normalized name; **lenient** = one contains the other at word boundaries ("1947" / "June 1947").
An entity that is never extracted cannot be reached by traversal, so strict recall bounds path
recall on the automatic graph (RQ4). HotpotQA has no gold chains, so only its answers are scored.

Rule baseline (no API):

| Dataset | Questions | Paragraphs | Mentions | Entities | Strict | Lenient |
|---|---|---|---|---|---|---|
| MuSiQue | 600 | 7,049 | 101,333 | 47,205 | 0.842 | 0.951 |
| 2Wiki | 200 | 1,462 | 20,194 | 12,019 | 0.930 | 0.995 |
| HotpotQA | 200 | 1,966 | 31,041 | 16,706 | 0.652 | 0.837 |

LLM extractor, live test (first 3 MuSiQue questions, 60 paragraphs, `gemini-3.5-flash-lite`):

| Extractor | Mentions | Entities | Strict | Bad offsets | Invalid answers |
|---|---|---|---|---|---|
| LLM | 1,193 | 886 | 1.00 (6/6) | 0 | 0 |
| Rule | 1,038 | 717 | 0.83 (5/6) | 0 | n/a |

LLM entity types on that sample: PERSON 312, ORGANIZATION 127, NUMBER 125, DATE 97, CONCEPT 84,
LOCATION 70, WORK 62, EVENT 9 (the rule baseline only produces CONCEPT and DATE).

Full LLM run over all three datasets: in progress as of 2026-11-12 (free tier, slowed by 503
"high demand" retries; it resumes from the cache when re-run). When it finishes, the numbers are
in each `manifest.json`; add them here.

## Known limits

- The rule baseline cannot type entities and merges run-on capitalized text from tables
  ("Wealth of Nations Author Adam Smith Country Scotland"); it is a baseline only.
- The LLM sometimes picks a debatable type (awards as EVENT, a film title as CONCEPT). Entity
  type is the majority over mentions, which smooths most of this.
- Linking is exact on normalized names: "NYC" and "New York City" stay separate entities.
  `linking.merge_by_embedding` implements the embedding merge; it is switched off
  (`linking.embedding_merge`) until the shared `embed()` service exists.
- Pronouns are not resolved (by design: the title mention covers the common "He ..." case).

# Dynamic context construction and LLM integration (Role 6)

Code: `src/context/` (`serializer.py`, `prompt.py`, `answerer.py`, `run.py`, `answer_run.py`).
Config: `configs/context.yaml`.

## Contract

Role 5 returns selected `Edge` objects (`H`) and the search diagnostics. Role 6:

1. Resolves each edge's `provenance` paragraph IDs against the **dataset-wide sampled corpus**.
2. Serializes each unique fact once, normalizing generated inverse edges back to the direction of the source fact.
3. Includes a readable triple and up to `max_sources_per_edge` supporting source paragraphs.
4. Uses the exact same `edge_block()` function during Role 5 selection and final serialization, so the configured budget counts the context representation rather than a triple-only proxy.
5. Builds the same evidence-grounded QA prompt for S1-S7 and records the answer, citations and model-reported input/output tokens when Gemini generation is enabled.

The graph edge schema stores paragraph provenance, not a sentence index. Consequently this implementation includes the full source paragraph(s) available in the selected context, not a claimed exact supporting sentence. `max_sources_per_edge: 0` is available for the proposal's triple-without-source-text ablation.

## Offline serialization smoke test

This mode does not call an answer model. For an API-free plumbing test with mock graph/query embeddings:

```bash
python -m src.context.run --dataset musique --kind auto --limit 5 \
  --tag offline --path-relevance-mode mean_edge
```

It writes `results/context/musique_auto_offline.jsonl` and a summary JSON. The run uses the full sampled corpus for provenance lookup even when `--limit` or a split manifest restricts the query count. The output includes `context_tokens` (shared regex count of evidence only), `prompt_tokens_shared` (same counter applied to the complete common prompt), and Gemini-reported usage when generation is enabled.

## Research answer generation

The default Role 5 config uses full-text path embeddings. After Role 3 graph/entity/edge embeddings and Role 4 question embeddings have been generated with the configured real embedding model, run:

```bash
python -m src.context.run --dataset musique --kind auto --tag s6 --answer-mode gemini
python -m src.context.run --dataset musique --kind oracle --tag s7 --answer-mode gemini
```

`GEMINI_API_KEY` is read from the local git-ignored `.env` or environment. The answer model is pinned in `configs/context.yaml` / `--answer-model`; temperature is 0. The model receives a common prompt that asks for an answer grounded only in the retrieved context and returns JSON with `answer`, `supporting_paragraph_ids`, and `evidence_sufficiency`.

To generate answers for a Role 7 retrieval file, use the same prompt and answerer:

```bash
python -m src.context.answer_run \
  --input results/baselines/musique_dev.jsonl \
  --output results/answers/musique_baselines.jsonl
```

## Shared split, alternate graph copies, and output

`src.context.run` accepts `--split-manifest` plus `--split-part tune|test`, and directory overrides
`--processed-dir`, `--query-dir`, `--graph-dir`, `--extraction-dir`, and `--out-dir`. These are
needed for the same fixed tuning/test IDs across S6/S7/baselines and for RQ4's isolated corrupted
graphs. The answer model is called once per evaluated question; use small `--limit` smoke tests
before starting full runs.

`src.context.answer_run` applies the same `build_qa_prompt()` and Gemini answerer to saved Role 7
S1-S3 retrieval rows, and filters model citations to paragraph IDs actually present in that row's
retrieved context. It records answer latency, shared prompt token count, and provider-reported usage.

## Token counts and limits

`regex_wordpunct_v1` is the shared, deterministic tokenizer used by Roles 5-8 for comparable local budgets and retrieval summaries. It is **not Gemini's native tokenizer**. When Gemini answer generation runs, its usage metadata is recorded separately as `prompt_tokens_model_reported` and `output_tokens_model_reported`. Do not call the regex count an exact provider token count. The same tokenizer and prompt template must be used for every system in a comparison.

The context budget covers the serialized evidence context, not the question/instruction portion of the full prompt. The full prompt's actual input-token usage is reported separately by Gemini when available.

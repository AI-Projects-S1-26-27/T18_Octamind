# Final runbook: Roles 1-8

This runbook is the reproducible execution order for the approved Research-track project. It distinguishes API-free integration checks from research runs. **Do not put API keys, raw datasets, embeddings, caches or result dumps in Git.** Keep `GEMINI_API_KEY` in the local, git-ignored `.env` file or in the shell environment; never paste it into chat or a commit.

## 0. Environment and repository checks

From the repository root on Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

`python -m pytest -q` is the API-free regression suite. It uses toy records, mock embeddings and injected answer services; passing it establishes code-contract checks, not research performance.

Check that `.env` is ignored before creating it:

```bash
git check-ignore .env
```

Add the team's approved `GEMINI_API_KEY` to `.env` locally, or export it in the current shell. Never include the key in command history, terminal screenshots, uploaded files or Git.

## 1. API-free integration checks

Use these only after Roles 1-4 have produced compatible local MuSiQue sample, graph and query files. Mock embeddings must be labelled as plumbing only.

```bash
# Role 6: Role 5 traversal + evidence serialization; no LLM answer call
python -m src.context.run --dataset musique --kind auto --limit 5 \
  --tag offline --path-relevance-mode mean_edge --answer-mode none

# Role 7: S1/S2/S3 retrieval smoke test
python -m src.baselines.run --dataset musique --limit 5 \
  --embedding-mode mock --tag offline
```

Expected files include `results/context/musique_auto_offline.jsonl` and
`results/baselines/musique_offline.jsonl`. A smoke test may have low/zero evidence scores with
mock vectors; do not present those scores as experimental findings.

## 2. Build research-grade inputs consistently

Before the final runs, confirm that each dataset has a fixed-seed processed sample under
`data/processed/`, and that Roles 2-4 and Role 3 graph embeddings use the **same real embedding
model/mode**. Follow the dataset-specific download/format steps in `docs/DATA_NOTES.md`.

For MuSiQue, the project commands are:

```bash
# Role 1: load, validate and sample the configured development dataset(s)
python -m data.sampling

# Role 2: use the LLM extractor for the research version (rule is a baseline/smoke mode)
python -m src.graph.run_extraction --dataset musique --extractor llm

# Role 3: build gold/oracle and automatic relationship graphs
python -m src.graph.oracle --dataset musique
python -m src.graph.relations --dataset musique --mode llm

# Role 3: rebuild graph/entity/edge vectors using the configured real embedding model
python -m src.graph.embeddings --dataset musique --mode gemini

# Role 4: regenerate query seeds and question vectors with the same embedding model
python -m src.query.run --dataset musique --mode gemini
```

Rebuild the graph and query embeddings after switching away from mock mode. Verify the graph
report has zero missing edge/entity embeddings and consistent dimensions. Verify the Role 4
manifest reports the same embedding model/mode as Role 3. Do not combine mock query vectors with
Gemini graph vectors, or vice versa.

Repeat the input-generation steps for `twiki` and `hotpotqa` only after their raw data has been
prepared using `docs/DATA_NOTES.md`. The repository's oracle graph builder supports MuSiQue and
2Wiki; do not run an oracle S7 experiment for HotpotQA unless a verified oracle source is added.

## 3. Create one fixed tuning/test split

Create the split once per dataset and retain it for every system/configuration:

```bash
python -m src.evaluation.splits --dataset musique --tune-fraction 0.2 --seed 0
```

This writes `results/evaluation/musique_split.json`. For the 600-question MuSiQue sample, this
should allocate approximately 120 questions to tuning and 480 to test, stratified by hop level.
Use the **same manifest** for every S1-S3, S6 and S7 run. Never select hyperparameters based on
test results.

## 4. Role 5 and Role 6: S6/S7 retrieval, evidence context and answers

Role 6's runner calls the Role 5 traversal and selector, then serializes selected relationships
with their source paragraphs and builds the common QA prompt. Use `path_relevance_mode: text` for
research; this embeds the complete ordered path using the shared model. `mean_edge` is an offline
approximation and must not be used for the main research claim.

Generate the Role 5/Role 6 configuration sweep files:

```bash
python -m src.evaluation.ablation
```

This creates `configs/ablations/budget_250.yaml`, `budget_500.yaml`, `budget_1000.yaml`,
`budget_2000.yaml`, and `budget_4000.yaml`, alongside the other ablation configs.

Run the S6 budget sweep on tuning IDs (one command per budget):

```bash
for budget in 250 500 1000 2000 4000; do
  python -m src.context.run --dataset musique --kind auto \
    --config "configs/ablations/budget_${budget}.yaml" \
    --tag "s6_budget${budget}_tune" --answer-mode gemini \
    --split-manifest results/evaluation/musique_split.json --split-part tune
done
```

Run the same S6 sweep on test IDs so the evaluation harness can report the locked configurations
on the held-out set. The evaluation code selects the S6 operating point using tune rows only; it
does not choose the best-looking test configuration.

```bash
for budget in 250 500 1000 2000 4000; do
  python -m src.context.run --dataset musique --kind auto \
    --config "configs/ablations/budget_${budget}.yaml" \
    --tag "s6_budget${budget}_test" --answer-mode gemini \
    --split-manifest results/evaluation/musique_split.json --split-part test
done
```

Run the oracle comparison S7 using the approved default traversal config, for both splits:

```bash
python -m src.context.run --dataset musique --kind oracle --tag s7_tune \
  --answer-mode gemini --split-manifest results/evaluation/musique_split.json --split-part tune
python -m src.context.run --dataset musique --kind oracle --tag s7_test \
  --answer-mode gemini --split-manifest results/evaluation/musique_split.json --split-part test
```

The answer model is pinned in `configs/context.yaml` (temperature 0). Run a 2-5 question answer
smoke test before the full split to verify quota, latency, and output schema. Full answer generation
can require many model requests; check the team's approved quota/billing before launching the
complete sweep.

## 5. Role 7: S1-S3 baselines

The local runner implements:

- S1: all paragraphs supplied with that question, including dataset-provided distractors.
- S2: vector RAG over chunks from the pooled sampled-dataset corpus; chunk sizes and `k` are swept.
- S3: unranked depth-limited BFS over the automatic graph, without query-aware edge scoring.

Run the configured S1-S3 sweep on each split with Gemini embeddings. The same complete sampled
corpus is used for indexing on both partitions; only the evaluated question IDs are filtered.

```bash
python -m src.baselines.run --dataset musique --embedding-mode gemini --tag baselines_tune \
  --split-manifest results/evaluation/musique_split.json --split-part tune
python -m src.baselines.run --dataset musique --embedding-mode gemini --tag baselines_test \
  --split-manifest results/evaluation/musique_split.json --split-part test
```

Generate answers for S1-S3 with exactly the same Role 6 prompt and answer model:

```bash
python -m src.context.answer_run \
  --input results/baselines/musique_baselines_tune.jsonl \
  --output results/answers/musique_baselines_tune_answers.jsonl
python -m src.context.answer_run \
  --input results/baselines/musique_baselines_test.jsonl \
  --output results/answers/musique_baselines_test_answers.jsonl
```

S2 has multiple chunk-size/`k` variants and S3 has depth/budget variants. This default sweep can
produce many answer-generation requests. It is legitimate to reduce the pre-registered sweep only
if the team documents the change and uses the same selected settings on the held-out test set.

**Not implemented:** S4 HippoRAG (LightRAG fallback) and optional S5 IRCoT are not in this local
runner. Do not describe S1-S3 as S4/S5. If the external S4 implementation cannot be configured,
run and verified reproducibly before submission, disclose it as a limitation. See `docs/BASELINES.md`.

## 6. Role 8: held-out evaluation and plots

Combine all answer-bearing tune/test files. The evaluator separates them using the shared split
manifest, chooses the S2 reference variant and S6 operating point from tuning IDs only, validates
exact test-ID coverage, computes EM/token F1/evidence metrics/path recall, paired bootstrap
comparisons, per-hop and distractor summaries, and held-out matched-context reduction.

```bash
python -m src.evaluation.run --dataset musique \
  --inputs \
    results/context/musique_auto_s6_budget250_tune.jsonl \
    results/context/musique_auto_s6_budget250_test.jsonl \
    results/context/musique_auto_s6_budget500_tune.jsonl \
    results/context/musique_auto_s6_budget500_test.jsonl \
    results/context/musique_auto_s6_budget1000_tune.jsonl \
    results/context/musique_auto_s6_budget1000_test.jsonl \
    results/context/musique_auto_s6_budget2000_tune.jsonl \
    results/context/musique_auto_s6_budget2000_test.jsonl \
    results/context/musique_auto_s6_budget4000_tune.jsonl \
    results/context/musique_auto_s6_budget4000_test.jsonl \
    results/context/musique_oracle_s7_tune.jsonl \
    results/context/musique_oracle_s7_test.jsonl \
    results/answers/musique_baselines_tune_answers.jsonl \
    results/answers/musique_baselines_test_answers.jsonl \
  --split-manifest results/evaluation/musique_split.json --tag heldout
```

Outputs: `results/evaluation/musique_heldout_scored.jsonl`, summary JSON and CSV. Optional plots:

```bash
python -m src.evaluation.plots --summary results/evaluation/musique_heldout_summary.json
```

Do not run this command until each named input file exists and contains answers for all IDs in
its split. If you intentionally run a smaller experiment, create a matching manifest for that
sample rather than silently mixing different ID sets.

## 7. RQ3 ablations and RQ4 error injection

The ablation generator creates configs; each config still needs a run on tune IDs and, after
selection, a held-out run using the selected config. The proposal's signal removals are
`alpha=0`, `beta=0`, `eta=0`, and `gamma=0`; search ablations include uniform-cost (`W=0`), greedy
best-first (`use_g=false`), A*-style (`W=1`), and weighted A*. Also sweep beam width/margin,
path relevance, budgets and source-sentence removal. Keep all configurations and result files.

RQ4 error injection creates isolated automatic-graph copies at rates 0.10, 0.20 and 0.30:

```bash
python -m src.evaluation.error_injection --dataset musique \
  --corruptions deletions tail merge --rates 0.1 0.2 0.3 --seed 0 \
  --embedding-mode gemini
```

For each graph copy, run Role 6 with `--graph-dir data/graph_perturbed/<corruption>_<rate>` and
the same fixed test IDs. Edge deletions are topology-only; tail corruption and entity merges must
re-embed changed triples with the same real embedding model. Do not run tail/merge corruption with
mock vectors for research claims.

## 8. What is not complete until a real run exists

- API-backed Role 2/3/4 data/embedding generation with one consistent model.
- S6/S7 and S1-S3 research answers on disjoint tune/test IDs.
- Actual held-out analysis, ablation and error-injection outputs.
- S4 HippoRAG/LightRAG fallback and optional S5 IRCoT (if attempted).
- Final written report, six-slide presentation, live demo and viva preparation.

The code modules and automated tests are reproducibility infrastructure. They are not a substitute
for the experiments above. Report null, negative or incomplete results honestly; never invent
metrics or call mock runs research results.

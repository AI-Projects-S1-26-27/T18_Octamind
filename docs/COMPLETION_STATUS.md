# Completion status and remaining verification

This status separates repository implementation from empirical evidence. A module being present and
passing synthetic tests does not mean its live research experiment is complete. Review this page with
the named role owners before using it as the team's final accountability record.

| Area | Repository implementation | What still needs to happen |
|---|---|---|
| Role 1: data | Loaders, validation, fixed-seed sampling | Confirm the final committed sample/manifests for every dataset can be rebuilt from a clean clone. |
| Role 2: entity extraction | Rule and Gemini extraction paths, linking and recall tools | Confirm approved extraction mode and final cache/output manifests; current user's local MuSiQue output was rule-based. |
| Role 3: graph | Automatic/oracle graph construction, provenance and embedding service | Regenerate graph and vectors with the approved real embedding model; current user's local MuSiQue output used mock vectors. |
| Role 4: query | Entity seed analysis and query embedding runner | Regenerate query vectors with the same real embedding model as Role 3; this integration work did not modify Role 4 source. |
| Role 5: traversal | Query-aware scoring, weighted best-first search, beam pruning, path selection and diagnostics | Tune weights/beam/budget on tuning IDs; run full-path embedding research configuration and held-out evaluation. |
| Role 6: context | Evidence-aware serialization, inverse-fact deduplication, common prompt and optional Gemini generation | Named owner review; live generation/citations and final token/cost measurements. |
| Role 7: baselines | S1 full question context, S2 pooled-corpus vector RAG, S3 naive k-hop BFS | Named owner review and live tuned/held-out runs. S4 HippoRAG/LightRAG adapter is not integrated; S5 IRCoT is optional and not integrated. |
| Role 8: evaluation | EM/F1, evidence metrics, path recall, paired bootstrap, split selection, ablation configs, graph corruption and plots | Named owner review; execute the actual sweeps, analyze results and retain run artifacts. |
| Final submission | README, role docs, runbook, AI-use/progress logs | Written report, six-slide presentation, demo and viva preparation are still team deliverables. |

## Verified in the integration workspace

- `python -m pytest -q`: 176 tests passed.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.
- CLI `--help` checks passed for data extraction, graph, query, context, baselines and evaluation modules.

These checks use API-free tests and mock/injected services. No live Gemini API calls, final answer-generation sweep,
weight tuning, held-out result table, or RQ4 corruption experiment was run as part of this code-integration work.

## User-provided local MuSiQue smoke-test history

The user previously reported: 600 sampled MuSiQue questions (200 each for 2/3/4 hops); rule extraction
of 7,049 paragraphs into 101,333 mentions and 47,205 entities (strict/lenient gold-name recall 0.8422/0.9511);
automatic graph with 44,495 base edges and oracle graph with 994 base edges; mock 64-dimensional embeddings;
600 Role 4 questions with seeds; and Role 5 offline traversal plumbing tests. These are user's reported local
runs, not runs performed by this integration workspace. Because mock embeddings are non-semantic and the
extractor was the rule mode, these numbers are smoke-test evidence only, not final research results.

# Octamind: Query-Aware Semantic Graph Retrieval for Multi-Hop QA

CS F407 / U407 Artificial Intelligence project, **Team 18 (Octamind)**, **Research** track, BITS Pilani Hyderabad Campus.

Repository: <https://github.com/AI-Projects-S1-26-27/T18_Octamind>

> **Status (2026-11-17).** Roles 1-4 and Role 5 traversal are implemented in the repository. Role 5 has been merged into `main`. Role 6 now has evidence-aware context serialization and optional Gemini answer generation; Role 7 has local S1-S3 retrievers (full context, vector RAG, naive k-hop); Role 8 has answer/evidence metrics, paired bootstrap, tuning/test split, ablation configuration generation and graph-error injection helpers. These are implementation components, not evidence of final research outcomes. The official S4 HippoRAG/LightRAG integration and optional S5 IRCoT are not bundled; real Gemini embeddings/answers, tuned weights and held-out experiments still require credentials and actual runs.

## What this project is about

Standard retrieval-augmented generation (RAG) retrieves text chunks by embedding similarity. For multi-hop questions, where the answer needs a chain of facts across entities and documents, no single chunk holds the chain, so similarity search either misses a link or pads the prompt with text that is similar but not needed.

We build a **Semantic Knowledge Network** (entities and typed relationships, each with a confidence and a pointer to its source text) and answer each question by **query-aware traversal**: score every relationship by its relevance to the question, search for the chain the question needs, and give the LLM only that evidence.

We do **not** claim novelty in knowledge graphs or GraphRAG in general. The question we test is narrow:

> Can query-specific traversal of an entity-relation graph assemble a much smaller and more relevant context than similarity-based retrieval, while keeping answer accuracy?

A partial or negative result, with analysis of where and why, is an acceptable outcome.

## Research questions

| | Question | Hypothesis |
|---|---|---|
| RQ1 (headline) | Does query-aware traversal reach the accuracy of tuned vector RAG with substantially fewer context tokens? | H1: at the best F1 reached by tuned vector RAG, our method needs fewer context tokens; at equal token budget its F1 is at least as high. |
| RQ2 | Does the advantage grow with hop depth and on distractor-heavy questions? | H2: the gain over vector RAG and naive k-hop expansion is larger at 4 hops than at 2 hops. |
| RQ3 | Which scoring signals matter? | H3: query relevance contributes most. |
| RQ4 | How sensitive is the method to graph-construction errors? | H4: performance degrades with graph errors; the oracle-graph gap separates construction errors from traversal errors. |

## Method

**Offline, built once:** corpus, then entity/concept/fact extraction, then relationship discovery, then the Semantic Knowledge Network (triples + confidence + provenance).

**Per query:** query analysis and seed entities, then relationship selection with beam traversal, then minimal relevant subgraph under a token budget, then context construction, then LLM answer.

Traversal is a search problem (CS F407 Modules 2 and 3):

- Edge cost: `cost(e | q) = α(1 − rel(e,q)) + β(1 − c(e)) + η(1 − s(e)) + γ·depth(e)`
- Evaluation function (weighted A*): `f(n) = g(n) + W·h(n)`, with a beam of width `k` or margin `m`
- Context selection: add candidate paths in decreasing utility-per-token order until budget `B` is reached or marginal utility falls below `ε`

The heuristic `h` is not admissible, so it is used for ordering and pruning, not to claim optimal paths. `W`, `k` and `m` are config values (`configs/traversal.yaml`), so the search-strategy ablations (uniform-cost, greedy best-first, A*, weighted A*) need no extra code.

## Systems compared

| ID | System |
|---|---|
| S1 | Full context |
| S2 | Vector RAG (chunk size and k swept) |
| S3 | Naive k-hop expansion (depth-limited BFS, no query scoring) |
| S4 | HippoRAG (LightRAG as fallback) |
| S5 | IRCoT (optional) |
| S6 | Proposed method on the automatically built graph |
| S7 | Proposed method on an oracle graph |

## Data and evaluation

- **Datasets:** MuSiQue-Ans (2-, 3- and 4-hop), 2WikiMultiHopQA, HotpotQA. 200 sampled dev questions per hop level, fixed seed.

> **Data note:** all three datasets download and sample (HotpotQA is rebuilt from the official
> HuggingFace parquet). Sources and commands: `docs/DATA_NOTES.md`.

- **Models (for research runs):** use one pinned Gemini answer model at temperature 0 for every system and the shared Gemini Embedding 2 model for entities, relationships, query vectors and vector-RAG chunks. Current model names are configured in `configs/context.yaml`, `configs/graph.yaml` and `configs/extraction.yaml`. Role 2 has both rule and Gemini extraction modes; the rule-based run is a baseline/smoke path, not a substitute for an approved LLM research run.
- **Metrics:** EM and token-level F1; context tokens sent to the LLM; evidence recall and precision (mapped to source paragraphs); path recall where a verified gold chain exists; context reduction at matched accuracy; nodes expanded, peak frontier size and latency.
- **Statistics:** paired bootstrap 95% confidence intervals.

## Current implementation status

- **Roles 1-4:** dataset processing, entity extraction, relationship/graph construction, and query analysis are present in `data/`, `src/graph/`, and `src/query/`.
- **Role 5:** query-aware scoring, weighted best-first traversal, beam search, path ranking, budgeted subgraph selection and traversal diagnostics are implemented in `src/traversal/`.
- **Role 6:** source-aware context serialization, a common evidence-grounded prompt, optional Gemini answer generation, and a shared answer runner for S1-S3 are implemented in `src/context/`.
- **Role 7:** the local S1 full-context, S2 vector-RAG and S3 naive k-hop baselines are implemented in `src/baselines/`. Proposal S4 HippoRAG/LightRAG and optional S5 IRCoT are not integrated.
- **Role 8:** answer/evidence metrics, paired bootstrap, hop-stratified split, ablation configs, graph corruption and plotting helpers are implemented in `src/evaluation/`.

Implementation and API-free tests do not mean the research experiments are complete. Real Gemini embeddings/answers, hyperparameter selection on tune IDs, held-out evaluation, RQ4 graph-corruption runs, and a reproducible S4 fallback still need to be executed and reviewed. See `docs/FINAL_RUNBOOK.md`.

## Repository layout

```
octamind-graph-rag/
├── README.md
├── configs/             # all run settings (models, W, k, m, budgets, seeds)
├── data/                # dataset loaders and sampling (raw data is not committed)
├── src/
│   ├── graph/           # entity extraction/linking, relationship discovery, oracle graph and embeddings
│   ├── query/           # query analysis, seed-entity identification and question embeddings
│   ├── traversal/       # Role 5: edge cost, weighted A*, beam search, context selection
│   ├── context/         # Role 6: source-aware serialization, prompts, optional answer generation
│   ├── baselines/       # Role 7: S1 full context, S2 vector RAG, S3 naive k-hop
│   └── evaluation/      # Role 8: EM/F1, evidence metrics, bootstrap, ablations, error injection
├── experiments/         # scripts for each experiment
├── tests/               # unit tests, including the Figure 1 toy graph
├── results/             # tables and figures
└── docs/
    ├── INTERFACE.md          # shared data contracts for Roles 3-8
    ├── DATA_NOTES.md         # dataset sources and ingestion notes (Role 1)
    ├── EXTRACTION.md         # entity extraction: how to run, results, limits (Role 2)
    ├── GRAPH.md              # relation discovery, oracle graph, embeddings (Role 3)
    ├── TRAVERSAL.md          # search and subgraph selection (Role 5)
    ├── CONTEXT.md            # context serialization and answer generation (Role 6)
    ├── BASELINES.md          # implemented and pending retrieval baselines (Role 7)
    ├── EVALUATION.md         # metrics, splits, ablations and error analysis (Role 8)
    ├── FINAL_RUNBOOK.md      # reproducible pipeline and experiment commands
    ├── COMPLETION_STATUS.md  # implementation vs. empirical completion status
    ├── PROGRESS_LOG.md       # running log of work and decisions
    └── AI_USAGE.md           # AI-tool disclosure log
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m data.sampling                 # Role 1: sample dev sets (see docs/DATA_NOTES.md)
python -m src.graph.run_extraction       # Role 2: entity extraction (see docs/EXTRACTION.md)
python -m src.graph.oracle              # Role 3: build oracle graphs (System S7)
python -m src.graph.relations           # Role 3: build automatic graphs (System S6)
python -m src.graph.embeddings          # Role 3: generate vectors & entity embeddings
python -m src.query.run                 # Role 4: seed entities + question embeddings
python -m pytest -q                     # run all unit/integration tests
```

For Roles 5-8 experiment order, tune/test splits, baselines, answer generation and evaluation,
follow `docs/FINAL_RUNBOOK.md` rather than running sweeps on the held-out test data.

Run `python -m pytest -q` for the API-free test suite. Gemini-backed embedding and answer generation require a local `.env` file with `GEMINI_API_KEY`; it is git-ignored and must never be committed. Mock embeddings are hash noise for plumbing tests only. Every report result must be reproducible from a clean clone using committed configs and fixed seeds.

## Working agreements

- **Commit steadily.** The course checks the commit history across the whole semester. Small, frequent commits with clear messages; no single last-minute upload.
- **Work on branches and open pull requests** so each member's contribution is visible and reviewed.
- **Seeds and configs, not magic numbers.** Anything tunable lives in `configs/`.
- **Keep `docs/PROGRESS_LOG.md` up to date** (date, who, what, decisions).
- **Log AI-tool use in `docs/AI_USAGE.md`.** Everyone must be able to explain what they submit.
- **Do not commit** API keys, large datasets or model weights.

## Team (Team 18)

| Member | Role in the proposal |
|---|---|
| Chirantan S | Role 1: Data Ingestion and Gold Evidence; Role 4: Query Understanding and Evidence Planning |
| Shantanu Pandey | Role 2: Semantic Entity and Concept Extraction |
| Utkarsh Gopal Bhartariya | Role 3: Relationship Discovery and Graph Construction |
| Pranjal Agrawal | Role 5: Relationship Ranking and Graph Traversal |
| Prashant Kadam | Role 6: Dynamic Context Construction and LLM Integration |
| Harshini Reddy Goli | Role 7: Baselines |
| Suchendra Kumar Pothukuchi | Role 8: Evaluation, Ablation and Error Analysis |

## AI-tool disclosure

The course allows AI tools but requires disclosure and that every member can explain what they submit. We record which tools were used, for what, and how the output was verified, in `docs/AI_USAGE.md`, and summarize it in the final report.

## References

1. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. <https://arxiv.org/abs/2005.11401>
2. Edge et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. 2024. <https://arxiv.org/abs/2404.16130>
3. Gutierrez et al. HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs. NeurIPS 2024. <https://arxiv.org/abs/2405.14831>
4. Guo et al. LightRAG: Simple and Fast Retrieval-Augmented Generation. 2024. <https://arxiv.org/abs/2410.05779>
5. Sun et al. Think-on-Graph: Deep and Responsible Reasoning of LLM on Knowledge Graph. ICLR 2024. <https://arxiv.org/abs/2307.07697>
6. He et al. G-Retriever: RAG for Textual Graph Understanding and QA. NeurIPS 2024. <https://arxiv.org/abs/2402.07630>
7. Trivedi et al. Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT). ACL 2023. <https://arxiv.org/abs/2212.10509>
8. Trivedi et al. MuSiQue: Multihop Questions via Single-hop Question Composition. TACL 2022. <https://arxiv.org/abs/2108.00573>
9. Ho et al. Constructing a Multi-hop QA Dataset (2WikiMultiHopQA). COLING 2020. <https://arxiv.org/abs/2011.01060>
10. Yang et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA. EMNLP 2018. <https://arxiv.org/abs/1809.09600>

## License

MIT

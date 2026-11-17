# Retrieval baselines (Role 7)

Code: `src/baselines/` (`retrievers.py`, `run.py`). Config: `configs/baselines.yaml`.

The implemented local runners cover the proposal's S1-S3 baselines:

| System | Implementation | Behaviour |
|---|---|---|
| S1 Full context | `FullContextRetriever` | All paragraphs supplied with each question, including its dataset-provided distractors. |
| S2 Vector RAG | `VectorRAGRetriever` | Shared corpus of sampled-dataset paragraphs, chunked, embedded with the same configured embedding model, ranked by cosine similarity to Role 4's query vector; `chunk_sizes` and `vector_k` are swept. |
| S3 Naive k-hop | `NaiveKHopRetriever` | Breadth-first expansion from Role 4 seed entities, no query scoring, depth and token-budget sweeps. |

The shared corpus uses globally unique Role 1 paragraph IDs. All retrieval outputs store the question ID, retrieved paragraph IDs, source text, common-tokenizer context size, retrieval latency, and the system parameters needed to reproduce each variant.

## Run

API-free plumbing test (mock embeddings only; not research evidence):

```bash
python -m src.baselines.run --dataset musique --limit 5 --embedding-mode mock --tag offline
```

Research run (Role 4 query vectors and Role 3 graph vectors must also be real Gemini Embedding 2 vectors):

```bash
python -m src.baselines.run --dataset musique --embedding-mode gemini --tag dev
python -m src.context.answer_run \
  --input results/baselines/musique_dev.jsonl \
  --output results/answers/musique_dev_answers.jsonl
```

The runner rejects a mismatch between the embedding mode recorded for Role 4 queries, Role 3 graph vectors, and the requested baseline mode. S2 indexes each configured chunk size once, then reuses a question's single ranking for all configured `k` values. It uses all sampled paragraphs to construct the S2 shared corpus, even when `--limit` or a split manifest restricts which questions are evaluated. It accepts `--split-manifest`, `--split-part`, `--processed-dir`, `--query-dir`, `--graph-dir`, and `--out-dir`.

For a split-consistent smoke test, create the split once with Role 8 and then run the same `--split-manifest ... --split-part tune` or `test` flags for each baseline. To generate answer-bearing S1-S3 rows using the same Role 6 prompt/model, pass each saved retrieval file through `src.context.answer_run`.

## External baselines not bundled here

- **S4 HippoRAG:** the approved proposal names the official implementation, with LightRAG as fallback. Neither external package is part of this repository's dependencies or verified in the current environment. Do not label S1-S3 or a local approximation as HippoRAG. The official HippoRAG repository currently documents a separate `hipporag` installation and model/provider configuration; its resource and API requirements must be checked before integrating it. See <https://github.com/OSU-NLP-Group/HippoRAG>.
- **S5 IRCoT:** explicitly optional in the proposal and not implemented in the local runner.

If S4 cannot be run reproducibly before submission, disclose the omission and reason in Results/Limitations rather than inventing a baseline result. Keep the answer model, prompt, sampling, and context-token measurement consistent across systems.

# Interface between graph construction (Role 3) and traversal (Role 5)

Agree this first; everything else depends on it. If you change a field, change it here, in
`src/traversal/types.py`, and tell the other role.

## Edge (one relationship)

| Field | Type | Meaning |
|---|---|---|
| `head`, `tail` | str | canonical entity ids (after normalization and linking) |
| `relation` | str | relation label, e.g. `works_in` |
| `confidence` | float in [0,1] | extractor's confidence, c(e) |
| `support` | int >= 1 | number of source sentences that support this edge |
| `provenance` | tuple of str | source paragraph ids (needed for evidence recall/precision) |
| `embedding` | tuple of float | phi(e): embedding of the text "head relation tail", from the shared embedding model |

## Paragraph ids (Role 1 to everyone)

`<dataset>::<title>` when that title has one text in the whole dataset; otherwise every variant
is `<dataset>::<title>#<sha1(text)[:8]>`. An id always names exactly one text across all
questions (enforced by `data.evidence.validate_records`), so a paragraph id is safe to use as
edge provenance and for evidence recall/precision.

## Entities (Role 2 to Role 3)

Written by `python -m src.graph.run_extraction` to `data/graph/extraction/<dataset>/` (git-ignored).
How to run it, results and known limits: `docs/EXTRACTION.md`.

`entities.jsonl`, one canonical entity per line. Its `id` is what goes in an edge's `head`/`tail`.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | canonical id, `ent:<normalized_name>` (e.g. `ent:new_york_city`) |
| `name` | str | most frequent surface form |
| `type` | str | PERSON, ORGANIZATION, LOCATION, WORK, EVENT, DATE, NUMBER or CONCEPT (untyped / other) |
| `aliases` | list of str | other surface forms linked to this entity |
| `paragraph_ids` | list of str | paragraphs that mention it (Role 1 paragraph ids) |
| `mention_count` | int | number of mentions |

`mentions.jsonl`, one mention per line:

| Field | Type | Meaning |
|---|---|---|
| `text` | str | surface form, exactly as in the paragraph |
| `type` | str | type given to this mention (an entity's type is the majority over its mentions) |
| `paragraph_id` | str | Role 1 paragraph id |
| `sentence_index` | int | sentence number from `src.graph.entities.split_sentences`; `-1` = title mention |
| `start`, `end` | int | character offsets into the paragraph text; `-1` for the title mention |
| `source` | str | `llm` or `rule` |
| `confidence` | float | 0.9 llm, 0.6 rule name, 0.9 rule date, 1.0 title |
| `paragraph_sha1` | str | `sha1(text)[:12]` of the paragraph the offsets refer to (consistency check) |
| `entity_id` | str | the entity this mention was linked to |

The title mention (`sentence_index = -1`) stands for the paragraph's subject, so a relation in
"He was born in Ohio" can use the title entity as its head.

## Graph

- A list of edges plus `entity_embeddings: dict[str, tuple]` (one embedding per entity, from the same
  model), used for the semantic-distance heuristic h(n).
- Persisted as JSON Lines (one edge per line) plus one JSON file for entity embeddings, so every
  member can load it without the extraction code. Files, per dataset (`musique`, `twiki`, `hotpotqa`):

  | File | Content |
  |---|---|
  | `data/graph/<dataset>/oracle_edges.jsonl` | oracle graph (S7); not built for `hotpotqa` (no gold chain) |
  | `data/graph/<dataset>/edges.jsonl` | automatic graph (S6), rule or LLM extractor (see `manifest_<mode>.json`) |
  | `data/graph/<dataset>/entity_embeddings.json` | `{entity id: vector}` for every entity in `entities.jsonl` **and** every edge endpoint Role 2 did not extract |

- Load with `src.graph.graph_io.load_graph(dataset, kind="oracle" | "auto")`, which returns a
  `Graph` and adds inverse edges; `graph_report(graph)` lists missing embeddings.
- **Inverse edges.** `Graph.out_edges` only follows an edge from its head, and an extracted edge
  points the way the sentence reads, so a question seed on the object of a sentence would have no
  way out. `load_graph` adds, for every `a --r--> b`, an edge `b --"inverse of r"--> a` with the same
  confidence, support, provenance and embedding. `graph_io.is_inverse(edge)` identifies them (for
  example to avoid listing one fact twice in the context). Pass `add_inverse=False` for the raw graph.
- Edge text for phi(e) is `"<head name> <relation> <tail name>"` with readable entity names
  (`embedding.use_entity_names` in `configs/graph.yaml`); `Edge.text` itself still uses ids.

## Shared services (one implementation, used by everyone)

- `embed(text) -> tuple[float]`: the fixed embedding model (`gemini-embedding-2`, Gemini Embedding 2).
  `from src.graph.embeddings import embed` (configured by `configs/graph.yaml`; `mode: mock` gives
  non-semantic hash vectors for tests only).
- `count_tokens(text) -> int`: one tokenizer for all systems, so context sizes are comparable.

## Query side (Role 4 to Role 5)

- `seeds: list[str]` entity ids linked from the question.
- `q_emb: tuple[float]` question embedding.

Role 4's initial baseline matches normalized entity names and aliases in the question,
prefers the longest non-overlapping matches, and returns every entity id for an ambiguous
alias. Candidates are ranked by canonical-name match, mention frequency and stable id, but
never capped or removed. It also records ambiguous matches for evaluation. A more advanced
ranking or capping policy must be measured as a query-understanding policy rather than
introduced as an undocumented heuristic.

Role 4 ignores single-word question function words (for example `which`, `was`, and `and`)
even if an upstream extraction run produced them as entities. Multi-word names are not removed
by this guard. The query manifest reports seed counts, non-gold seed names, and recall for
gold names that are explicitly present in the question; gold entities absent from the question
are not counted against query-side seed recall.

## Output of traversal (Role 5 to Role 6)

- `traverse(graph, seeds, q_emb, cfg)` returns a `TraversalResult` (candidate `Path`s, nodes expanded,
  initial/peak frontier sizes, beam/relevance prune counts, and flags). In the research config,
  `path_relevance_mode: text` uses the shared embedding service to embed complete candidate paths in
  batches; `mean_edge` is reserved for API-free tests/debugging. `select_context(result.paths, q_emb, token_counter, cfg, edge_text_fn)` returns
  `H`: the selected list of `Edge`, in the order they were added. Role 6 serializes H (triple plus source
  sentence) into the prompt. Pass the serializer's text for an edge as `edge_text_fn` so the token budget
  counts what the prompt really contains.
- **Inverse edges (answer to Role 3):** keep them in `load_graph`. `select_context` counts an edge and its
  inverse as one fact, so a fact is never listed twice. Role 6: use `graph_io.is_inverse(edge)` to print the
  fact in its natural direction.
- Run on the sampled questions with `python -m src.traversal.run` (see `docs/TRAVERSAL.md`).

## Role 6: selected subgraph to evidence context and answer

`src/context/ContextSerializer` consumes Role 5's selected `list[Edge]` and a dataset-wide `CorpusIndex` built from Role 1 `QuestionRecord.paragraphs`. Each serialized fact contains:

- one readable triple in the source fact's natural direction (inverse edges are normalized and duplicates are removed);
- up to `max_sources_per_edge` paragraphs resolved from the edge's `provenance` IDs;
- paragraph IDs beside their source text, so the answer model can cite only evidence it received.

The exact `ContextSerializer.edge_block` function is passed to `select_context` as `edge_text_fn`, so the selection budget covers the same text Role 6 later renders. The current graph edge schema contains paragraph IDs but not source sentence indices; Role 6 therefore serializes the full supporting paragraph rather than claiming exact sentence extraction. See `docs/CONTEXT.md`.

The common QA output row stores `question_id`, `system` (S6/S7/S1/S2/S3), `context`, `retrieved_paragraph_ids`, `gold_paragraph_ids`, `answer`, `supporting_paragraph_ids`, model name, and provider-reported prompt/output token counts when generation is enabled. Every system uses `src/context/prompt.py`'s same prompt template.

## Role 7: retrieval baseline output

S1-S3 are implemented in `src/baselines/retrievers.py`; `src.baselines.run` writes JSONL rows with `question_id`, `system`, sweep `parameters`, `retrieved_paragraph_ids`, `context`, `context_tokens`, `retrieval_latency_ms`, evidence metrics and embedding mode. S2 embeds the shared sampled-corpus chunks using the same embedding service as Roles 3/4. S3 uses depth-limited BFS and no query-aware scoring. S4 HippoRAG/LightRAG and optional S5 IRCoT remain external/not bundled; see `docs/BASELINES.md`.

## Role 8: evaluation result rows

`src.evaluation.run` consumes common JSONL result rows from Roles 6/7 and adds answer EM/F1, paragraph evidence precision/recall/F1, optional oracle-chain path recall and failure classification. `src.evaluation.metrics.paired_bootstrap_ci` expects values aligned by question ID. The deterministic split manifest from `src.evaluation.splits` separates tuning IDs from held-out test IDs. `docs/EVALUATION.md` defines the run order and known limitations.

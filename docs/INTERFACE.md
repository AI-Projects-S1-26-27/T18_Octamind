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
alias in stable order. It also records the ambiguous matches for evaluation. Ranking and
capping candidates are deliberately deferred: they must be measured as a query-understanding
policy rather than introduced as an undocumented heuristic.

## Output of traversal (Role 5 to Role 6)

- `H`: the selected list of `Edge` (the selected subgraph), plus counters: nodes expanded, peak
  frontier size. Role 6 serializes H (triple plus source sentence) into the prompt.

# Relationship discovery and graph construction (Role 3)

Code: `src/graph/` (`oracle.py`, `relations.py`, `embeddings.py`, `graph_io.py`).
Config: `configs/graph.yaml`. Output contract: `docs/INTERFACE.md`, section "Graph".

## What it builds

| Graph | System | Built by | File |
|---|---|---|---|
| Oracle | S7 | `src/graph/oracle.py`, from the gold reasoning chains (MuSiQue, 2Wiki) | `data/graph/<dataset>/oracle_edges.jsonl` |
| Automatic | S6 | `src/graph/relations.py`, from Role 2's mentions | `data/graph/<dataset>/edges.jsonl` |

Both use the same `Edge` (head, relation, tail, confidence, support, provenance, embedding) and the
same entity ids as Role 2 (`ent:<normalized_name>`), so S6 and S7 can be compared directly (RQ4).
HotpotQA has no gold chain, so it has an automatic graph only.

## How to run (in this order)

```bash
python -m src.graph.oracle                       # oracle graphs; manifest shows how many gold steps were kept
python -m src.graph.relations                    # automatic graph, rule extractor (default in configs/graph.yaml)
python -m src.graph.relations --mode llm         # automatic graph, Gemini extractor (needs GEMINI_API_KEY in .env)
python -m src.graph.embeddings                   # LAST: entity + edge embeddings (needs GEMINI_API_KEY)
```

Run `embeddings` last: building a graph again rewrites its edge file without vectors.
Every command takes `--dataset musique|twiki|hotpotqa`; `embeddings` also takes `--mode mock|gemini`.
Load a graph for traversal with `src.graph.graph_io.load_graph(dataset, kind="oracle" | "auto")`
and check it with `graph_report(graph)` (all `missing_*` fields should be 0).

## Oracle graph

Each gold step becomes edges `head --relation--> tail`, with provenance from the step's support
paragraphs. Step formats handled (`oracle.py` docstring has the details):

- 2Wiki: explicit head, relation, tail. Provenance: gold paragraphs whose title is the head or tail.
- MuSiQue `X >> relation`, `#N >> relation`: the relation is the part after `>>`.
- MuSiQue free-text sub-question: every `#N` is replaced by the answer of step N; one edge per
  referenced step, so a merge step ("who is older, #1 or #2") is linked to both branches.

`data/graph/oracle_manifest.json` reports `gold_steps`, `edges` and `gold_steps_skipped` by reason
(`skip_no_head`, `skip_no_tail`, `skip_self_loop`). Check that the skipped share is small before
using S7 as an upper bound.

## Automatic graph

- **Rule extractor (baseline):** the text between two neighbouring entities in a sentence is the relation
  (confidence `pair_confidence`); a sentence with one entity is paired with the paragraph title
  (`title_confidence`). Relation labels are free text, edges point the way the sentence reads.
- **LLM extractor:** Gemini, temperature 0, sees the numbered sentences and the paragraph's entity ids and
  returns `head, relation, tail, confidence, sentence`. Every item is validated (known ids, no self-loop,
  confidence clamped to [0.5, 1], sentence index inside the paragraph), so `support` counts distinct
  sentences. Same operating rules as Role 2: only a valid answer is cached (an empty list counts), API
  errors and unparseable answers are retried by re-running the command, a daily quota stops the run
  cleanly, and the cache key includes the entity ids, so a Role 2 re-run never reuses stale relations.
  A run is complete when `failed_paragraphs` is empty and `invalid_responses` is 0 in `manifest_llm.json`.
- `relation.min_confidence` drops weaker relations before aggregation; `support` is the number of
  distinct source sentences, `confidence` the maximum seen.

## Embeddings

- Model: `gemini-embedding-2` (the project's fixed model). `text-embedding-004` was shut down by Google
  in January 2026 and must not be used.
- `mode: mock` produces deterministic hash vectors for tests. They carry **no semantics** (a paraphrase is
  as far away as an unrelated sentence), so a mock run is flagged in `manifest_embeddings.json`
  (`"mock": true`) and prints a warning. Never use mock vectors for experiments.
- Entity vectors cover Role 2's entities and every edge endpoint it did not extract (gold entities that
  extraction missed), so `h(n)` always finds a vector.
- Edge vectors embed `"<head name> <relation> <tail name>"`.
- The service never drops or reorders vectors: a response with the wrong number of vectors raises.

## Inverse edges

`load_graph` adds `b --"inverse of r"--> a` for every `a --r--> b` (same confidence, support, provenance,
embedding), because traversal only follows outgoing edges and the extractor orients edges by word order.
`is_inverse(edge)` tells them apart. The stored files do not contain them.

## Known limits

- Rule relation labels are raw connective text ("was born in the city of"), so the label vocabulary is
  large; the LLM extractor gives short labels.
- Only neighbouring entity pairs are linked by the rule extractor; pronouns are not resolved (Role 2).
- One graph per dataset is built over all sampled questions together, so entities with the same
  normalized name are merged across questions and `support` counts across questions.
- The real-Gemini paths (`relations --mode llm`, `embeddings --mode gemini`) are covered by tests with
  fake models only; run a small `--dataset ... ` check with a real key before the full runs.

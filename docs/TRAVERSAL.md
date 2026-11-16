# Traversal and context selection (Role 5)

Code: `src/traversal/` (`types.py`, `scoring.py`, `frontier.py`, `search.py`, `select.py`, `run.py`).  
Config: `configs/traversal.yaml`. Tests: `tests/test_traversal*.py` (synthetic and API-free unless a path embedder is explicitly injected).  
Hand-checkable example: the Figure 1 toy graph, `src/traversal/toy_graph.py`.

## What it does

Input: the loaded graph (`load_graph`, Role 3) and, per question, `seeds` and `q_emb` (Role 4).
`traverse` runs best-first search with beam pruning from the seed entities and returns candidate
paths. `select_context` chooses edges from those paths under the dynamic context-token budget.
Output `H` remains a list of the agreed `Edge` objects and is passed to Role 6. The Role 5 runner
also measures evidence recall/precision against the gold paragraph IDs.

## Formulas and proposal mapping

| Proposal | Code |
|---|---|
| rel(e,q) = cos(phi(e), phi(q)) | `scoring.cosine`, `Scorer.rel` |
| cost(e\|q) = alpha(1-rel) + beta(1-c) + eta(1-s) + gamma*depth | `scoring.cost_from_parts` (one implementation) |
| s(e) = support / max support | `scoring.strength` |
| h(n) = 1 - cos(phi(end(n)), phi(q)) | `scoring.semantic_distance`, `Scorer.h` |
| f(n) = g(n) + W*h(n) | `search.traverse`, inner `f_score` |
| beam: k best f, or within margin m | `Frontier.keep_best` |
| rho(P) = cos(phi(P), phi(q)), prune rho < tau | `scoring.path_relevance`, end of `traverse` |
| dynamic context budget, utility-per-token, stop at B or epsilon | `select.select_context` |

`Scorer` uses cached unit-length float32 vectors to avoid repeatedly normalizing high-dimensional
graph embeddings. The plain scoring functions remain available for checks. Cosine inputs with
incompatible dimensions or non-finite values fail clearly instead of producing silent NaNs.

## Path relevance implementation

The proposal defines `rho(P)` using an embedding of the complete ordered path text. The research
configuration therefore sets `path_relevance_mode: text` and the runner calls the shared
`EmbeddingService.embed_many` from `src.graph.embeddings`. Unique candidate path texts are embedded
in batches and use the same configured embedding model as the graph/query vectors; the service's
cache avoids repeated API work across runs.

`path_relevance_mode: mean_edge` is retained only for API-free tests and offline debugging. It
uses the mean of the edge vectors and is an approximation, not the proposal's exact `phi(P)`.
Do not use that mode for final reported research results unless the report explicitly labels it as
a deviation and the team approves that choice. In `mean_edge` mode, a path with a missing edge
embedding has unavailable `rho` and is not pruned using a fabricated relevance value. In `text`
mode, path relevance can still be embedded from the path text, but the missing edge vector makes
`rel(e,q)` fall back to 0 for that edge. In either case, `graph_report` should show zero missing
embeddings for final experiments.

The path text uses the ordered triples with `ent:` prefixes removed and underscores converted to
spaces. It is a canonical readable representation because the traversal graph stores canonical
entity IDs rather than original surface names. Keep this representation fixed across experiments.

## Module 2 mapping

State = a partial path (the `SearchNode` stores the end entity, parent, action/edge, path cost g,
and depth). The frontier is a priority queue ordered by f. `W = 0` gives uniform-cost ordering;
`use_g: false` gives greedy best-first; `W = 1` gives A*-style ordering; `W > 1` gives weighted A*
(satisficing). The beam is memory-bounded and can sacrifice completeness and optimality.

The beam is applied both after all initial seeds are inserted and after each expansion. This matters
because Role 4 can return multiple seeds. `initial_frontier` is the retained seed frontier after
pruning, `nodes_pruned_by_beam` counts removals (including seed removals), and `peak_frontier`
records the largest generated frontier before pruning so it captures the temporary successor set.

## Design decisions and limits

1. **Candidate-path goal test.** A path is a candidate when its end node has no unvisited outgoing
   edge or reaches `max_depth`. Traversal is not given an answer-type schema, so this is a retrieval
   candidate test, not a claim that the terminal entity is the answer. A useful path prefix can be
   selected when a longer path shares those edges with another selected candidate.
2. **Heuristic guarantees.** `h` is not guaranteed admissible or consistent, so no A* optimality
   guarantee is claimed. Weighted ordering and beam pruning are used as a satisficing strategy.
3. **No revisits.** A path cannot return to an entity already on that same path. This prevents cycles
   and stops an inverse edge from immediately walking back. Alternate paths to a shared entity are
   still allowed; there is no global visited set that would discard a distinct evidence chain.
4. **Inverse edges.** `load_graph` adds `b --"inverse of r"--> a`. They carry the same confidence,
   support, provenance and embedding. `select_context` maps a fact and its inverse to one
   `fact_key`, so the same fact is not charged to the context budget twice.
5. **Missing entity embeddings.** `h = 1.0` when an entity vector is absent or zero-norm. The metric
   counts distinct affected entities, not repeated visits. Missing or inconsistent vector dimensions
   are surfaced as errors; `graph_report` should show zero missing embeddings for final runs.
6. **Expansion cap.** `max_expansions` (default 2000 popped nodes per question) bounds pathological
   searches, including uniform-cost search on hubs. If it triggers, `hit_expansion_cap` is reported.
7. **Selection objective.** At each iteration, candidate paths are ranked by the utility of new
   (not-yet-selected) facts divided by their additional token cost. This greedy selection is an
   approximation to the budgeted objective, not a globally optimal knapsack solution. The budget is
   hard: an addition that would exceed `B` is skipped. Inverse duplicates are not re-added.
8. **Tokenizer / prompt serializer.** The repository does not yet expose the shared `count_tokens`
   service or Role 6's final serializer. Until those are connected, the runner's default
   `simple_token_count` and readable triple text are explicitly placeholders. The output summary
   records `tokenizer_is_placeholder: true`. For final experiments, call `run_dataset` with the
   shared token counter and a serializer-compatible `edge_text_fn`; otherwise `context_tokens` is
   only an estimate and must not be reported as actual LLM prompt tokens.
9. **Weight tuning.** Values in `configs/traversal.yaml` are initial settings, not tuned findings.
   Tune alpha, beta, eta, gamma, W, beam settings and budget on a held-out development slice; do not
   tune on the same questions used for the reported result.
10. **Real-data status.** Unit and synthetic end-to-end tests do not establish real-data performance.
    Run the smoke commands below against the team's built graphs and Role 4 query files before
    claiming dataset results.

## First real run

The default config uses exact text-path embeddings and therefore requires the configured Gemini
embedding service and its API credentials. For a no-API smoke test only, override the path relevance
mode to `mean_edge`:

```bash
# Research configuration: whole-path embedding, shared model/cache
python -m src.traversal.run --dataset musique --kind oracle --limit 20 --tag smoke
python -m src.traversal.run --dataset musique --kind auto   --limit 20 --tag smoke

# Offline diagnostic only (not the exact proposal path-relevance method)
python -m src.traversal.run --dataset musique --kind oracle --limit 5 --tag offline \
  --path-relevance-mode mean_edge
```

Inspect `results/traversal/*_summary.json`. Red flags: `share_with_no_edges` high (seeds missing or
weights too strict), `share_hit_cap` > 0, `missing_entity_embeddings` > 0, `nodes_pruned_by_beam`
higher than expected, or recall near 0 on the oracle graph (seed/entity-ID or provenance mismatch
between Roles 1, 3 and 4). Also verify `tokenizer_is_placeholder` is false before treating context
counts as actual prompt-token measurements.

## Ablations

| Experiment | Command (add `--dataset`, `--kind`, `--tag`) |
|---|---|
| Weighted A* (default) | `--W 1.5 --beam-k 5` |
| Uniform-cost | `--W 0 --no-beam` |
| A* | `--W 1 --no-beam` |
| Greedy best-first | `--no-g --no-beam` |
| Beam size sweep | `--beam-k 1`, `3`, `5`, `10` |
| Budget sweep | `--budget 250`, `500`, `1000`, `2000`, `4000` |
| Zero a signal | edit alpha, beta, eta or gamma to 0 in a copy of the config and pass `--config` |
| Path relevance approximation check | run once with `--path-relevance-mode mean_edge`, label it separately, and compare with `text` |

## Likely viva questions

1. **Where is weighted A*?** `search.traverse`: a priority queue ordered by f = g + W*h; g accumulates
   `cost(e|q)` along the path and h is the semantic distance of the end entity. W=1 is A*-style
   ordering, but the heuristic is not admissible and the beam can prune states, so no optimality
   guarantee is claimed.
2. **Why use query relevance and semantic distance?** `rel(e,q)` scores the current relationship;
   `h(n)` estimates how semantically close the path's current endpoint is to the question. They act
   on different objects and have different roles in the proposal's equations.
3. **What does W=0 do?** It ignores h; priority is by accumulated path cost g, giving uniform-cost
   ordering (Dijkstra-style on non-negative costs).
4. **What does the beam lose?** It bounds frontier memory but may discard a branch, sacrificing
   completeness and possibly the best-scoring path. It applies to the initial seed set too.
5. **Why can a path contribute only some edges?** Selection charges only facts not already selected;
   shared prefixes are not paid for twice, and a candidate may contribute its not-yet-selected facts.
6. **How is the context budget enforced?** A path addition that would exceed B is skipped; selection
   stops when no candidate fits or the best marginal utility is below epsilon.
7. **What happens when an embedding is missing?** Missing entity vectors give h=1 and are counted
   once per entity. A path with a missing edge vector is retained without rho-based pruning, while
   the graph report flags the underlying issue.
8. **Why inverse edges?** A seed may match the object of a sentence. The inverse permits traversal
   from that endpoint; selection deduplicates the inverse and original as one fact.
9. **What is not yet finalized?** The team still needs to connect the shared tokenizer and Role 6
   prompt serializer, tune weights on a held-out slice, and verify the actual dataset runs.
10. **Course connection?** Module 2 formal search problem, weighted best-first search, beam search
    and search-effort metrics; Module 3 knowledge base, declarative facts and soft data-driven
    chaining over graph relationships.

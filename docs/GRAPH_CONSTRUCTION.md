# Relationship Discovery and Graph Construction (Role 3)

Owner: **Utkarsh Gopal Bhartariya**.  
Code: `src/graph/` (`oracle.py`, `relations.py`, `embeddings.py`).  
Config: `configs/graph.yaml`.  
Output format: `docs/INTERFACE.md`, sections "Edge (one relationship)" and "Graph".

---

## What It Does

Role 3 connects the canonical entities extracted by Role 2 into the **Semantic Knowledge Network (SKN)** and provides the vector representations required by Role 5's heuristic traversal:

1. **Oracle Graph Construction (`oracle.py`, System S7):**
   - Translates human-verified gold reasoning chains (MuSiQue decompositions and 2Wiki explicit relations) into canonical `Edge` objects.
   - Resolves multi-hop step placeholders (`#1`, `#2`) and falls back to supporting paragraph titles for step 1 subjects.
   - Aggregates duplicate triples, counts support $s(e)$, and merges provenance paragraph IDs.
   - Serves as the upper-bound control benchmark for **RQ4** (*separating graph-construction errors from traversal errors*).

2. **Automatic Relationship Discovery (`relations.py`, System S6):**
   - **Rule-based extractor (baseline):** Extracts connective verb phrases and prepositions between co-occurring entities in a sentence; pairs lone entities with paragraph title mentions (`sentence_index = -1`). Fast, deterministic, and 100% offline.
   - **LLM extractor:** Prompts Gemini Flash-Lite to extract factual relations between known entity IDs in a passage, cached per paragraph in `data/cache/relations/`.
   - Aggregates raw triples into deduplicated `Edge` objects with sentence support counts $s(e) \ge 1$ and provenance IDs.

3. **Shared Embedding Service & Vector Store (`embeddings.py`):**
   - Implements the project-wide `embed(text) -> tuple[float, ...]` contract.
   - Supports both `mock` (deterministic unit-normalized vectors in RAM) and `gemini` (`text-embedding-004`).
   - Populates edge vectors $\phi(e)$ on `edges.jsonl` and `oracle_edges.jsonl`.
   - Generates `entity_embeddings.json` mapping each entity ID to $\phi(\text{entity})$ for Role 5's $h(n)$ semantic distance heuristic.

---

## How to Run

```bash
# 1. Build the Oracle Graphs (System S7)
python -m src.graph.oracle

# 2. Build the Automatic Graphs (System S6, rule baseline)
python -m src.graph.relations --mode rule

# 3. Generate Vector Embeddings (Edges and entity_embeddings.json)
python -m src.graph.embeddings --mode mock          # fast offline mock mode (no API needed)
python -m src.graph.embeddings --mode gemini        # production mode (needs GEMINI_API_KEY in .env)

# Run on a single dataset:
python -m src.graph.oracle --dataset twiki
python -m src.graph.relations --dataset musique --mode rule
python -m src.graph.embeddings --dataset hotpotqa --mode mock
```

---

## Graph Statistics (Sampled Dev Sets, Seed 0)

Outputs are saved under `data/graph/<dataset>/`:

| Dataset | Questions | Canonical Entities | Oracle Edges (S7) | Automatic Edges (S6, Rule) | Entity Embeddings |
|---|---|---|---|---|---|
| **MuSiQue** | 600 | 47,205 | **990** | **44,495** | `entity_embeddings.json` (47.2k) |
| **2WikiMultiHopQA** | 200 | 12,019 | **497** | **8,543** | `entity_embeddings.json` (12.0k) |
| **HotpotQA** | 200 | 16,706 | *n/a (no gold chains)* | **13,103** | `entity_embeddings.json` (16.7k) |

---

## Output Data Contracts

### 1. `oracle_edges.jsonl` and `edges.jsonl`
One JSON line per edge matching `src/traversal/types.py:Edge`:
```json
{
  "head": "ent:secret_lives",
  "relation": "director",
  "tail": "ent:edmond_t_greville",
  "confidence": 1.0,
  "support": 1,
  "provenance": ["twiki::Secret Lives (film)"],
  "embedding": [0.0178, -0.1043, ...]
}
```

### 2. `entity_embeddings.json`
Dictionary mapping entity IDs to float tuples:
```json
{
  "ent:secret_lives": [0.012, -0.045, ...],
  "ent:edmond_t_greville": [0.038, 0.091, ...]
}
```

---

## Running the Unit Tests

All Role 3 logic is covered by unit tests using synthetic fixtures (requiring zero API keys and zero network calls):

```bash
pytest tests/test_oracle_graph.py tests/test_relations.py tests/test_embeddings.py -v
```
*(All 19 Role 3 unit tests pass in under 0.2 seconds).*


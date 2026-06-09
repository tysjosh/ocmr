# Ontology-Constrained Memory (OCM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tysjosh/ocmr/blob/main/OCM_Colab.ipynb)

OCM is a **write-time-governed, ontology-constrained memory layer** for long-horizon
LLM agents. Its central claim is that memory should be governed at *write time* —
not just filtered at read time. Every incoming fact is turned into a typed graph
assertion, checked against an ontology and a set of constraints, run through a
contradiction gate, and stamped with provenance before it is allowed to commit.
Retrieval then packages **hybrid symbolic + semantic evidence** rather than
returning raw text chunks.

The implementation lives under the [`ocm/`](ocm/) package.

## Why write-time governance

Most agent memory systems store whatever the model emits and try to sort out the
mess during retrieval. OCM inverts that: a candidate fact only becomes durable
memory if it is structurally valid, satisfies ontology constraints, and does not
contradict existing high-confidence knowledge. Conflicting or low-trust writes are
*quarantined* or *rejected* instead of silently overwriting good data, and every
accepted assertion keeps a provenance trail back to its source.

## Project layout

| Path | Responsibility |
| --- | --- |
| `ocm/core/` | Settings, dependency container, ID generation, research logging |
| `ocm/ontology/` | Entity/relation types, enums, ontology models |
| `ocm/extraction/` | `Mock_Extractor` (offline) and optional `LLM_Extractor` |
| `ocm/resolution/` | Normalization and entity resolution |
| `ocm/memory/` | Write pipeline, assertion builder, commit manager, SQLite repo, graph store, provenance + quarantine stores |
| `ocm/validation/` | Schema validator, constraint validator (C1–C10), contradiction checker |
| `ocm/retrieval/` | Query classifier, symbolic + semantic retrievers, reranker, evidence packager, vector index |
| `ocm/agent/` | Agent loop, memory tool, answer policy |
| `ocm/evaluation/` | Benchmark generation, baselines B0–B4, runner, metrics |
| `ocm/app/` | FastAPI application and the five HTTP endpoints |
| `ocm/scripts/` | CLI entry points (serve + experiment commands) |
| `ocm/tests/` | Unit and property-based (Hypothesis) test suite |

## Setup

OCM targets **Python 3.11+**. Install the package (with dev/test extras) in
editable mode from the repository root:

```bash
pip install -e ".[dev]"
```

Runtime dependencies (declared in [`pyproject.toml`](pyproject.toml)):
`fastapi`, `uvicorn`, `pydantic` (v2), `networkx`, `chromadb`, and
`sentence-transformers`. The `dev` extra adds `pytest`, `hypothesis`, and `httpx`.

> On this machine the interpreter is invoked as `python3` (e.g.
> `python3 -m pip install -e ".[dev]"` and `python3 -m pytest`). The
> `python -m ocm.scripts.*` commands below work the same with `python3`.

## Offline / deterministic configuration

OCM runs **fully offline by default** — no API key and no network access are
required. The defaults select the `Mock_Extractor` plus a local embedding model,
so you can build the benchmark, run the API, and execute the whole test suite
hermetically.

Configuration is centralized in [`ocm/core/config.py`](ocm/core/config.py)
(`Settings`, a Pydantic v2 model):

| Setting | Default | Purpose |
| --- | --- | --- |
| `extractor` | `"mock"` | `"mock"` (offline, no network) or `"llm"` (opt-in) |
| `llm_base_url` / `llm_api_key` / `llm_model` | `None` / `None` / `"gpt-4o-mini"` | LLM extractor connection (only used when `extractor="llm"`) |
| `embedding_model` | `"sentence-transformers/all-MiniLM-L6-v2"` | Local embedding model |
| `embedding_mode` | `"local"` | Embedding provider mode |
| `sqlite_path` | `"ocm.db"` | Durable SQLite path |
| `chroma_mode` | `"persistent"` | `"persistent"` (on disk) or `"memory"` (in-memory vector index) |
| `chroma_path` | `".chroma"` | Chroma persistence directory |
| `deterministic_test_mode` | `False` | Reproducible IDs + in-memory storage + deterministic embeddings |
| `rerank_weights` | see below | Reranker score weights |
| `contradiction_high_confidence` | `0.8` | Confidence floor for the contradiction gate |
| `decision_evidence_floor` | `1` | Minimum supporting evidence for a decision answer |

Determinism knobs:

- **`deterministic_test_mode=True`** yields reproducible IDs, backs storage with
  an in-memory SQLite database, and uses a dependency-free deterministic
  embedding provider (so `sentence-transformers` is not required). It also mounts
  the read-only debug routes (see below).
- **`chroma_mode="memory"`** keeps the vector index in memory (with a pure-Python
  fallback when `chromadb` is unavailable).

The LLM extractor is strictly opt-in: set `extractor="llm"` together with
`llm_base_url`, `llm_api_key`, and `llm_model`.

The reranker score (weights in `RerankWeights`) is:

```
score = alpha*semantic_similarity + beta*graph_relevance + gamma*confidence
      + delta*provenance_quality + eta*recency - lambda*contradiction_penalty
```

with defaults `alpha=0.40`, `beta=0.25`, `gamma=0.15`, `delta=0.10`, `eta=0.05`,
`lambda=0.30`.

## API service

The FastAPI app is defined in [`ocm/app/main.py`](ocm/app/main.py) (exposed as
`ocm.app.main:app`) and the routes in
[`ocm/app/api/routes.py`](ocm/app/api/routes.py). Start it with:

```bash
python -m ocm.scripts.serve                       # 127.0.0.1:8000
python -m ocm.scripts.serve --host 0.0.0.0 --port 8080
python -m ocm.scripts.serve --reload              # dev auto-reload
```

(Or directly via `uvicorn ocm.app.main:app`.)

### The five endpoints

**1. `POST /memory/write`** — run the full write pipeline (W1–W8).

```jsonc
// request
{ "text": "...", "source_ref": "...", "write_intent": "new_fact" }
// response
{ "accepted": [...], "superseded": [...], "quarantined": [...],
  "rejected": [...], "summary": { ... } }
```

**2. `POST /memory/query`** — run the retrieval pipeline (R0–R4).

```jsonc
// request
{ "query": "...", "top_k": 5, "include_conflicts": false }
// response
{ "query_type": "...", "answer": "...", "confidence": 0.0,
  "supporting_assertions": [...], "supporting_sources": [...],
  "conflicts": [...], "missing_information": [...], "retrieved_items": [...] }
```

**3. `POST /memory/validate`** — validate a candidate (W5→W6→W7) **without
committing**.

```jsonc
// request
{ "candidate": { ... } }
// response
{ "valid": true, "decision": "accept", "reason": null, "severity": null,
  "failed_check": null, "conflicting_ids": [] }
```

**4. `GET /memory/entity/{id}`** — fetch an entity and every accepted assertion
it participates in.

```jsonc
{ "entity": { ... }, "entity_type": "...", "assertions": [...] }
```

**5. `GET /memory/conflicts`** — list unresolved conflicts and quarantined
candidates.

```jsonc
{ "unresolved_conflicts": [...], "quarantined_candidates": [...] }
```

In addition, read-only `/debug/*` inspection routes are mounted when
`deterministic_test_mode` is enabled (or debug routes are explicitly requested).
They are non-production helpers for inspecting internal state.

## Architecture

### Write pipeline (W1–W8)

A write request flows through eight stages before any data becomes durable:

1. **W1 Extract** — turn source text into candidate facts (`Mock_Extractor` by
   default, `LLM_Extractor` when configured).
2. **W2 Normalize** — canonicalize values and surface forms.
3. **W3 Resolve** — resolve mentions to existing entities.
4. **W4 Build** — construct typed `CandidateAssertion`s.
5. **W5 Schema validate** — structural validation against ontology types.
6. **W6 Constraint validate** — graph-level constraints **C1–C10**.
7. **W7 Contradiction check** — the contradiction gate against existing
   high-confidence knowledge.
8. **W8 Commit** — route each candidate to **accept / supersede / quarantine /
   reject**, embedding accepted assertions and recording provenance.

### Retrieval pipeline (R0–R4)

1. **R0 Classify** — determine the query type.
2. **R1 Symbolic** — graph/assertion retrieval.
3. **R2 Semantic** — vector retrieval over embeddings.
4. **R3 Rerank** — merge symbolic + semantic results into one ordered set using
   the weighted score above.
5. **R4 Evidence package** — assemble the answer, confidence, supporting
   assertions/sources, conflicts, and missing-information fields.

### Storage

- **SQLite** is the durable store behind a repository interface
  (`StorageRepository` / `SQLiteRepository`).
- A **NetworkX** graph holds the accepted-only projection and is **rebuilt from
  the repository on startup**.
- A **Chroma** vector index holds assertion/memory embeddings.
- Dedicated **provenance** and **quarantine** stores track source trails and
  withheld candidates.

All components are wired together once by `CoreContainer`
([`ocm/core/container.py`](ocm/core/container.py)); the API and agent resolve
their dependencies from it.

## Experiments

The evaluation harness compares baselines over a seeded, reproducible benchmark.
All commands run offline.

### 1. Build the benchmark

```bash
python -m ocm.scripts.build_benchmark --out benchmark.jsonl --seed 1337
```

`--out` defaults to `benchmark.jsonl` and `--seed` defaults to `1337`. The script
prints the total example/question counts and a per-category breakdown.

### 2. Run the baselines

```bash
python -m ocm.scripts.run_benchmark \
  --benchmark benchmark.jsonl \
  --baselines B0,B1,B2,B3 \
  --log research_log.jsonl \
  --out results.jsonl \
  --top-k 10 \
  --limit 50
```

If `--benchmark` is omitted, one is generated on the fly (using `--seed`).
`--baselines` defaults to `B0,B1,B2,B3`. `--out` writes the per-question result
records that `report_metrics` consumes; `--log` writes a research log;
`--top-k` (default 10) sets retrieval depth; `--limit` caps the number of
examples for smoke checks.

### 3. Report metrics

```bash
# from saved result records
python -m ocm.scripts.report_metrics --results results.jsonl --json metrics.json

# or run inline over a benchmark, then report
python -m ocm.scripts.report_metrics --benchmark benchmark.jsonl
```

Provide exactly one source: `--results` (records from `run_benchmark --out`) or
`--benchmark` (runs the default baselines inline first). `--json PATH` dumps the
structured metrics as JSON (use `--json` with no path to print to stdout). The
report prints retrieval, answer, write-time, and agent metric families with
deltas vs. B0.

### Baselines

Baselines are the same memory strategy with different governance toggles
([`ocm/evaluation/baselines.py`](ocm/evaluation/baselines.py)):

| Baseline | Description |
| --- | --- |
| **B0** | Vector retrieval only |
| **B1** | Graph / symbolic retrieval only (no vectors) |
| **B2** | Graph + semantic, no governance |
| **B3** | Full governance (contradiction + quarantine + provenance) |
| **B4** | B3 + the `Answer_Policy` |

`B0–B3` run by default; `B4` layers the answer policy on `B3` for the
answer-quality comparison.

## Testing

The full suite is offline and hermetic:

```bash
python3 -m pytest
```

It contains both **unit tests** and **property-based tests** (Hypothesis, with at
least 100 iterations each). Property tests are tagged
`Feature: ontology-constrained-memory, Property N` via the `property` pytest
marker.

## License

MIT (see `pyproject.toml`).

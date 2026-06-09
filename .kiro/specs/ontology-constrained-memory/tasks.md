# Implementation Plan: Ontology-Constrained Memory (OCM)

## Overview

This plan converts the OCM design into an incremental, test-driven build that follows the design's
five-phase order (Req 27.4): (1) scaffold + ontology + storage + minimal write path, (2) write-time
governance (validators, contradiction gate, commit/quarantine/provenance), (3) extraction +
resolution + full write pipeline W1–W8, (4) embeddings + retrieval pipeline R0–R4, and (5) API,
agent, and the evaluation harness.

All implementation lives under the `ocm/` package (Req 27.1). Every test runs offline and
reproducibly with `deterministic_test_mode=True`, `chroma_mode="memory"`, and the `Mock_Extractor`
(Req 27.2, 27.5, 13.6). Property-based tests (Hypothesis) cover the 11 correctness properties and are
tagged `Feature: ontology-constrained-memory, Property N`. Each task builds on the prior ones and ends
by wiring its output into the pipeline so no code is orphaned.

## Tasks

- [x] 1. Project scaffold and property-based test harness
  - [x]* 1.1 Property-based testing primer and deterministic test harness
    - Add `pytest` and `hypothesis` to dev dependencies; create `ocm/tests/conftest.py`
    - Establish the property test tagging convention `Feature: ontology-constrained-memory, Property N` (shared marker/docstring helper) and set Hypothesis to a minimum of 100 iterations per property
    - Provide a `deterministic_settings` fixture (`deterministic_test_mode=True`, `chroma_mode="memory"`, `extractor="mock"`) and an in-memory container/repository fixture so every later test is hermetic and offline
    - _Requirements: 27.2, 27.5, 13.6, 3.4_
  - [x] 1.2 Create the `ocm/` package skeleton and build config
    - Create `pyproject.toml` (Python 3.11+, deps: fastapi, pydantic v2, networkx, chromadb, sentence-transformers, pytest, hypothesis) and the package directories `ocm/{core,ontology,extraction,resolution,validation,memory,retrieval,agent,evaluation,app/api,scripts,tests}` with `__init__.py` files
    - _Requirements: 27.1, 27.4_
  - [x] 1.3 Implement core config, ID generation, and research logger
    - Write `ocm/core/config.py` (`Settings` with extractor/embedding selection, sqlite/chroma paths, `deterministic_test_mode`, `RerankWeights`, `contradiction_high_confidence`, `decision_evidence_floor`) defaulting to offline Mock_Extractor + local embeddings
    - Write `ocm/core/ids.py` (`IdGenerator` with random and deterministic-seeded modes) and `ocm/core/logging.py` (`ResearchLogger` with `log_write`, `log_query`, `log_benchmark` JSONL record methods)
    - _Requirements: 27.2, 27.3, 27.5, 25.1, 25.2, 25.3_
  - [x]* 1.4 Write unit tests for config defaults and deterministic ID generation
    - Assert offline defaults (mock extractor, local embeddings) and that deterministic mode derives IDs from type/normalized_name/source_ref/counter
    - _Requirements: 27.2, 27.3, 27.5_

- [x] 2. Ontology layer: schemas, enums, relation registry
  - [x] 2.1 Implement enums, status-defaulting mixin, and entity/assertion models
    - Write `ocm/ontology/enums.py` (all status/priority/severity/intent/resolution enums) and `ocm/ontology/models.py` with `StatusDefaultMixin` and Pydantic v2 models: Person, Organization, Project, Task, Event (no status), Claim, Document (no status), Decision, Assertion, QuarantineRecord, Provenance
    - Enforce `confidence` in [0,1] via `confloat`, default status to `unknown` with `status_defaulted` metadata, and reject out-of-enum values
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15_
  - [x]* 2.2 Write property test for schema round-trip identity
    - **Property 1: Schema round-trip identity** — `model_validate_json(model_dump_json(x)) == x` for every model
    - **Validates: Requirements 1.12, 3.2**
    - _Requirements: 1.12, 3.2_
  - [x]* 2.3 Write property test for confidence bounds
    - **Property 2: Confidence always in [0,1]** — valid confidences stay in range; out-of-range raises validation error
    - **Validates: Requirements 1.6, 1.9, 8.7**
    - _Requirements: 1.6, 1.9, 8.7_
  - [x]* 2.4 Write schema unit tests for enum rejection and status defaulting
    - Reject `confidence > 1` and `confidence < 0`; reject invalid status/priority/severity/write_intent; confirm default-to-unknown sets `status_defaulted`; confirm Event/Document have no status field
    - _Requirements: 1.11, 1.13, 1.14, 1.15, 26.1_
  - [x] 2.5 Implement the relation signature registry and task transition map
    - Write `ocm/ontology/relations.py` with `Cardinality`, `RelationSignature`, the frozen `RELATION_SIGNATURES` for all 13 relations (incl. SUPERSEDES), the `get_relation_signature` lookup, and `TASK_STATUS_TRANSITIONS`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 8.11_
  - [x]* 2.6 Write unit tests for registry lookup and transitions
    - Assert lookup returns declared source/target types and cardinality per predicate, raises on unknown predicate, and that `TASK_STATUS_TRANSITIONS` matches the design map
    - _Requirements: 2.14, 8.11_

- [x] 3. Storage and graph foundation
  - [x] 3.1 Implement pipeline contract models
    - Write `ocm/memory/contracts.py`: `ExtractionResult`, `ResolutionOutcome`, `CandidateAssertion`, `ValidationResult`, `ContradictionResult`, `WriteOutcome`, `WriteSummary`
    - _Requirements: 6.1, 6.2, 8.1, 9.7, 10.1, 19.2_
  - [x] 3.2 Implement Storage_Repository interface and SQLite adapter with DDL
    - Write `ocm/memory/repository.py` (`StorageRepository` ABC) and `ocm/memory/sqlite_repository.py` (`SQLiteRepository`) provisioning the 7 tables (entities, assertions, claims, documents, quarantine_records, provenance, embeddings) and all CRUD methods, behind the interface so a Postgres adapter is a drop-in
    - _Requirements: 11.1, 11.2, 11.3, 11.4_
  - [x] 3.3 Implement Graph_Store and rebuild-on-restart
    - Write `ocm/memory/graph_store.py` (`GraphStore` over NetworkX `MultiDiGraph`, accepted-only edges) and `rebuild_graph(repo)` that reloads entities as nodes and accepted assertions as edges
    - _Requirements: 11.5, 11.6, 11.8_
  - [x]* 3.4 Write unit tests for repository persistence and graph rebuild
    - Round-trip entities/assertions/quarantine/provenance through SQLite; confirm rebuilt graph contains only accepted assertions and equals pre-restart state
    - _Requirements: 11.1, 11.5, 11.8_

- [x] 4. Assertion Builder (W4) and minimal manual write path
  - [x] 4.1 Implement the Assertion_Builder (W4)
    - Write `ocm/memory/assertion_builder.py` constructing `CandidateAssertion` with `operation="upsert_assertion"`, populating subject_id/predicate/object_id/confidence/source_ref/write_intent, defaulting `write_intent` to `new_fact`
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 4.2 Implement a minimal manual write path
    - Write `ocm/memory/manual_write.py` that takes pre-resolved entities + a `CandidateAssertion`, persists entities and an accepted assertion through the repository, and reflects it in the Graph_Store (write-through), to exercise storage end-to-end before the validators exist
    - _Requirements: 6.1, 11.6, 11.5_
  - [x]* 4.3 Write unit tests for the Assertion_Builder and manual write path
    - Assert builder defaults and field population; assert manual write persists and graph-reflects an accepted assertion
    - _Requirements: 6.1, 6.2, 6.3, 11.6_

- [x] 5. Checkpoint - Phase 1 foundation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Schema validation (W5) and constraint validation (W6, C1–C10)
  - [x] 6.1 Implement the Schema_Validator (W5, structural only)
    - Write `ocm/validation/schema_validator.py` checking required fields, registered predicate, valid status enum, confidence in [0,1], subject/object reference existing entities, and the static registry signature — structural only, no graph-level domain/range
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_
  - [x]* 6.2 Write schema-validator unit tests
    - Reject invalid predicate, reject invalid source/target type (structural), accept a valid assertion
    - _Requirements: 7.2, 7.6, 26.1_
  - [x] 6.3 Implement the Constraint_Validator C1–C10 as separate validators
    - Write `ocm/validation/constraints.py` with one function per constraint: C1 identity uniqueness, C2 temporal sanity, C3 acyclic PRECEDES, C4 done-task completion event, C5 inactive assignee, C6 confidence bounds, C7 contradiction gate (delegates to W7), C8 decision evidence floor, C9 graph-level domain/range, C10 task status transition; aggregate into a `ValidationResult` with valid/reason/severity/conflicting_ids/recommended_action
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.9, 8.10, 8.11, 8.12, 8.13_
  - [x]* 6.4 Write temporal constraint unit tests (C2, C3)
    - Reject `timestamp_end < timestamp_start`, reject a PRECEDES cycle, accept a valid PRECEDES chain
    - _Requirements: 8.3, 8.4, 26.2_
  - [x]* 6.5 Write task constraint unit tests (C4, C5)
    - Reject a done task without a completion event, accept a done task with one; reject assignment to an inactive person, accept assignment to an active person
    - _Requirements: 8.5, 8.6, 26.3_
  - [x]* 6.6 Write consolidated constraint coverage tests (>=10 passing)
    - At least 10 passing constraint tests spanning C1–C10 (identity uniqueness, confidence bounds, domain/range, transition map, decision evidence floor, etc.)
    - _Requirements: 8.2, 8.7, 8.9, 8.10, 8.11, 8.12, 26.6, 28.4_

- [x] 7. Contradiction Checker (W7)
  - [x] 7.1 Implement the Contradiction_Checker
    - Write `ocm/validation/contradiction_checker.py` detecting hard, soft, and temporal contradictions: exact-predicate conflicts, status conflicts, explicit CONTRADICTS links, single-valued (m:1/1:1) cardinality conflicts, and temporal overlaps; return `has_conflict/severity/reason/conflicting_assertion_ids/kind/recommended_action`
    - Wire C7 in the Constraint_Validator to invoke this checker (single source of contradiction truth)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 8.8_
  - [x]* 7.2 Write contradiction-detection unit tests
    - Detect a high-confidence exact-predicate conflict, a single-valued ASSIGNED_TO conflict, an explicit CONTRADICTS conflict, and a temporal overlap; confirm a low-confidence contradiction yields only a soft warning
    - _Requirements: 9.2, 9.4, 9.5, 9.6, 26.4_

- [x] 8. Commit Manager (W8), Quarantine Store, and Provenance Tracker
  - [x] 8.1 Implement the Quarantine_Store and Provenance_Tracker
    - Write `ocm/memory/quarantine_store.py` (`add/list/set_status` over `quarantine_records`, persists across restarts) and `ocm/memory/provenance_tracker.py` (`record/for_subject` over the `provenance` table)
    - _Requirements: 11.7, 12.1, 12.4_
  - [x] 8.2 Implement the Commit_Manager (W8) routing
    - Write `ocm/memory/commit_manager.py` routing each candidate to accept (graph + assertion store + vector index hook + provenance), supersede (mark old superseded, accept new, add SUPERSEDES edge, preserve both provenances), quarantine (QuarantineRecord, excluded from accepted memory), or reject (logged, excluded); enforce that quarantined/rejected never enter the graph as accepted and report all validation failures
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 12.3, 2.13_
  - [x]* 8.3 Write property test for the contradiction-gate invariant
    - **Property 6: Contradiction-gate invariant** — no two accepted assertions with confidence > 0.8 contradict each other; at most one survives accepted
    - **Validates: Requirements 8.8, 9.1, 9.2, 9.5**
    - _Requirements: 8.8, 9.1, 9.2, 9.5_
  - [x]* 8.4 Write property test for ASSIGNED_TO single-valued invariant
    - **Property 10: ASSIGNED_TO single-valued (m:1) invariant** — every Task has at most one accepted ASSIGNED_TO edge; a second distinct assignee is detected as a conflict
    - **Validates: Requirements 2.5, 9.5**
    - _Requirements: 2.5, 9.5_
  - [x]* 8.5 Write property test for done-task completion event
    - **Property 11: Accepted done Task has a completion event** — every accepted done-task write has a RESULTS_IN completion Event; otherwise it is quarantined
    - **Validates: Requirements 8.5**
    - _Requirements: 8.5_
  - [x]* 8.6 Write property test for PRECEDES acyclicity
    - **Property 5: PRECEDES graph stays acyclic** — the Event/PRECEDES projection is always a DAG; a cycle-closing edge is never accepted
    - **Validates: Requirements 8.4**
    - _Requirements: 8.4_
  - [x]* 8.7 Write property test for supersession integrity
    - **Property 7: Supersession preserves provenance and leaves exactly one accepted** — after a correction supersedes B: A accepted, B superseded, SUPERSEDES(A→B) edge exists, exactly one accepted, provenance for both
    - **Validates: Requirements 10.2, 12.3, 2.13**
    - _Requirements: 10.2, 12.3, 2.13_
  - [x]* 8.8 Write property test for provenance coverage
    - **Property 4: Every accepted assertion has provenance** — each accepted assertion has at least one provenance record with matching subject_id
    - **Validates: Requirements 12.1, 12.2, 12.4**
    - _Requirements: 12.1, 12.2, 12.4_
  - [x]* 8.9 Write commit/supersede/quarantine unit tests
    - Reject/quarantine a high-confidence contradiction (C7), allow a correction to supersede, and retrieve unresolved conflicts from the Quarantine_Store
    - _Requirements: 10.2, 10.3, 26.4, 28.5, 28.6_

- [x] 9. Checkpoint - Phase 2 governance
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Extraction, normalization, resolution, and full write pipeline (W1–W3, wire W1–W8)
  - [x] 10.1 Implement the Extractor interface and Mock_Extractor (default)
    - Write `ocm/extraction/base.py` (`Extractor` protocol + `ExtractionError`) and `ocm/extraction/mock_extractor.py` producing deterministic, offline, strict-JSON-validated `ExtractionResult` (entities/events/claims/documents/decisions/relations); reject input on Pydantic validation failure
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7_
  - [x] 10.2 Implement the opt-in LLM_Extractor
    - Write `ocm/extraction/llm_extractor.py` calling an OpenAI-compatible JSON-mode endpoint (gated by `settings.extractor == "llm"`), validating output into `ExtractionResult`, raising `ExtractionError` on timeout/non-JSON
    - _Requirements: 3.2, 3.3, 3.6_
  - [x] 10.3 Implement the Normalizer (W2)
    - Write `ocm/resolution/normalizer.py` normalizing names/aliases (canonical, non-merging), timestamps (ISO-8601 UTC), status synonyms (incl. "completed"→"done"), priority synonyms (incl. "high priority"→"high"), relation names → predicates, and confidence → [0,1]
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [x]* 10.4 Write Normalizer unit tests
    - Assert status/priority synonym mapping, confidence clamping, predicate canonicalization, and that distinct entities are preserved (no merge)
    - _Requirements: 4.3, 4.4, 4.6, 4.7_
  - [x] 10.5 Implement the Entity_Resolver (W3)
    - Write `ocm/resolution/entity_resolver.py` applying the exact priority order (exact ID → exact normalized name+type → alias+type → contextual → create new), emitting POSSIBLY_SAME_AS on uncertain matches, returning `resolution_status/entity_id/candidate_matches`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
  - [x]* 10.6 Write Entity_Resolver unit tests
    - Cover each resolution branch and the priority ordering, including POSSIBLY_SAME_AS creation on uncertainty
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6, 5.8_
  - [x] 10.7 Wire the full WritePipeline W1–W8
    - Write `ocm/memory/write_pipeline.py` orchestrating W1→W2→W3→W4→W5→W6→W7→W8 per-candidate (independent failure routing), aggregating `accepted/superseded/quarantined/rejected` lists + `WriteSummary`; embed accepted claims/documents/assertions/events via the vector-index hook; record per-write research logs
    - _Requirements: 3.1, 10.1, 10.6, 10.7, 13.5, 16.6, 19.2, 25.1_
  - [x]* 10.8 Write property test for deterministic IDs across runs
    - **Property 8: Deterministic IDs across runs** — under `deterministic_test_mode`, two runs over a fixed input batch produce identical entity/assertion ID sequences
    - **Validates: Requirements 27.5, 3.5**
    - _Requirements: 27.5, 3.5_

- [x] 11. Checkpoint - Phase 3 write pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Embeddings and vector index
  - [x] 12.1 Implement the EmbeddingProvider and LocalEmbeddingProvider
    - Write `ocm/retrieval/embeddings.py` (`EmbeddingProvider` protocol + `LocalEmbeddingProvider` wrapping all-MiniLM-L6-v2, 384-dim, offline, swappable; selected by default when no embedding config)
    - _Requirements: 13.1, 13.2, 13.3_
  - [x] 12.2 Implement the Chroma Vector_Index
    - Write `ocm/retrieval/vector_index.py` (`VectorIndex` over Chroma with persistent + in-memory modes, `add`/`query` with `{memory_id, memory_type, status}` metadata and cosine→similarity); add embedding-text construction for assertion/claim/document/event and connect the Commit_Manager embed hook
    - _Requirements: 13.4, 13.5, 13.6, 16.6_
  - [x]* 12.3 Write vector-index unit tests (in-memory mode)
    - Assert add/query round-trip, status metadata filtering, and similarity conversion
    - _Requirements: 13.4, 13.6_

- [x] 13. Retrieval pipeline (R0–R4) and core integration test
  - [x] 13.1 Implement the Query_Classifier (R0)
    - Write `ocm/retrieval/query_classifier.py` classifying into direct_fact/temporal/planning/contradiction_check/open_ended/provenance_request and returning `query_type/entities/predicates/needs_semantic_fallback`
    - _Requirements: 14.1, 14.2_
  - [x] 13.2 Implement the Symbolic_Retriever (R1)
    - Write `ocm/retrieval/symbolic_retriever.py` answering owner (OWNS), assignee (ASSIGNED_TO), and preceding-events (PRECEDES) queries from the Graph_Store; flag exact matches so the reranker forces semantic_similarity = 1.0
    - _Requirements: 15.1, 15.2, 15.3, 15.4_
  - [x] 13.3 Implement the Semantic_Retriever (R2)
    - Write `ocm/retrieval/semantic_retriever.py` embedding the query and searching top-k claims/assertions/documents/events; accepted-by-default; include quarantined items for conflict queries and conflict-relevant cases; exclude irrelevant quarantined items otherwise
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  - [x] 13.4 Implement the Reranker (R3)
    - Write `ocm/retrieval/reranker.py` computing `score = alpha*sim + beta*graph + gamma*conf + delta*prov + eta*recency - lambda*contradiction` with default weights, and applying the contradiction penalty
    - _Requirements: 17.1, 17.2, 17.3_
  - [x]* 13.5 Write property test for reranker contradiction monotonicity
    - **Property 9: Reranker contradiction monotonicity** — an item with contradiction_penalty > 0 scores strictly lower than an otherwise-identical item with penalty 0 (lambda = 0.30)
    - **Validates: Requirements 17.1, 17.2, 17.3**
    - _Requirements: 17.1, 17.2, 17.3_
  - [x] 13.6 Implement the Evidence_Packager (R4) and wire the RetrievalPipeline
    - Write `ocm/retrieval/evidence_packager.py` assembling `EvidencePackage` (answer optional, confidence, supporting_assertions with id+confidence, supporting_sources via provenance, conflicts, missing_information, retrieved_items) and `ocm/retrieval/retrieval_pipeline.py` orchestrating R0→R4; record per-query research logs
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 25.2_
  - [x]* 13.7 Write property test for accepted-only default retrieval
    - **Property 3: Quarantined/rejected never appear in accepted retrieval** — for any write stream and any default query, every accepted-memory result has status accepted
    - **Validates: Requirements 10.3, 10.4, 10.5, 16.2, 16.5**
    - _Requirements: 10.3, 10.4, 10.5, 16.2, 16.5_
  - [x]* 13.8 Write retrieval unit tests
    - Symbolic retrieval returns the correct owner, semantic retrieval returns a relevant claim, a conflict query retrieves a quarantined contradiction, the reranker penalizes a contradicted assertion, an evidence package includes sources
    - _Requirements: 15.1, 16.1, 16.3, 17.3, 18.3, 26.5_
  - [x]* 13.9 Write the end-to-end Task T1 integration test
    - Three sequential writes (Alice owns Project Orion + Bob assigned to T1 → accepted; Bob completed T1 with completion Event → accepted; T1 not started at high confidence → quarantined), then query "What is the current status of Task T1?" and assert the package reports `done` and surfaces the quarantined conflict
    - _Requirements: 28.5, 28.7, 28.8_

- [x] 14. Checkpoint - Phase 4 retrieval
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. API service (FastAPI)
  - [x] 15.1 Implement the CoreContainer and request/response models
    - Write `ocm/core/container.py` wiring repo, ids, graph (rebuild on startup), embeddings, vector index, quarantine, provenance, extractor, write + retrieval pipelines, and logger; write `ocm/app/api/schemas.py` for all endpoint request/response models
    - _Requirements: 11.8, 19.1, 27.2, 27.3_
  - [x] 15.2 Implement the five memory endpoints
    - Write `ocm/app/api/routes.py` and `ocm/app/main.py`: POST /memory/write (outcome lists + summary), POST /memory/query (query_type + evidence fields), POST /memory/validate (decision without committing), GET /memory/entity/{id}, GET /memory/conflicts
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 28.1, 28.2, 28.7_
  - [x] 15.3 Implement the routes_debug router
    - Write `ocm/app/api/routes_debug.py` (read-only /debug/graph, /debug/quarantine, /debug/provenance/{id}) mounted only in debug/deterministic mode
    - _Requirements: 19.1_
  - [x]* 15.4 Write API tests with FastAPI TestClient
    - Cover all five endpoints (write returns accepted/quarantined + summary; query returns symbolic + semantic results; validate does not mutate state; entity returns entity + assertions; conflicts returns unresolved + quarantined) and a service start smoke test
    - _Requirements: 28.1, 28.2, 28.7, 28.8, 19.4, 19.5, 19.6_

- [x] 16. Agent loop and answer policy
  - [x] 16.1 Implement the MemoryTool and Agent_Loop
    - Write `ocm/agent/memory_tool.py` (`query`/`write` mapping to the pipelines) and `ocm/agent/loop.py` with nodes receive→retrieve→answer→extract→validate→commit, calling memory query per turn and memory write for new memory, with no LangGraph dependency
    - _Requirements: 20.1, 20.2, 20.3, 20.4_
  - [x] 16.2 Implement the Answer_Policy (P1–P5)
    - Write `ocm/agent/answer_policy.py` preferring accepted high-confidence assertions (P1), surfacing conflicts (P2), keeping conflicting claims separate (P3), including provenance when high_stakes/decision-support (P4), and stating missing evidence (P5)
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5_
  - [x]* 16.3 Write agent loop and answer policy unit tests
    - Assert a turn triggers query then write; assert P1–P5 rendering (conflicts surfaced separately, provenance on high_stakes, missing evidence stated)
    - _Requirements: 20.2, 20.3, 21.2, 21.3, 21.4, 21.5_

- [x] 17. Evaluation harness: baselines, benchmark, metrics, logging
  - [x] 17.1 Implement the MemoryStrategy and baselines B0–B4
    - Write `ocm/evaluation/strategies.py` (`MemoryStrategy` + `StrategyToggles`) and `ocm/evaluation/baselines.py` defining B0 (vectors only), B1 (graph/symbolic only), B2 (graph+semantic, no governance), B3 (full hybrid + contradiction + quarantine + provenance), B4 (B3 + Answer_Policy) via feature toggles
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5_
  - [x] 17.2 Implement the Benchmark_Generator
    - Write `ocm/evaluation/benchmark.py` producing a seeded JSONL dataset (id/category/sessions/questions with expected_answer_contains, expected_conflict, optional expected_supporting_ids) across 6 categories, >=25 per category and >=150 total, plus the 6 hand-authored anchor examples; identical output across runs for a fixed seed
    - _Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 23.6_
  - [x] 17.3 Implement the Baseline_Runner and Research_Logger benchmark logging
    - Write `ocm/evaluation/runner.py` executing baselines B0–B3 against the JSONL benchmark and recording per-benchmark-example logs (baseline_name, answer, retrieved_ids, conflicts, expected_conflict, score, latency_ms)
    - _Requirements: 22.6, 25.3, 28.9_
  - [x] 17.4 Implement the Metrics_Reporter
    - Write `ocm/evaluation/metrics.py` computing retrieval (hit@1/3/5, supporting-evidence precision/recall), answer (factual precision/recall, contradiction rate, conflict-surfacing rate, hallucination rate), write-time (invalid-write detection, false-quarantine, contradiction precision/recall, entity-resolution accuracy), and agent metrics, with comparisons against B0
    - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5_
  - [x]* 17.5 Write benchmark and metrics unit tests
    - Assert seeded benchmark reproducibility, category/count thresholds, anchor inclusion, and metric computation against a small fixture
    - _Requirements: 23.3, 23.4, 23.5, 24.1, 24.5_

- [x] 18. Scripts and README
  - [x] 18.1 Implement entry-point scripts
    - Write `ocm/scripts/serve.py` (start API), `ocm/scripts/build_benchmark.py` (generate JSONL), `ocm/scripts/run_benchmark.py` (run B0–B3), `ocm/scripts/report_metrics.py` (compute/report metrics)
    - _Requirements: 28.9, 28.10_
  - [x] 18.2 Write the README
    - Document setup, the five API endpoints, offline/deterministic configuration, and the experiment commands (build benchmark, run baselines, report metrics)
    - _Requirements: 28.11_

- [x] 19. Final checkpoint - Definition of Done
  - Ensure all tests pass (>=10 constraint tests, contradiction gate, quarantine persistence, integration test, evidence package), ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each sub-task references the specific requirement clauses it satisfies for traceability; together they cover all 28 requirements.
- Property-based tests (Hypothesis) validate the 11 universal correctness properties and run >=100 iterations each, tagged `Feature: ontology-constrained-memory, Property N`.
- Unit tests cover the Req 26 required cases (schema, temporal, task, contradiction, retrieval) and >=10 constraint tests; the Task T1 scenario is the end-to-end integration test.
- All tests run with `deterministic_test_mode=True`, `chroma_mode="memory"`, and the `Mock_Extractor` for hermetic, offline, reproducible runs.
- Checkpoints (tasks 5, 9, 11, 14, 19) mark phase boundaries for incremental validation.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "2.5"] },
    { "id": 2, "tasks": ["1.4", "2.2", "2.3", "2.4", "2.6", "3.1"] },
    { "id": 3, "tasks": ["3.2"] },
    { "id": 4, "tasks": ["3.3"] },
    { "id": 5, "tasks": ["3.4", "4.1", "4.2"] },
    { "id": 6, "tasks": ["4.3", "6.1", "6.3"] },
    { "id": 7, "tasks": ["6.2", "6.4", "6.5", "6.6", "7.1"] },
    { "id": 8, "tasks": ["7.2", "8.1"] },
    { "id": 9, "tasks": ["8.2"] },
    { "id": 10, "tasks": ["8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9", "10.1", "10.2", "10.3", "10.5"] },
    { "id": 11, "tasks": ["10.4", "10.6", "10.7"] },
    { "id": 12, "tasks": ["10.8", "12.1"] },
    { "id": 13, "tasks": ["12.2"] },
    { "id": 14, "tasks": ["12.3", "13.1", "13.2", "13.3", "13.4"] },
    { "id": 15, "tasks": ["13.5", "13.6"] },
    { "id": 16, "tasks": ["13.7", "13.8", "13.9", "15.1"] },
    { "id": 17, "tasks": ["15.2", "15.3"] },
    { "id": 18, "tasks": ["15.4", "16.1", "16.2", "17.1", "17.2"] },
    { "id": 19, "tasks": ["16.3", "17.3", "17.4"] },
    { "id": 20, "tasks": ["17.5", "18.1"] },
    { "id": 21, "tasks": ["18.2"] }
  ]
}
```

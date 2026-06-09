# Requirements Document

## Introduction

This document specifies the requirements for **Ontology-Constrained Memory (OCM)**, a research prototype of a reusable, ontology-constrained memory layer for long-horizon LLM agents. The central research claim is that agent memory should be governed at write time via typed graph assertions, ontology constraints, contradiction gates, and provenance, and then retrieved through hybrid symbolic-semantic evidence packaging.

The system accepts unstructured inputs (user messages, tool outputs, documents), extracts candidate memory items, normalizes and resolves entities, converts information into typed graph assertions, validates those assertions against an ontology and constraint rules, detects contradictions before committing, stores accepted assertions in a graph-backed memory store, stores rejected or conflicting candidates in quarantine, supports hybrid retrieval (symbolic graph queries plus semantic vector search), and returns evidence packages including provenance and unresolved conflicts.

OCM is a memory module pluggable into agent frameworks, not an end-user application. It lives under an `ocm/` Python package within the workspace.

### Technology and Scope Decisions

- **Language/Runtime**: Python 3.11+, FastAPI, Pydantic v2, NetworkX, pytest
- **Vector store**: Chroma with local persistence
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local, offline) behind a swappable embedding interface
- **LLM extractor**: deterministic mock extractor by default (no API key required); real OpenAI-compatible extraction is opt-in via configuration
- **Database**: SQLite only at present, behind a clean storage/repository interface so a Postgres adapter can be added later
- **Benchmark**: seeded programmatic generator (~25 examples each across 6 categories, ~150 total) plus hand-authored anchor examples; fixed random seed for reproducibility
- **Agent demo**: lightweight custom agent loop mimicking LangGraph node structure; LangGraph is optional, not a required dependency
- **Excluded from scope**: authentication, frontend, multi-user permissions, deployment

## Glossary

- **OCM_System**: The complete ontology-constrained memory layer being specified.
- **Ontology_Schema**: The set of Pydantic v2 models defining entity types, their fields, and enumerations per Ontology v1.
- **Relation_Registry**: The component holding directed relation signatures, including source types, target types, and cardinality.
- **Write_Pipeline**: The ordered processing stages W1 through W8 that transform unstructured input into committed, superseded, quarantined, or rejected outcomes.
- **Extractor**: The component (W1) that produces candidate memory items as strict JSON validated by Pydantic. Has a deterministic Mock_Extractor (default) and an opt-in LLM_Extractor.
- **Mock_Extractor**: A deterministic extractor producing reproducible candidate output without external API calls.
- **LLM_Extractor**: An optional OpenAI-compatible extractor enabled via configuration.
- **Normalizer**: The component (W2) that normalizes names, aliases, timestamps, statuses, relation names, and confidence values.
- **Entity_Resolver**: The component (W3) that performs conservative entity resolution and produces a resolution status.
- **Assertion_Builder**: The component (W4) that constructs candidate assertions.
- **Schema_Validator**: The component (W5) that validates candidate assertions against the Ontology_Schema and Relation_Registry.
- **Constraint_Validator**: The component (W6) that evaluates graph-level constraints C1 through C10.
- **Contradiction_Checker**: The component (W7) that detects hard, soft, and temporal contradictions.
- **Commit_Manager**: The component (W8) that commits, supersedes, quarantines, or rejects candidate assertions.
- **Graph_Store**: The in-memory NetworkX graph holding entities and accepted assertions.
- **Assertion_Store**: The persistent store (SQLite) of assertions.
- **Quarantine_Store**: The persistent store of quarantined candidates as QuarantineRecord items.
- **Vector_Index**: The Chroma-backed semantic index of embedded memory items.
- **Embedding_Provider**: The swappable embedding interface; default implementation uses sentence-transformers/all-MiniLM-L6-v2.
- **Storage_Repository**: The clean repository interface abstracting persistence (SQLite now, Postgres later).
- **Provenance_Tracker**: The component that records and preserves the origin of memory items, including source_ref, created_at, extractor_version, and supporting evidence IDs, across creation, retrieval, and supersession.
- **Retrieval_Pipeline**: The ordered retrieval stages R0 through R4.
- **Query_Classifier**: The component (R0) that classifies queries into query types.
- **Symbolic_Retriever**: The component (R1) that answers graph queries.
- **Semantic_Retriever**: The component (R2) that performs vector search.
- **Reranker**: The component (R3) that performs constraint-aware reranking.
- **Evidence_Packager**: The component (R4) that assembles evidence packages.
- **Evidence_Package**: The structured retrieval result containing answer, confidence, supporting assertions, supporting sources, conflicts, missing information, and retrieved items.
- **API_Service**: The FastAPI service exposing the memory endpoints.
- **Agent_Loop**: The lightweight LangGraph-style agent demonstration loop.
- **Answer_Policy**: The agent answer policy P1 through P5.
- **Baseline_Runner**: The component that executes baselines B0 through B4 against the benchmark.
- **Benchmark_Generator**: The seeded component that produces the evaluation benchmark dataset.
- **Metrics_Reporter**: The component that computes and reports evaluation metrics.
- **Research_Logger**: The component that records per-write, per-query, and per-benchmark-example logs.
- **Assertion**: A typed graph statement (subject_id, predicate, object_id, confidence, status, source_ref, created_at, optional valid_from/valid_to, optional extractor_version, write_intent).
- **write_intent**: One of new_fact, update, correction, deletion, hypothesis.
- **Provenance**: The recorded origin and supporting references of an assertion or claim, including source_ref, created_at, extractor_version where available, and supporting evidence IDs where available.
- **SUPERSEDES**: A directed relation from a new accepted Assertion to the prior Assertion it replaces, used to link superseded and accepted assertions during supersession.
- **deterministic_test_mode**: A configuration flag that, when enabled, causes the OCM_System to generate deterministic IDs derived from entity type, normalized name, source_ref, and/or seeded counters so that repeated runs produce identical IDs.
- **Quarantine**: The outcome where a candidate is stored as a QuarantineRecord and not exposed as accepted memory.
- **Supersession**: The outcome where an existing accepted assertion is marked superseded and a new assertion is accepted in its place.

## Requirements

### Requirement 1: Ontology Entity Schema Definition

**User Story:** As a memory layer developer, I want all ontology entity types defined as strict typed models, so that every memory item has a validated, consistent structure.

#### Acceptance Criteria

1. THE Ontology_Schema SHALL define a Person model with fields id, name, roles (list), status (one of active, inactive, unknown), and aliases (list).
2. THE Ontology_Schema SHALL define an Organization model with fields id, name, type, and status.
3. THE Ontology_Schema SHALL define a Project model with fields id, name, optional goal, status (one of active, inactive, completed, cancelled, unknown), and optional owner_id.
4. THE Ontology_Schema SHALL define a Task model with fields id, title, status (one of todo, in_progress, blocked, done, cancelled, unknown), priority (one of low, medium, high, urgent, unknown), optional project_id, optional assignee_id, and optional due_at.
5. THE Ontology_Schema SHALL define an Event model with fields id, type, timestamp_start, optional timestamp_end, and description.
6. THE Ontology_Schema SHALL define a Claim model with fields id, text, source_ref, confidence (a value in the range 0 through 1), status (one of accepted, rejected, quarantined, superseded, unknown), and created_at.
7. THE Ontology_Schema SHALL define a Document model with fields id, title, path_or_url, created_at, and tags (list).
8. THE Ontology_Schema SHALL define a Decision model with fields id, summary, timestamp, optional made_by, optional rationale, and status (one of draft, final, superseded, rejected, unknown).
9. THE Ontology_Schema SHALL define an Assertion model with fields id, subject_id, predicate, object_id, confidence (a value in the range 0 through 1), status, source_ref, created_at, optional valid_from, optional valid_to, optional extractor_version, and write_intent (one of new_fact, update, correction, deletion, hypothesis).
10. THE Ontology_Schema SHALL define a QuarantineRecord model with fields id, candidate_payload, reason, severity (one of low, medium, high), conflicting_ids (list), created_at, and status (one of unresolved, resolved, dismissed).
11. WHEN a value outside a model's declared enumeration is supplied for a status, priority, severity, or write_intent field, THE Ontology_Schema SHALL reject the value with a validation error.
12. THE Ontology_Schema SHALL be implemented using Pydantic v2 models.
13. WHEN an entity or assertion model that defines a status field is constructed and no status value is supplied, THE Ontology_Schema SHALL default the status to unknown.
14. THE Ontology_Schema SHALL define the Event model and the Document model without a status field, such that no status defaulting applies to the Event model or the Document model.
15. WHERE metadata is available, THE Ontology_Schema SHALL record that the status was defaulted.

### Requirement 2: Relation Signature Registry

**User Story:** As a memory layer developer, I want every relation type defined with directed source and target types and cardinality, so that assertions can be validated against allowed relations.

#### Acceptance Criteria

1. THE Relation_Registry SHALL define PARTICIPATES_IN with source type Person, target type Event, and cardinality m:n.
2. THE Relation_Registry SHALL define MEMBER_OF with source type Person, target type Organization, and cardinality m:n.
3. THE Relation_Registry SHALL define OWNS with source types Person or Organization, target type Project, and cardinality m:n.
4. THE Relation_Registry SHALL define CONTAINS with source type Project, target type Task, and cardinality 1:n.
5. THE Relation_Registry SHALL define ASSIGNED_TO with source type Task, target type Person, and cardinality m:1.
6. THE Relation_Registry SHALL define PRECEDES with source type Event, target type Event, and cardinality m:n.
7. THE Relation_Registry SHALL define SUPPORTS with source type Claim, target types Claim or Decision, and cardinality m:n.
8. THE Relation_Registry SHALL define CONTRADICTS with source types Claim or Assertion, target types Claim or Assertion, and cardinality m:n.
9. THE Relation_Registry SHALL define EVIDENCE_FOR with source types Document or Event, target types Claim, Decision, or Assertion, and cardinality m:n.
10. THE Relation_Registry SHALL define RESULTS_IN with source types Event or Decision, target types Event, Task, or Project, and cardinality m:n.
11. THE Relation_Registry SHALL define ABOUT with source types Document or Claim, target types Person, Project, Task, Event, or Decision, and cardinality m:n.
12. THE Relation_Registry SHALL define POSSIBLY_SAME_AS with source and target types drawn from Person, Organization, Project, Task, or Event, and cardinality m:n.
13. THE Relation_Registry SHALL define SUPERSEDES with source type Assertion, target type Assertion, and cardinality m:n.
14. WHEN the Relation_Registry is queried for a predicate, THE Relation_Registry SHALL return the declared source types, target types, and cardinality for that predicate.

### Requirement 3: Candidate Extraction (W1)

**User Story:** As a memory layer developer, I want unstructured input converted into validated candidate memory items, so that downstream stages operate on structured data.

#### Acceptance Criteria

1. WHEN unstructured input text is submitted to the Write_Pipeline, THE Extractor SHALL produce candidate memory items containing entities, events, claims, documents, decisions, and relations.
2. THE Extractor SHALL return candidate output as strict JSON validated by Pydantic models.
3. IF the extractor output fails Pydantic validation, THEN THE Write_Pipeline SHALL reject the input and record a validation failure.
4. WHERE no extractor configuration is provided, THE OCM_System SHALL use the Mock_Extractor as the default extractor.
5. WHEN the Mock_Extractor processes identical input with an identical configuration, THE Mock_Extractor SHALL produce identical candidate output.
6. WHERE the LLM_Extractor is enabled via configuration, THE Extractor SHALL use an OpenAI-compatible extraction backend.
7. THE Mock_Extractor SHALL operate without requiring any external API key or network access.

### Requirement 4: Normalization (W2)

**User Story:** As a memory layer developer, I want extracted values normalized to canonical forms, so that equivalent inputs map to consistent representations without aggressive merging.

#### Acceptance Criteria

1. THE Normalizer SHALL normalize entity names and aliases to canonical forms.
2. THE Normalizer SHALL normalize timestamps to a consistent representation.
3. WHEN a status synonym is encountered, THE Normalizer SHALL map the synonym to its canonical enumeration value, including mapping "completed" to "done".
4. WHEN a priority synonym is encountered, THE Normalizer SHALL map the synonym to its canonical enumeration value, including mapping "high priority" to "high".
5. THE Normalizer SHALL normalize relation names to their canonical predicate identifiers.
6. THE Normalizer SHALL normalize confidence values to numeric values in the range 0 through 1.
7. THE Normalizer SHALL preserve distinct entities as distinct and SHALL NOT merge entities solely on the basis of normalization.

### Requirement 5: Entity Resolution (W3)

**User Story:** As a memory layer developer, I want conservative entity resolution, so that references are linked to existing entities only when confident and otherwise tracked as possible matches.

#### Acceptance Criteria

1. WHEN resolving an entity that includes an exact existing ID, THE Entity_Resolver SHALL resolve to that existing entity and set resolution_status to resolved_existing.
2. WHEN no ID match exists but an exact normalized name and type match exists, THE Entity_Resolver SHALL resolve to that existing entity and set resolution_status to resolved_existing.
3. WHEN no name match exists but an alias and type match exists, THE Entity_Resolver SHALL resolve to that existing entity and set resolution_status to resolved_existing.
4. WHEN no exact or alias match exists but contextual evidence indicates an existing entity, THE Entity_Resolver SHALL resolve using contextual matching.
5. IF no match of any kind is found, THEN THE Entity_Resolver SHALL create a new entity and set resolution_status to created_new.
6. IF a match is uncertain, THEN THE Entity_Resolver SHALL create a POSSIBLY_SAME_AS relation and set resolution_status to possible_match.
7. THE Entity_Resolver SHALL return resolution_status (one of resolved_existing, created_new, possible_match, unresolved), entity_id, and candidate_matches.
8. THE Entity_Resolver SHALL apply the matching priority order of exact ID, then exact normalized name and type, then alias and type, then contextual, then create new.

### Requirement 6: Candidate Assertion Construction (W4)

**User Story:** As a memory layer developer, I want resolved information turned into candidate assertions, so that all memory writes are expressed as typed graph statements.

#### Acceptance Criteria

1. THE Assertion_Builder SHALL construct candidate assertions with operation upsert_assertion.
2. THE Assertion_Builder SHALL populate each candidate assertion with subject_id, predicate, object_id, confidence, source_ref, and write_intent.
3. WHERE the input does not specify a write_intent, THE Assertion_Builder SHALL assign write_intent new_fact by default.

### Requirement 7: Schema Validation (W5)

**User Story:** As a memory layer developer, I want each candidate assertion validated against the schema before constraint checks, so that structurally invalid assertions are caught early.

#### Acceptance Criteria

1. THE Schema_Validator SHALL verify that all required fields of a candidate assertion are present.
2. THE Schema_Validator SHALL verify that the predicate is a registered relation predicate.
3. THE Schema_Validator SHALL verify that the status value is a valid enumeration value.
4. THE Schema_Validator SHALL verify that the confidence value is within the range 0 through 1.
5. THE Schema_Validator SHALL verify that the subject_id and object_id reference valid existing entity IDs.
6. THE Schema_Validator SHALL verify that the candidate assertion satisfies the static relation source and target type signature declared in the Relation_Registry, performing structural validation only and not graph-level domain and range validation against resolved entity types.
7. IF a candidate assertion fails any structural schema validation check, THEN THE Schema_Validator SHALL return a failure result identifying the failed check.

### Requirement 8: Constraint Validation (W6)

**User Story:** As a memory layer developer, I want all ontology constraints enforced at write time, so that the committed graph remains internally consistent.

#### Acceptance Criteria

1. THE Constraint_Validator SHALL return a result containing valid, reason, severity, and conflicting_ids for each evaluated candidate.
2. IF two nodes of the same type would share an ID, THEN THE Constraint_Validator SHALL fail constraint C1 (identity uniqueness).
3. IF an Event has timestamp_end earlier than timestamp_start, THEN THE Constraint_Validator SHALL fail constraint C2 (temporal sanity), and WHERE timestamp_end is missing THE Constraint_Validator SHALL pass constraint C2.
4. IF a PRECEDES relation would form a cycle, THEN THE Constraint_Validator SHALL fail constraint C3 (acyclic PRECEDES).
5. IF a Task has status done without at least one completion Event related by RESULTS_IN to that Task, THEN THE Constraint_Validator SHALL fail constraint C4 and recommend quarantine.
6. IF an ASSIGNED_TO target Person has status inactive, THEN THE Constraint_Validator SHALL fail constraint C5 and recommend quarantine, and WHERE the target Person status is active or unknown THE Constraint_Validator SHALL pass constraint C5.
7. IF a confidence value is outside the range 0 through 1, THEN THE Constraint_Validator SHALL fail constraint C6 (confidence bounds).
8. WHERE assertion A CONTRADICTS assertion B and confidence of A exceeds 0.8 and confidence of B exceeds 0.8 and status of B is accepted, THE Constraint_Validator SHALL fail constraint C7 by preventing silent acceptance of A, AND WHERE write_intent of A is correction THE Constraint_Validator SHALL permit supersession, AND WHERE write_intent of A is new_fact THE Constraint_Validator SHALL recommend quarantine. THE Constraint_Validator SHALL evaluate constraint C7 by relying on and invoking the contradiction results produced by the Contradiction_Checker (W7) rather than duplicating contradiction detection logic.
9. IF a Decision has status final without at least one EVIDENCE_FOR relation from a Document or Event to that Decision, THEN THE Constraint_Validator SHALL fail constraint C8 and recommend quarantine.
10. IF an assertion predicate does not satisfy its declared source and target type signature when evaluated at graph level against the actual resolved entity types of the subject and object in the Graph_Store, THEN THE Constraint_Validator SHALL fail constraint C9 (graph-level relation domain and range validation).
11. WHERE a Task status transition is not permitted by the transition map (todo to in_progress, blocked, or cancelled; in_progress to blocked, done, or cancelled; blocked to in_progress or cancelled; done to none; cancelled to none), THE Constraint_Validator SHALL fail constraint C10 and recommend quarantine, AND WHERE write_intent is correction THE Constraint_Validator SHALL permit the transition.
12. THE Constraint_Validator SHALL implement each constraint C1 through C10 as a separate validator.
13. WHEN a candidate assertion passes graph-level relation domain and range validation (C9), THE Constraint_Validator SHALL mark the candidate as eligible for further validation, AND THE Constraint_Validator SHALL NOT treat domain and range validation success as acceptance, where final acceptance requires passing all applicable schema, constraint, contradiction, provenance, and write-intent checks.

### Requirement 9: Contradiction Detection (W7)

**User Story:** As a memory layer developer, I want contradictions detected before commit, so that conflicting memory is never silently accepted.

#### Acceptance Criteria

1. THE Contradiction_Checker SHALL detect hard contradictions, soft contradictions, and temporal contradictions.
2. THE Contradiction_Checker SHALL detect exact predicate conflicts between a candidate assertion and existing accepted assertions.
3. THE Contradiction_Checker SHALL detect status conflicts between a candidate assertion and existing accepted assertions.
4. THE Contradiction_Checker SHALL detect conflicts arising from explicit CONTRADICTS links.
5. THE Contradiction_Checker SHALL detect single-valued relation conflicts where a cardinality constraint permits only one target.
6. THE Contradiction_Checker SHALL detect temporal overlap conflicts.
7. THE Contradiction_Checker SHALL return a result containing has_conflict, severity, reason, conflicting_assertion_ids, and recommended_action.

### Requirement 10: Commit, Supersede, Quarantine, and Reject (W8)

**User Story:** As a memory layer developer, I want each candidate assertion routed to the correct outcome, so that only valid, non-conflicting memory becomes accepted while problematic memory is retained for inspection.

#### Acceptance Criteria

1. WHEN a candidate assertion passes schema validation, constraint validation, and contradiction checking, THE Commit_Manager SHALL set its status to accepted and write it to the Graph_Store, the Assertion_Store, and the Vector_Index.
2. WHEN a candidate assertion with write_intent correction supersedes an existing accepted assertion, THE Commit_Manager SHALL set the existing assertion status to superseded, set the new assertion status to accepted, and link the old assertion and the new assertion via the SUPERSEDES relation.
3. WHEN a candidate assertion is quarantined, THE Commit_Manager SHALL set its status to quarantined, write a QuarantineRecord to the Quarantine_Store, and exclude it from accepted memory.
4. WHEN a candidate assertion is rejected, THE Commit_Manager SHALL log the rejection and SHALL NOT include the assertion in default retrieval results.
5. THE Commit_Manager SHALL ensure that quarantined and rejected candidates are not written to the Graph_Store as accepted memory.
6. THE Commit_Manager SHALL exclude all validation failures from accepted memory.
7. WHEN a validation failure occurs, THE Commit_Manager SHALL report the validation failure.
8. WHERE a candidate is malformed or unusable, THE Commit_Manager SHALL reject the candidate.
9. WHERE a candidate is reviewable or conflicting, THE Commit_Manager SHALL store the candidate as a QuarantineRecord.

### Requirement 11: Storage and Repository Interface

**User Story:** As a memory layer developer, I want persistence behind a clean repository interface, so that the backend can move from SQLite to Postgres without changing callers.

#### Acceptance Criteria

1. THE Storage_Repository SHALL provide tables for entities, assertions, claims, documents, quarantine_records, provenance, and embeddings.
2. THE Storage_Repository SHALL use SQLite as the persistence backend.
3. THE Storage_Repository SHALL expose a repository interface that abstracts persistence operations from callers.
4. THE Storage_Repository SHALL be designed so that a Postgres adapter can be added without modifying caller code.
5. THE Graph_Store SHALL maintain an in-memory NetworkX representation of entities and accepted assertions.
6. WHEN an accepted assertion is committed, THE OCM_System SHALL persist it to the Assertion_Store and reflect it in the Graph_Store.
7. WHEN the OCM_System restarts, THE Quarantine_Store SHALL retain previously persisted QuarantineRecord items.
8. WHEN the OCM_System restarts, THE Graph_Store SHALL be rebuildable from the persisted entities and accepted assertions in the Storage_Repository (SQLite).

### Requirement 12: Provenance Tracking

**User Story:** As a memory layer developer, I want provenance recorded and preserved for every memory item, so that accepted answers can be traced to their origins and supersession history is retained.

#### Acceptance Criteria

1. WHEN an assertion, claim, document, or quarantine record is created, THE OCM_System SHALL record source_ref, created_at, extractor_version where available, and supporting evidence IDs where available.
2. WHEN an accepted assertion is retrieved, THE Evidence_Packager SHALL include its provenance in supporting_sources.
3. WHEN an assertion is superseded, THE OCM_System SHALL preserve provenance for both the old superseded assertion and the new accepted assertion.
4. THE Provenance_Tracker SHALL persist provenance records in the provenance table of the Storage_Repository.

### Requirement 13: Embeddings and Vector Index

**User Story:** As a memory layer developer, I want a swappable embedding interface backed by a local model, so that semantic retrieval runs offline and the embedding model can be replaced.

#### Acceptance Criteria

1. THE Embedding_Provider SHALL expose a swappable embedding interface.
2. WHERE no embedding configuration is provided, THE Embedding_Provider SHALL use the sentence-transformers/all-MiniLM-L6-v2 model.
3. THE Embedding_Provider SHALL produce embeddings using a local model without requiring network access.
4. THE Vector_Index SHALL use Chroma with local persistence.
5. WHEN an assertion or memory item is accepted, THE OCM_System SHALL add its embedding to the Vector_Index.
6. THE Vector_Index SHALL persist embeddings locally across process restarts, unless explicitly configured to use an in-memory mode for tests.

### Requirement 14: Query Classification (R0)

**User Story:** As a memory layer developer, I want incoming queries classified, so that retrieval can route to the appropriate strategy.

#### Acceptance Criteria

1. WHEN a query is received, THE Query_Classifier SHALL classify the query into one of direct_fact, temporal, planning, contradiction_check, open_ended, or provenance_request.
2. THE Query_Classifier SHALL return a result containing query_type, entities, predicates, and needs_semantic_fallback.

### Requirement 15: Symbolic Retrieval (R1)

**User Story:** As an agent integrator, I want graph-based symbolic retrieval, so that precise structural questions are answered from typed assertions.

#### Acceptance Criteria

1. WHEN a query targets a project owner, THE Symbolic_Retriever SHALL return the owner from the Graph_Store via the OWNS relation.
2. WHEN a query targets a task assignee, THE Symbolic_Retriever SHALL return the assignee from the Graph_Store via the ASSIGNED_TO relation.
3. WHEN a query targets events preceding a given event, THE Symbolic_Retriever SHALL return the preceding events from the Graph_Store via the PRECEDES relation.
4. WHEN a symbolic query produces an exact match, THE Reranker SHALL treat the semantic_similarity for that match as 1.0.

### Requirement 16: Semantic Retrieval (R2)

**User Story:** As an agent integrator, I want semantic vector retrieval, so that relevant claims and documents are found even without exact structural matches.

#### Acceptance Criteria

1. WHEN a semantic query is received, THE Semantic_Retriever SHALL embed the query and search the Vector_Index for the top-k claims, assertions, documents, and events.
2. THE Semantic_Retriever SHALL include accepted assertions in semantic results by default.
3. WHERE the query is a conflict query, THE Semantic_Retriever SHALL include quarantined items in the results.
4. WHERE a quarantined item is relevant to a conflict involving accepted memory, THE Semantic_Retriever SHALL include that quarantined item in the results.
5. IF a query is not a conflict query and a quarantined item is not relevant to an accepted-memory conflict, THEN THE Semantic_Retriever SHALL exclude quarantined items from the results.
6. WHEN an accepted Claim or an accepted Document is committed, THE OCM_System SHALL embed the Claim or Document into the Vector_Index so that it is retrievable semantically.

### Requirement 17: Constraint-Aware Reranking (R3)

**User Story:** As an agent integrator, I want retrieved items reranked using constraints and provenance, so that high-quality, non-contradicted evidence ranks highest.

#### Acceptance Criteria

1. THE Reranker SHALL compute a score as alpha times semantic_similarity plus beta times graph_relevance plus gamma times confidence plus delta times provenance_quality plus eta times recency minus lambda times contradiction_penalty.
2. WHERE no reranking weights are configured, THE Reranker SHALL use the default weights alpha 0.40, beta 0.25, gamma 0.15, delta 0.10, eta 0.05, and lambda 0.30.
3. WHEN an assertion is contradicted, THE Reranker SHALL apply the contradiction_penalty so that the contradicted assertion scores lower than an equivalent non-contradicted assertion.

### Requirement 18: Evidence Packaging (R4)

**User Story:** As an agent integrator, I want retrieval to return an evidence package, so that the calling agent receives provenance and conflicts rather than just a raw answer.

#### Acceptance Criteria

1. THE Evidence_Packager SHALL return an evidence package containing answer, confidence, supporting_assertions, supporting_sources, conflicts, missing_information, and retrieved_items.
2. THE Evidence_Packager SHALL include the IDs and confidence of supporting assertions in the evidence package.
3. THE Evidence_Packager SHALL include provenance for supporting sources in the evidence package.
4. WHERE unresolved conflicts are relevant to the query, THE Evidence_Packager SHALL include those conflicts in the evidence package.
5. THE Retrieval_Pipeline SHALL return an evidence package and SHALL NOT be required to produce a final natural-language answer.

### Requirement 19: API Service

**User Story:** As an agent integrator, I want HTTP endpoints for writing, querying, validating, and inspecting memory, so that the memory layer can be used over a standard interface.

#### Acceptance Criteria

1. THE API_Service SHALL be implemented using FastAPI.
2. WHEN a POST request is made to /memory/write with text, source_ref, write_intent, and extractor_version, THE API_Service SHALL return accepted, superseded, quarantined, and rejected lists plus a summary containing num_candidates, num_accepted, num_quarantined, num_rejected, and num_superseded.
3. WHEN a POST request is made to /memory/query with query, top_k, and include_conflicts, THE API_Service SHALL return query_type, confidence, supporting_assertions, supporting_sources, conflicts, missing_information, and retrieved_items.
4. WHEN a POST request is made to /memory/validate with a candidate, THE API_Service SHALL return valid, decision, reason, and severity without committing the candidate.
5. WHEN a GET request is made to /memory/entity/{id}, THE API_Service SHALL return the entity and its associated assertions.
6. WHEN a GET request is made to /memory/conflicts, THE API_Service SHALL return the unresolved conflicts and quarantined candidates.

### Requirement 20: Agent Integration Loop

**User Story:** As a researcher, I want a lightweight agent loop demonstrating memory integration, so that the memory layer can be exercised end to end without requiring LangGraph.

#### Acceptance Criteria

1. THE Agent_Loop SHALL implement nodes for receive input, retrieve memory, generate answer or action, extract new memory, validate, and commit or quarantine.
2. WHEN the Agent_Loop processes a turn, THE Agent_Loop SHALL call memory query with the user input.
3. WHEN the Agent_Loop produces new memory, THE Agent_Loop SHALL call memory write with the turn content and a source_ref.
4. THE Agent_Loop SHALL operate without requiring LangGraph as a dependency.

### Requirement 21: Agent Answer Policy

**User Story:** As a researcher, I want an explicit answer policy applied to agent outputs, so that conflicts and provenance are surfaced and evidence gaps are stated.

#### Acceptance Criteria

1. WHERE accepted high-confidence typed assertions are available, THE Answer_Policy SHALL prefer them over raw text (P1).
2. WHEN conflicts exist for an answered query, THE Answer_Policy SHALL surface the conflicts explicitly (P2).
3. THE Answer_Policy SHALL present conflicting claims separately and SHALL NOT collapse them into a single claim (P3).
4. WHERE a request sets high_stakes to true or the output is decision-support, THE Answer_Policy SHALL include provenance (P4).
5. IF required evidence is missing, THEN THE Answer_Policy SHALL state what evidence is missing (P5).

### Requirement 22: Baselines

**User Story:** As a researcher, I want multiple comparable baseline configurations, so that the contribution of each system component can be measured.

#### Acceptance Criteria

1. THE Baseline_Runner SHALL provide baseline B0 using vector retrieval only, without ontology, graph, contradiction gate, or quarantine.
2. THE Baseline_Runner SHALL provide baseline B1 using graph assertions and symbolic retrieval only, without vectors.
3. THE Baseline_Runner SHALL provide baseline B2 using graph assertions and semantic retrieval, without contradiction detection, without quarantine, and without the provenance-aware answer policy.
4. THE Baseline_Runner SHALL provide baseline B3 using full hybrid memory with contradiction detection, quarantine, and provenance.
5. THE Baseline_Runner SHALL provide baseline B4 consisting of baseline B3 plus the Answer_Policy.
6. THE Baseline_Runner SHALL execute baselines B0 through B3 against the JSONL benchmark dataset.

### Requirement 23: Evaluation Benchmark

**User Story:** As a researcher, I want a reproducible benchmark dataset across all reasoning categories, so that baselines can be evaluated consistently.

#### Acceptance Criteria

1. THE Benchmark_Generator SHALL produce a JSONL dataset where each example contains id, category, sessions (each with session_id and input), and questions (each with query, expected_answer_contains, and expected_conflict).
2. THE Benchmark_Generator SHALL produce examples in six categories: longitudinal factual QA, multi-step planning requiring entity consistency, contradiction-heavy update stream, temporal reasoning over ordered events, entity resolution ambiguity, and evidence-required decisions.
3. THE Benchmark_Generator SHALL produce at least 25 examples per category and at least 150 examples in total.
4. THE Benchmark_Generator SHALL include hand-authored anchor examples covering Task T1 conflict, the Joseph and Pharaoh case, project owner conflict, inactive assignee, final decision without evidence, and temporal cycle.
5. WHEN the Benchmark_Generator runs with the configured fixed random seed, THE Benchmark_Generator SHALL produce an identical dataset across runs.
6. THE Benchmark_Generator SHALL include in each question an optional expected_supporting_ids field identifying the expected supporting memory facts for retrieval evaluation, in addition to expected_answer_contains.

### Requirement 24: Metrics Reporting

**User Story:** As a researcher, I want all evaluation metrics computed and reported, so that system performance and the research claim can be assessed against baselines.

#### Acceptance Criteria

1. THE Metrics_Reporter SHALL compute retrieval metrics hit@1, hit@3, hit@5, supporting_evidence_precision, and supporting_evidence_recall.
2. THE Metrics_Reporter SHALL compute answer metrics factual_precision, factual_recall, contradiction_rate_per_100_responses, conflict_surfacing_rate, and memory_induced_hallucination_rate.
3. THE Metrics_Reporter SHALL compute write-time metrics invalid_write_detection_rate, false_quarantine_rate, contradiction_detection_precision, contradiction_detection_recall, and entity_resolution_accuracy.
4. THE Metrics_Reporter SHALL compute agent metrics long_horizon_plan_success, correction_turns_after_injected_error, latency_overhead, and token_overhead.
5. THE Metrics_Reporter SHALL report metric comparisons of the proposed system against baseline B0.

### Requirement 25: Research Logging

**User Story:** As a researcher, I want detailed structured logs for writes, queries, and benchmark runs, so that experiments are traceable and reproducible.

#### Acceptance Criteria

1. WHEN a write operation completes, THE Research_Logger SHALL record input_id, source_ref, number_of_candidates, number_accepted, number_quarantined, number_rejected, validation_failures, contradiction_failures, latency_ms, and token_count_if_llm_used.
2. WHEN a query operation completes, THE Research_Logger SHALL record query_id, query_type, symbolic_results_count, semantic_results_count, top_k_ids, conflicts_returned, latency_ms, and token_count_if_llm_used.
3. WHEN a benchmark example is evaluated, THE Research_Logger SHALL record baseline_name, answer, retrieved_ids, conflicts, expected_conflict, score, and latency_ms.

### Requirement 26: Required Test Coverage

**User Story:** As a researcher, I want the specified test suite implemented, so that correctness of validation, constraints, contradiction handling, and retrieval is demonstrated.

#### Acceptance Criteria

1. THE OCM_System SHALL include schema tests that reject confidence greater than 1, reject confidence less than 0, reject an invalid predicate, reject an invalid source or target type, and accept a valid assertion.
2. THE OCM_System SHALL include temporal tests that reject timestamp_end earlier than timestamp_start, reject a PRECEDES cycle, and accept a valid PRECEDES chain.
3. THE OCM_System SHALL include task tests that reject a done task without a completion event, accept a done task with a completion event, reject assignment to an inactive person, and accept assignment to an active person.
4. THE OCM_System SHALL include contradiction tests that reject or quarantine a high-confidence contradiction, allow a low-confidence contradiction with a warning, allow a correction to supersede, and retrieve unresolved conflicts.
5. THE OCM_System SHALL include retrieval tests that confirm symbolic retrieval returns the correct owner, semantic retrieval returns a relevant claim, a conflict query retrieves a quarantined contradiction, the reranker penalizes a contradicted assertion, and an evidence package includes sources.
6. THE OCM_System SHALL include at least 10 passing constraint tests.

### Requirement 27: Configuration and Build Phasing

**User Story:** As a memory layer developer, I want the system organized by implementation phases and configurable defaults, so that the memory layer is built before the agent demo and can run offline by default.

#### Acceptance Criteria

1. THE OCM_System SHALL reside under an ocm package within the workspace.
2. WHERE no configuration is provided, THE OCM_System SHALL operate fully offline using the Mock_Extractor and the local embedding model.
3. THE OCM_System SHALL allow the extractor and embedding implementations to be selected via configuration.
4. THE OCM_System SHALL be built in phase order such that the memory layer (schemas, storage, graph store, write API, validators, contradiction checker, quarantine, extraction, resolution, retrieval) precedes the agent demonstration.
5. WHERE deterministic_test_mode is enabled, THE OCM_System SHALL generate deterministic IDs derived from entity type, normalized name, source_ref, and/or seeded counters, so that repeated runs produce identical IDs.

### Requirement 28: Definition of Done

**User Story:** As a researcher, I want clear completion criteria, so that the prototype can be confirmed as functionally complete.

#### Acceptance Criteria

1. WHEN the API_Service is started, THE API_Service SHALL run successfully.
2. WHEN a valid write request is processed, THE /memory/write endpoint SHALL return accepted and quarantined results.
3. THE OCM_System SHALL enforce Pydantic schema validation on memory items.
4. THE OCM_System SHALL pass at least 10 constraint tests.
5. WHEN a high-confidence conflict is encountered, THE contradiction gate SHALL prevent silent acceptance.
6. THE Quarantine_Store SHALL persist quarantined candidates across restarts.
7. WHEN a query request is processed, THE /memory/query endpoint SHALL return both symbolic and semantic results.
8. THE Evidence_Package SHALL include IDs, confidence, provenance, and conflicts.
9. THE Baseline_Runner SHALL run baselines B0 through B3 on the JSONL benchmark.
10. THE Metrics_Reporter SHALL report the specified metrics from an evaluation script.
11. THE OCM_System SHALL include a README documenting setup, API, and experiment commands.

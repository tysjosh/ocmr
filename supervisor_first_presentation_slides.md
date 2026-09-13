# Ontology-Constrained Memory for Agent Reasoning

First supervisor presentation

---

## 1. Project in One Sentence

Long-horizon AI agents need persistent memory, but current memory stacks can store stale or contradictory facts; this project proposes **write-time governance** so only admissible memory updates become durable state.

Key claim:

OCMR improves **durable memory-state integrity**, not necessarily raw answer accuracy in every setting.

---

## 2. Important Terms: Core Concepts

- **Long-horizon agent**: an AI system that acts across many interactions, sessions, tools, and decisions instead of answering one isolated prompt.
- **Persistent memory**: external memory that survives beyond the current context window and can be reused later.
- **Durable state**: memory that the agent treats as currently true or operationally valid.
- **Memory write**: the act of adding, updating, or rejecting a fact in long-term memory.
- **Write-time governance**: checking a memory update before it becomes durable state.
- **Ontology**: a typed structure that says what entities, relations, and constraints are allowed.
- **Assertion**: one candidate memory fact, usually represented as subject-predicate-object plus metadata.
- **Provenance**: evidence about where a memory came from, such as source text, timestamp, or tool output.

---

## 3. Important Terms: Decisions and Metrics

- **Accepted**: a memory assertion enters active memory.
- **Superseded**: a newer valid assertion replaces an older stale one.
- **Quarantined**: a risky or unresolved assertion is stored for audit but excluded from normal answers.
- **Rejected**: an invalid or unsupported assertion is not stored.
- **Contradiction gate**: the rule that blocks or quarantines conflicting writes before they contaminate active memory.
- **Durable-write constraint violation**: an invalid active memory state, such as two current values for a single-valued relation.
- **Task success**: whether the answer or plan matches the expected result.
- **Ablation**: a controlled experiment that removes one mechanism to test whether it matters.

---

## 4. Why This Matters

Persistent agents reuse memory across sessions, tools, and plans.

If memory admits bad state, later reasoning can remain fluent while depending on invalid facts:

- two active owners for one task
- stale user preferences
- expired access states treated as current
- contradicted facts retrieved as if both are true

The core risk is not only hallucination at answer time; it is **contaminated durable state**.

---

## 5. Problem Statement

Most agent memory systems optimize:

- what to retrieve
- how much context to keep
- how to summarize past interactions

They often under-specify:

- what is allowed to enter long-term memory
- how corrections supersede stale facts
- what happens to unresolved contradictions
- how provenance is preserved for future audit

Research question:

Can write-time constraints reduce durable memory violations while preserving useful recall under trusted update semantics?

---

## 6. Core Idea

Treat memory writes as governed state transitions.

Each candidate memory assertion is routed to one of four states:

- **accepted**: enters active memory
- **superseded**: replaces stale accepted state
- **quarantined**: preserved for audit but excluded from normal answers
- **rejected**: invalid or unsupported write

Only accepted and superseding writes affect ordinary retrieval.

---

## 7. OCMR Pipeline

Input interaction  
→ candidate extraction  
→ normalization and entity resolution  
→ schema/type checks  
→ temporal and cardinality checks  
→ contradiction detection  
→ accept / supersede / quarantine / reject  
→ hybrid symbolic-semantic retrieval

The key difference from RAG-style repair:

OCMR prevents invalid durable state from forming; read-time filtering only hides problems after the store is already contaminated.

---

## 8. Memory Representation

Memory is represented as a typed graph with first-class assertion nodes.

Each assertion stores:

- subject, predicate, object
- assertion type
- confidence
- evidence count
- status
- provenance
- validity interval
- extractor/schema version

This supports both retrieval and auditability.

---

## 9. Example: Why Write-Time Governance Matters

Suppose an agent stores a user's travel preference over time.

Session 1:

- user says: "Book my trips from **Boston**."
- memory write: `user home_airport Boston`
- decision: **accepted**

Session 4:

- user says: "I moved. Use **Chicago O'Hare** from now on."
- memory write: `user home_airport Chicago O'Hare`

What can go wrong:

- a plain vector memory may retrieve both Boston and Chicago
- an LLM-managed memory may insert Chicago but fail to retire Boston
- read-time filtering may hide the conflict for one answer, but the store is still contaminated

OCMR behavior:

- detects `home_airport` as a single-valued relation
- supersedes Boston with Chicago
- keeps provenance for both assertions
- exposes only Chicago as active durable state

---

## 10. Example: Same Pattern in the Experiments

The same failure pattern appears in the evaluation.

MultiWOZ:

- dialogue state slots behave like single-valued facts
- if the restaurant area changes from "north" to "centre", the old value should not remain active
- OCMR tests whether stale slots are superseded rather than accumulated

LongMemEval:

- user facts change across sessions
- examples include preferences, locations, jobs, relationships, and plans
- raw extraction can produce noisy or partial memory writes

Why the example matters:

The project is not only asking whether the final answer is fluent. It asks whether the memory store remains structurally valid after many updates.

---

## 11. Related Work: Retrieval and Memory

- **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
  Important concept: retrieval can ground generation in external evidence, but it does not decide whether memory should have been stored in the first place.

- **CRAG: Corrective Retrieval Augmented Generation** and **Self-RAG**  
  Important concept: systems can critique or correct retrieved evidence at answer time; OCMR asks whether correction should happen earlier, at memory-write time.

- **GraphRAG: From Local to Global** and **HippoRAG**  
  Important concept: graph structure improves retrieval over connected facts; OCMR also uses graph structure, but for admissibility and contradiction control.

- **MemGPT: Towards LLMs as Operating Systems** and **MemoryBank**  
  Important concept: long-term conversational memory is useful, but LLM-managed or summary-based memory can still admit stale or conflicting state.

---

## 12. Related Work: Governance and Evaluation

- **SHACL: Shapes Constraint Language**  
  Important concept: typed constraints can define what counts as a valid graph state.

- **PROV-DM: The PROV Data Model**  
  Important concept: provenance makes memory auditable by tracking source, evidence, and derivation.

- **A Truth Maintenance System** and **An Assumption-Based TMS**  
  Important concept: conflicting beliefs should be managed explicitly instead of silently merged.

- **LongMemEval**, **LoCoMo**, **AMA-Bench**, and **UltraHorizon**  
  Important concept: long-horizon agents fail through stale facts, entity confusion, and accumulated memory errors.

OCMR's distinction:

It combines retrieval, graph memory, provenance, and consistency ideas into a **write-time governance layer** for durable agent memory.

---

## 13. Baselines

Main comparison arms:

- **B0 Text**: vector retrieval over serialized memory
- **B1 Ontology**: symbolic graph only, schema checks only
- **B2 Hybrid**: graph + semantic retrieval without governance
- **B_RAG**: vanilla vector text-evidence RAG
- **B_RTCF**: read-time contradiction filtering, no write gate
- **B_sup**: latest-value supersession only
- **B_memgpt**: MemGPT-style LLM-managed insert/update/skip
- **B3 OCMR**: full schema + gate + quarantine + provenance

Planned addition to discuss:

- **MemoryBank-style comparator** for raw LongMemEval, if time permits.

---

## 14. Evaluation Settings

Synthetic stress benchmark:

- factual recall
- contradiction-heavy updates
- temporal reasoning
- planning under evolving task state

Real-data validations:

- **MultiWOZ 2.2** with oracle slot extraction
- **LongMemEval cached-oracle** knowledge updates
- **LongMemEval raw end-to-end** extraction from multi-session text

Metrics:

- contradiction rate
- durable-write constraint violations
- task success / answer-token recall
- abstention accuracy
- write outcomes

---

## 15. Datasets

- **Synthetic Qwen stress benchmark**  
  Controlled trajectories designed around persistent-memory failure modes: factual recall, conflicting updates, temporal inconsistency, entity ambiguity, and planning with changing task state. This dataset tests whether governance prevents invalid durable memory under known pressure cases.

- **MultiWOZ 2.2**  
  Task-oriented dialogue dataset with structured dialogue-state slots. We use oracle slot extraction so the experiment isolates single-valued update governance: when a user changes a slot, the new value should supersede the stale one.

- **LongMemEval**  
  Long-term conversational memory benchmark with multi-session user facts and knowledge updates. We use it in two ways: cached-oracle fact trajectories to isolate memory-state transitions, and raw end-to-end extraction to test behavior under noisy real text.

Why these three:

Together they separate controlled contradiction pressure, trusted structured updates, and realistic long-session memory extraction.

---

## 16. Main Synthetic Result

OCMR reduces durable state violations from **50.72 to 0.00** and contradiction leakage from **14.49 to 1.26**.

Trade-off:

B0 has higher raw task success, while OCMR is more conservative under ambiguous conflicts.

Interpretation:

This is not a Pareto-dominance claim. It is a state-integrity result for settings where durable contamination is costlier than missing or quarantining uncertain information.

---

## 17. Real-Data Result: MultiWOZ

MultiWOZ isolates trusted single-valued updates.

Because changed slots are authoritative, OCMR supersedes stale values rather than quarantining them.

Result:

- B0/B2 slot violations: **7.14**
- OCMR slot violations: **0.012**
- task success preserved at **99.96**

Takeaway:

When update semantics are trusted, governance improves integrity without hurting recall.

---

## 18. Real-Data Result: LongMemEval

Cached-oracle LongMemEval:

- B0/B2 durable violations: **106.38**
- B_sup and OCMR violations: **0.00**
- task success: **100.0** for all arms

Raw end-to-end LongMemEval:

- B0/B2: **54.17** task success, **801.39** violations
- B_memgpt: **52.78** task success, **4.17** violations
- OCMR: **54.17** task success, **0.00** violations

Takeaway:

The remaining end-to-end recall ceiling is mostly extraction/linking quality, not the governance gate.

---

## 19. What the Ablations Show

The contradiction gate is the decisive measured lever.

Removing it:

- raises contradiction rate from **1.26 to 14.49**
- raises durable violations from **0.00 to 50.72**
- increases raw task success

Important caveat:

This workload is contradiction-dominated, so only the gate visibly moves the metrics. That does **not** mean the other checks are redundant — it means this workload cannot separate them. The next slide shows a targeted diagnostic that isolates each component.

---

## 20. Do the Other Checks Matter? (Targeted Diagnostic)

Concern this addresses:

If only the contradiction gate (C7) moves the main ablation, are the schema, temporal, evidence, and status checks doing anything?

Method:

- build a workload of writes that are **not** single-valued contradictions (so C7 has nothing to fire on)
- each write violates exactly one other check: schema/domain-range (C9), temporal sanity (C2), decision-evidence floor (C8), or task status (C4/C10)
- feed the **same inputs** to four governance configurations and count invalid state left in durable memory, by type

Result (invalid active-state assertions left; identical inputs):

| Arm (schema / constraints / gate) | schema | evidence | temporal | status | Total |
|---|---|---|---|---|---|
| Ungoverned (off/off/off) | 4 | 2 | 4 | 2 | 12 |
| Gate-only (off/off/**on**) | 4 | 2 | 4 | 2 | 12 |
| Schema+Constraints (**on/on**/off) | 0 | 0 | 0 | 0 | **0** |
| Full OCMR (**on/on/on**) | 0 | 0 | 0 | 0 | **0** |

Takeaway:

Fed the same inputs, the gate alone leaves all four violation types active; enabling the schema/constraint checks removes every one. So each check carries independent governance value — C7 is dominant *on the original workload*, not a substitute for the others.

Honesty scope:

This is a constructed diagnostic, not a natural-frequency claim. It required one small additive guard (gating the status/decision checks by the existing constraint-validation toggle) that is behavior-preserving in the default configuration; all prior results are unchanged and the full test suite stays green.

---

## 21. Honest Limitations

Current limits:

- conservative quarantine can reduce recall
- results depend on extraction quality
- component independence is shown via a *constructed* diagnostic (Slide 20), not yet under a natural real-data distribution
- synthetic benchmark is controlled rather than naturally distributed
- B_memgpt is single-seed and scoped to raw LongMemEval
- MemoryBank-style comparator still needs to be added if supervisor expects memory-agent baselines beyond MemGPT

Positioning:

The contribution is a governance layer for durable memory, not a universal answer-quality improvement.

---

## 22. What I Need Feedback On

Feedback requests:

- Is the core claim scoped correctly as durable state integrity?
- Are the baselines sufficient for a first submission?
- Should MemoryBank-style memory be added before submission?
- Is the conservative recall trade-off acceptable, or should the policy expose a tunable operating point?
- Is the constructed component-isolating diagnostic (Slide 20) convincing, or should I also seek a natural real-data workload that exercises the non-gate checks?
- Which venue framing is strongest: agent memory, neuro-symbolic systems, or reliability/safety?

---

## 23. Next Steps

Near-term:

- add or justify MemoryBank-style baseline
- tighten Table I and baseline prose
- run final compile and visual table check on Overleaf
- prepare a short artifact/readme for result reproducibility

Submission-focused:

- fold the component-isolating diagnostic (Slide 20) into the paper's ablation section
- sharpen contribution wording
- reduce overclaiming
- foreground state-integrity metrics
- add a limitations paragraph that anticipates reviewer concerns

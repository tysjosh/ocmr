# Evaluation Methodology and Threshold Selection

Reviewer-facing supplement covering:

- **M4** — selection of the contradiction-gate threshold τ.
- **W2 / W5** — evaluation and benchmark transparency: categories, gold-label
  construction, and scoring, documented directly from the implementation.

All numbers below are the harness's own measured values. File/symbol references
point at the source of truth so every claim is auditable.

---

## M4 — Selecting the contradiction threshold τ

### Mechanism

The contradiction gate (Algorithm 1, implemented in
`ocm/validation/constraints.py::c7_contradiction_gate`, delegating detection to
`ocm/validation/contradiction_checker.py`) blocks a write only when a detected
conflict is **hard**: both the incoming candidate and the conflicting *accepted*
assertion carry confidence strictly above τ
(`settings.contradiction_high_confidence`, default `0.8`). Concretely, the
checker computes

```
candidate_high  = candidate.confidence > τ
counterpart_high = counterpart_confidence > τ
both_high       = candidate_high and counterpart_high
```

and grades the conflict `hard` (→ quarantine, or supersede for a `correction`)
when `both_high`, else `soft` (→ accept as a warning). τ therefore trades two
failure modes against each other:

- **τ too low** → benign updates are quarantined (false quarantines).
- **τ too high** → genuine contradictions are silently accepted, accumulating
  mutually contradictory accepted state (constraint violations).

### What the sweep optimizes

`ocm/evaluation/experiment.py::threshold_sweep` runs the full system (B3) over a
fixed benchmark for τ ∈ {0.6, 0.7, 0.8, 0.9, 0.95} and reports a selection
objective

```
J(τ) = ContrRate + λ_q · FalseQuar + λ_c · ECE      (λ_q = 0.5, λ_c = 10)
```

By this objective **J is minimized at τ = 0.95** (J ≈ 16.2 vs ≈ 39.3 at τ ≤ 0.9),
driven almost entirely by the false-quarantine rate dropping from 72.9 to 24.9:

| τ    | ContrRate | FalseQuar | ECE   | Brier | J(τ)   |
|------|-----------|-----------|-------|-------|--------|
| 0.6  | 1.45      | 72.88     | 0.137 | 0.199 | 39.263 |
| 0.7  | 1.45      | 72.88     | 0.137 | 0.199 | 39.263 |
| 0.8  | 1.45      | 72.88     | 0.137 | 0.199 | 39.263 |
| 0.9  | 1.45      | 71.19     | 0.220 | 0.259 | 39.247 |
| 0.95 | 1.45      | 24.86     | 0.229 | 0.253 | **16.172** |

### Why we do **not** adopt τ = 0.95

The sweep's selection is informative precisely because it is misleading: **J
omits `constraint_violations`**, the one metric the gate exists to protect, so it
cannot see the cost of raising τ. The decisive five-seed A/B (decisive-metrics
block, `run_full_suite(tau=...)`) makes that cost explicit:

| Config   | TaskSuccess ↑        | Contradiction ↓     | ConstraintViol ↓     | B3 write outcomes (accept / quar) |
|----------|----------------------|---------------------|----------------------|-----------------------------------|
| τ = 0.8  | 60.0 [57.2, 62.8]    | 1.3 [0.9, 1.6]      | **0.0 [0.0, 0.0]**   | 654 / 1198                        |
| τ = 0.95 | 73.0 [72.1, 74.0]    | 1.3 [0.9, 1.6]      | **50.7 [47.9, 53.5]**| 1296 / 556 (= B0 baseline)        |

At τ = 0.95 the contradiction rate is unchanged and task success recovers, but
constraint violations jump to the **ungoverned-baseline value** and B3's write
outcomes collapse onto B0's. In other words, at τ = 0.95 the gate is effectively
*off*: the benchmark asserts conflicting relations at ≈ 0.90 confidence and
status flips at 0.95, so once τ reaches 0.95 neither side clears the strict
`> τ` bar, every conflict is regraded `soft`, and the durable store accumulates
mutually contradictory accepted state. The task-success "gain" at τ = 0.95 is
therefore not free accuracy — it is the system answering from ungoverned,
contradictory memory.

The false-quarantine reduction that drives J toward τ = 0.95 is also partly
illusory: as shown in the false-quarantine reconciliation below (W6 / M7,
governed-write replay, `ocm/evaluation/replay_governed_writes.py`), the
quarantines counted at τ ≤ 0.9 are dominated by *correct* single-valued
cardinality enforcement — on the real-LLM run 835 of 1,198 (69.7%) quarantines
fall in non-conflict-labeled examples, every one because the task already has a
different accepted assignee. Raising τ "fixes" them only by ceasing to enforce
the constraint at all.

### Decision

We treat the τ-sweep as a **sensitivity frontier, not a selector**:

- **τ = 0.95** is its permissive extreme — maximal admission, no durable safety
  guarantee.
- **τ = 0.8** is the safety-preserving regime, and the only operating point that
  holds *both* safety metrics (contradiction rate and constraint violations) at
  their floor.

We adopt **τ = 0.8**, accepting a measured task-success cost (B3 60.0 vs the
text-only B0 77.2, ≈ 17 points on the full real-LLM run). The per-scenario
breakdown shows this cost does **not** fall in the contradiction-heavy category
(where every arm floors near 3.7) but in the planning, longitudinal, and
temporal categories — the *mechanism's* signature: B3 quarantines conflicting
writes at ingest (1,198 vs 556 quarantined), so the losing side of a conflict is
no longer retrievable, the deliberate price of a durably consistent store.

> **Optional, recommended:** make the objective violation-aware,
> `J'(τ) = ContrRate + λ_q·FalseQuar + λ_c·ECE + λ_v·ConstraintViol`. With any
> non-trivial `λ_v`, J' selects τ = 0.8, so the reported sweep selection agrees
> with the adopted configuration and no Table VI / chosen-τ discrepancy remains.

---

## W6 / M7 — Reconciling the false-quarantine rate

The threshold sweep (Table VI) reports a write-time false-quarantine rate that,
on its face, suggests the governed system rejects a large share of benign
updates. A governed-write replay (`ocm/evaluation/replay_governed_writes.py`)
shows the quarantines are overwhelmingly **correct 1:1 cardinality enforcement**
on a benchmark where many tasks receive multiple distinct assignees, not
rejection of consistent updates.

**Real-LLM run (Qwen2.5-14B + real embeddings, 5 seeds, `per_category=25`,
τ=0.8), totals summed across seeds.** Under the harness protocol — all examples
ingested into one shared governed store — write-time quarantines that fall in
examples not labeled conflict-expected are **835 of 1,198 (69.7%)**. Every such
quarantine carries the same reason: the task already has a different *accepted*
`ASSIGNED_TO` target, so the later, conflicting assignment (ingested under
`new_fact` intent) is quarantined — exactly what the single-valued cardinality
rule is for. The label "false quarantine" here is a heuristic (any quarantine in
a non-conflict-labeled example); it does **not** mean the governance was wrong.

**Where the conflicting assignments come from.** Re-running with each example
isolated in its own store (`isolate_per_example=True`) reduces total quarantines
from **1,198 to 434** and false quarantines from **835 to 299** — so
**cross-example identifier reuse accounts for ≈ 64% of all quarantines** (the
synthetic generator recycles a small task-id pool `T1…T16` across examples, which
entity resolution then merges). Crucially, unlike the deterministic offline mock
(where isolation drives within-example false quarantines to zero), a substantial
**within-example residual remains on the real stack: 299 of 434 (68.9%)**. This
residual is real extraction / entity-resolution behaviour — the LLM emits, or
real embeddings merge mentions into, more than one distinct assignee for a task
*within a single example* — which the gate then correctly quarantines under
`new_fact`. Partitioning the residual precisely into extractor over-generation
vs. alias over-merge requires a per-example extraction audit (cf.
`validate_anchor_extractions.py`) and is left to future work.

**What this means for the reported rate.** The write-time false-quarantine rate
is an **upper bound** that conflates three things, none of which is wrongful
rejection of a consistent update: (i) cross-example identifier reuse (≈ 64% of
volume, a benchmark-construction artifact), (ii) real-extractor multiplicity
within an example, and (iii) genuine reassignments that, under `new_fact` intent,
are quarantined rather than superseded. A deployment that issued reassignments
under `correction`/`update` intent would supersede instead of quarantine; the
benchmark uses `new_fact` uniformly, which maximises quarantines by construction.

> **Honest correction.** An earlier draft of this section reported isolation
> driving the within-example false-quarantine rate to *0.0* — that holds only for
> the deterministic offline mock, **not** for the real-LLM run (68.9%). The
> offline figure must not be cited for the LLM configuration.

**Reproduction.**

```bash
# shared-store (protocol) vs isolated, single seed
python -m ocm.evaluation.replay_governed_writes --per-category 25 --seed 1337
python -m ocm.evaluation.replay_governed_writes --per-category 25 --seed 1337 --isolate-per-example
```

```python
# multi-seed totals (shared vs isolated), real extractor/embeddings
from ocm.evaluation.replay_governed_writes import replay_governed_writes
replay_governed_writes(per_category=25, seeds=(1337, 7, 42, 99, 2024),
                       extractor=qwen_extractor, embeddings=real_embeddings,
                       isolate_per_example=False)  # 835/1198 false-quar
replay_governed_writes(per_category=25, seeds=(1337, 7, 42, 99, 2024),
                       extractor=qwen_extractor, embeddings=real_embeddings,
                       isolate_per_example=True)   # 299/434 false-quar
```


---

## W2 / W5 — Evaluation and benchmark transparency

Source of truth: `ocm/evaluation/benchmark.py` (dataset),
`ocm/evaluation/runner.py` (scoring), `ocm/evaluation/experiment.py` (decisive
metrics), `ocm/evaluation/stress.py` (entity resolution).

### Dataset construction

The benchmark is **fully synthetic, seeded, and offline** — a single
`random.Random(seed)` (default seed `1337`) drives every sample and categories
are iterated in a fixed order, so the dataset is byte-identical across runs for a
fixed seed (`BenchmarkGenerator.generate`). No held-out human data is used; this
is a controlled probe of governance behavior, not a natural-language benchmark.

**Six reasoning categories** (`benchmark.CATEGORIES`), `per_category` examples
each (default 25 → 150 generated) plus six hand-authored anchors (156 total at
default scale):

| Category (impl.)                          | Paper class        | Probes |
|-------------------------------------------|--------------------|--------|
| `longitudinal_factual_qa`                 | Recall             | facts established over turns, recalled later |
| `multi_step_planning_entity_consistency`  | Planning           | owner/assignee consistency across sessions |
| `contradiction_heavy_update_stream`       | Contradiction-heavy| conflicting status updates to the same task |
| `temporal_reasoning_ordered_events`       | Temporal           | ordered/`PRECEDES` event reasoning |
| `entity_resolution_ambiguity`             | (Table VIII)       | same entity under alias surface forms |
| `evidence_required_decisions`             | (governance / C8)  | decisions that require supporting evidence |

The paper reports four headline scenario classes; the mapping to the six
implementation categories is `benchmark.PAPER_SCENARIO_CLASSES`. Entity
resolution is reported separately (Table VIII) and evidence-required decisions
via the C8/governance results. **Scale is configuration, not a hard-coded
result:** the paper's 120 trajectories/class is reproduced with
`per_category=120` over the four headline categories; the default `25` is a
faster offline reference.

Session `input` strings are templated so the deterministic `MockExtractor`
recognizes them, keeping the pipeline offline and reproducible end to end:

| Phrase pattern                    | Extracted |
|-----------------------------------|-----------|
| `"X owns Project Y"`              | `OWNS` relation |
| `"X is assigned to Task Y"`       | `ASSIGNED_TO` relation |
| `"X completed Task Y"`            | status `done` + completion `Event` |
| `"Task Y is not started"`         | status assertion |
| `"We decided to ..."`             | `Decision` |
| `http(s)://...`                   | `Document` |

### Gold labels

Each `Question` (`benchmark.Question`) carries:

- **`expected_answer_contains`** — a list of canonical tokens that must appear in
  the answer/evidence (the answer gold label). For status, the canonical token is
  the enum value (e.g. the phrase "not started" maps to the token `todo`; see
  `benchmark._STATUS_PHRASES`).
- **`expected_conflict`** — boolean gold flag for whether a known contradiction
  should be surfaced for this question.
- **`expected_supporting_ids`** *(optional)* — expected supporting memory ids for
  retrieval scoring (Req 23.6), populated on the anchors.

For the entity-resolution stress scenarios, `BenchmarkExample.gold_entity_groups`
gives `{canonical_id: [mention surface form, ...]}`, the gold clustering used to
score F1 and false-merge.

The **six anchors** (`BenchmarkGenerator._anchors`) are hand-authored with known
correct outcomes and target specific constraints: Task-T1 done-vs-not-started
conflict (C7), Joseph/Pharaoh longitudinal recall, project owner conflict
(single-owner), inactive assignee (C5), final decision without evidence (C8), and
a temporal `PRECEDES` cycle (C3). Their extraction preconditions are separately
auditable via `ocm/evaluation/validate_anchor_extractions.py`.

### Scoring (operational definitions)

Per `(method, example, question)` the runner (`runner._run_question`) produces a
record; the decisive metrics (`experiment.decisive_metrics`) are computed per
`(method, seed)`:

- **Task success ↑** — `100 × mean(score)`, where `score` is the fraction of
  `expected_answer_contains` tokens found in a case-insensitive haystack of the
  rendered answer plus retrieved-item text/ids (`runner._answer_score`). It is
  **answering only**, deliberately decoupled from conflict-surfacing so the three
  decisive metrics are independent.
- **Contradiction rate ↓** — per 100 responses, the rate at which a *known*
  contradiction (`expected_conflict=True`) was **not** surfaced
  (`conflict_surfaced=False`) — a governance miss that leaks a contradiction into
  the response.
- **Constraint violations ↓** — durable-write constraint-violation rate per 100
  responses. The primary measure is taken from the arm's *durable store*
  (`experiment.durable_constraint_violations`): a single-valued relation (1:1 or
  m:1, e.g. `HAS_STATUS`, `ASSIGNED_TO`) left with ≥ 2 distinct accepted objects
  for one subject. Because every arm shares the governed write path, the durable
  store is identical across arms; the governance difference is whether an arm
  *surfaces* constraint-violating state unflagged at answer time
  (`runner`'s `surfaced_violation`).

**Calibration** (`stats.expected_calibration_error`, `stats.brier_score`) uses
the package confidence vs whether the answer was fully correct (all expected
tokens present). **False-quarantine rate** (`experiment._false_quarantine_rate`)
is, over non-conflict questions, the share whose writes were quarantined.

**Entity resolution** (`stress.evaluate_entity_resolution`), over gold mention
pairs within each example:

```
precision = tp / (tp + fp);  recall = tp / (tp + fn)
F1 = 2·precision·recall / (precision + recall)
false_merge_rate = false_merges / distinct_pairs
```

where a "merge" means two mentions resolved to the same node; `false_merges`
counts distinct-entity pairs (different gold groups, same example) wrongly
merged.

### Comparison baselines (extended)

Beyond the canonical B0–B4 toggle matrix, two extended baselines isolate
alternative designs (opt-in via `baselines=(..., "Brag", "Brtcf")`;
`ocm/evaluation/baselines.py`). All baselines share the governed write pipeline;
these differ in retrieval composition and write-time gating.

- **`Brag` — RAG-only.** Vectors-only similarity retrieval with the answer read
  **only from retrieved text** (the evidence package is built without the
  Graph_Store, so no graph-assisted structural answer is derived), and no
  write-time governance. This is the vanilla retrieval-augmented baseline. It is
  deliberately distinct from B0, which is also vectors-only but derives exact
  structural answers from the graph — hence Brag scores lower on the structured
  benchmark (task success 62.1 vs B0's 77.2 on the full real-LLM run), the gap
  attributable to graph-assisted answering.

- **`Brtcf` — retrieval-time contradiction filter.** Full hybrid retrieval with
  the **write-time contradiction gate off** (durable memory accumulates
  conflicting accepted state, like B2), but contradictions are detected **at
  query time**: a live scan of accepted single-valued relations
  (`MemoryStrategy._read_time_contradicted_ids`) flags every conflicting group so
  the reranker excludes them from confident support and the packager surfaces
  them. This isolates "filter at read time" against OCMR's "gate at write time".

  Headline contrast — full real-LLM run (Qwen2.5-14B + real embeddings, 5 seeds,
  `per_category=25`, τ=0.8), mean [95% CI]:

  | Method | TaskSuccess ↑ | Contradiction ↓ | ConstraintViol ↓ |
  |--------|---------------|-----------------|------------------|
  | B0 (text-only)          | 77.2 [76.5, 77.9] | 14.5 [14.5, 14.5] | 50.7 [47.9, 53.5] |
  | B2 (hybrid, no governance) | 73.0 [72.1, 74.0] | 14.5 [14.5, 14.5] | 50.7 [47.9, 53.5] |
  | `Brag` (RAG-only)       | 62.1 [61.4, 62.8] | 14.5 [14.5, 14.5] | 50.7 [47.9, 53.5] |
  | `Brtcf` (read-time filter) | 72.3 [71.6, 73.0] | **1.4 [1.4, 1.4]** | 50.7 [47.9, 53.5] |
  | B3 (write-time gate)    | 60.0 [57.2, 62.8] | 1.3 [0.9, 1.6] | **0.0 [0.0, 0.0]** |

  The read-time filter matches the write-time gate on **contradiction
  surfacing** — B3 vs `Brtcf` on contradiction rate is **not significant**
  (paired t-test, Holm-corrected p = 0.178, d = −0.73) — **but constraint
  violations stay at the ungoverned-baseline level (50.7)** because the durable
  store is never repaired. Only write-time governance (B3) drives durable
  violations to zero, and that advantage is decisive (B3 vs B0, Holm-corrected
  p < 0.0001, d = −22.6). The task-success cost of B3 (60.0 vs `Brtcf` 72.3) is
  the *mechanism's* signature: B3 quarantines conflicting writes at ingest (1,198
  vs 556 quarantined), so they are not retrievable, whereas `Brtcf` keeps the
  full store and filters only the answer. The design choice is therefore explicit
  — durable integrity (and protection of any consumer that does not re-run the
  read-time filter) in exchange for conservative, conflict-aware answering. Any
  downstream consumer that does not re-run the same read-time filter sees the
  corrupted state; this is the core argument for gating at write time.



### Statistics

Multi-seed (default 5 seeds: `1337, 7, 42, 99, 2024`). Per-metric aggregates use a
Student-t 95% CI (`stats.mean_ci`) with a nonparametric percentile-bootstrap CI
reported alongside (`stats.bootstrap_mean_ci`); per-seed raw values are exported
so degenerate (zero-variance) metrics are auditable. Significance vs the
strongest non-OCMR baseline uses a paired test chosen by a normality heuristic
(t-test, else Wilcoxon; `stats.paired_test_auto`) with Holm–Bonferroni correction
across the three decisive metrics (`stats.holm_bonferroni`).

### Reproduction

```bash
# Decisive metrics + sweep + stress + efficiency (offline mock), τ = 0.8 default
python -c "from ocm.evaluation.experiment import run_full_suite, print_report; \
print_report(run_full_suite(per_category=25))"

# τ A/B
python -c "from ocm.evaluation.experiment import run_full_suite; \
run_full_suite(per_category=25, tau=0.95)"

# Governed-write evidence + false-quarantine reconciliation
python -m ocm.evaluation.replay_governed_writes --per-category 25
python -m ocm.evaluation.replay_governed_writes --per-category 25 --isolate-per-example
```

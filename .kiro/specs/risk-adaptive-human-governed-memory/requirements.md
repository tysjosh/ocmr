# Requirements — Risk-Adaptive Human-Governed Memory (RAHGM)

Implements the methodology (§3) and experiments (§4) of
*Risk-Adaptive Human-Governed Memory for Trustworthy Analytic Agents*, which
extends Ontology-Constrained Memory for Agent Reasoning (OCMR).

RAHGM is **additive**: it reuses the existing OCMR write path (W1–W8, C1–C10, the
Contradiction_Checker, the Quarantine_Store, and `typed_violations`) and inserts a
risk-adaptive router at the single `Commit_Manager.commit` seam. No existing
OCMR module may change behavior when RAHGM is disabled.

---

## R1 — Escalation feature vector

1.1 The system SHALL derive a typed status vector `f(u) = [f_e, f_s, f_t, f_v, f_c]`
from OCMR's own constraint outcomes: entity, schema, temporal, evidentiary, and
contradiction checks.

1.2 Each component SHALL be encoded `0.0` for pass, `0.5` for unresolved, and
`1.0` for fail.

1.3 `f_e` SHALL be derived from C1 identity uniqueness and entity-resolution
status; an unresolved alias / `possible_match` SHALL encode `0.5`.

1.4 `f_s` SHALL be derived from W5 structural schema validation and C9 graph
domain/range; single-valued cardinality pressure without a hard violation SHALL
encode `0.5`.

1.5 `f_t` SHALL be derived from C2 temporal sanity, C3 acyclic PRECEDES, and C10
task status transitions; a missing or unordered timestamp SHALL encode `0.5`.

1.6 `f_v` SHALL be derived from the evidence count used by C8 and Algorithm 1
`e_min`: zero evidence SHALL encode `1.0`, evidence below the required floor
SHALL encode `0.5`, and evidence at or above the floor SHALL encode `0.0`.

1.7 `f_c` SHALL be derived from the Contradiction_Checker (W7/C7): a hard or
high-severity conflict SHALL encode `1.0`, a soft or temporal conflict `0.5`, no
conflict `0.0`.

1.8 The router SHALL additionally receive consequence `q ∈ [0,1]`, reversibility
`v ∈ [0,1]`, source authority `a ∈ [0,1]`, and `k`, the number of simultaneously
unresolved or failed checks.

1.9 `q`, `v`, and `a` SHALL be assigned by preregistered, inspectable rubrics
shipped with the evaluation suite; a corpus case MAY supply explicit values that
override the rubric.

1.10 Feature extraction SHALL be deterministic and SHALL NOT invoke a language
model.

## R2 — Transparent escalation score

2.1 The system SHALL compute
`z(u) = β₀ + βf·f(u) + β_k[k−1]₊ + β_q·q − β_v·v − β_a·a` and
`r(u) = σ(z(u))`.

2.2 Coefficients SHALL be fitted by L2-regularized logistic regression on the
training partition with label `y = 1` when autonomous execution would produce an
incorrect transition.

2.3 The admissible set `B` SHALL constrain failure, interaction, and consequence
coefficients to be nonnegative and the displayed authority and reversibility
discounts to be nonnegative, yielding an inspectable monotonic policy.

2.4 Fitting SHALL be pure-Python/NumPy projected gradient descent (no new
dependency) and SHALL be deterministic for a given seed and data ordering.

2.5 Monotonicity SHALL be verifiable: increasing any failure component, `k`, or
`q` SHALL never decrease `r`; increasing `v` or `a` SHALL never increase `r`.

## R3 — Threshold selection

3.1 Thresholds `(τ_l, τ_h)` SHALL be selected on the development partition to
maximize `F₂` subject to the missed-consequential-conflict constraint
`FN_cons / N_cons ≤ 0.02`.

3.2 The search SHALL enforce `0 ≤ τ_l < τ_h ≤ 1`.

3.3 When no feasible pair satisfies the constraint, the system SHALL return the
pair minimizing MCR and SHALL record that the constraint was infeasible.

## R4 — Deterministic three-tier routing

4.1 Routing SHALL be the deterministic policy
`π(u) = reject if g(u)=1; accept if r<τ_l ∧ m(u)=0; supersede if h(u)=1 ∧ r<τ_h; review otherwise`.

4.2 `g(u)` SHALL indicate a malformed, prohibited, or unattributed write.

4.3 `m(u)` SHALL indicate failure of a mandatory constraint. Mandatory
constraints SHALL be immutable and SHALL NOT be disableable by adaptation.

4.4 `h(u)` SHALL require authority ≥ 0.90, a resolved temporal relation, and a
recoverable incumbent assertion.

4.5 Each routing decision SHALL carry the features, the rule that fired, and the
score; the score SHALL NEVER be presented as an unexplained confidence value.

4.6 The router SHALL be inserted at the `Commit_Manager.commit` seam so both the
relation path and the status-reconcile path are governed, without editing
`write_pipeline.py`, `commit_manager.py`, or `contracts.py`.

## R5 — Review queue and review-and-release

5.1 A `review` route SHALL create a durable OCMR quarantine record and a linked
review-queue item, so no accepted memory is overwritten.

5.2 A review item SHALL present the incumbent assertion, proposed assertion,
requested operation, source, timestamp, failed/unresolved checks, and the
available actions: accept, supersede, quarantine, reject, request evidence.

5.3 Adjudication SHALL be able to **release** a held write by committing it
through the Commit_Manager and transitioning the quarantine record to
`resolved`/`dismissed`. This closes the review-and-release gap OCMR lacked.

5.4 Release SHALL be reversible: the prior assertion and its provenance SHALL be
retained on supersession.

## R6 — Explanation depth

6.1 Three explanation depths SHALL be rendered from the same review item:
`minimal` (recommended action plus failed/unresolved constraints), `evidence`
(adds supporting and conflicting evidence snippets with provenance), and `full`
(adds the memory timeline, alternative actions, reversibility, and predicted
downstream consequence).

6.2 Depth SHALL affect presentation only, never the route.

6.3 Depth assignment across a participant's cases SHALL follow a Latin-square
schedule so each participant sees every level and no case twice.

## R7 — Bounded feedback adaptation

7.1 Adaptation SHALL update only the registered coefficients and the two
thresholds, after each block of 20 adjudicated writes.

7.2 The candidate update SHALL be a projected gradient step
`β̃ = Π_B(β − η∇L_B(β))`.

7.3 Trust regions SHALL enforce `‖β̃ − β‖₂ ≤ 0.05` and `|τ̃_x − τ_x| ≤ 0.02`.

7.4 Tier definitions, mandatory constraints, feature encodings, and rejection
rules SHALL be immutable; no adaptation path may alter them.

## R8 — Canary gate and policy versioning

8.1 A candidate parameter set SHALL deploy only if
`A(θ̃,θ) = 1[ΔDVR = 0 ∧ ΔMCR ≤ 0.01 ∧ ΔRR ≤ 0.05]` on the fixed canary
partition.

8.2 A failed candidate SHALL be discarded and logged; the deployed parameters
SHALL remain unchanged.

8.3 Every accepted policy version SHALL record its parent version, training
cases, parameter delta, canary results, and rollback target.

8.4 Rollback to any recorded version SHALL be supported.

8.5 Frozen RAHGM SHALL execute the identical pipeline with updates disabled.

## R9 — Evaluation corpus and ground truth

9.1 The corpus SHALL contain 1,500 candidate writes in 50 scenarios of 30
temporally ordered writes each.

9.2 The corpus SHALL be balanced across three write classes: 500 routine
compatible updates, 500 authoritative corrections or temporal supersessions, and
500 ambiguous or consequential conflicts.

9.3 Within each class, cases SHALL vary entity-alias ambiguity, schema
cardinality, source authority, temporal consistency, evidentiary support,
contradiction type, consequence, and reversibility.

9.4 Poisoned or unsupported evidence SHALL appear in 20% of scenarios.

9.5 Scenarios (not individual writes) SHALL be partitioned into training (25),
development (10), canary (5), and test (10).

9.6 Every write SHALL carry ground truth: the correct transition, whether an
error would be consequential, and the least evidence required to resolve it.

9.7 Ground truth SHALL be objectively derivable from construction, and the
generator SHALL be deterministic for a given seed.

9.8 Inter-annotator agreement SHALL be reportable via Krippendorff's alpha; for
the synthetic corpus, two rubric-based annotator simulators with independent
noise SHALL stand in for human annotators, and this substitution SHALL be
disclosed in every report.

## R10 — Governance conditions

10.1 Five conditions SHALL be implemented over one shared transition interface
`M_{t+1} = T(M_t, u_t, d_t)`, `d ∈ {accept, supersede, review, reject}`:
C1 universal review, C2 autonomous OCMR, C3 fixed-threshold escalation
(confidence < 0.80 or high consequence), C4 frozen RAHGM, C5 adaptive RAHGM.

10.2 Conditions SHALL receive identical proposed writes, evidence, incumbent
memory states, and model outputs.

10.3 Replay SHALL preserve write order so an erroneous transition at `t` remains
available to influence all later states.

## R11 — Measures

11.1 The suite SHALL compute `MCR`, `R100`, and `DVR` per equation (10), with
`DVR` sourced from the existing `typed_violations` report.

11.2 Secondary outcomes SHALL include false escalations, false quarantines,
analytic answer accuracy after memory replay, correction quality, review-queue
precision and recall, canary-regression rate, and rollback frequency.

11.3 Human outcomes SHALL include adjudication accuracy, decision time,
workload, evidence use, recommendation-following rate, and calibration via
Brier score and ECE per equation (11).

11.4 Reviewer minutes in the replay experiment SHALL come from an explicit,
inspectable review-cost model, and every report SHALL label it as a model rather
than measured human data.

## R12 — Statistical analysis

12.1 The primary model SHALL be a random-intercept logistic regression
`logit Pr(Y=1) = α + xᵍγ + b_p + c_s` with participant and scenario random
intercepts, fitted by Laplace approximation (no statsmodels dependency).

12.2 Decision time SHALL use the same structure on `log t` with Gaussian error;
workload SHALL use a cumulative-logit link.

12.3 The three primary contrasts (C4 vs C3, C5 vs C4, explanation depth) SHALL
be reported with Holm-corrected familywise error.

12.4 Results SHALL be reported as marginal effects or odds ratios with 95%
confidence intervals alongside raw counts.

12.5 The preregistered success criteria SHALL be evaluated explicitly:
`R100 < R100(universal)`, `DVR_C5 − DVR_C2 ≤ 0.005`, and `MCR_C5 < MCR_C3`.

## R13 — Experiments

13.1 **Quarantine audit** SHALL reanalyze the 1,198 quarantined writes recorded
in `governance_examples.json`, assigning a primary cause, validity, and
review-worthiness, and SHALL populate Table 2.

13.2 **Experiment 1** SHALL apply the five conditions to held-out trajectories
and populate Table 3, plus a routing ablation (full, quarantine-only, scalar
threshold, failure-pattern only, reversibility only, without consequence,
without authority) with risk–coverage AUC for Table 4.

13.3 **Experiment 2** SHALL run the explanation-depth and reliance study and
populate Table 5. Because no human participants are available, it SHALL use a
simulated-analyst model, and every artifact SHALL be labelled as simulated.

13.4 **Experiment 3** SHALL evaluate frozen, bounded+canary, bounded-no-canary,
and unconstrained adaptation under clean, noisy, biased, and adversarial
feedback across multiple seeds, and populate Table 6.

13.5 **Experiment 4** SHALL replay each resulting memory state through a
downstream analytic agent and populate Table 7 with answer accuracy, unsupported
conclusions, and stale-value propagation.

13.6 A single command SHALL run every experiment and emit machine-readable JSON
plus rendered tables.

## R14 — Honest reporting

14.1 Any quantity that is modelled or simulated rather than measured SHALL be
labelled as such in the emitted artifacts, including a top-level scope note.

14.2 The suite SHALL NOT substitute simulated human data for the paper's
preregistered human study; it SHALL report simulated results under an explicitly
separate name.

## R15 — Non-regression and tests

15.1 The existing 582 OCMR tests SHALL continue to pass unchanged.

15.2 With RAHGM disabled, the write path SHALL be byte-identical in behavior to
current OCMR.

15.3 New tests SHALL cover feature encoding, monotonicity, threshold feasibility,
routing rules, mandatory-constraint immutability, trust-region bounds, canary
gating, adversarial-feedback rejection, corpus balance and partition disjointness,
metric definitions, and review-and-release.

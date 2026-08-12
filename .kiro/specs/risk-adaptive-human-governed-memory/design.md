# Design — Risk-Adaptive Human-Governed Memory (RAHGM)

## 1. Integration strategy

RAHGM is additive. The paper's transition interface
`M_{t+1} = T(M_t, u_t, d_t)`, `d ∈ {accept, supersede, review, reject}` maps onto
OCMR's existing `Commit_Manager.commit(candidate, vr)` seam, which every write
funnels through — both `_process_relation` (W4–W8) and `_reconcile_entity_status`.

```
                      ┌──────────────── MEMORY-GOVERNANCE PLANE ────────────────┐
CandidateAssertion ─► OCMR checks (W5, C1..C10, W7) ─► f(u) ─► z(u), r(u)
                                                              │
                                            π(u) ─┬─ accept ──► Commit_Manager._accept
                                                  ├─ supersede ► Commit_Manager._supersede
                                                  ├─ reject ───► Commit_Manager._reject
                                                  └─ review ───► Commit_Manager._quarantine
                                                                 + ReviewQueue item
                      └──────────────────────────────────────────────────────────┘
                      ┌──────────────── POLICY-GOVERNANCE PLANE ────────────────┐
FeedbackRecord ─► BoundedUpdater (projected step + trust region) ─► CanaryGate
                                                                    │
                                            PolicyRegistry (parent, delta, canary, rollback)
                      └──────────────────────────────────────────────────────────┘
```

`GovernedCommitManager` implements the same `.commit(candidate, vr, *, created_at)`
signature as `CommitManager`, so it can be dropped in via
`container.write_pipeline.commit_manager = governed`. Nothing in
`write_pipeline.py`, `commit_manager.py`, or `contracts.py` changes.

`review` is realized as an OCMR quarantine plus a `ReviewItem` keyed by the
returned `quarantine_id`. This deliberately reuses the durable
`QuarantineRecord` + `QuarantineStatus{unresolved,resolved,dismissed}` surface, so
the "review" tier inherits OCMR's integrity guarantee (accepted memory is never
overwritten by a held write) and gains the release path OCMR lacked.

## 2. Module layout

```
ocm/governance/
  features.py     RiskFeatures, CheckCode, FeatureExtractor, Rubric, WriteContext
  policy.py       PolicyParameters, EscalationPolicy, Tier, RouteGuards,
                  fit_policy (projected-gradient logistic), select_thresholds
  router.py       RoutingDecision, RiskAdaptiveRouter, GovernedCommitManager
  review_queue.py ExplanationDepth, ReviewItem, ReviewQueue, EvidenceBundle,
                  render_explanation, latin_square
  adaptation.py   FeedbackRecord, BoundedUpdater, CanaryGate, PolicyVersion,
                  PolicyRegistry, AdaptationOutcome
  conditions.py   Condition enum + CONDITIONS registry + build_governance()

ocm/evaluation/rahgm/
  corpus.py       CandidateWrite, Scenario, RahgmCorpus, generate_corpus, Partition
  annotate.py     rubric annotator simulators + krippendorff_alpha
  review_cost.py  ReviewCostModel (explicit, labelled model — not measured)
  metrics.py      ReplayMetrics, mcr/r100/dvr, queue P/R, risk_coverage_auc,
                  brier, ece, calibration_bins
  replay.py       ScenarioReplayer, run_experiment1 (Table 3)
  ablation.py     ROUTING_VARIANTS, run_ablation (Table 4)
  analyst.py      SimulatedAnalyst (labelled simulation)
  human_study.py  run_experiment2 (Table 5)
  adaptation_study.py run_experiment3 (Table 6)
  end_to_end.py   run_experiment4 (Table 7)
  audit.py        run_quarantine_audit (Table 2)
  stats.py        RandomInterceptLogit, holm, bootstrap_ci, cumulative_logit
  report.py       table renderers + SCOPE_NOTE
  run_all.py      CLI: python -m ocm.evaluation.rahgm.run_all
```

## 3. Feature extraction (`features.py`)

`CheckCode` groups OCMR checks into the paper's five families:

| Component | OCMR sources | 0.5 (unresolved) condition |
|---|---|---|
| `f_e` entity | C1 identity uniqueness, entity resolution status | `possible_match` / unresolved alias |
| `f_s` schema | W5 structural checks, C9 domain/range | single-valued predicate already has an incumbent |
| `f_t` temporal | C2 sanity, C3 acyclic PRECEDES, C10 transition | missing/unordered `valid_from` vs incumbent |
| `f_v` evidentiary | C8 floor, Algorithm 1 `_evidence_count` | `0 < evidence < floor` |
| `f_c` contradiction | W7 Contradiction_Checker via C7 | `kind ∈ {soft, temporal}` or `severity ≤ medium` |

`FeatureExtractor.extract(candidate, graph, vr, context)` runs the real OCMR
checks (never re-implements them) and returns `RiskFeatures`. `k` counts
components `> 0`.

`Rubric` supplies `q`, `v`, `a` when the context omits them:

- **consequence `q`**: predicate/entity class table (`HAS_STATUS` on `Decision`
  final = 0.9, `OWNS` = 0.8, `ASSIGNED_TO` = 0.6, `HAS_VALUE` = 0.4,
  `ABOUT`/`SUPPORTS` = 0.2), raised to 0.9 when the write retires a terminal
  status, raised by severity.
- **reversibility `v`**: 0.9 when an incumbent exists and is recoverable by
  supersession, 0.5 when the write creates new state with no incumbent, 0.2 for
  `deletion` intent or a write that would retire more than one incumbent.
- **authority `a`**: `SourceAuthority` registry keyed by `source_ref` scheme
  (`analyst:`/`system-of-record:` = 0.95, `tool:` = 0.75, `observation:` = 0.5,
  `inferred:` = 0.35, `untrusted:`/`unattributed` = 0.1).

## 4. Policy (`policy.py`)

```python
@dataclass(frozen=True)
class PolicyParameters:
    beta_0: float
    beta_f: tuple[float, float, float, float, float]   # nonneg
    beta_k: float                                       # nonneg
    beta_q: float                                       # nonneg
    beta_v: float                                       # nonneg (displayed discount)
    beta_a: float                                       # nonneg (displayed discount)
    tau_l: float
    tau_h: float
```

`z = β₀ + Σ βf_i·f_i + β_k·max(0, k−1) + β_q·q − β_v·v − β_a·a`, `r = σ(z)`.

`project()` clamps every constrained coefficient at 0 and keeps
`0 ≤ τ_l < τ_h ≤ 1`, which makes the fitted policy monotone by construction:
`∂r/∂f_i ≥ 0`, `∂r/∂k ≥ 0`, `∂r/∂q ≥ 0`, `∂r/∂v ≤ 0`, `∂r/∂a ≤ 0`.

`fit_policy(samples, lam, lr, iters, seed)` minimizes the L2-regularized
log-loss of eq. (4) by projected gradient descent over `B`. Deterministic:
full-batch gradients, fixed iteration count, no shuffling.

`select_thresholds(samples, policy)` grid-searches `τ_l < τ_h` on a 0.01 lattice
maximizing `F₂` of the review decision against gold `review`, subject to
`MCR ≤ 0.02`; returns `ThresholdSelection(tau_l, tau_h, f2, mcr, feasible)`.

`RouteGuards(g, m, h)` are computed from the OCMR verdict:
`g` = W5 structural failure, C6/C9 hard reject, or blank `source_ref`;
`m` = failure of any `MANDATORY_CHECKS` frozenset (`{C1,C2,C3,C6,C9}` plus W5) —
immutable and unreachable by adaptation;
`h` = `authority ≥ 0.90 ∧ f_t == 0 ∧ incumbent_recoverable ∧ intent ∈ {update, correction}`.

## 5. Router (`router.py`)

`RiskAdaptiveRouter.decide(candidate, vr, graph, context) -> RoutingDecision`
carrying `tier`, `risk`, `score`, `features`, `guards`, `rule`, and a
human-readable `rationale` listing the failed checks and the rule that fired.

`GovernedCommitManager.commit(...)` translates the tier into a `ValidationResult`
for the inner `CommitManager`:

| Tier | Inner action | Notes |
|---|---|---|
| `accept` | `accept` | `vr.valid=True, recommended_action="accept"` |
| `supersede` | `supersede` | requires `conflicting_ids`; falls back to review when empty |
| `reject` | `reject` | preserves OCMR's reason |
| `review` | `quarantine` | plus `ReviewQueue.enqueue(...)` keyed by `quarantine_id` |

The original OCMR verdict is always preserved on the decision record so C2
(autonomous OCMR) is recoverable and the audit can compare routes.

## 6. Review queue (`review_queue.py`)

`ReviewItem` holds the incumbent edge, the proposed candidate, the requested
operation, source and timestamp, the `RoutingDecision`, an `EvidenceBundle`
(supporting/conflicting snippets with provenance), the memory timeline, and the
five available `ReviewAction`s.

`render_explanation(item, depth)` returns a dict whose keys are strictly nested:
`minimal ⊂ evidence ⊂ full`. Depth never touches the route.

`ReviewQueue.adjudicate(item_id, action, *, analyst_id, confidence, seconds,
evidence_opened)` records an `Adjudication` and, for `accept`/`supersede`,
**releases** the held write by committing it through the inner `CommitManager`
and flipping the quarantine record to `resolved`; `reject`/`quarantine` flip to
`dismissed`/leave `unresolved`. `request_evidence` returns the item to the queue
with an incremented evidence request count.

`latin_square(n_levels, n_blocks, offset)` yields the balanced depth schedule.

## 7. Bounded adaptation (`adaptation.py`)

- `BoundedUpdater(eta, beta_radius=0.05, tau_radius=0.02)`:
  `step(params, block)` computes the block log-loss gradient, takes one step,
  projects onto `B`, then **radially clips** the coefficient delta to
  `‖Δβ‖₂ ≤ 0.05` and each threshold delta to `0.02`. Threshold deltas are
  proposed by the same F₂/MCR objective restricted to the trust region.
- `CanaryGate(canary_scenarios, replay_fn)`: measures `DVR`, `MCR`, `RR` under
  `θ` and `θ̃`; accepts iff `ΔDVR == 0 ∧ ΔMCR ≤ 0.01 ∧ ΔRR ≤ 0.05`.
- `PolicyVersion(version_id, parent_id, params, delta_norm, canary, training_case_ids,
  rollback_target, accepted, reason)`; `PolicyRegistry` keeps the full lineage,
  exposes `current`, `history`, `rollback(version_id)`, and
  `regression_rate = rejected / proposed`.
- Immutability: `BoundedUpdater` only ever returns a `PolicyParameters`;
  `MANDATORY_CHECKS`, the tier semantics, feature encodings, and the reject rule
  live in module constants the updater has no handle on. A test asserts that no
  reachable update can make a mandatory-failure write route to `accept`.

## 8. Conditions (`conditions.py`)

| Id | Name | Router |
|---|---|---|
| C1 | `universal_review` | every candidate → `review` (except `g(u)` → `reject`) |
| C2 | `autonomous_ocmr` | pass-through: native OCMR verdict, no review tier |
| C3 | `fixed_threshold` | `review` if `candidate.confidence < 0.80` or `q ≥ 0.7`, else native |
| C4 | `frozen_rahgm` | fitted policy, adaptation disabled |
| C5 | `adaptive_rahgm` | fitted policy + `BoundedUpdater` + `CanaryGate` |

`build_governance(condition, container, policy, ...) -> GovernanceHarness`
wires the router, queue, and (for C5) the registry, and swaps the container's
commit manager.

## 9. Corpus (`corpus.py`)

Deterministic generator, `seed=1337`. 50 scenarios × 30 writes = 1,500.

Per scenario: a seeded entity population (2 persons, 1 org, 1 project, 3 tasks,
2 documents, 1 decision, 1 event, 2 slots) installed directly into the repo and
graph as accepted incumbent state, then 30 temporally ordered `CandidateWrite`s —
10 `routine`, 10 `correction`, 10 `conflict` — shuffled within the scenario by the
scenario RNG so class does not correlate with position.

Each write is instantiated from a template whose correct transition is known by
construction:

| Class | Template family | `gold_transition` |
|---|---|---|
| routine | compatible new edge, fresh slot value, legal status step, evidence link | `accept` |
| correction | authoritative slot correction, temporal supersession, evidence-backed status fix | `supersede` |
| conflict | alias-ambiguous update, low-authority contradiction, cardinality clash, unsupported final decision, terminal-status flip, poisoned evidence | `review` |
| conflict (malformed subset) | unregistered predicate, out-of-range confidence, unattributed source, domain/range violation | `reject` |

`perturbations` records which of the eight axes the case exercises, so §9.3
coverage is checkable. `consequential = q ≥ 0.6 or v ≤ 0.3`.
`minimum_evidence` names the least evidence that resolves the case.

Poisoned evidence appears in exactly 10 of 50 scenarios (20%). Partitions by
scenario index: `train` 0–24, `dev` 25–34, `canary` 35–39, `test` 40–49 — no
entity id or alias is shared across partitions (ids are namespaced by scenario).

Each scenario also carries `questions`: `(query, gold_value, subject_id,
predicate)` triples answered from the final memory state, for Experiment 4.

## 10. Replay (`replay.py`)

`ScenarioReplayer.run(scenario, harness)`:
1. fresh deterministic `CoreContainer` (in-memory SQLite, deterministic ids and
   embeddings);
2. install seed entities and accepted incumbent assertions;
3. for each write in order: build `CandidateAssertion`, run W5 then W6 (the real
   `SchemaValidator` / `ConstraintValidator`), extract features, route, commit;
4. record per-write `WriteRecord(gold, routed, final_transition, escalated,
   consequential, error, review_seconds, risk)`;
5. after the scenario, compute `typed_violations(container)` for `DVR` and answer
   the scenario questions for `accuracy`.

Order preservation is intrinsic: the container is shared across the scenario's 30
writes, so a bad transition at `t` is visible to `t+1..30`.

## 11. Metrics (`metrics.py`)

Per eq. (10)/(11), computed over test writes only:

- `MCR = Σ c·e·(1−z) / Σ c`; `R100 = 100·Σ t / N`; `DVR = (1/N)·Σ v`
  with `v` from `typed_violations(...).total`.
- `RR = Σ z / N`; false escalation = routed `review` with gold `accept|supersede`;
  false quarantine = held write whose gold is `accept|supersede` (the OCMR
  false-quarantine measure this work targets).
- queue precision = review-worthy among escalated; recall = escalated among
  review-worthy.
- `risk_coverage_auc`: sweep coverage by ascending `r`, integrate autonomous
  error rate by trapezoid; lower is better.
- `brier`, `ece(bins=10)`, `calibration_bins`.

`ReviewCostModel` (in `review_cost.py`) maps a review item to minutes:
`base[depth] + 0.35·k + 0.8·(f_c==1) + 0.5·(f_e>0) + 0.4·q`. It is an explicit
model; every table that uses it carries the scope note.

## 12. Statistics (`stats.py`)

`RandomInterceptLogit.fit(y, X, participant, scenario)` maximizes the Laplace
approximation to the marginal likelihood with two crossed random intercepts,
using `scipy.optimize.minimize(method="L-BFGS-B")` over
`(γ, log σ_p, log σ_s)` with the inner mode found by Newton iterations. Returns
coefficients, SEs from the observed information, odds ratios, and 95% CIs.
`gaussian_random_intercept` handles `log t`; `cumulative_logit` handles TLX.
`holm(pvalues)` gives the familywise correction for the three primary contrasts.
`bootstrap_ci(values, stat, iters=2000, seed)` gives cluster bootstrap CIs.

## 13. Honest-reporting contract

`report.SCOPE_NOTE` is embedded at the top level of every emitted JSON and
printed above every table:

> Experiment 1, 3, 4 and the quarantine audit are fully computed. Experiment 2
> reports a **simulated** analyst, not the paper's preregistered 80-participant
> human study; its numbers characterize the harness and the explanation
> renderer, not human behavior. Reviewer minutes come from an explicit
> review-cost model. Krippendorff's alpha is computed over two rubric-based
> annotator simulators, not human annotators.

Table 5, Table 6's analyst-driven rows, and `R100` each carry a per-table
`modelled: true` flag.

## 14. Test plan

`ocm/tests/test_rahgm_features.py` encoding + rubric bounds + determinism;
`test_rahgm_policy.py` monotonicity (property), projection, fit convergence,
threshold feasibility;
`test_rahgm_router.py` the four routing rules, mandatory immutability, the
`GovernedCommitManager` contract, and OCMR non-regression when disabled;
`test_rahgm_review_queue.py` explanation nesting, Latin square balance,
review-and-release, provenance retention;
`test_rahgm_adaptation.py` trust-region bounds, canary accept/reject,
versioning/rollback, adversarial-feedback rejection, no tier disablement;
`test_rahgm_corpus.py` counts (1500/50/30), class balance (500/500/500), 20%
poisoned, partition sizes and disjointness, ground-truth derivability;
`test_rahgm_metrics.py` eq. (10)/(11) against hand-computed fixtures, risk–coverage
AUC ordering;
`test_rahgm_experiments.py` smoke: each experiment runs on a reduced corpus and
emits the expected keys.

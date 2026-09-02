# Discriminating metric for write-policy baselines

Design for the metric that separates OCMR's gate from Toki's four resolution operators
(and from the existing `Bsup`). Written before implementing `Bevi` because the metric
determines whether the experiment is worth running at all.

## 1. This is an existing problem, not a new one

`docs/evaluation_methodology.md:603` already records it:

| Result version | B0/B2 task | `Bsup` task | B3 task |
|---|---:|---:|---:|
| paper draft | 54.17 | — | 54.17 |
| older local JSON | 22.22 | 18.06 | 18.06 |
| new Colab run | 15.28 | 18.06 | **16.67** |

`Bsup` is last-writer-wins restricted to `HAS_VALUE`. It **ties B3 in one run and beats it
in another**, with both at zero durable violations. The same doc already concedes the
consequence at line 629: *"Do not claim B3 uniquely explains any LongMemEval Arm-B gain
until `Bsup` is rerun under the same fingerprint."*

Adding `Blww` (LWW generalized past `HAS_VALUE`) and `Bevi` without a new metric would
reproduce that tie across more arms and both real datasets. The fallback position in the
methodology doc is a retreat to synthetic results. **The metric is what lets the real-data
tables carry the claim instead.**

## 2. Why no existing metric can separate these arms

Three independent reasons, each verified:

**`durable_constraint_violations` measures consistency, not correctness.**
`experiment.py:200` groups accepted assertions by `(subject_id, predicate)` and counts
`len(objects) - 1`. Every correctly-implemented Toki operator retires the loser, so each
leaves exactly one value and scores 0. The metric cannot see *which* value survived.

**`task_success` rewards hoarding.** `runner.py:_answer_score` computes the fraction of
`expected_answer_contains` tokens present in a haystack built from the answer plus every
retrieved item's text. A store holding *both* "New York" and "San Francisco" contains the
gold token and scores 1.0 — identically to a store holding only the correct value. This is
the likeliest reason B0/B2 tied B3 at 54.17 in the paper draft. An arm that never resolves
anything is scored as well as one that resolves correctly.

**Neither metric credits correct abstention.** A quarantine is indistinguishable from a
miss, so OCMR's central behaviour is invisible in the numbers.

There is no store-level correctness metric in the codebase. Stale-value logic exists only in
`run_longmemeval_diagnostics.py` (lines 191, 352), and it measures *extraction and ranking*,
not what the durable store ended up holding.

## 3. The metric: durable-state outcome partition

For each single-valued key `k = (subject_id, predicate)` that has a gold current value `v*`,
let `S(k)` be the set of objects accepted for `k`, and let `Q(k)` be true when a quarantine
record names `k`. Classify each key into exactly one bucket:

| Bucket | Condition | Meaning |
|---|---|---|
| **Correct** | `S(k) == {v*}` | one value, and it is the right one |
| **Stale** | `\|S(k)\| == 1`, `S(k) != {v*}`, not `Q(k)` | **silently wrong** |
| **Split** | `\|S(k)\| >= 2` | cardinality breach (the existing metric) |
| **Abstained** | `Q(k)` and not Correct | wrong or absent, but **declared** |
| **Missing** | `S(k)` empty and not `Q(k)` | extraction never produced the fact |

The `not Q(k)` clause on **Stale** is the load-bearing detail. When OCMR quarantines an
incoming value, the incumbent stays accepted, so `S(k) = {v_old} != {v*}`. Without that
clause OCMR would be scored as silently stale for behaving exactly as designed. With it, the
partition distinguishes *wrong and silent* from *wrong but flagged* — which is the paper's
thesis stated as a measurement.

**Missing** is separated so extraction failure does not contaminate governance measurement.
This is the direct fix for the concern at `evaluation_methodology.md:621` that Arm B is
"extraction/linking-bound, not governance-bound."

Reported rates, over `N` gold-labelled keys:

- **SSR** (Silent Stale Rate) = Stale / N — **primary, minimize**
- **DSC** (Durable State Correctness) = Correct / N — maximize
- **ABS** (Abstention Rate) = Abstained / N — the cost, reported not optimized
- Split / N — the existing violations measure, kept for continuity
- Missing / N — the extraction floor, diagnostic

Plus governance-scoped variants excluding Missing (`DSC_gov = Correct / (N - Missing)`) so
the arm comparison is not diluted by a shared extraction bottleneck.

SSR is orthogonal to `durable_constraint_violations` by construction: Split and Stale are
disjoint buckets, so the two metrics never double-count. That orthogonality is the argument
for why this metric is necessary rather than gratuitous.

## 4. Why it separates

| Arm | Split | SSR | ABS |
|---|---|---|---|
| B0 / B2 (ungoverned) | high | — | 0 |
| `Blww` (last-writer-wins) | 0 | **>0 when recency is wrong** | 0 |
| `Bevi` (evidence-weighted) | 0 | **>0 when the stale fact is more confident** | 0 |
| `Bawait` (await-confirmation) | 0 | ~0 | high (needs an oracle; report as ceiling) |
| `Brule` (per-rule policy) | 0 | depends on the policy table | 0 |
| B3 (OCMR) | 0 | **~0 by design** | >0 |

The ungoverned arms stop scoring well because Split is no longer counted as Correct — this
alone removes the hoarding reward that produced 54.17 == 54.17.

The claim the table supports is not "OCMR is more accurate." It is **"OCMR converts silent
staleness into declared abstention,"** which is what the abstract already says it does
("deliberately paying a recall cost").

## 5. The metric alone is not enough — the workload must contain cases where recency is wrong

Both real datasets are currently built so **the newest value is gold by construction**:

- `longmemeval_adapter.py:231` — `intent = "update" if belief_set else "new_fact"`, and the
  trajectory is bucketed by ascending session index, so the last-written value *is*
  `current_value`.
- `multiwoz_adapter.py:113` — a changed slot is emitted as `correction`, and
  `run_multiwoz_suite` sets `authoritative_update_supersede=True`.

On this data last-writer-wins is **correct by design**, which is exactly why `Bsup` ties B3.
Reporting SSR on the current workload would show ties everywhere and prove nothing.

### Proposed 2×2 over the existing gold trajectories

The LongMemEval oracle already controls arrival order, intent, and confidence in one loop
(`build_from_kupdate_oracle`, `longmemeval_adapter.py:230-245`), and `ann["current_value"]`
is an independent gold label. Two orthogonal perturbations, each a small change in that loop,
with the gold answer held fixed:

- **Arrival order**: `aligned` (current value written last — today's behaviour) vs
  `permuted` (a stale value written last).
- **Confidence order**: `aligned` (current value more confident — today's behaviour) vs
  `inverted` (stale value more confident).

Pre-registered predictions, one distinct failure signature per operator:

| order | confidence | `Blww` | `Bevi` | B3 |
|---|---|---|---|---|
| aligned | aligned | correct | correct | correct (**tie, expected**) |
| permuted | aligned | **stale** | correct | abstain / correct |
| aligned | inverted | correct | **stale** | abstain / correct |
| permuted | inverted | **stale** | **stale** | abstain |

This is falsifiable in the direction that matters: if B3 does *not* abstain in the
off-diagonal cells, the OCMR claim is wrong and the experiment says so.

**Framing discipline.** The permutation is a *constructed* perturbation and must be labelled
one, with the aligned cell reported as the natural-workload anchor. Toki applies the same
discipline (its constructed controls saturate at 1.00 and it says so explicitly). The
justification for permuted arrival is that late-surfacing of older facts is a real
long-conversation phenomenon ("actually, back in March I was in Boston"); the intervention
changes only arrival order on real data, not the gold answer.

## 6. Implementation

**New metric** — `ocm/evaluation/durable_state.py`, mirroring `durable_constraint_violations`:

```python
def durable_state_outcomes(container, gold: dict[tuple[str, str], str]) -> DurableStateReport
```

- group `container.repo.list_assertions("accepted")` by `(subject_id, predicate)` for
  single-valued predicates — same grouping as `experiment.py:214-224`
- resolve object id → value string via `graph.get_entity_payload(oid)["value"]`, the pattern
  `typed_violations._status_value` already uses for `StatusValue`
- read `container.quarantine_store.list()` and map `conflicting_ids` / candidate payload back
  to keys to compute `Q(k)`
- return the five counts plus the derived rates

**Gold map** — `build_from_kupdate_oracle` already computes `slot_name` and `current` in the
same scope; return `{(slot_id, "HAS_VALUE"): current}` as a third element. MultiWOZ's
`build_from_dialogues` can do the same from its final cumulative belief state.

**Workload knob** — one parameter on `build_from_kupdate_oracle`, e.g.
`order: Literal["aligned","permuted"]` and `confidence: Literal["aligned","inverted"]`,
threaded into the `per_session_values` bucketing and the `conf` assignment. Fold both into
the checkpoint `key_suffix` so cells don't collide.

Estimated cost: metric ~80 lines, workload knob ~20, plus tests. Under a day before run time.

## 7. Open decisions, and one caveat

**Decide before implementing:**

1. Does the 2×2 go in the paper as a table, or in an appendix as a mechanism probe? It is
   constructed-workload evidence, so it cannot carry a headline utility claim.
2. `Bawait` needs a confirmation oracle. With gold labels it is a ceiling, not a baseline —
   label it as such, or defer it to the RAHGM paper where the reviewer models already exist.
3. Whether to keep `authoritative_update_supersede=True` for MultiWOZ in the perturbed
   cells. Leaving it on is the honest choice (it is a semantics decision about source trust,
   per `evaluation_methodology.md:574`), but it means B3 supersedes rather than abstains
   there.

**Caveat I could not fix in the metric.** This measures the *store*. The read-path metric
`task_success` still rewards hoarding, because token containment cannot tell "holds the right
value" from "holds every value." If you want the answer-side number to reflect exclusivity
too, that needs a separate stale-surfacing rate ("the stale token must be *absent* from the
retrieved evidence"). `run_longmemeval_diagnostics.py:379` already computes something close
to this per-question and could be promoted. Worth doing, but it is a distinct change and
should not be bundled into this one.

# Positioning OCMR against Toki (arXiv 2606.06240)

Insert-ready prose for the OCMR paper's related work. The paper has no LaTeX source in
this repo, so this is drafted for paste-in rather than applied directly.

**Reference being positioned against:** Ziming Wang, "TOKI: A Bitemporal Operator Algebra
for Contradiction Resolution in LLM-Agent Persistent Memory," arXiv:2606.06240v1 [cs.DB],
4 Jun 2026 (PVLDB-track preprint, single author, HKUST). Full text extracted to
`docs/_paper3_extracted.txt`.

**Why this is needed:** Toki's central claim is that contradiction resolution *is*
write-time concurrency control. OCMR's is write-time state admissibility. That is the same
control point, published three months before submission, and the current §II positions only
against retrieval-time memory work. A DB-aware reviewer will ask.

**Good news:** a grep for absolute novelty claims (`first`, `no prior`, `to our knowledge`,
`only system`, ...) returns nothing in the OCMR draft, and the abstract already hedges the
contribution to "durable state governance rather than an unconditional task-success
improvement." Nothing has to be retracted. All three edits below are additive.

---

## Edit 1 — new paragraph, end of §II-D

Append after the existing paragraph in *"D. Contradiction Handling, Truth Maintenance, and
Provenance"* (the paragraph ending "...assertion states with attached source lineage.").

> A recent line reframes contradiction resolution itself as write-time concurrency control.
> Toki [31] types the four resolution strategies deployed in production memory
> systems—last-writer-wins, evidence-weighted merge, await-confirmation, and per-rule
> policy—as one family of bitemporal operators, each declaring an isolation precondition and
> retaining the losing fact in an audit row, and reports that every agent-memory baseline in
> its eight-system audit admits at least one write-time anomaly. That contract is
> complementary to ours, and it operates one step downstream: its operators presuppose that
> a contradiction has already been detected on a subject–predicate key and that one of the
> two facts should survive, then fix which one and under what isolation guarantee. OCMR
> addresses the prior question of whether a candidate is admissible durable state at all,
> under typed signatures, cardinality, temporal coherence, and evidence support, and may
> quarantine an unresolved conflict rather than elect a winner. We therefore adopt these
> four strategies as comparison baselines (Section IV).

## Edit 2 — second new paragraph, immediately after Edit 1

This is the one that earns reviewer trust: it claims the one anomaly OCMR genuinely
excludes and concedes the two it does not.

> Toki's anomaly vocabulary also sharpens what OCMR does and does not guarantee. Because a
> superseded assertion is retained with its provenance and linked to its successor rather
> than overwritten, OCMR's lifecycle excludes audit erasure, the anomaly Toki reports in
> five of its six agent-memory baselines. OCMR declares no isolation level, however, and its
> confidence estimates originate from an LLM extractor, so it does not address replay
> inconsistency or belief-drift skew; pinning those would require the keyed-judge logging
> Toki proves necessary, which we leave to future work.

## Edit 3 — one clause in §II-F

§II-F ("Positioning and Novelty") already runs a parallel enumeration of adjacent lines.
Add a clause for Toki so the list stays exhaustive. Current text:

> ...proactive memory agents decide when stored state should be injected into an action
> trajectory [17]; OCMR treats memory formation itself as the reliability-critical operation.

Revised:

> ...proactive memory agents decide when stored state should be injected into an action
> trajectory [17]; write-time operator algebras specify how a detected contradiction is
> resolved and under what isolation guarantee [31]; OCMR treats memory formation itself as
> the reliability-critical operation.

## Edit 4 — reference list

Appends as [31]; the current list ends at [30].

```
[31] Z. Wang, "TOKI: A bitemporal operator algebra for contradiction resolution in
     LLM-agent persistent memory," arXiv preprint arXiv:2606.06240, 2026.
```

---

## Accuracy ledger

Every factual claim above, and where it is verifiable in `docs/_paper3_extracted.txt`:

| Claim in the draft | Verified at |
| --- | --- |
| Four strategies: last-writer-wins, evidence-weighted merge, await-confirmation, per-rule policy | abstract, §1, Table 1 |
| Typed as bitemporal operators, each with an isolation precondition | §3.2, Definition 1 |
| Losing fact retained in an audit row | §3.1 (dual-row schema), Table 2 |
| Eight-system audit; every agent-memory baseline admits ≥1 anomaly | §4.1, Table 4 |
| Audit erasure in five of six agent-memory baselines | line 1040, "covers five of the six baselines" |
| Keyed judge logging is *necessary* for replay consistency | Theorem 5, §3.3 |
| The three anomalies are N1 replay inconsistency, N2 belief-drift skew, N3 audit erasure | Table 3, §1 |

OCMR-side claim: superseded assertions are retained, not deleted. Verified in
`ocm/memory/commit_manager.py` — `_mark_superseded` flips status via `UPDATE`
(`sqlite_repository.py:274`), `_add_supersedes_edge` links successor to predecessor, and
provenance is preserved on both sides. There is no `DELETE` on assertions anywhere in the
repository.

## Deliberately not claimed

- **That OCMR excludes N1 or N2.** It does not. OCMR's gate is deterministic given a
  candidate's confidence, but that confidence comes from an LLM extractor, so
  re-adjudication can diverge — exactly Toki's N1. Edit 2 concedes this in the paper's own
  voice rather than leaving a reviewer to find it.
- **That OCMR implements Toki.** §IV will implement Toki's four *winner-selectors* as write
  policies. The bitemporal algebra, isolation preconditions, and soundness theorems are out
  of scope, and the baseline arms should be named for the heuristics (`Blww`, `Bevi`,
  `Bawait`, `Brule`) with a scope note, not for Toki.
- **That Toki is weaker on utility.** It is fair to note Toki draws no superiority claim (it
  says so at §4.6, §6, and in the abstract: its cross-system comparison "stays underpowered
  and claims no superiority"), but that is a scope statement about its own evidence, not a
  deficiency to score points on. Asserting OCMR beats Toki on utility would require running
  Toki, which we are not doing.

## Open question for §IV

Toki's operators all retire the loser, so each one scores ~0 on
`durable_constraint_violations` (`ocm/evaluation/experiment.py:200` counts distinct accepted
objects per `(subject, predicate)`). Reported on that metric alone, the new baselines will
tie with B3 and the table will say nothing. The discriminating measurement is on conflicts
where electing *any* winner is wrong — ambiguous, low-margin updates where OCMR quarantines
and last-writer-wins and evidence-weighted must still pick. Decide that metric before
implementing the operators.

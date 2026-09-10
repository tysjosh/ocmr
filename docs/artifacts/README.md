# Paper artifact map

This directory contains the tracked, frozen artifacts used by the controlled
entity-linking-evasion evaluation in *Write-Time Governance for Persistent-State
Integrity in AI Agents*.

Frozen development revision: `cc56f21ffb3bccd70694e8bc1090ff9624267038`.

## Entity-linking-evasion evaluation

- `entity_linking_evasion_broad/paper_grade_suite.json` — five-seed paper suite:
  seven attack axes, original and mutated attacks, B0/B2/Bsup/B3 comparisons,
  the C7-off ablation, benign utility, and latency summaries.
- `entity_linking_evasion/pre_fix_confirmed_bypass_config_off.json` — original
  config-off bypass reproduction.
- `entity_linking_evasion/after_fix_original_seed1337.json` — post-fix original
  attack suite.
- `entity_linking_evasion/after_fix_mutated_seed2026.json` — post-fix mutated
  suite.
- `entity_linking_evasion/benign_false_positive_seed1337.json` — initial benign
  false-positive check.
- `entity_linking_evasion/release_gate_summary.json` — scoped containment gate.

## LongMemEval durable-state (LM-O, cached oracle)

- `longmemeval_durable_state/lmo_durable_state_qwen14b.json` — five-arm
  durable-state scoring of the cached-oracle LongMemEval arm, produced at
  revision `5cfcffe159b08b9c466bdf33bddbb32733b93d19` with a clean tracked tree
  (`code_dirty: false`) and `all-MiniLM-L6-v2` embeddings.

This is the artifact behind the LM-O row. It exists because `task_success` on
this surface is **100.0 for every arm**, governed and ungoverned alike, so the
headline recall number separates nothing while durable violations differ by two
orders of magnitude. Durable-state correctness (DSC) separates the arms
completely:

| Arm  | DSC ↑  | SSR ↓ | split ↓ | task_success | violations/100 ↓ |
|------|--------|-------|---------|--------------|------------------|
| B0   | 0.00   | 0.00  | 100.00  | 100.00       | 106.38           |
| B2   | 0.00   | 0.00  | 100.00  | 100.00       | 106.38           |
| Bsup | 100.00 | 0.00  | 0.00    | 100.00       | 0.00             |
| Bevi | 100.00 | 0.00  | 0.00    | 100.00       | 0.00             |
| B3   | 100.00 | 0.00  | 0.00    | 100.00       | 0.00             |

All 47 annotated keys are `split` (two or more accepted values) under the
ungoverned arms and `correct` under every governed arm. `missing` is 0, so no
extraction floor confounds this surface — the gap is governance alone. The
`106.38` reproduces the published figure and is recorded directly as
`durable_violations_per_100_responses` rather than left as `50 / 47 * 100`.

Independently reproduced on a second machine against the *other* LongMemEval
corpus — `longmemeval_oracle.json` (15.4 MB, evidence sessions only) rather than
the `longmemeval_s.json` (277.4 MB, full haystack) used for the tracked file —
yielding byte-identical buckets, task success and violation counts, and the same
`instances=72 annotations=47` denominators. The row is therefore invariant to
which corpus is loaded, as expected: writes are emitted only at trajectory
placements, so the store is identical, and the recall question resolves through
the `[[slot]]` marker and the `HAS_VALUE` rule rather than retrieval ranking.
The extra distractor sessions are inert here. Reproducing from the oracle file is
the cheaper path, and `_meta.dataset_sha256` records which was used.

Gold value trajectories: `../longmemeval_kupdate_annotations__Qwen_Qwen2.5-14B-Instruct.json`,
47 annotations validated against the benchmark answers (0 orphans, 0 field
gaps, 0 answer mismatches, 0 unknown session ids, trajectory length 2–3).
LM-O is annotation-scoped, so 47 — not the 72 `knowledge-update` questions in
the dataset — is the denominator.

Two caveats for anyone reading these numbers:

- `Bevi` is **identical** to `Bsup` here. Under uniform annotation confidence
  the evidence surface has nothing to discriminate on, so this is structural,
  not a measurement.
- `task_success` must be read only from an `--embeddings local` run. The
  deterministic provider hashes text into vectors, making semantic retrieval
  near-random; it reports `51.06 / 68.09` on this same store. The durable-state
  buckets are unaffected, since they read the store rather than retrieval.

## Metric saturation and the revised tables

The published Table VIII columns do not separate the arms. Reading `Task` down
each block gives 0.09 points on MultiWOZ (99.87 → 99.96) and **zero** on LM-O
(100.0 for every arm, governed and ungoverned). `Contr.` is 0.00 for every arm
on both datasets. Of the reported metric columns, LM-O separates arms on `Viol`
alone.

The cause is mechanical rather than a tuning artifact. `task_success` is
answer-token recall over the rendered answer *plus every retrieved item*, so it
is monotone in retrieved volume; on MultiWOZ,
`EvidencePackager._slot_value_answer` deliberately returns the most recent
accepted `HAS_VALUE` when several are active — which only happens on an arm that
failed to supersede — so an arm carrying single-valued violations is still handed
the current gold value and scores ~99.9.

Durable-state correctness (DSC) is measured on the same runs and reads the store
rather than an answer:

| Data | Arm | Task ↑ | DSC ↑ | split ↓ | Viol ↓ | correct/stale/split | A/S/Q (5 seeds) |
|------|-----|--------|-------|---------|--------|---------------------|-----------------|
| MW   | B0 Text     | 99.88 | 93.09  | 6.87   | 7.14   | 7544/3/557 | 43550/0/0 |
| MW   | B2 Hybrid   | 99.88 | 93.09  | 6.87   | 7.14   | 7544/3/557 | 43550/0/0 |
| MW   | Bsup        | 99.96 | 99.95  | 0.00   | 0.00   | 8100/4/0   | not recorded |
| MW   | Bevi        | 99.96 | 99.95  | 0.00   | 0.00   | 8100/4/0   | not recorded |
| MW   | B3 Governed | 99.96 | 99.95  | 0.00   | 0.00   | 8100/4/0   | 40540/3010/0 |
| LM-O | B0 Text     | 100.0 | 0.00   | 100.00 | 106.38 | 0/0/47     | 485/0/0 |
| LM-O | B2 Hybrid   | 100.0 | 0.00   | 100.00 | 106.38 | 0/0/47     | 485/0/0 |
| LM-O | Bsup        | 100.0 | 100.00 | 0.00   | 0.00   | 47/0/0     | 235/250/0 |
| LM-O | Bevi        | 100.0 | 100.00 | 0.00   | 0.00   | 47/0/0     | 235/250/0 |
| LM-O | B3 Governed | 100.0 | 100.00 | 0.00   | 0.00   | 47/0/0     | 235/250/0 |

The A/S/Q column is retained deliberately rather than replaced. It and the
outcome buckets are not substitutes: A/S/Q is what the gate *decided*, the
buckets are what the store ended up *holding*, and identical write counts can
accompany very different end states. On raw end-to-end LongMemEval, `B3` and
`Bmemgpt` hold near-identical accepted counts (11,715 vs 11,479 per seed) yet
score 34.7 against 48.6 task success — a difference invisible in A/S/Q. Read as
a pair, A/S/Q is what governance cost and DSC is what it bought; the `Q` and `R`
columns are also what support the bounded-review-burden claim, since review
effort is a count rather than a rate.

LM-O A/S/Q are reproduced by `run_lmo_durable_state.py`, which reports
single-pass counts (`97/0/0` and `47/50/0`); multiplying by the five seeds gives
the values above. MultiWOZ `Bsup` and `Bevi` write counts were never recorded —
the published table omits them — and are marked as such rather than assumed
equal to `B3`. They are not safe to infer: on raw LongMemEval `Bsup` and `Bevi`
differ in write counts (11,714/2,810 vs 11,715/2,809) even where their outcomes
match.

MultiWOZ: 1,000 validation dialogues, 8,104 probed slots. LM-O: 47 annotated
`knowledge-update` questions. DSC turns 0.09 points into 6.86 on MultiWOZ and
zero into a complete 0.00-vs-100.00 split on LM-O. The `split` bucket is what
does the work — two or more accepted values on one durable key, which is the
cardinality breach `durable_constraint_violations` counts.

### Reading the published A/S/Q columns

The accepted/superseded/quarantined counts are **summed across all five seeds**,
which is why LM-O reports 485 writes for 47 questions. Divided through they
reconcile with the single-pass durable-state runs, and the last line is a useful
internal check — every write B3 supersedes is exactly one violation B0 leaves
behind:

| | published | ÷ 5 seeds | durable-state run |
|---|---|---|---|
| LM-O B0 accepted    | 485 | 97 | 47 keys × 2.06 trajectory values ≈ 97 |
| LM-O B3 accepted    | 235 | 47 | 47 keys holding one value each |
| LM-O B3 superseded  | 250 | 50 | 50, equal to B0's violation count |

### Caveats

- `Bsup` and `Bevi` are **identical** on both datasets. This is structural, not
  a measurement: under uniform annotation confidence the evidence surface has
  nothing to discriminate on. The published table omits `Bsup`; showing it
  requires stating this, or a reader will suspect a duplicated column.
- Two small gaps against the published row remain: `Task` 99.87 vs 99.88 on
  MultiWOZ, and superseded `3010 / 5 = 602` vs 579 violations. The first is
  plausibly the write-side embedding batching in `ocm/retrieval/vector_index.py`
  (batched `embed()` pads to the longest sequence, so it is not bit-identical to
  per-item `embed_one`). The second is expected: a slot reaching three distinct
  values contributes two supersessions while the violation metric counts
  per-key breaches, so the two need not be equal.
- The MultiWOZ DSC figures above come from a completed run whose output uses the
  pre-refactor bucket names (`exact` / `ambig` / `wrong_value` for
  `correct` / `split` / `stale`) and carries no provenance stamp. The numbers
  are sound and reproduce the published 7.14 violation rate; regenerating
  through `run_multiwoz_durable_state.py` would only add the canonical names and
  the `_meta` provenance block, so it is a presentation improvement rather than
  a correctness requirement.

## Reproduction entry points

- `run_entity_linking_evasion.py --paper-suite` regenerates the broad synthetic
  security suite.
- `run_lmo_durable_state.py --annotations <file> --embeddings local` regenerates
  the LM-O durable-state artifact; `--fixture` runs the same scorer offline with
  no annotations file or dataset. Note that `_meta.annotations_sha256` digests
  the *parsed* annotations (sorted keys, normalized separators), so it is
  invariant to reformatting and will not match `shasum` on the raw file.
- `run_multiwoz_durable_state.py --arms B0,B2,Bsup,Bevi,B3` regenerates the
  MultiWOZ row (full validation split by default; `--limit` for a smoke run,
  `--fixture` for a 3-dialogue offline check). It fetches dialogues from
  `raw.githubusercontent.com`, so it needs network access to GitHub specifically.
- `ocm/evaluation/artifact_meta.py` supplies the provenance stamps. Both
  durable-state runners are replay scorers over gold facts, so identical code
  with different inputs yields different numbers; `code_revision`, `code_dirty`
  and the input digests are what let a reader tell two such runs apart.
- `ocm/evaluation/durable_state.py` defines the outcome taxonomy. The
  `stale`/`abstained` distinction is load-bearing: a quarantining arm leaves a
  non-gold incumbent accepted by design, and scoring that as silently stale
  would penalise it for behaving correctly.
- `ocm/evaluation/entity_linking_evasion.py` defines the attack generator,
  aggregation, ablations, and benign workload.
- `ocm/tests/test_entity_linking_evasion.py` contains permanent regression
  coverage.
- `docs/evaluation_methodology.md` documents the primary evaluation protocol,
  threshold sensitivity, statistical units, and quarantine-audit reconciliation.

## Scope and unavailable frozen outputs

The tracked directory freezes the entity-linking-evasion artifacts and the LM-O
durable-state run. Still outstanding for the final anonymous artifact bundle:
the primary Qwen2.5-14B benchmark, bounded-review fold outputs, the end-to-end
(non-oracle) LongMemEval and MultiWOZ runs, and efficiency traces. They are not
represented by the security-suite JSON and must not be reconstructed from
mock-extractor runs.

The LM-O artifact here is oracle-driven by construction: it scores governance
*given* gold facts, isolating it from extraction error. It is therefore not a
substitute for the end-to-end run, which is where noisy extraction and entity
resolution are exercised.

## Checkpoint validity when archiving notebook runs

The notebook's 7d and 7e cells run through `run_multiseed`, which checkpoints
per `(method, seed)` under keys of the form
`ms__{method}__seed{seed}__pc{per_category}__fp{run_fingerprint}`. That
fingerprint digests the *extraction stack* — embeddings, dataset, annotating
model, annotations digest — and **not the code revision**.

Those checkpoints and their result files remain accurate records of what ran,
and archiving them is safe. The durable-state runners in this repository cannot
affect them: they bypass `run_multiseed` entirely, drive `BaselineRunner`
directly, use an oracle extractor with no LLM calls, and run with
`chroma_mode="memory"`, so they write nothing but the file named by `--out`.

The hazard is confined to *resuming* a run against an existing checkpoint
directory after the scoring code has changed. `ocm/retrieval/vector_index.py`
now batches write-side embedding (one `embed()` per flush instead of
`embed_one()` per item), which is semantically equivalent but not bit-identical,
since batching pads to the longest sequence in the batch. A partial re-run would
therefore blend batched and unbatched cells with nothing in the key to detect
it. Two ways to avoid that: point at a fresh `--checkpoint-dir` so every cell
recomputes under one configuration, or fold the code revision into the
fingerprint so the mismatch forces recomputation instead of passing silently.

A practical consequence for reviewers: reproducing MultiWOZ from the current
revision may yield 99.88 task success where 99.87 was published. That is the
batching difference, not a discrepancy in the artifact.

The paper should link to the anonymous artifact bundle rather than the
author-identifying development repository.

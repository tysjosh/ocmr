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

## Reproduction entry points

- `run_entity_linking_evasion.py --paper-suite` regenerates the broad synthetic
  security suite.
- `run_lmo_durable_state.py --annotations <file> --embeddings local` regenerates
  the LM-O durable-state artifact; `--fixture` runs the same scorer offline with
  no annotations file or dataset. Note that `_meta.annotations_sha256` digests
  the *parsed* annotations (sorted keys, normalized separators), so it is
  invariant to reformatting and will not match `shasum` on the raw file.
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

The paper should link to the anonymous artifact bundle rather than the
author-identifying development repository.

"""LM-O (cached-oracle LongMemEval): durable-state scoring per arm.

Why
---
``task_success`` is answer-token recall over the rendered answer *plus every
retrieved item*, so it is monotone in retrieved volume and saturates wherever the
answer deriver can find the current value regardless of what else is active. On
the cached-oracle LongMemEval arm the published row is ``100.0`` task success for
B0, B2, Bsup and B3 alike, while durable violations differ by two orders of
magnitude (106.38 vs 0.00) -- so the headline recall number separates nothing.

This script reports the durable-state buckets instead, using the project's
existing :mod:`ocm.evaluation.durable_state` metric (not a new one). Its taxonomy
is what makes the comparison fair to a governed arm:

* ``correct``   -- the key holds exactly the gold value
* ``stale``     -- exactly one accepted value, wrong, and **nothing was flagged**
                   (silently wrong)
* ``split``     -- two or more accepted values (the cardinality breach that
                   ``durable_constraint_violations`` counts)
* ``abstained`` -- wrong or absent, but the key **was** flagged: the arm declined
                   and said so
* ``missing``   -- never extracted; the extraction floor, a diagnostic

The ``stale`` / ``abstained`` split is load-bearing. When the gate quarantines an
incoming value the incumbent stays accepted, so the store holds a non-gold value
*by design*; scoring that as silently stale would penalise the governed arm for
behaving correctly. Any run of ``intent_mode="corroborated"`` (which quarantines
rather than overwriting) must be scored with this distinction or its results are
meaningless.

Requires the gold annotations JSON (Qwen-produced value trajectories, written by
``longmemeval_annotate.annotate_file``). LM-O is annotation-scoped: instances
without an annotation are skipped by the builder, so the denominator is the
annotated subset.

Usage:

    python run_lmo_durable_state.py --annotations path/to/annotations.json

    # offline smoke test on the built-in fixture, no annotations file needed:
    python run_lmo_durable_state.py --fixture
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Mirrors the notebook's download cells. LM-O (7e) reads the *oracle* file --
# evidence sessions only, ~tens of MB -- and the gold annotations were produced
# from it by ``annotate_file``. ``longmemeval_s.json`` is the 277 MB full-haystack
# file used by the end-to-end arm (7f); it carries the same question ids, so it
# loads here without error, which is precisely why the mismatch is easy to miss.
_LME_URLS = {
    "longmemeval_oracle.json":
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_oracle.json",
    "longmemeval_s.json":
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json",
}


def _ensure_dataset(path: Path) -> None:
    """Download the LongMemEval file if absent, as the notebook cells do.

    ``data/`` is gitignored, so a freshly cloned machine has the code and the
    annotations but not the corpus, and the failure is a bare ``FileNotFoundError``
    from deep inside the adapter.
    """
    if path.exists():
        return
    url = _LME_URLS.get(path.name)
    if url is None:
        raise SystemExit(
            f"{path} not found and no download URL is known for '{path.name}'.\n"
            f"Known files: {', '.join(sorted(_LME_URLS))}"
        )
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"{path} missing; downloading {url}", flush=True)
    # Download to a sidecar first so an interrupted transfer cannot leave a
    # truncated file that later loads as valid-but-incomplete JSON.
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    print(f"downloaded -> {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def _dataset_provenance(path: Path, *, fixture: bool) -> dict[str, Any]:
    """Which corpus was scored.

    Recorded because ``longmemeval_oracle.json`` and ``longmemeval_s.json`` share
    question ids: pointing at the wrong one raises no error, it just changes how
    many distractor sessions are ingested. Without this field a result cannot be
    told apart from one run against the other file.
    """
    from ocm.evaluation.artifact_meta import file_provenance
    if fixture:
        return {"dataset": "builtin-fixture"}
    return file_provenance(path, prefix="dataset")


def _provenance(path: Path | None, annotations: Any) -> dict[str, Any]:
    """Identity of the gold annotations and the code that scored them.

    The annotations are LLM-generated, so the result is only interpretable
    alongside *which* model produced them and *which* file. The notebook stamps
    the model into the filename for exactly this reason; digesting the loaded
    content as well means a silently edited file cannot masquerade as the
    original. ``code_revision`` is recorded because the durable-state taxonomy
    itself is code under active change.
    """
    from ocm.evaluation.artifact_meta import code_provenance
    from ocm.evaluation.run_identity import json_digest
    meta: dict[str, Any] = {}
    if path is not None:
        meta["annotations_path"] = str(path)
        # Digests the *parsed* annotations (sorted keys, normalized separators),
        # so it is invariant to reformatting and will not match `shasum` on the file.
        meta["annotations_sha256"] = json_digest(annotations, length=0)
        # ``longmemeval_kupdate_annotations__<model-slug>.json`` -- kept as the raw
        # slug because un-slugging "/" from "_" is ambiguous for arbitrary model ids.
        stem = path.stem
        meta["annotating_model_slug"] = (
            stem.split("__", 1)[1] if "__" in stem else None)
    meta.update(code_provenance())
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=None,
                        help="Gold trajectory JSON from longmemeval_annotate.")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s.json"))
    parser.add_argument("--arms", default="B0,B2,Bsup,Bevi,B3")
    parser.add_argument("--embeddings", choices=("local", "deterministic"),
                        default="deterministic",
                        help="Durable-state scoring reads the store, not retrieval, so "
                             "the provider cannot change the buckets; deterministic is "
                             "far faster. Use local only to also compare task_success.")
    parser.add_argument("--fixture", action="store_true",
                        help="Use the built-in offline fixture instead of real data.")
    parser.add_argument("--out", type=Path,
                        default=Path("local_results/lmo_durable_state.json"))
    args = parser.parse_args()

    from ocm.core.config import Settings
    from ocm.evaluation.arms import build_arm
    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_from_kupdate_oracle, load_longmemeval, sample_annotations, sample_instances,
    )
    from ocm.evaluation.durable_state import durable_state_outcomes, resolve_gold_keys
    from ocm.evaluation.experiment import durable_constraint_violations
    from ocm.evaluation.runner import BaselineRunner
    from ocm.retrieval.embeddings import (
        DeterministicEmbeddingProvider, LocalEmbeddingProvider)

    if args.fixture:
        instances, annotations = sample_instances(), sample_annotations()
        print("using the offline fixture")
    else:
        if args.annotations is None:
            parser.error("--annotations is required unless --fixture is passed")
        from ocm.evaluation.datasets.longmemeval_annotate import load_annotations
        _ensure_dataset(args.data)
        instances = load_longmemeval(
            str(args.data), question_type="knowledge-update", limit=None)
        annotations = load_annotations(str(args.annotations))

    examples, oracle = build_from_kupdate_oracle(instances, annotations)
    print(f"instances={len(instances)} annotations={len(annotations)} "
          f"examples={len(examples)} gold_keys={len(oracle.gold_current_values)}")
    if not examples:
        print("no annotated instances -- nothing to score", flush=True)
        return 1

    def settings_factory() -> Settings:
        return Settings(
            deterministic_test_mode=True, chroma_mode="memory", extractor="mock",
            authoritative_update_supersede=True,
        )

    embeddings = (
        LocalEmbeddingProvider() if args.embeddings == "local"
        else DeterministicEmbeddingProvider()
    )
    runner = BaselineRunner(settings_factory=settings_factory, top_k=10)

    header = (f"{'arm':7}{'DSC':>8}{'SSR':>8}{'split':>8}{'abst':>8}{'miss':>8}"
              f"{'task':>8}{'viol':>7}   correct/stale/split/abst/missing")
    print("\n" + header)
    print("-" * len(header))

    out: dict[str, Any] = {"_meta": {
        "n_instances": len(instances),
        "n_annotations": len(annotations),
        "n_examples": len(examples),
        "n_gold_keys": len(oracle.gold_current_values),
        "embeddings": args.embeddings,
        "fixture": bool(args.fixture),
        **_dataset_provenance(args.data, fixture=bool(args.fixture)),
        **_provenance(args.annotations, annotations),
    }}

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        strategy = build_arm(arm, settings_factory, extractor=oracle, embeddings=embeddings)
        records = []
        # Write-side outcomes (the paper's A/S/Q(/R) columns). Reported alongside
        # the durable-state buckets because the two are not substitutes: A/S/Q is
        # what the gate *decided*, the buckets are what the store ended up
        # holding, and identical A/S/Q counts can accompany very different end
        # states -- a latest-wins policy that supersedes the correct value looks
        # the same here as one that keeps it.
        writes = {"candidates": 0, "accepted": 0, "superseded": 0,
                  "quarantined": 0, "rejected": 0}
        for example in examples:
            counts = runner._ingest_sessions(strategy, example)
            for key in writes:
                writes[key] += counts[key]
            for q_index, question in enumerate(example.questions):
                records.append(runner._run_question(
                    arm, strategy, example, q_index, question,
                    write_quarantined=counts["quarantined"]))

        # Gold keys are recorded by slot *name*; entity ids are minted per run, so
        # they must be read back from the store after the replay.
        gold = resolve_gold_keys(strategy.container, oracle.gold_current_values)
        report = durable_state_outcomes(strategy.container, gold)
        violations, _accepted = durable_constraint_violations(strategy.container)
        task = 100.0 * sum(r["score"] for r in records) / max(1, len(records))

        out[arm] = {
            **report.as_dict(),
            "durable_violations": violations,
            # The paper reports violations per 100 responses, not the raw count;
            # recording it here keeps the table cell traceable to this file.
            "durable_violations_per_100_responses": (
                100.0 * violations / len(records) if records else 0.0),
            "n_responses": len(records),
            "task_success": task,
            # A/S/Q(/R), single-pass. The paper's columns sum these over five
            # seeds, so multiply by len(seeds) to compare (LM-O: 97 -> 485).
            "write_outcomes": dict(writes),
            "outcomes": {f"{k[0]}|{k[1]}": v for k, v in report.outcomes.items()},
        }
        print(f"{arm:7}{report.dsc:8.2f}{report.ssr:8.2f}{report.split_rate:8.2f}"
              f"{report.abstention_rate:8.2f}{report.missing_rate:8.2f}"
              f"{task:8.2f}{violations:7}   "
              f"{report.correct}/{report.stale}/{report.split}/"
              f"{report.abstained}/{report.missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {args.out}")
    print("\nDSC=durable state correctness (up)  SSR=silent stale rate (down)")
    print("split=two+ accepted values (down)   abst=declined and flagged")
    print("miss=never extracted (extraction floor, diagnostic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

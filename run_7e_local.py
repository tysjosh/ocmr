#!/usr/bin/env python3
"""Run notebook Cell 7e locally: LongMemEval Arm A / oracle.

This mirrors OCM_Colab.ipynb Cell 7e, but uses local filesystem defaults and
requires the cached annotation file by default so a CPU-only machine does not
accidentally start a very slow Qwen annotation pass.

Expected annotation cache:
    local_results/longmemeval_kupdate_annotations.json

Example:
    python run_7e_local.py
    python run_7e_local.py --embeddings deterministic --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


LME_ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_oracle.json"
)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _parse_csv(value))


def _ci(row: dict, key: str) -> str:
    x = row[key]
    return f"{x['mean']:.1f} [{x['low']:.1f},{x['high']:.1f}]"


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run LongMemEval Cell 7e locally using cached annotations."
    )
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional question cap.")
    parser.add_argument("--seeds", default="1337,7,42,99,2024")
    parser.add_argument("--baselines", default="B0,B2,B3")
    parser.add_argument(
        "--embeddings",
        choices=("local", "deterministic"),
        default="local",
        help="Use local sentence-transformers embeddings or cheap deterministic vectors.",
    )
    parser.add_argument(
        "--skip-abstention",
        action="store_true",
        help="Only run knowledge-update, skip the oracle abstention plumbing metric.",
    )
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from ocm.evaluation.datasets.longmemeval_adapter import (
        evaluate_abstention,
        load_longmemeval,
        run_longmemeval_suite,
    )
    from ocm.evaluation.datasets.longmemeval_annotate import load_annotations
    from ocm.retrieval.embeddings import (
        DeterministicEmbeddingProvider,
        LocalEmbeddingProvider,
    )

    data_dir = (args.data_dir or repo_dir / "data").resolve()
    output_dir = (args.output_dir or repo_dir / "local_results").resolve()
    checkpoint_dir = (
        args.checkpoint_dir or output_dir / "checkpoints" / "qwen"
    ).resolve()
    ann_path = (
        args.annotations
        or output_dir / "longmemeval_kupdate_annotations.json"
    ).resolve()
    lme_path = data_dir / "longmemeval_oracle.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _download_if_missing(lme_path, LME_ORACLE_URL)
    print(f"LongMemEval data: {lme_path}")

    if not ann_path.exists():
        raise SystemExit(
            "Missing annotation cache:\n"
            f"  {ann_path}\n\n"
            "Generate Cell 7e annotations once on GPU/Colab, then copy the file "
            "here. This local runner intentionally refuses to start a slow CPU "
            "Qwen annotation pass."
        )

    annotations = load_annotations(str(ann_path))
    print(f"loaded {len(annotations)} cached annotations from {ann_path}")

    if args.embeddings == "deterministic":
        embeddings = DeterministicEmbeddingProvider()
    else:
        embeddings = LocalEmbeddingProvider()

    seeds = _parse_seeds(args.seeds)
    baselines = _parse_csv(args.baselines)

    ku_instances = load_longmemeval(
        str(lme_path), question_type="knowledge-update", limit=args.limit
    )
    print(f"{len(ku_instances)} knowledge-update questions; {len(annotations)} annotated")
    lme_report = run_longmemeval_suite(
        ku_instances,
        annotations,
        baselines=baselines,
        seeds=seeds,
        embeddings=embeddings,
        checkpoint_dir=str(checkpoint_dir),
    )

    print("\n=== LongMemEval knowledge-update decisive metrics (mean [95% CI]) ===")
    print(f"{'Method':<8}{'TaskSuccess up':<22}{'Contradiction dn':<22}{'ConstraintViol dn':<22}")
    for method in lme_report["methods"]:
        row = lme_report["decisive_metrics"][method]
        print(
            f"{method:<8}"
            f"{_ci(row, 'task_success'):<22}"
            f"{_ci(row, 'contradiction_rate'):<22}"
            f"{_ci(row, 'constraint_violations'):<22}"
        )
    print(
        "\nwrite outcomes:",
        {m: lme_report["write_outcomes"][m] for m in lme_report["methods"]},
    )

    abstention = None
    if not args.skip_abstention:
        abst_instances = load_longmemeval(
            str(lme_path), question_type=None, abstention=True, limit=args.limit
        )
        abstention = evaluate_abstention(
            abst_instances,
            baselines=baselines,
            embeddings=embeddings,
        )
        print(f"\nabstention accuracy (n={len(abst_instances)}):", abstention)

    out_path = output_dir / "results_longmemeval.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"knowledge_update": lme_report, "abstention": abstention},
            fh,
            indent=2,
            default=str,
        )
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit a results/checkpoint folder for recorded model provenance.

Different artifacts record their identity in different places, and some record
nothing at all. That asymmetry makes it easy to attribute a metric to the wrong
model, so this script reports what each file *actually* says rather than what a
filename or a directory implies.

Probed locations, in order:

============================================  ==========================================
Location                                      Written by
============================================  ==========================================
``__meta__.fingerprint``                      :class:`~ocm.extraction.caching_extractor.CachingExtractor`
``__meta__.namespace``                        ``CachedChat`` (run_7f_local / notebook 7f)
``_run_manifest.run_identity``                ``run_longmemeval_e2e`` result payloads
``run_identity``                              ``run_manifest.json``
============================================  ==========================================

Anything else — notably :func:`~ocm.evaluation.experiment.run_full_suite`
reports and the ``ms__*`` / ``tau__*`` / ``stress__*`` checkpoints — carries **no
model identity**. Those are reported as ``UNRECORDED``, and the only way to
attribute them is by correlating their mtime with a cache that does record one.
That is circumstantial; the script prints timestamps so the correlation is at
least visible rather than assumed.

Usage::

    python -m ocm.scripts.audit_artifacts /content/drive/MyDrive/ocm_results
    python -m ocm.scripts.audit_artifacts <dir> --max-mb 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Files larger than this are skipped by default (a full dataset dump is not an
#: artifact worth parsing here, and parsing it can take minutes).
DEFAULT_MAX_MB = 25.0

UNRECORDED = "UNRECORDED"


def _iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M")
    )


def _dig(payload: Any, *path: str) -> Optional[Any]:
    """Walk a nested mapping, returning None on any miss."""
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def extract_identity(payload: Any) -> tuple[str, str, dict[str, Any]]:
    """Return ``(model, source, detail)`` for a parsed artifact.

    ``source`` names which provenance slot the model came from, so a reader can
    judge how direct the evidence is.
    """
    probes = (
        ("__meta__.fingerprint", ("__meta__", "fingerprint"), "model"),
        ("__meta__.namespace", ("__meta__", "namespace"), "llm_model"),
        ("_run_manifest.run_identity", ("_run_manifest", "run_identity"), "llm_model"),
        ("run_identity", ("run_identity",), "llm_model"),
    )
    for source, path, model_key in probes:
        block = _dig(payload, *path)
        if isinstance(block, dict):
            model = block.get(model_key)
            if model:
                return str(model), source, block
            # The slot exists but names no model: still worth reporting.
            return UNRECORDED, f"{source} (no {model_key})", block
    return UNRECORDED, "-", {}


def describe(path: Path, max_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    row: dict[str, Any] = {
        "path": path,
        "size": size,
        "mtime": path.stat().st_mtime,
        "model": UNRECORDED,
        "source": "-",
        "detail": {},
        "note": "",
    }
    if size > max_bytes:
        row["note"] = f"skipped (>{max_bytes / 1e6:.0f} MB)"
        return row
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:  # unreadable / not JSON / truncated
        row["note"] = f"unparseable: {type(exc).__name__}"
        return row
    model, source, detail = extract_identity(payload)
    row["model"], row["source"], row["detail"] = model, source, detail
    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, dict):
            row["note"] = f"{len(entries)} cache entries"
        elif "decisive_metrics" in payload:
            methods = payload.get("methods") or []
            seeds = payload.get("seeds") or []
            row["note"] = f"suite report: {len(methods)} arms, seeds={seeds}"
        elif "decisive" in payload:
            row["note"] = "single (method, seed) checkpoint"
    return row


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocm.scripts.audit_artifacts",
        description="Report recorded model provenance for every JSON artifact in a folder.",
    )
    parser.add_argument("directory", type=Path, help="Folder to audit (recursive).")
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_MB,
        help=f"Skip files larger than this (default {DEFAULT_MAX_MB} MB).",
    )
    parser.add_argument(
        "--only-recorded",
        action="store_true",
        help="Show only files that record a model.",
    )
    args = parser.parse_args(argv)

    root = args.directory.expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rows = [
        describe(p, int(args.max_mb * 1e6))
        for p in sorted(root.rglob("*.json"))
        if p.is_file()
    ]
    if args.only_recorded:
        rows = [r for r in rows if r["model"] != UNRECORDED]
    rows.sort(key=lambda r: r["mtime"])

    if not rows:
        print("no JSON artifacts found")
        return 0

    print(f"{len(rows)} artifact(s) under {root}\n")
    width = max(len(str(r["path"].relative_to(root))) for r in rows)
    header = f"{'modified':16}  {'size':>10}  {'model':38}  {'from':30}  file"
    print(header)
    print("-" * (len(header) + width - 4))
    for r in rows:
        rel = r["path"].relative_to(root)
        size = f"{r['size']:,}"
        model = r["model"] if r["model"] != UNRECORDED else "UNRECORDED"
        note = f"  [{r['note']}]" if r["note"] else ""
        print(
            f"{_iso(r['mtime']):16}  {size:>10}  {model:38}  {r['source']:30}  {rel}{note}"
        )

    recorded = [r for r in rows if r["model"] != UNRECORDED]
    models = sorted({r["model"] for r in recorded})
    print()
    print(f"files recording a model : {len(recorded)}/{len(rows)}")
    print(f"distinct models found   : {models or 'none'}")
    if len(models) > 1:
        print(
            "\nWARNING: more than one model appears in this folder. Metrics from "
            "different models must not be combined into one table."
        )
    unrecorded = [r for r in rows if r["model"] == UNRECORDED and not r["note"].startswith("skipped")]
    if unrecorded:
        print(
            f"\n{len(unrecorded)} file(s) record no model. Attribute them only by "
            "mtime proximity to a file above that does, and treat that as "
            "circumstantial."
        )
    for r in recorded:
        detail = {
            k: v
            for k, v in r["detail"].items()
            if k in {"model", "llm_model", "max_new_tokens", "llm_max_tokens",
                     "prompt_sha256", "extract_prompt", "code_revision",
                     "dataset_sha256", "version"}
        }
        if detail:
            print(f"\n{r['path'].relative_to(root)}:")
            for k, v in sorted(detail.items()):
                print(f"    {k:18} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

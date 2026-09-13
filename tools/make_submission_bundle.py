"""Materialise the anonymous submission bundle from the tracked tree.

Builds a filtered copy of the repository for double-blind review. Nothing is
deleted from the working tree: the source of truth is ``git ls-files`` minus the
globs in ``.submission-exclude``.

Two properties this enforces, because both are easy to get wrong by hand:

1. **Only tracked files are copied.** Untracked scratch (``run_with_patch.py``,
   downloaded PDFs, ``local_results/``) never reaches the bundle, so a stray
   file holding results or a machine path cannot leak.
2. **The bundle is checked for identity leaks** before it is declared good.
   Author names, usernames, internal hostnames and home-directory paths are
   grepped for, and the run fails if any survive.

Usage::

    python tools/make_submission_bundle.py --out /tmp/ocmr-anon
    python tools/make_submission_bundle.py --out /tmp/ocmr-anon --verify-tests

``--verify-tests`` runs the bundle's own suite inside the bundle, which is the
only way to be sure an exclusion did not remove something load-bearing.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXCLUDE_FILE = ".submission-exclude"

#: Patterns that must not appear in any bundled file. Author identity, account
#: names, internal hosts, and developer home paths.
IDENTITY_PATTERNS = (
    r"tysjosh",
    r"olukotunjosh",
    r"ojolukot",
    r"olatayo",
    r"vclvm",
    r"ncsu",
    r"/Users/[A-Za-z0-9._-]+",
    r"/home/[A-Za-z0-9._-]+",
)

#: Third-party strings that look like identity but are legitimate citations of
#: the datasets used, and must survive for reproducibility.
IDENTITY_ALLOWLIST = ("xiaowu0162",)


def load_excludes(root: Path) -> list[str]:
    path = root / EXCLUDE_FILE
    if not path.exists():
        raise SystemExit(f"missing {EXCLUDE_FILE}; nothing would be excluded")
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def is_excluded(rel: str, patterns: list[str]) -> str | None:
    """Return the matching pattern, or ``None`` when the path is kept."""
    for pat in patterns:
        # A trailing slash means "this directory and everything under it".
        if pat.endswith("/"):
            if rel.startswith(pat):
                return pat
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat):
            return pat
    return None


def scan_identity(root: Path) -> list[tuple[str, int, str]]:
    """Grep the built bundle for identity leaks. Text files only."""
    rx = re.compile("|".join(IDENTITY_PATTERNS), re.IGNORECASE)
    hits: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = rx.search(line)
            if not m:
                continue
            if any(a in line for a in IDENTITY_ALLOWLIST):
                continue
            hits.append((str(path.relative_to(root)), n, line.strip()[:120]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="Destination directory; created, and overwritten if present.")
    ap.add_argument("--verify-tests", action="store_true",
                    help="Run the bundle's test suite inside the bundle.")
    ap.add_argument("--allow-identity-hits", action="store_true",
                    help="Report identity matches without failing (inspection only).")
    args = ap.parse_args()

    root = Path(
        subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True, check=True).stdout.strip())
    patterns = load_excludes(root)
    tracked = subprocess.run(["git", "ls-files"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.split("\n")
    tracked = [t for t in tracked if t]

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    kept, dropped = [], {}
    for rel in tracked:
        pat = is_excluded(rel, patterns)
        if pat:
            dropped.setdefault(pat, []).append(rel)
            continue
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst)
        kept.append(rel)

    print(f"tracked: {len(tracked)}   kept: {len(kept)}   excluded: {len(tracked) - len(kept)}")
    for pat in sorted(dropped):
        print(f"  -{len(dropped[pat]):3} {pat}")

    hits = scan_identity(args.out)
    print(f"\nidentity scan: {len(hits)} hit(s)")
    for f, n, line in hits[:25]:
        print(f"  {f}:{n}: {line}")
    if hits and not args.allow_identity_hits:
        print("\nFAILED: identity strings present in the bundle.", file=sys.stderr)
        return 1

    if args.verify_tests:
        print("\nrunning the bundle's test suite...")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "ocm/tests", "-q"],
            cwd=args.out, capture_output=True, text=True)
        tail = [l for l in proc.stdout.splitlines() if l.strip()][-6:]
        print("\n".join(f"  {l}" for l in tail))
        if proc.returncode != 0:
            print("\nFAILED: the bundle's own tests do not pass, so an exclusion "
                  "removed something load-bearing.", file=sys.stderr)
            return 1

    print(f"\nbundle -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

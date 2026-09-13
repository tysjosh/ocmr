"""Remove internal spec cross-references ("Req 8.1", "task 15.6") from the tree.

Why not a plain find-and-replace
-------------------------------
Almost every reference is *wrapped* in punctuation that only exists to hold it::

    raise a ``ValidationError`` (Req 1.11). Every status enum...

Deleting the matched text alone leaves ``(.11)``, or an empty ``()``, or a
double space before a full stop. A naive pass over this repository produced 40+
such artifacts across 11 files, including a partial match on ``Req 1`` that left
orphaned ``.11`` and ``.13`` fragments. Each rule here removes the reference
*together with its enclosing construct*, and a tidy pass repairs the punctuation
that legitimately remains.

The IDs point at spec documents no longer in the tree, so they are dangling
cross-references rather than useful provenance.

The ``task`` vocabulary needs more care than ``req``
---------------------------------------------------
"task" is a domain concept in this project, so a loose pattern is destructive:

* ``Task {task}`` and ``Task A`` appear in the synthetic benchmark's generated
  text. These are safe only because the pattern requires a digit and is
  case-sensitive on a lowercase "task".
* ``"published B0 task 77.20 / contrad 14.49"`` in ``summarize_ocmr_arm.py`` is
  a **task_success metric value**, not a reference. It is protected by an
  explicit snippet allowlist rather than by a regex guard: a lookahead on ``/``
  looked like it would work, but it both let the metric through (the ID group
  backtracks to ``77.2``, leaving ``0`` before the separator) and wrongly
  skipped two genuine references -- ``(task 1.3 / API wiring)`` and a
  sentence-final ``task 15.4.``

Safety
------
* Byte-exact round-trip via ``surrogateescape``, so a file containing invalid
  UTF-8 is not silently rewritten.
* Every ``.py`` is re-compiled after rewriting; a failure aborts before writing.
* ``--check`` reports what would change without writing.

Usage::

    python tools/strip_spec_refs.py --check
    python tools/strip_spec_refs.py --kinds task
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: One or more spec IDs: "1.11", "9.x", "23", optionally comma- or dash-joined.
#:
#: The separator class includes the *typographic* dashes. Omitting them is not
#: cosmetic: the tree contains ranges written "Req 19.5-19.6" with an en-dash, so
#: an ASCII-only class matched "Req 19.5" and left a stranded "(-19.6)" behind in
#: seven files.
_IDS = (r"\d+(?:\.(?:\d+|x))?"
        r"(?:\s*[,;&\-\u2013\u2014]\s*\d+(?:\.(?:\d+|x))?)*")

_REQ = rf"[Rr]eq(?:uirement)?s?\.?\s*{_IDS}"
_TASK = rf"tasks?\.?\s*{_IDS}"

#: Literal snippets that must survive untouched. A line containing one is left
#: alone entirely. Used where the surrounding text is indistinguishable from a
#: reference by pattern, so a regex guard would be either leaky or over-broad.
PROTECTED = ("published B0 task",)


def _inline_rules(ref: str, *, bare: bool) -> tuple[tuple[str, str], ...]:
    """Removal rules for one reference vocabulary.

    Ordered: the parenthesised forms must run before the bare one, or the bare
    pattern strips the text and leaves the brackets behind.

    ``bare`` controls whether a reference standing in running prose is removed.
    It must be off for vocabularies whose references are *sentence elements*
    rather than asides. "task" is such a vocabulary::

        belongs to task 15.4.      -> "belongs to."       (broken)
        if task 16.2 has landed    -> "if has landed"     (broken)

    Those need rewording by a human, so they are reported instead of mangled.
    """
    rules = [
        # Sole content of a parenthetical, with any preceding space.
        (rf"[ \t]*\((?:see[ \t]+)?{ref}\)", ""),
        # Trailing item in a larger parenthetical: "(W5 checks, Req 8.1)".
        (rf",[ \t]*(?:see[ \t]+)?{ref}(?=[)\]])", ""),
        # Leading item in a larger parenthetical, comma/semicolon or slash
        # separated: "(Req 8.1; W5 checks)", "(task 1.3 / API wiring)".
        (rf"\((?:see[ \t]+)?{ref}[ \t]*[;,/][ \t]*", "("),
        # Bracketed variant.
        (rf"[ \t]*\[(?:see[ \t]+)?{ref}\]", ""),
        # Dash-attached aside.
        (rf"[ \t]*[-\u2014\u2013]{{1,2}}[ \t]*{ref}(?=[.,;:)\]]|\s|$)", ""),
    ]
    if bare:
        rules.append((rf"[ \t]*\b{ref}\b", ""))
    return tuple(rules)


#: Punctuation repair, applied ONLY to lines a removal changed and only after the
#: leading indent. Applied globally these corrupt valid text: a space-before-colon
#: rule eats the space in a Sphinx role (`` :class:``) and collapsing double
#: spaces destroys reST list-continuation indentation.
TIDY: tuple[tuple[str, str], ...] = (
    (r"\(\s*\)", ""),
    (r"\(\s*[,;]\s*", "("),
    (r"[ \t]+([,.;])", r"\1"),
    (r"[ \t]+\)", ")"),
    (r"\([ \t]+", "("),
    (r"(?<=\S)[ \t]{2,}(?=\S)", " "),
)

#: A line that is only a requirement listing, e.g. "Requirements: 9.x, 11.x."
RX_REQ_LINE = re.compile(
    rf"^[ \t]*(?:#[ \t]*)?(?:\*[ \t]*)?"
    rf"(?:Implements|Validates|Covers|Satisfies)?[ \t]*:?[ \t]*"
    rf"Requirements?:?[ \t]*{_IDS}[ \t]*\.?[ \t]*$",
    re.I,
)

#: (matcher, wrapped-reference fixer, rules, drop listing-only lines)
KINDS = {
    "req": (re.compile(_REQ), re.compile(rf"([Rr]eq(?:uirement)?s?\.?)[ \t]*\r?\n[ \t]*(?={_IDS})"),
            _inline_rules(_REQ, bare=True), True),
    "task": (re.compile(_TASK), re.compile(rf"(tasks?\.?)[ \t]*\r?\n[ \t]*(?={_IDS})"),
             _inline_rules(_TASK, bare=False), False),
}


def transform(text: str, kinds: tuple[str, ...] = ("req", "task")) -> str:
    for kind in kinds:
        text = _transform_one(text, *KINDS[kind])
    return text


def _transform_one(text, rx_any, rx_wrapped, inline_rules, drop_listing_lines):
    text = rx_wrapped.sub(r"\1 ", text)
    out: list[str] = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        eol = raw[len(body):]

        if any(p in body for p in PROTECTED):
            out.append(raw)
            continue
        if drop_listing_lines and RX_REQ_LINE.match(body) and rx_any.search(body):
            continue
        if not rx_any.search(body):
            out.append(raw)
            continue

        indent = body[: len(body) - len(body.lstrip())]
        content = body[len(indent):]
        for pat, repl in inline_rules:
            content = re.sub(pat, repl, content)
        for pat, repl in TIDY:
            content = re.sub(pat, repl, content)
        content = content.rstrip()

        # A wrapped line can *begin* with the reference, stranding its sentence's
        # terminator: "(Req 10.1). Per arm it builds...". The full stop belongs to
        # the previous line's sentence, so move it there.
        orphan = re.match(r"^([.;,:!?])[ \t]+(.*)$", content)
        if orphan:
            punct, rest = orphan.group(1), orphan.group(2)
            for i in range(len(out) - 1, -1, -1):
                prev = out[i].rstrip("\r\n")
                if not prev.strip():
                    continue
                if prev.rstrip()[-1:] not in ".;,:!?":
                    out[i] = prev.rstrip() + punct + out[i][len(prev):]
                break
            content = rest

        if not content or content in {"(", ")", "()", ".", "-", "*", "#", "# "}:
            continue
        out.append(indent + content + eol)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="Report without writing.")
    ap.add_argument("--kinds", default="req,task",
                    help="Comma-separated vocabularies: req, task (default both).")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        ap.error(f"unknown --kinds {bad}; choose from {sorted(KINDS)}")
    matchers = [KINDS[k][0] for k in kinds]

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           check=True).stdout.split()
    planned: list[tuple[str, str, int]] = []
    retained: list[str] = []

    for rel in files:
        path = Path(rel)
        if path.suffix not in (".py", ".md", ".ipynb") or rel.startswith("tools/"):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="surrogateescape")
        except OSError:
            continue
        before = sum(len(rx.findall(src)) for rx in matchers)
        if not before:
            continue
        new = transform(src, kinds)
        leftover = [l for l in new.splitlines()
                    if any(rx.search(l) for rx in matchers)
                    and not any(p in l for p in PROTECTED)]
        if leftover:
            retained.append(f"{rel}: {leftover[0].strip()[:90]}")
        if new == src:
            continue
        if path.suffix == ".py":
            try:
                compile(new, rel, "exec")
            except SyntaxError as exc:
                print(f"ABORT: rewriting {rel} would not compile: {exc}", file=sys.stderr)
                return 1
        planned.append((rel, new, before))

    print(f"{len(planned)} file(s), {sum(n for _, _, n in planned)} reference(s)")
    for rel, _, n in planned[:12]:
        print(f"   {n:4}  {rel}")
    if len(planned) > 12:
        print(f"   ... and {len(planned) - 12} more")
    if retained:
        print(f"\n{len(retained)} line(s) would retain a reference -- inspect:")
        for r in retained[:10]:
            print(f"   {r}")

    if args.check:
        print("\n--check: nothing written")
        return 0
    for rel, new, _ in planned:
        Path(rel).write_text(new, encoding="utf-8", errors="surrogateescape")
    print(f"\nrewrote {len(planned)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

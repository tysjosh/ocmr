"""Remove internal requirement-ID references ("Req 8.1") from the tree.

Why not a plain find-and-replace
-------------------------------
Almost every reference is *wrapped* in punctuation that only exists to hold it::

    raise a ``ValidationError`` (Req 1.11). Every status enum...

Deleting the matched text alone leaves ``(.11)``, or an empty ``()``, or a
double space before a full stop. A naive pass over this repository produced 40+
such artifacts across 11 files, including a partial match on ``Req 1`` that left
orphaned ``.11`` and ``.13`` fragments.

So each pattern here removes the reference *together with its enclosing
construct*, and a final tidy pass repairs the punctuation that legitimately
remains. The IDs point at spec documents that are no longer in the tree, so they
are dangling cross-references rather than useful provenance.

Safety
------
* Byte-exact round-trip via ``surrogateescape``, so a file containing invalid
  UTF-8 is not silently rewritten.
* Every ``.py`` file is re-compiled after rewriting; a failure aborts before
  anything is written to disk.
* ``--check`` reports what would change without writing.

Usage::

    python tools/strip_req_refs.py --check
    python tools/strip_req_refs.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: One or more requirement IDs: "1.11", "9.x", "23", optionally comma/dash joined.
_IDS = r"\d+(?:\.(?:\d+|x))?(?:\s*[,;&-]\s*\d+(?:\.(?:\d+|x))?)*"
_REQ = rf"Req(?:uirement)?s?\.?\s*{_IDS}"

#: Ordered: each pattern consumes the reference *and* the syntax holding it.
#: Order matters -- the parenthesised forms must run before the bare one, or the
#: bare pattern strips the text and leaves the brackets behind.
INLINE_RULES: tuple[tuple[str, str], ...] = (
    # Sole content of a parenthetical, with any preceding space: "foo (Req 8.1)."
    (rf"[ \t]*\((?:see[ \t]+)?{_REQ}\)", ""),
    # Trailing item inside a larger parenthetical: "(W5 checks, Req 8.1)"
    (rf",[ \t]*(?:see[ \t]+)?{_REQ}(?=[)\]])", ""),
    # Leading item inside a larger parenthetical: "(Req 8.1; W5 checks)"
    (rf"\((?:see[ \t]+)?{_REQ}[;,][ \t]*", "("),
    # Bracketed variant.
    (rf"[ \t]*\[(?:see[ \t]+)?{_REQ}\]", ""),
    # Dash-attached aside: "-- Req 8.1" / "— Req 8.1"
    (rf"[ \t]*[-—–]{{1,2}}[ \t]*{_REQ}(?=[.,;:)\]]|\s|$)", ""),
    # Anything left over in prose.
    (rf"[ \t]*\b{_REQ}\b", ""),
)

#: Punctuation repair, applied ONLY to lines a removal actually changed and only
#: to the content after the leading indent. Applying these globally corrupts
#: valid text: ``\s+:`` eats the space in a Sphinx role (`` :class:``), and
#: collapsing double spaces destroys reST list-continuation indentation.
TIDY: tuple[tuple[str, str], ...] = (
    (r"\(\s*\)", ""),            # emptied parenthetical
    (r"\(\s*[,;]\s*", "("),      # leading separator inside parens
    (r"[ \t]+([,.;])", r"\1"),   # space before punctuation (NOT ':' -- see above)
    (r"[ \t]+\)", ")"),          # space before close paren
    (r"\([ \t]+", "("),          # space after open paren
    (r"(?<=\S)[ \t]{2,}(?=\S)", " "),  # runs of spaces *inside* the text only
)

#: A line whose entire content is a requirement listing, e.g.
#: "Requirements: 9.x, 11.x." or "# Implements: Requirements 3.1, 3.2".
RX_REQ_LINE = re.compile(
    rf"^[ \t]*(?:#[ \t]*)?(?:\*[ \t]*)?"
    rf"(?:Implements|Validates|Covers|Satisfies)?[ \t]*:?[ \t]*"
    rf"Requirements?:?[ \t]*{_IDS}[ \t]*\.?[ \t]*$",
    re.I,
)

RX_ANY = re.compile(_REQ, re.I)


#: A reference wrapped across a line break, e.g. "``x = 1.0`` (Req\n    15.4)".
#: Neither line matches on its own -- the first has no digit, the second no
#: "Req" -- so the number is pulled up before the line-by-line pass runs.
RX_WRAPPED = re.compile(
    rf"(Req(?:uirement)?s?\.?)[ \t]*\r?\n[ \t]*(?={_IDS})", re.I)


def transform(text: str) -> str:
    """Strip requirement references line-by-line, tidying only what changed."""
    text = RX_WRAPPED.sub(r"\1 ", text)
    out: list[str] = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        eol = raw[len(body):]

        # A line that is nothing but a requirement listing goes entirely.
        if RX_REQ_LINE.match(body) and RX_ANY.search(body):
            continue

        if not RX_ANY.search(body):
            out.append(raw)
            continue

        indent = body[: len(body) - len(body.lstrip())]
        content = body[len(indent):]
        for pat, repl in INLINE_RULES:
            content = re.sub(pat, repl, content, flags=re.I)
        for pat, repl in TIDY:
            content = re.sub(pat, repl, content)
        content = content.rstrip()

        # A wrapped line can *begin* with the reference, leaving its sentence's
        # terminator stranded:
        #     ...feeds the same examples to every arm
        #     (Req 10.1). Per arm it builds a ``CoreContainer``...
        # The full stop belongs to the sentence on the previous line, so move it
        # there rather than dropping it or leaving ". Per arm..." behind.
        orphan = re.match(r"^([.;,:!?])[ \t]+(.*)$", content)
        if orphan:
            punct, rest = orphan.group(1), orphan.group(2)
            for i in range(len(out) - 1, -1, -1):
                prev = out[i].rstrip("\r\n")
                if not prev.strip():
                    continue
                if prev.rstrip()[-1:] not in ".;,:!?":
                    prev_eol = out[i][len(prev):]
                    out[i] = prev.rstrip() + punct + prev_eol
                break
            content = rest

        # A line reduced to nothing but leftover punctuation (e.g. it held only
        # "(Req 8.1)") is dropped rather than left as a stub.
        if not content or content in {"(", ")", "()", ".", "-", "*", "#", "# "}:
            continue
        out.append(indent + content + eol)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="Report changes without writing.")
    args = ap.parse_args()

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           check=True).stdout.split()
    planned: list[tuple[str, str, int]] = []
    skipped: list[str] = []

    for rel in files:
        path = Path(rel)
        if path.suffix not in (".py", ".md", ".ipynb"):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="surrogateescape")
        except OSError:
            continue
        before = len(RX_ANY.findall(src))
        if not before:
            continue
        new = transform(src)
        if RX_ANY.search(new):
            skipped.append(rel)
        if new == src:
            continue
        if path.suffix == ".py":
            try:
                compile(new, rel, "exec")
            except SyntaxError as exc:
                print(f"ABORT: rewriting {rel} would not compile: {exc}",
                      file=sys.stderr)
                return 1
        planned.append((rel, new, before))

    total = sum(n for _, _, n in planned)
    print(f"{len(planned)} file(s), {total} reference(s)")
    for rel, _, n in planned[:15]:
        print(f"   {n:4}  {rel}")
    if len(planned) > 15:
        print(f"   ... and {len(planned) - 15} more")
    if skipped:
        print(f"\n{len(skipped)} file(s) would retain a reference the rules did "
              f"not match -- inspect these:")
        for rel in skipped[:10]:
            print(f"   {rel}")

    if args.check:
        print("\n--check: nothing written")
        return 0

    for rel, new, _ in planned:
        Path(rel).write_text(new, encoding="utf-8", errors="surrogateescape")
    print(f"\nrewrote {len(planned)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

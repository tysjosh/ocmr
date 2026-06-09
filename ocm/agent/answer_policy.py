"""Agent Answer Policy — P1–P5 (Req 21.1, 21.2, 21.3, 21.4, 21.5).

`Answer_Policy` (`ocm/agent/answer_policy.py`) shapes the agent's final,
human-readable answer from an :class:`EvidencePackage` (the R4 retrieval
contract). It is the last hop in the agent loop (task 16.1) and the toggled
feature that distinguishes baseline B4 from B3 (task 17.x).

The policy is a deterministic, pure transform — same package in, same string
out — so its output can be asserted directly in tests (task 16.3) and compared
across baselines.

Policies
--------
- **P1 — Prefer typed assertions (Req 21.1).** Lead with the accepted,
  high-confidence answer: ``pkg.answer`` when retrieval derived one, otherwise
  the text of the top supporting assertion(s). Raw, unsupported text is never
  promoted ahead of accepted assertions.
- **P2 — Surface conflicts (Req 21.2).** When ``pkg.conflicts`` is non-empty the
  rendered answer calls them out explicitly under a dedicated heading; conflicts
  are never silently dropped.
- **P3 — Keep conflicts separate (Req 21.3).** Each conflicting claim is printed
  as its own labeled line; they are never merged into a single statement.
- **P4 — Include provenance when high-stakes (Req 21.4).** When
  ``high_stakes=True`` (or the output is decision-support) the supporting
  ``source_ref`` provenance is attached to the answer.
- **P5 — State missing evidence (Req 21.5).** When ``pkg.missing_information`` is
  set — or nothing supports the query — the gaps are enumerated rather than
  fabricating an answer.

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5.
"""

from __future__ import annotations

from ocm.retrieval.evidence_packager import EvidencePackage
from ocm.retrieval.reranker import RankedItem

#: Cap on the number of leading supporting assertions surfaced for P1.
DEFAULT_MAX_SUPPORTING = 3


class AnswerPolicy:
    """Render an :class:`EvidencePackage` into a final answer string (P1–P5)."""

    def __init__(self, max_supporting: int = DEFAULT_MAX_SUPPORTING) -> None:
        """Args:
        max_supporting: Maximum supporting assertions surfaced under P1.
        """
        self.max_supporting = max_supporting

    def render(self, pkg: EvidencePackage, high_stakes: bool = False) -> str:
        """Shape ``pkg`` into a deterministic, human-readable answer.

        Args:
            pkg: The :class:`EvidencePackage` produced by the retrieval pipeline.
            high_stakes: When ``True`` (or decision-support) provenance is always
                attached to the answer (P4).

        Returns:
            A human-readable answer string applying policies P1–P5.
        """
        # Index retrieved items by id so supporting assertions (id + confidence)
        # can recover their text/type for display.
        items_by_id: dict[str, RankedItem] = {
            item.memory_id: item for item in (pkg.retrieved_items or [])
        }

        sections: list[str] = []

        # --- P1: lead with accepted high-confidence assertions ------------- #
        sections.append(self._render_answer(pkg, items_by_id))

        # --- P2 / P3: surface conflicts, each kept separate ---------------- #
        conflict_section = self._render_conflicts(pkg)
        if conflict_section:
            sections.append(conflict_section)

        # --- P4: attach provenance when high-stakes / decision-support ----- #
        if high_stakes:
            sections.append(self._render_provenance(pkg))

        # --- P5: state what evidence is missing ---------------------------- #
        missing_section = self._render_missing(pkg)
        if missing_section:
            sections.append(missing_section)

        return "\n\n".join(section for section in sections if section).strip()

    # ------------------------------------------------------------------ #
    # P1 — Prefer typed assertions (Req 21.1)
    # ------------------------------------------------------------------ #
    def _render_answer(
        self, pkg: EvidencePackage, items_by_id: dict[str, RankedItem]
    ) -> str:
        """Lead with the accepted, high-confidence answer (P1)."""
        supporting = pkg.supporting_assertions or []

        # Prefer a derived, typed answer when retrieval produced one.
        if pkg.answer:
            line = f"Answer: {pkg.answer}"
            if supporting:
                line += f" (confidence {pkg.confidence:.2f})"
            return line

        # Otherwise lead with the top supporting assertion(s) — typed/accepted
        # assertions preferred over raw text.
        if supporting:
            lines = [f"Answer (confidence {pkg.confidence:.2f}):"]
            for assertion in supporting[: self.max_supporting]:
                item = items_by_id.get(assertion.id)
                text = (item.text if item and item.text else assertion.id)
                lines.append(f"- {text} [{assertion.id}] (confidence {assertion.confidence:.2f})")
            return "\n".join(lines)

        # No accepted support — defer to P5 (missing evidence). Avoid fabricating.
        return "Answer: No accepted assertions support this query."

    # ------------------------------------------------------------------ #
    # P2 / P3 — Surface conflicts, kept separate (Req 21.2, 21.3)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_conflicts(pkg: EvidencePackage) -> str:
        """Surface each conflict as its own labeled statement (P2, P3)."""
        conflicts = pkg.conflicts or []
        if not conflicts:
            return ""

        lines = ["Conflicts detected — claims are kept separate:"]
        for index, conflict in enumerate(conflicts, start=1):
            descriptor = conflict.text or conflict.reason or "conflicting claim"
            label = f"Claim {index}"
            if conflict.memory_id:
                label += f" [{conflict.memory_id}]"
            detail = descriptor
            if conflict.status:
                detail += f" (status: {conflict.status})"
            if conflict.conflicting_ids:
                detail += f" (conflicts with: {', '.join(conflict.conflicting_ids)})"
            lines.append(f"- {label}: {detail}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # P4 — Include provenance when high-stakes (Req 21.4)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_provenance(pkg: EvidencePackage) -> str:
        """Attach supporting provenance (source_refs) for decision-support (P4)."""
        sources = pkg.supporting_sources or []
        if not sources:
            return "Provenance: none recorded for the supporting assertions."

        lines = ["Provenance:"]
        for source in sources:
            parts = [f"- {source.source_ref}"]
            if source.subject_id:
                parts.append(f"(subject: {source.subject_id})")
            if source.extractor_version:
                parts.append(f"(extractor: {source.extractor_version})")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # P5 — State missing evidence (Req 21.5)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_missing(pkg: EvidencePackage) -> str:
        """Enumerate evidence gaps rather than fabricating (P5)."""
        missing = pkg.missing_information or []
        if not missing:
            return ""
        lines = ["Missing evidence:"]
        for note in missing:
            lines.append(f"- {note}")
        return "\n".join(lines)

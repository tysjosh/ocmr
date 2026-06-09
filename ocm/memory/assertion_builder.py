"""W4 — Assertion Builder.

Turns a normalized relation plus its resolved subject/object entities into a
:class:`CandidateAssertion`, the typed write unit consumed by the rest of the
write pipeline (W5 onward).

Behavior (Req 6):

- The operation is always ``upsert_assertion`` (Req 6.1) — enforced by the
  ``CandidateAssertion`` default, set here explicitly for clarity.
- Populates ``subject_id``, ``predicate``, ``object_id``, ``confidence``,
  ``source_ref``, and ``write_intent`` (Req 6.2).
- Defaults ``write_intent`` to ``new_fact`` when the relation does not specify
  one (Req 6.3).

The expected inputs follow the W3 -> W4 contract: a ``relation`` dict shaped like
``{subject, predicate, object, confidence, write_intent?, source_ref?}`` and a
``resolved`` mapping from each entity reference (the relation's ``subject`` and
``object`` values) to its :class:`ResolutionOutcome` produced by W3.

Requirements: 6.1, 6.2, 6.3.
"""

from __future__ import annotations

from ocm.memory.contracts import CandidateAssertion, ResolutionOutcome
from ocm.ontology.enums import WriteIntent


class AssertionBuilder:
    """Constructs :class:`CandidateAssertion` instances from resolved relations (W4)."""

    def build(
        self,
        relation: dict,
        resolved: dict[str, ResolutionOutcome],
        source_ref: str | None = None,
    ) -> CandidateAssertion:
        """Build a candidate assertion from a normalized relation and its resolved ends.

        Args:
            relation: Normalized relation dict with ``subject``, ``predicate``,
                ``object``, and ``confidence``; may also carry ``write_intent``
                and ``source_ref``.
            resolved: Mapping from entity reference (the relation's ``subject``
                and ``object`` values) to the W3 :class:`ResolutionOutcome`
                holding the resolved ``entity_id``.
            source_ref: Provenance reference for this write. Overrides any
                ``source_ref`` carried in ``relation`` when provided.

        Returns:
            A :class:`CandidateAssertion` with ``operation="upsert_assertion"``
            and all Req 6.2 fields populated.

        Raises:
            KeyError: If ``relation`` is missing ``subject``, ``predicate``, or
                ``object``, or if a referenced end is absent from ``resolved``.
            ValueError: If a resolved end has no ``entity_id`` (unresolved), or
                if no ``source_ref`` is available from either argument.
        """
        subject_ref = relation["subject"]
        object_ref = relation["object"]
        predicate = relation["predicate"]

        subject_id = self._resolved_id(subject_ref, resolved, role="subject")
        object_id = self._resolved_id(object_ref, resolved, role="object")

        effective_source_ref = source_ref if source_ref is not None else relation.get("source_ref")
        if effective_source_ref is None:
            raise ValueError(
                "source_ref is required to build a CandidateAssertion (Req 6.2); "
                "provide it via the source_ref argument or in the relation dict."
            )

        write_intent = self._coerce_write_intent(relation.get("write_intent"))

        return CandidateAssertion(
            operation="upsert_assertion",
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=relation["confidence"],
            source_ref=effective_source_ref,
            write_intent=write_intent,
            valid_from=relation.get("valid_from"),
            valid_to=relation.get("valid_to"),
            extractor_version=relation.get("extractor_version"),
        )

    @staticmethod
    def _resolved_id(ref: str, resolved: dict[str, ResolutionOutcome], *, role: str) -> str:
        """Return the resolved entity id for ``ref`` or raise if unresolved."""
        if ref not in resolved:
            raise KeyError(f"No ResolutionOutcome provided for {role} reference {ref!r}.")
        outcome = resolved[ref]
        if outcome.entity_id is None:
            raise ValueError(
                f"{role.capitalize()} reference {ref!r} is unresolved "
                f"(resolution_status={outcome.resolution_status.value}); cannot build assertion."
            )
        return outcome.entity_id

    @staticmethod
    def _coerce_write_intent(value: object) -> WriteIntent:
        """Coerce a relation's ``write_intent`` to a :class:`WriteIntent`.

        Defaults to ``new_fact`` when unspecified (Req 6.3). A provided string or
        enum is validated against :class:`WriteIntent`.
        """
        if value is None:
            return WriteIntent.new_fact
        if isinstance(value, WriteIntent):
            return value
        return WriteIntent(value)

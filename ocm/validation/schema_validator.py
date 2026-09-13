"""W5 — Schema Validator (structural only).

The :class:`SchemaValidator` is the first validation stage of the write
pipeline (W5), running on each :class:`CandidateAssertion` before the
constraint validator (W6). It performs **structural** checks only and never
reaches into graph-level domain/range reasoning — that is deliberately deferred
to constraint C9 at W6.

Structural checks performed:

1. All required fields are present and non-empty.
2. ``predicate`` is a registered relation predicate — looked up via
   :func:`ocm.ontology.relations.get_relation_signature`.
3. Any ``status`` carried on the candidate is a valid enum value. The
   :class:`CandidateAssertion` contract does not carry a status field, so this
   is validated defensively only when one is present.
4. ``confidence`` is within ``[0, 1]``.
5. ``subject_id`` and ``object_id`` reference existing entity nodes in the
   Graph_Store.
6. The candidate satisfies the **static** registry signature: the
   predicate is registered and its declared signature is internally
   well-formed (non-empty source/target type sets and a known cardinality).
   This explicitly does **not** check the *resolved* subject/object entity
   types against the declared domain/range — that graph-level check is C9 at
   W6.

On the first failed check the validator returns
``ValidationResult(valid=False, failed_check=..., reason=...)`` naming the
failed check, with ``severity=high`` and ``recommended_action="reject"`` because
a structural failure means the candidate is malformed/unusable (design
"Reject" routing). When every check passes it returns
``ValidationResult(valid=True)``.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7.
"""

from __future__ import annotations

from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.memory.graph_store import GraphStore
# ``AssertionStatus`` backs the defensive optional-status check; the
# CandidateAssertion contract has no status field today.
from ocm.ontology.enums import AssertionStatus, Severity
from ocm.ontology.relations import (
    Cardinality,
    RelationSignature,
    UnknownPredicateError,
    get_relation_signature,
)

# Stable check identifiers surfaced in ``ValidationResult.failed_check`` so the
# Commit Manager and research logs can attribute a rejection precisely.
CHECK_REQUIRED_FIELDS = "schema.required_fields"
CHECK_REGISTERED_PREDICATE = "schema.registered_predicate"
CHECK_STATUS_ENUM = "schema.status_enum"
CHECK_CONFIDENCE_BOUNDS = "schema.confidence_bounds"
CHECK_ENTITY_REFERENCES = "schema.entity_references"
CHECK_STATIC_SIGNATURE = "schema.static_signature"

# Required candidate fields that must be present and non-empty.
_REQUIRED_STRING_FIELDS = ("subject_id", "predicate", "object_id", "source_ref")


class SchemaValidator:
    """Performs structural schema validation on a candidate assertion (W5)."""

    def validate(self, c: CandidateAssertion, graph: GraphStore) -> ValidationResult:
        """Validate ``c`` structurally against the schema and registry.

        Returns the first failing check as a failure result, or
        ``ValidationResult(valid=True)`` when all structural checks pass. No
        graph-level domain/range validation is performed here.
        """
        # 1) Required fields present and non-empty.
        result = self._check_required_fields(c)
        if result is not None:
            return result

        # 2) Predicate is a registered relation predicate.
        try:
            signature = get_relation_signature(c.predicate)
        except UnknownPredicateError:
            return self._fail(
                CHECK_REGISTERED_PREDICATE,
                f"Predicate {c.predicate!r} is not a registered relation predicate.",
            )

        # 3) Any status carried on the candidate is a valid enum value.
        result = self._check_status_enum(c)
        if result is not None:
            return result

        # 4) Confidence within [0, 1].
        result = self._check_confidence_bounds(c)
        if result is not None:
            return result

        # 5) subject_id / object_id reference existing entity nodes.
        result = self._check_entity_references(c, graph)
        if result is not None:
            return result

        # 6) Static registry signature is internally well-formed.
        #    Structural only — does NOT check resolved entity types (that is C9).
        result = self._check_static_signature(signature)
        if result is not None:
            return result

        return ValidationResult(valid=True)

    # -- individual structural checks --------------------------------------
    def _check_required_fields(self, c: CandidateAssertion) -> ValidationResult | None:
        """Fail if any required field is missing or blank."""
        missing: list[str] = []
        for field in _REQUIRED_STRING_FIELDS:
            value = getattr(c, field, None)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing.append(field)
        if c.confidence is None:  # type: ignore[comparison-overlap]
            missing.append("confidence")
        if missing:
            return self._fail(
                CHECK_REQUIRED_FIELDS,
                f"Candidate assertion is missing required field(s): {', '.join(missing)}.",
            )
        return None

    def _check_status_enum(self, c: CandidateAssertion) -> ValidationResult | None:
        """Fail if a present ``status`` value is not a valid enum.

        The :class:`CandidateAssertion` contract does not carry a status, so this
        is a defensive guard: it only validates a status when one is actually
        present on the candidate.
        """
        status = getattr(c, "status", None)
        if status is None:
            return None
        if isinstance(status, AssertionStatus):
            return None
        try:
            AssertionStatus(status)
        except ValueError:
            return self._fail(
                CHECK_STATUS_ENUM,
                f"Status {status!r} is not a valid status enumeration value.",
            )
        return None

    def _check_confidence_bounds(self, c: CandidateAssertion) -> ValidationResult | None:
        """Fail if confidence is outside [0, 1]."""
        confidence = float(c.confidence)
        if not (0.0 <= confidence <= 1.0):
            return self._fail(
                CHECK_CONFIDENCE_BOUNDS,
                f"Confidence {confidence} is outside the valid range [0, 1].",
            )
        return None

    def _check_entity_references(
        self, c: CandidateAssertion, graph: GraphStore
    ) -> ValidationResult | None:
        """Fail if subject/object do not reference existing entities."""
        missing: list[str] = []
        if not graph.has_entity(c.subject_id):
            missing.append(f"subject_id={c.subject_id!r}")
        if not graph.has_entity(c.object_id):
            missing.append(f"object_id={c.object_id!r}")
        if missing:
            return self._fail(
                CHECK_ENTITY_REFERENCES,
                "Candidate references non-existent entity ID(s): " + ", ".join(missing) + ".",
            )
        return None

    def _check_static_signature(
        self, signature: RelationSignature
    ) -> ValidationResult | None:
        """Fail if the static registry signature is not well-formed.

        This is a *structural* check on the declared signature itself — the
        predicate is registered (already verified) and the signature declares
        non-empty source/target type sets and a recognized cardinality. It does
        **not** compare the resolved subject/object entity types against the
        declared domain/range; that graph-level validation is constraint C9 at
        W6.
        """
        if not signature.source_types:
            return self._fail(
                CHECK_STATIC_SIGNATURE,
                f"Relation {signature.predicate!r} declares no source types.",
            )
        if not signature.target_types:
            return self._fail(
                CHECK_STATIC_SIGNATURE,
                f"Relation {signature.predicate!r} declares no target types.",
            )
        if not isinstance(signature.cardinality, Cardinality):
            return self._fail(
                CHECK_STATIC_SIGNATURE,
                f"Relation {signature.predicate!r} has an invalid cardinality "
                f"{signature.cardinality!r}.",
            )
        return None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _fail(failed_check: str, reason: str) -> ValidationResult:
        """Build a structural-failure result.

        Structural failures route to ``reject`` (the candidate is
        malformed/unusable) with ``high`` severity.
        """
        return ValidationResult(
            valid=False,
            failed_check=failed_check,
            reason=reason,
            severity=Severity.high,
            recommended_action="reject",
        )

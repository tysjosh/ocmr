"""Property 1: Schema round-trip identity (task 2.2).

Validates: Requirements 1.12, 3.2

For any valid ontology/contract model instance ``x`` (Person, Organization,
Project, Task, Event, Claim, Document, Decision, Assertion, QuarantineRecord,
Provenance), serializing to JSON and parsing it back reconstructs an equal
instance::

    type(x).model_validate_json(x.model_dump_json()) == x

The strategies below build *valid* instances of every ontology model, drawing
enums via ``sampled_from``, confidences via ``floats(0, 1)``, timestamps via
``datetimes()``, JSON-safe text for string/id fields, and JSON-serializable
payloads for ``QuarantineRecord.candidate_payload``. Optional fields are mixed
with ``none()`` so both present and absent variants are exercised.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ocm.ontology.enums import (
    AssertionStatus,
    ClaimStatus,
    DecisionStatus,
    OrgStatus,
    PersonStatus,
    Priority,
    ProjectStatus,
    QuarantineStatus,
    Severity,
    TaskStatus,
    WriteIntent,
)
from ocm.ontology.models import (
    Assertion,
    Claim,
    Decision,
    Document,
    Event,
    Organization,
    Person,
    Project,
    Provenance,
    QuarantineRecord,
    Task,
)
from ocm.tests.markers import pbt_property

# ---------------------------------------------------------------------------
# Shared field strategies.
# ---------------------------------------------------------------------------
# Exclude lone surrogates ("Cs") so every generated string is UTF-8 encodable
# and therefore JSON-serializable without error.
safe_text = st.text(st.characters(blacklist_categories=("Cs",)), max_size=24)

# Confidence stays strictly within the confloat(ge=0, le=1) bound; no NaN/inf
# (those are not valid JSON and are rejected by the confloat constraint anyway).
confidence = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

timestamps = st.datetimes()

string_lists = st.lists(safe_text, max_size=3)

# A JSON-serializable value tree. Dict keys are text only, because JSON object
# keys are always strings; integer keys would survive a JSON round-trip as
# strings and break structural equality.
json_value = st.recursive(
    st.none() | st.booleans() | st.integers() | safe_text,
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(safe_text, children, max_size=3),
    max_leaves=6,
)
json_payload = st.dictionaries(safe_text, json_value, max_size=4)


def optional(strategy: st.SearchStrategy) -> st.SearchStrategy:
    """A field that is either ``None`` or a value from ``strategy``."""
    return st.none() | strategy


# ---------------------------------------------------------------------------
# One Hypothesis strategy per ontology model. Every field is supplied an
# explicit strategy because Pydantic v2's generic ``__init__(**data)`` signature
# carries no per-field type info for ``builds`` to infer from.
# ---------------------------------------------------------------------------
person = st.builds(
    Person,
    id=safe_text,
    name=safe_text,
    roles=string_lists,
    status=st.sampled_from(PersonStatus),
    aliases=string_lists,
    status_defaulted=st.booleans(),
)

organization = st.builds(
    Organization,
    id=safe_text,
    name=safe_text,
    type=safe_text,
    status=st.sampled_from(OrgStatus),
    status_defaulted=st.booleans(),
)

project = st.builds(
    Project,
    id=safe_text,
    name=safe_text,
    goal=optional(safe_text),
    status=st.sampled_from(ProjectStatus),
    owner_id=optional(safe_text),
    status_defaulted=st.booleans(),
)

task = st.builds(
    Task,
    id=safe_text,
    title=safe_text,
    status=st.sampled_from(TaskStatus),
    priority=st.sampled_from(Priority),
    project_id=optional(safe_text),
    assignee_id=optional(safe_text),
    due_at=optional(timestamps),
    status_defaulted=st.booleans(),
)

event = st.builds(
    Event,
    id=safe_text,
    type=safe_text,
    timestamp_start=timestamps,
    timestamp_end=optional(timestamps),
    description=safe_text,
)

claim = st.builds(
    Claim,
    id=safe_text,
    text=safe_text,
    source_ref=safe_text,
    confidence=confidence,
    status=st.sampled_from(ClaimStatus),
    created_at=timestamps,
    status_defaulted=st.booleans(),
)

document = st.builds(
    Document,
    id=safe_text,
    title=safe_text,
    path_or_url=safe_text,
    created_at=timestamps,
    tags=string_lists,
)

decision = st.builds(
    Decision,
    id=safe_text,
    summary=safe_text,
    timestamp=timestamps,
    made_by=optional(safe_text),
    rationale=optional(safe_text),
    status=st.sampled_from(DecisionStatus),
    status_defaulted=st.booleans(),
)

assertion = st.builds(
    Assertion,
    id=safe_text,
    subject_id=safe_text,
    predicate=safe_text,
    object_id=safe_text,
    confidence=confidence,
    status=st.sampled_from(AssertionStatus),
    source_ref=safe_text,
    created_at=timestamps,
    valid_from=optional(timestamps),
    valid_to=optional(timestamps),
    extractor_version=optional(safe_text),
    write_intent=st.sampled_from(WriteIntent),
    status_defaulted=st.booleans(),
)

quarantine_record = st.builds(
    QuarantineRecord,
    id=safe_text,
    candidate_payload=json_payload,
    reason=safe_text,
    severity=st.sampled_from(Severity),
    conflicting_ids=string_lists,
    created_at=timestamps,
    status=st.sampled_from(QuarantineStatus),
)

provenance = st.builds(
    Provenance,
    id=safe_text,
    subject_id=safe_text,
    source_ref=safe_text,
    created_at=timestamps,
    extractor_version=optional(safe_text),
    supporting_evidence_ids=string_lists,
)

MODEL_STRATEGIES = {
    "Person": person,
    "Organization": organization,
    "Project": project,
    "Task": task,
    "Event": event,
    "Claim": claim,
    "Document": document,
    "Decision": decision,
    "Assertion": assertion,
    "QuarantineRecord": quarantine_record,
    "Provenance": provenance,
}


@pbt_property(1, "Schema round-trip identity")
@pytest.mark.parametrize("model_name", list(MODEL_STRATEGIES))
@given(data=st.data())
def test_schema_round_trip_identity(model_name: str, data: st.DataObject) -> None:
    """Every ontology model survives a JSON dump/parse cycle unchanged."""
    x = data.draw(MODEL_STRATEGIES[model_name], label=model_name)
    rebuilt = type(x).model_validate_json(x.model_dump_json())
    assert rebuilt == x
    assert type(rebuilt) is type(x)

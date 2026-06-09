"""Schema validation unit tests for the ontology models (task 2.4).

These tests pin down four behaviors of ``ocm.ontology.models`` that the
ontology layer guarantees:

* Confidence is bounded to ``[0, 1]`` on the confidence-bearing models
  (``Claim``, ``Assertion``) — values ``> 1`` or ``< 0`` raise a
  ``ValidationError`` (Req 1.11; supports C6).
* Out-of-enum ``status`` / ``priority`` / ``severity`` / ``write_intent``
  values raise a ``ValidationError`` (Req 1.11, 26.1).
* Omitting ``status`` defaults it to ``unknown`` and records
  ``status_defaulted=True``; supplying an explicit status leaves
  ``status_defaulted=False`` (Req 1.13, 1.15).
* ``Event`` and ``Document`` have neither a ``status`` nor a
  ``status_defaulted`` field (Req 1.14).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from ocm.ontology.models import (
    Assertion,
    Claim,
    Decision,
    Document,
    Event,
    Organization,
    Person,
    Project,
    QuarantineRecord,
    Task,
)

_NOW = datetime(2024, 1, 1, 12, 0, 0)


def _claim_kwargs(**overrides):
    base = dict(
        id="claim-1",
        text="The sky is blue.",
        source_ref="src-1",
        confidence=0.5,
        created_at=_NOW,
    )
    base.update(overrides)
    return base


def _assertion_kwargs(**overrides):
    base = dict(
        id="assert-1",
        subject_id="ent-1",
        predicate="OWNS",
        object_id="ent-2",
        confidence=0.5,
        source_ref="src-1",
        created_at=_NOW,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Confidence bounds (Req 1.11; supports C6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_confidence", [1.0001, 1.5, 2.0, 42.0])
def test_claim_rejects_confidence_above_one(bad_confidence):
    with pytest.raises(ValidationError):
        Claim(**_claim_kwargs(confidence=bad_confidence))


@pytest.mark.parametrize("bad_confidence", [-0.0001, -0.5, -1.0])
def test_claim_rejects_confidence_below_zero(bad_confidence):
    with pytest.raises(ValidationError):
        Claim(**_claim_kwargs(confidence=bad_confidence))


@pytest.mark.parametrize("bad_confidence", [1.0001, 1.5, 2.0])
def test_assertion_rejects_confidence_above_one(bad_confidence):
    with pytest.raises(ValidationError):
        Assertion(**_assertion_kwargs(confidence=bad_confidence))


@pytest.mark.parametrize("bad_confidence", [-0.0001, -0.5, -1.0])
def test_assertion_rejects_confidence_below_zero(bad_confidence):
    with pytest.raises(ValidationError):
        Assertion(**_assertion_kwargs(confidence=bad_confidence))


@pytest.mark.parametrize("good_confidence", [0.0, 0.25, 0.5, 1.0])
def test_confidence_bounds_accept_in_range(good_confidence):
    assert Claim(**_claim_kwargs(confidence=good_confidence)).confidence == good_confidence
    assert (
        Assertion(**_assertion_kwargs(confidence=good_confidence)).confidence
        == good_confidence
    )


# ---------------------------------------------------------------------------
# Out-of-enum rejection (Req 1.11, 26.1)
# ---------------------------------------------------------------------------
def test_task_rejects_invalid_status():
    with pytest.raises(ValidationError):
        Task(id="task-1", title="T1", status="not_a_status")


def test_task_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        Task(id="task-1", title="T1", priority="super-urgent")


def test_person_rejects_invalid_status():
    with pytest.raises(ValidationError):
        Person(id="p-1", name="Alice", status="busy")


def test_assertion_rejects_invalid_status():
    with pytest.raises(ValidationError):
        Assertion(**_assertion_kwargs(status="maybe"))


def test_assertion_rejects_invalid_write_intent():
    with pytest.raises(ValidationError):
        Assertion(**_assertion_kwargs(write_intent="guess"))


def test_quarantine_record_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        QuarantineRecord(
            id="q-1",
            candidate_payload={},
            reason="conflict",
            severity="catastrophic",
            created_at=_NOW,
        )


# ---------------------------------------------------------------------------
# Status defaulting + status_defaulted metadata (Req 1.13, 1.15)
# ---------------------------------------------------------------------------
def test_task_defaults_status_to_unknown_and_flags_defaulted():
    task = Task(id="task-1", title="T1")
    assert task.status.value == "unknown"
    assert task.status_defaulted is True


def test_task_explicit_status_keeps_defaulted_false():
    task = Task(id="task-1", title="T1", status="todo")
    assert task.status.value == "todo"
    assert task.status_defaulted is False


def test_none_status_is_treated_as_defaulted():
    task = Task(id="task-1", title="T1", status=None)
    assert task.status.value == "unknown"
    assert task.status_defaulted is True


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (Person, dict(id="p-1", name="Alice")),
        (Organization, dict(id="o-1", name="Acme", type="company")),
        (Project, dict(id="pr-1", name="Orion")),
        (Decision, dict(id="d-1", summary="Ship it", timestamp=_NOW)),
        (Claim, _claim_kwargs()),
        (Assertion, _assertion_kwargs()),
    ],
)
def test_status_defaulting_applies_across_status_bearing_models(model, kwargs):
    instance = model(**kwargs)
    assert instance.status.value == "unknown"
    assert instance.status_defaulted is True


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (Person, dict(id="p-1", name="Alice", status="active")),
        (Project, dict(id="pr-1", name="Orion", status="active")),
        (Claim, _claim_kwargs(status="accepted")),
        (Assertion, _assertion_kwargs(status="accepted")),
    ],
)
def test_explicit_status_keeps_defaulted_false_across_models(model, kwargs):
    instance = model(**kwargs)
    assert instance.status_defaulted is False


# ---------------------------------------------------------------------------
# Event and Document have no status / status_defaulted fields (Req 1.14)
# ---------------------------------------------------------------------------
def test_event_has_no_status_fields():
    fields = set(Event.model_fields)
    assert "status" not in fields
    assert "status_defaulted" not in fields


def test_document_has_no_status_fields():
    fields = set(Document.model_fields)
    assert "status" not in fields
    assert "status_defaulted" not in fields


def test_event_constructs_without_status():
    event = Event(
        id="e-1",
        type="completion",
        timestamp_start=_NOW,
        description="Task completed",
    )
    assert not hasattr(event, "status")
    assert not hasattr(event, "status_defaulted")


def test_document_constructs_without_status():
    doc = Document(
        id="doc-1",
        title="Spec",
        path_or_url="/tmp/spec.md",
        created_at=_NOW,
    )
    assert not hasattr(doc, "status")
    assert not hasattr(doc, "status_defaulted")

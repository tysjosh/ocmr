"""Unit tests for the Quarantine Store and Provenance Tracker (task 8.1).

Validates: Requirements 11.7, 12.1, 12.4

These example-based tests exercise the two stores end to end against an
in-memory :class:`SQLiteRepository`:

* ``QuarantineStore.add/list/set_status`` round-trips through the
  ``quarantine_records`` table; records persist for a fresh store built on the
  same repository (the restart guarantee, Req 11.7).
* ``ProvenanceTracker.record/for_subject`` round-trips through the
  ``provenance`` table (Req 12.4), recording ``source_ref`` / ``created_at`` /
  ``extractor_version`` / ``supporting_evidence_ids`` (Req 12.1), and preserves
  multiple provenance rows per subject (supersession case, Req 12.3).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ocm.core.ids import IdGenerator
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.ontology.enums import QuarantineStatus, Severity

TS = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture
def repo():
    """A hermetic in-memory repository, closed after each test."""
    r = SQLiteRepository(":memory:")
    yield r
    r.close()


@pytest.fixture
def ids():
    """Deterministic id generator so generated ids are reproducible."""
    return IdGenerator(deterministic=True)


# --- QuarantineStore ------------------------------------------------------
def test_quarantine_add_returns_persisted_record(repo, ids):
    store = QuarantineStore(repo, ids)
    record = store.add(
        candidate_payload={"subject_id": "a", "predicate": "OWNS", "object_id": "b"},
        reason="high-confidence contradiction",
        severity=Severity.high,
        conflicting_ids=["asr_1"],
        created_at=TS,
    )

    assert record.id
    assert record.reason == "high-confidence contradiction"
    assert record.severity is Severity.high
    assert record.conflicting_ids == ["asr_1"]
    assert record.status is QuarantineStatus.unresolved


def test_quarantine_add_list_roundtrip(repo, ids):
    store = QuarantineStore(repo, ids)
    record = store.add(
        candidate_payload={"k": "v"},
        reason="needs review",
        severity=Severity.medium,
        created_at=TS,
    )

    listed = store.list()
    assert len(listed) == 1
    assert listed[0] == record


def test_quarantine_list_filters_by_status(repo, ids):
    store = QuarantineStore(repo, ids)
    unresolved = store.add(
        candidate_payload={"k": 1}, reason="r1", severity=Severity.low, created_at=TS
    )
    resolved = store.add(
        candidate_payload={"k": 2},
        reason="r2",
        severity=Severity.low,
        created_at=TS,
        status=QuarantineStatus.resolved,
    )

    only_unresolved = store.list(status=QuarantineStatus.unresolved)
    assert [r.id for r in only_unresolved] == [unresolved.id]

    # Accepts the raw string value too.
    only_resolved = store.list(status="resolved")
    assert [r.id for r in only_resolved] == [resolved.id]


def test_quarantine_set_status_updates_record(repo, ids):
    store = QuarantineStore(repo, ids)
    record = store.add(
        candidate_payload={"k": 1}, reason="r", severity=Severity.high, created_at=TS
    )

    updated = store.set_status(record.id, QuarantineStatus.dismissed)
    assert updated.status is QuarantineStatus.dismissed

    reloaded = store.list()
    assert len(reloaded) == 1
    assert reloaded[0].status is QuarantineStatus.dismissed


def test_quarantine_set_status_unknown_id_raises(repo, ids):
    store = QuarantineStore(repo, ids)
    with pytest.raises(KeyError):
        store.set_status("does-not-exist", QuarantineStatus.resolved)


def test_quarantine_persists_across_restarts(repo, ids):
    """A fresh store on the same repo sees previously persisted records (Req 11.7)."""
    QuarantineStore(repo, ids).add(
        candidate_payload={"k": 1},
        reason="persisted",
        severity=Severity.high,
        created_at=TS,
    )

    # Simulate a restart: a brand-new store over the same durable repository.
    fresh = QuarantineStore(repo)
    listed = fresh.list()
    assert len(listed) == 1
    assert listed[0].reason == "persisted"


# --- ProvenanceTracker ----------------------------------------------------
def test_provenance_record_returns_persisted(repo, ids):
    tracker = ProvenanceTracker(repo, ids)
    prov = tracker.record(
        subject_id="asr_1",
        source_ref="doc://notes#1",
        created_at=TS,
        extractor_version="mock-1",
        supporting_evidence_ids=["ev_1", "ev_2"],
    )

    assert prov.id
    assert prov.subject_id == "asr_1"
    assert prov.source_ref == "doc://notes#1"
    assert prov.extractor_version == "mock-1"
    assert prov.supporting_evidence_ids == ["ev_1", "ev_2"]


def test_provenance_record_for_subject_roundtrip(repo, ids):
    tracker = ProvenanceTracker(repo, ids)
    prov = tracker.record(subject_id="asr_1", source_ref="src", created_at=TS)

    fetched = tracker.for_subject("asr_1")
    assert fetched == [prov]


def test_provenance_optional_fields_default(repo, ids):
    tracker = ProvenanceTracker(repo, ids)
    prov = tracker.record(subject_id="claim_1", source_ref="src", created_at=TS)
    assert prov.extractor_version is None
    assert prov.supporting_evidence_ids == []


def test_provenance_preserves_multiple_records_per_subject(repo, ids):
    """Supersession keeps provenance for both old and new (Req 12.3)."""
    tracker = ProvenanceTracker(repo, ids)
    tracker.record(subject_id="asr_old", source_ref="src-old", created_at=TS)
    tracker.record(subject_id="asr_new", source_ref="src-new", created_at=TS)

    assert len(tracker.for_subject("asr_old")) == 1
    assert len(tracker.for_subject("asr_new")) == 1


def test_provenance_for_unknown_subject_is_empty(repo, ids):
    tracker = ProvenanceTracker(repo, ids)
    assert tracker.for_subject("nope") == []

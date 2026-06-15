"""Tests for the governed-write replay diagnostic (qualitative evidence dump)."""

from __future__ import annotations

import json

from ocm.evaluation.replay_governed_writes import replay_governed_writes


def test_replay_produces_real_bucketed_outcomes():
    report = replay_governed_writes(per_category=4, seeds=(1337,), verbose=False)
    totals = report["totals"]
    # The full governed arm must accept the bulk and quarantine the conflicts
    # the benchmark plants (single-valued ASSIGNED_TO + illegal status flips).
    assert totals["accepted"] > 0
    assert totals["quarantined"] > 0
    assert set(report["samples"]) == {"accepted", "superseded", "quarantined", "rejected"}


def test_quarantine_rows_carry_reason_and_provenance():
    report = replay_governed_writes(per_category=4, seeds=(1337,), verbose=False)
    q_rows = report["samples"]["quarantined"]
    assert q_rows, "expected at least one quarantine example"
    row = q_rows[0]
    # Every row is grounded in a real session + a governing reason.
    assert row["reason"]
    assert row["session_text"]
    assert row["source_ref"].startswith(row["example_id"])
    assert "-[" in row["triple"]  # readable subject -[predicate]-> object


def test_reason_histogram_and_false_quarantine_shape():
    report = replay_governed_writes(per_category=4, seeds=(1337,), verbose=False)
    hist = report["quarantine_reason_histogram"]
    assert sum(hist.values()) == report["totals"]["quarantined"]
    # False quarantines are a subset of quarantines, each flagged with a note.
    for fq in report["false_quarantine"]:
        assert fq["decision"] == "quarantined"
        assert "note" in fq


def test_replay_writes_json_report(tmp_path):
    out = tmp_path / "governance_examples.json"
    report = replay_governed_writes(
        per_category=2, seeds=(1337,), out_path=str(out), verbose=False
    )
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["totals"] == report["totals"]
    assert on_disk["method"] == "B3"


def test_replay_is_deterministic_for_a_seed():
    a = replay_governed_writes(per_category=3, seeds=(1337,), verbose=False)
    b = replay_governed_writes(per_category=3, seeds=(1337,), verbose=False)
    assert a["totals"] == b["totals"]
    assert a["quarantine_reason_histogram"] == b["quarantine_reason_histogram"]


def test_per_seed_totals_sum_to_aggregate():
    report = replay_governed_writes(
        per_category=4, seeds=(1337, 7), max_rows_per_bucket=0, verbose=False
    )
    ps = report["per_seed_totals"]
    assert [p["seed"] for p in ps] == [1337, 7]
    for bucket in ("accepted", "superseded", "quarantined", "rejected"):
        assert sum(p[bucket] for p in ps) == report["totals"][bucket]
    assert sum(p["false_quarantine"] for p in ps) == report["false_quarantine_total"]


def test_isolation_removes_cross_example_false_quarantines():
    # Cross-example identifier reuse (shared store) drives most false
    # quarantines; isolating each example in its own store must drive the
    # within-example false-quarantine count to zero, while accepting more.
    shared = replay_governed_writes(
        per_category=8, seeds=(1337,), isolate_per_example=False,
        max_rows_per_bucket=0, verbose=False,
    )
    isolated = replay_governed_writes(
        per_category=8, seeds=(1337,), isolate_per_example=True,
        max_rows_per_bucket=0, verbose=False,
    )
    assert shared["false_quarantine_total"] > 0
    assert isolated["false_quarantine_total"] == 0
    assert isolated["totals"]["accepted"] > shared["totals"]["accepted"]
    assert isolated["isolate_per_example"] is True

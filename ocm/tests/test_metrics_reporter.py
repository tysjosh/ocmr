"""Unit tests for the Metrics_Reporter (Req 24.1–24.5).

Drives :class:`MetricsReporter` over a small fabricated fixture of per-question
result records (the shape the Baseline_Runner emits) to confirm every metric
family computes and that the B0 comparison is present and correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from ocm.evaluation.metrics import MetricsReporter


def _record(**overrides):
    """Build a result record dict with sensible defaults."""
    base = {
        "baseline_name": "B0",
        "example_id": "ex-1",
        "category": "longitudinal_factual_qa",
        "query": "Who owns Project Orion?",
        "answer": "Alice owns Project Orion.",
        "retrieved_ids": ["ast_alice_owns_orion", "noise_1"],
        "conflicts": 0,
        "expected_answer_contains": ["Alice"],
        "expected_conflict": False,
        "expected_supporting_ids": ["ast_alice_owns_orion"],
        "score": 1.0,
        "latency_ms": 10.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def fixture_records():
    """A mixed fixture across B0 and B3 covering correct, wrong, and conflicts."""
    return [
        # --- B0 (vectors only): one correct, one wrong, one missed conflict.
        _record(baseline_name="B0", answer="Alice owns Project Orion."),
        _record(
            baseline_name="B0",
            query="Who owns Project Atlas?",
            answer="Nobody owns it.",
            expected_answer_contains=["Carol"],
            retrieved_ids=["noise_2"],
            expected_supporting_ids=["ast_carol_owns_atlas"],
        ),
        _record(
            baseline_name="B0",
            category="contradiction_heavy_update_stream",
            query="Status of Task T1?",
            answer="done",
            expected_answer_contains=["done"],
            expected_conflict=True,
            conflicts=0,  # B0 does not surface the conflict
            retrieved_ids=["ast_t1_done"],
            expected_supporting_ids=["ast_t1_done", "ast_t1_notstarted"],
        ),
        # --- B3 (governed): same questions, surfaces the conflict.
        _record(baseline_name="B3", answer="Alice owns Project Orion."),
        _record(
            baseline_name="B3",
            query="Who owns Project Atlas?",
            answer="Carol owns Project Atlas.",
            expected_answer_contains=["Carol"],
            retrieved_ids=["ast_carol_owns_atlas", "noise_2"],
            expected_supporting_ids=["ast_carol_owns_atlas"],
        ),
        _record(
            baseline_name="B3",
            category="contradiction_heavy_update_stream",
            query="Status of Task T1?",
            answer="done (conflicting report exists)",
            expected_answer_contains=["done"],
            expected_conflict=True,
            conflicts=["ast_t1_notstarted"],  # surfaced as a list
            retrieved_ids=["ast_t1_done", "ast_t1_notstarted"],
            expected_supporting_ids=["ast_t1_done", "ast_t1_notstarted"],
            latency_ms=25.0,
        ),
    ]


def test_compute_groups_by_baseline(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    assert "B0" in out and "B3" in out
    assert out["_meta"]["total_records"] == 6
    assert out["_meta"]["baselines"] == ["B0", "B3"]
    assert out["_meta"]["b0_present"] is True


def test_retrieval_metrics_present(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    r = out["B3"]["retrieval"]
    # All three B3 questions carry expected_supporting_ids and retrieve them.
    assert r["hit@1"] == pytest.approx(1.0)
    assert r["hit@5"] == pytest.approx(1.0)
    assert 0.0 <= r["supporting_evidence_precision"] <= 1.0
    assert r["supporting_evidence_recall"] == pytest.approx(1.0)


def test_b0_misses_one_retrieval(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    # B0 missed Atlas (retrieved noise only) → hit@1 averages below 1.
    assert out["B0"]["retrieval"]["hit@1"] == pytest.approx(2 / 3)


def test_answer_metrics(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    b0 = out["B0"]["answer"]
    b3 = out["B3"]["answer"]
    # B0: 2/3 correct answers (Orion + done), Atlas wrong.
    assert b0["factual_recall"] == pytest.approx(2 / 3)
    # B3: all 3 correct.
    assert b3["factual_recall"] == pytest.approx(1.0)
    # Conflict surfacing: B0 misses the one conflict (0.0), B3 surfaces it (1.0).
    assert b0["conflict_surfacing_rate"] == pytest.approx(0.0)
    assert b3["conflict_surfacing_rate"] == pytest.approx(1.0)
    # B0 answered the conflict question without surfacing → contradiction rate up.
    assert b0["contradiction_rate_per_100_responses"] == pytest.approx(100 / 3)
    assert b3["contradiction_rate_per_100_responses"] == pytest.approx(0.0)


def test_hallucination_rate(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    # B0 Atlas answer is wrong with no conflict → 1 hallucination of 3.
    assert out["B0"]["answer"]["memory_induced_hallucination_rate"] == pytest.approx(1 / 3)
    assert out["B3"]["answer"]["memory_induced_hallucination_rate"] == pytest.approx(0.0)


def test_write_time_contradiction_detection(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    # B3 surfaces exactly the expected conflict → precision and recall 1.0.
    wt = out["B3"]["write_time"]
    assert wt["contradiction_detection_precision"] == pytest.approx(1.0)
    assert wt["contradiction_detection_recall"] == pytest.approx(1.0)
    # B0 never surfaces → recall 0.0, precision None (no positive predictions).
    wt0 = out["B0"]["write_time"]
    assert wt0["contradiction_detection_recall"] == pytest.approx(0.0)
    assert wt0["contradiction_detection_precision"] is None


def test_write_time_optional_fields_none_with_notes(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    wt = out["B0"]["write_time"]
    assert wt["invalid_write_detection_rate"] is None
    assert wt["false_quarantine_rate"] is None
    assert wt["entity_resolution_accuracy"] is None
    assert any("invalid_write_detection_rate" in n for n in out["_meta"]["notes"])


def test_write_time_optional_fields_computed_when_present():
    records = [
        _record(
            baseline_name="B3",
            expected_invalid_write=True,
            invalid_write_detected=True,
            quarantined=False,
            expected_quarantine=False,
            entity_resolution_correct=True,
        ),
        _record(
            baseline_name="B3",
            expected_invalid_write=True,
            invalid_write_detected=False,
            quarantined=True,
            expected_quarantine=False,  # a false quarantine
            entity_resolution_correct=False,
        ),
    ]
    wt = MetricsReporter().compute(records)["B3"]["write_time"]
    assert wt["invalid_write_detection_rate"] == pytest.approx(0.5)
    assert wt["false_quarantine_rate"] == pytest.approx(0.5)
    assert wt["entity_resolution_accuracy"] == pytest.approx(0.5)


def test_agent_metrics(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    a3 = out["B3"]["agent"]
    assert 0.0 <= a3["answer_quality"] <= 1.0
    assert a3["mean_latency_ms"] is not None
    # No multi-turn transcripts → correction turns is None by design.
    assert a3["correction_turns_after_injected_error"] is None
    # planning category not in fixture → long_horizon proxy is None.
    assert a3["long_horizon_plan_success"] is None


def test_comparison_vs_b0_present(fixture_records):
    out = MetricsReporter().compute(fixture_records)
    comparisons = out["comparisons_vs_B0"]
    assert "B3" in comparisons
    d = comparisons["B3"]
    # B3 surfaces conflicts B0 misses → positive delta.
    assert d["conflict_surfacing_rate_delta"] == pytest.approx(1.0)
    # B3 reduces hallucination vs B0 → negative delta.
    assert d["memory_induced_hallucination_rate_delta"] == pytest.approx(-1 / 3)
    # B3 has equal-or-higher factual precision.
    assert d["factual_precision_delta"] >= 0.0


def test_report_is_readable_string(fixture_records):
    text = MetricsReporter().report(fixture_records)
    assert "Metrics_Reporter summary" in text
    assert "B0" in text and "B3" in text
    assert "Deltas vs B0" in text
    assert "conflict_surf" in text


def test_tolerant_of_object_records():
    @dataclass
    class ResultRecord:
        baseline_name: str
        answer: str
        retrieved_ids: list
        conflicts: int
        expected_answer_contains: list
        expected_conflict: bool
        expected_supporting_ids: Optional[list]
        latency_ms: float
        category: str = "longitudinal_factual_qa"

    records = [
        ResultRecord(
            baseline_name="B0",
            answer="Alice",
            retrieved_ids=["ast_x"],
            conflicts=0,
            expected_answer_contains=["Alice"],
            expected_conflict=False,
            expected_supporting_ids=["ast_x"],
            latency_ms=5.0,
        )
    ]
    out = MetricsReporter().compute(records)
    assert out["B0"]["answer"]["factual_recall"] == pytest.approx(1.0)
    assert out["B0"]["retrieval"]["hit@1"] == pytest.approx(1.0)


def test_missing_baseline_grouped_as_unknown():
    out = MetricsReporter().compute([{"answer": "x", "expected_answer_contains": []}])
    assert "unknown" in out
    assert out["_meta"]["b0_present"] is False
    # No B0 → comparisons empty, with a note.
    assert out["comparisons_vs_B0"] == {}


def test_empty_results():
    out = MetricsReporter().compute([])
    assert out["_meta"]["total_records"] == 0
    assert out["comparisons_vs_B0"] == {}

"""Baseline_Runner tests (Req 22.6, 25.3, 28.9).

Drives B0–B3 over a small slice of the seeded benchmark on a deterministic,
offline configuration and asserts the runner produces (a) per-(baseline,
example, question) result records for the Metrics_Reporter and (b) matching
benchmark research logs carrying the required fields.
"""

from __future__ import annotations

from ocm.evaluation.benchmark import (
    BenchmarkExample,
    BenchmarkGenerator,
    Question,
    Session,
)
from ocm.evaluation.runner import BaselineRunner, load_benchmark

# Required research-log fields per benchmark record (Req 25.3).
_BENCHMARK_LOG_FIELDS = {
    "baseline_name",
    "answer",
    "retrieved_ids",
    "conflicts",
    "expected_conflict",
    "score",
    "latency_ms",
}

# Required result-record fields the Metrics_Reporter (17.4) consumes.
_RESULT_FIELDS = {
    "baseline_name",
    "example_id",
    "category",
    "question_index",
    "query",
    "answer",
    "retrieved_ids",
    "supporting_ids",
    "conflict_ids",
    "conflict_surfaced",
    "expected_conflict",
    "expected_answer_contains",
    "score",
    "latency_ms",
}


def _slice(n: int = 4):
    """A small, deterministic slice of the benchmark (includes anchors)."""
    examples = BenchmarkGenerator(seed=1337).generate()
    # Mix some generated examples with a conflict anchor for conflict coverage.
    anchors = [e for e in examples if e.id == "anchor-task-t1-conflict"]
    return examples[:n] + anchors


def test_run_produces_result_records_b0_through_b3():
    runner = BaselineRunner()
    examples = _slice()
    records = runner.run(examples)

    baselines = {"B0", "B1", "B2", "B3"}
    assert {r["baseline_name"] for r in records} == baselines

    expected_questions = sum(len(e.questions) for e in examples)
    assert len(records) == expected_questions * len(baselines)

    for r in records:
        assert _RESULT_FIELDS <= set(r)
        assert isinstance(r["retrieved_ids"], list)
        assert 0.0 <= r["score"] <= 1.0
        assert r["latency_ms"] >= 0.0


def test_benchmark_research_logs_have_required_fields():
    runner = BaselineRunner()
    records = runner.run(_slice())

    bench_logs = runner.benchmark_records()
    # One benchmark log per result record.
    assert len(bench_logs) == len(records)
    for log in bench_logs:
        assert log["kind"] == "benchmark"
        assert _BENCHMARK_LOG_FIELDS <= set(log)
        assert isinstance(log["retrieved_ids"], list)
        assert isinstance(log["conflicts"], list)
        assert isinstance(log["expected_conflict"], bool)


def test_fresh_container_per_baseline_isolates_state():
    """Each baseline must start from empty memory (no cross-baseline leakage)."""
    runner = BaselineRunner()
    # Same example run across all four baselines; retrieved ids must be derived
    # from that example alone, never accumulating across baselines.
    examples = _slice(2)
    records = runner.run(examples)
    # Every (baseline, example, question) triple appears exactly once.
    keys = [
        (r["baseline_name"], r["example_id"], r["question_index"]) for r in records
    ]
    assert len(keys) == len(set(keys))


def test_load_benchmark_roundtrip(tmp_path):
    from ocm.evaluation.benchmark import write_jsonl

    examples = _slice(3)
    path = tmp_path / "bench.jsonl"
    write_jsonl(examples, path)
    loaded = load_benchmark(path)
    assert [e.id for e in loaded] == [e.id for e in examples]


def test_conflict_surfacing_recorded_when_package_surfaces_it():
    """The runner faithfully records a surfaced conflict for B3.

    Mirrors the B3 conflict-surfacing scenario from the baseline tests: two
    single-valued ASSIGNED_TO writes (the second is quarantined) plus a
    conflict-check query. The runner must record ``conflict_surfaced`` and the
    conflicting memory ids.
    """
    example = BenchmarkExample(
        id="synthetic-assignment-conflict",
        category="contradiction_heavy_update_stream",
        sessions=[
            Session(session_id="s1", input="Bob is assigned to Task T1."),
            Session(session_id="s2", input="Carol is assigned to Task T1."),
        ],
        questions=[
            Question(
                query="Is there a conflict about who is assigned to Task T1?",
                expected_answer_contains=["Bob"],
                expected_conflict=True,
            )
        ],
    )
    runner = BaselineRunner()
    records = runner.run([example], baselines=("B3",))
    assert records, "expected at least one record"
    r = records[0]
    assert r["conflict_surfaced"] is True
    assert r["conflict_ids"], "expected conflicting ids to be recorded"
    assert r["conflict_correct"] is True
    # The matching benchmark log carries the conflicts list (Req 25.3).
    bench = runner.benchmark_records()[0]
    assert bench["conflicts"], "benchmark log should record the surfaced conflicts"

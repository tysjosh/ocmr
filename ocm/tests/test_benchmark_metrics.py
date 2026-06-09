"""Benchmark reproducibility, thresholds, anchors, and a metric cross-check.

This module focuses on the Benchmark_Generator contract (Req 23.3, 23.4, 23.5)
plus a small, focused cross-check that the Metrics_Reporter computes retrieval
hit@k and a B0 comparison over benchmark-shaped result records (Req 24.1, 24.5).

It deliberately does NOT re-test the full metric suite — that lives in
``test_metrics_reporter.py``. Here we only confirm the metrics layer composes
sensibly with benchmark-derived records.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ocm.evaluation.benchmark import (
    CATEGORIES,
    DEFAULT_SEED,
    PER_CATEGORY,
    BenchmarkGenerator,
    generate_jsonl,
    write_jsonl,
)
from ocm.evaluation.metrics import MetricsReporter

# The six curated anchors required by Req 23.4.
EXPECTED_ANCHOR_IDS = {
    "anchor-task-t1-conflict",
    "anchor-joseph-pharaoh",
    "anchor-project-owner-conflict",
    "anchor-inactive-assignee",
    "anchor-final-decision-no-evidence",
    "anchor-temporal-cycle",
}


# --------------------------------------------------------------------------- #
# Reproducibility (Req 23.5)
# --------------------------------------------------------------------------- #
def test_generate_is_reproducible_for_fixed_seed():
    """Two runs at the fixed seed produce identical examples (Req 23.5)."""
    first = BenchmarkGenerator(seed=1337).generate()
    second = BenchmarkGenerator(seed=1337).generate()

    assert len(first) == len(second)
    dumps_first = [ex.model_dump() for ex in first]
    dumps_second = [ex.model_dump() for ex in second]
    assert dumps_first == dumps_second


def test_default_seed_matches_explicit_seed():
    """The default constructor seed equals the documented DEFAULT_SEED (Req 23.5)."""
    assert DEFAULT_SEED == 1337
    default_run = [ex.model_dump() for ex in BenchmarkGenerator().generate()]
    explicit_run = [ex.model_dump() for ex in BenchmarkGenerator(seed=DEFAULT_SEED).generate()]
    assert default_run == explicit_run


def test_different_seed_changes_generated_examples():
    """A different seed perturbs the generated (non-anchor) examples (Req 23.5)."""
    base = BenchmarkGenerator(seed=1337).generate()
    other = BenchmarkGenerator(seed=2024).generate()
    # Anchors are hand-authored and seed-independent, so compare the generated
    # portion only; with different seeds at least one example must differ.
    base_generated = [ex.model_dump() for ex in base if not ex.id.startswith("anchor-")]
    other_generated = [ex.model_dump() for ex in other if not ex.id.startswith("anchor-")]
    assert base_generated != other_generated


def test_write_jsonl_is_byte_identical_across_runs(tmp_path: Path):
    """Writing the seeded benchmark twice yields byte-identical files (Req 23.5)."""
    examples = BenchmarkGenerator(seed=1337).generate()
    path_a = tmp_path / "bench_a.jsonl"
    path_b = tmp_path / "bench_b.jsonl"
    write_jsonl(examples, path_a)
    write_jsonl(examples, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_generate_jsonl_round_trips_to_byte_identical_file(tmp_path: Path):
    """generate_jsonl twice at the same seed is byte-identical (Req 23.5)."""
    path_a = tmp_path / "gen_a.jsonl"
    path_b = tmp_path / "gen_b.jsonl"
    returned = generate_jsonl(path_a, seed=1337)
    generate_jsonl(path_b, seed=1337)
    assert path_a.read_bytes() == path_b.read_bytes()
    # The returned list reflects what was written (line count matches).
    assert len(returned) == len(path_a.read_text().splitlines())


# --------------------------------------------------------------------------- #
# Category / count thresholds (Req 23.3)
# --------------------------------------------------------------------------- #
def test_all_six_categories_present():
    """Every one of the six required categories appears (Req 23.3, 23.2)."""
    examples = BenchmarkGenerator().generate()
    present = {ex.category for ex in examples}
    assert set(CATEGORIES) == present
    assert len(CATEGORIES) == 6


def test_at_least_25_examples_per_category():
    """Each category carries at least 25 examples (Req 23.3)."""
    examples = BenchmarkGenerator().generate()
    counts = Counter(ex.category for ex in examples)
    for category in CATEGORIES:
        assert counts[category] >= 25, f"{category} had only {counts[category]}"
    # The generated-per-category constant also meets the threshold.
    assert PER_CATEGORY >= 25


def test_total_at_least_150_examples():
    """The full dataset has at least 150 examples total (Req 23.3)."""
    examples = BenchmarkGenerator().generate()
    assert len(examples) >= 150


def test_every_example_has_required_fields():
    """Each example carries id, category, sessions, and questions (Req 23.1)."""
    examples = BenchmarkGenerator().generate()
    for ex in examples:
        assert ex.id
        assert ex.category in CATEGORIES
        assert ex.sessions, f"{ex.id} has no sessions"
        for session in ex.sessions:
            assert session.session_id
            assert session.input
        for question in ex.questions:
            assert question.query
            assert isinstance(question.expected_answer_contains, list)
            assert isinstance(question.expected_conflict, bool)


# --------------------------------------------------------------------------- #
# Anchor inclusion (Req 23.4)
# --------------------------------------------------------------------------- #
def test_all_six_anchors_present():
    """The six hand-authored anchors are present exactly once each (Req 23.4)."""
    examples = BenchmarkGenerator().generate()
    anchor_ids = [ex.id for ex in examples if ex.id.startswith("anchor-")]
    assert set(anchor_ids) == EXPECTED_ANCHOR_IDS
    assert len(anchor_ids) == 6
    # No duplicate anchor ids.
    assert len(anchor_ids) == len(set(anchor_ids))


def test_anchors_carry_expected_supporting_ids():
    """Anchor questions provide expected_supporting_ids for retrieval scoring (Req 23.4, 23.6)."""
    examples = BenchmarkGenerator().generate()
    anchors = [ex for ex in examples if ex.id in EXPECTED_ANCHOR_IDS]
    assert anchors, "no anchors found"
    for anchor in anchors:
        assert anchor.questions, f"{anchor.id} has no questions"
        # At least one question on each anchor pins supporting ids.
        assert any(q.expected_supporting_ids for q in anchor.questions), anchor.id


def test_anchor_ids_are_unique_across_full_dataset():
    """All example ids (anchors + generated) are unique (Req 23.4)."""
    examples = BenchmarkGenerator().generate()
    ids = [ex.id for ex in examples]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Metric cross-check over benchmark-shaped records (Req 24.1, 24.5)
# --------------------------------------------------------------------------- #
def _benchmark_result(**overrides):
    """A minimal benchmark-shaped result record for the metrics cross-check."""
    base = {
        "baseline_name": "B0",
        "example_id": "anchor-joseph-pharaoh",
        "category": "longitudinal_factual_qa",
        "query": "Who owns Project Pharaoh?",
        "answer": "Joseph owns Project Pharaoh.",
        "retrieved_ids": ["ast_joseph_owns_pharaoh"],
        "conflicts": 0,
        "expected_answer_contains": ["Joseph"],
        "expected_conflict": False,
        "expected_supporting_ids": ["ast_joseph_owns_pharaoh"],
        "latency_ms": 10.0,
    }
    base.update(overrides)
    return base


def test_metrics_compute_retrieval_hit_at_k_and_b0_comparison():
    """A tiny B0/B3 fixture yields retrieval hit@k and a B0 comparison (Req 24.1, 24.5)."""
    records = [
        # B0 retrieves the supporting id at rank 1 → hit.
        _benchmark_result(baseline_name="B0"),
        # B0 misses the supporting id entirely → not a hit.
        _benchmark_result(
            baseline_name="B0",
            example_id="anchor-project-owner-conflict",
            query="Who owns Project Orion?",
            answer="Unknown.",
            retrieved_ids=["noise"],
            expected_answer_contains=["Alice"],
            expected_supporting_ids=["ast_alice_owns_orion"],
        ),
        # B3 retrieves both supporting ids at the top → hit.
        _benchmark_result(
            baseline_name="B3",
            example_id="anchor-project-owner-conflict",
            query="Who owns Project Orion?",
            answer="Alice owns Project Orion.",
            retrieved_ids=["ast_alice_owns_orion", "ast_carol_owns_orion"],
            expected_answer_contains=["Alice"],
            expected_supporting_ids=["ast_alice_owns_orion"],
        ),
    ]

    out = MetricsReporter().compute(records)

    # Req 24.1: retrieval hit@k computed for each baseline.
    for baseline in ("B0", "B3"):
        retrieval = out[baseline]["retrieval"]
        for key in ("hit@1", "hit@3", "hit@5"):
            assert key in retrieval

    # B0 hits 1 of 2 supporting-id questions at rank 1 → hit@1 == 0.5.
    assert out["B0"]["retrieval"]["hit@1"] == 0.5
    # B3 hits its single supporting-id question → hit@1 == 1.0.
    assert out["B3"]["retrieval"]["hit@1"] == 1.0

    # Req 24.5: comparisons against B0 are present for the other baseline.
    assert out["_meta"]["b0_present"] is True
    assert "B3" in out["comparisons_vs_B0"]
    assert "hit@1_delta" in out["comparisons_vs_B0"]["B3"]
    # B3 retrieves better than B0 on hit@1 → non-negative delta.
    assert out["comparisons_vs_B0"]["B3"]["hit@1_delta"] == 0.5

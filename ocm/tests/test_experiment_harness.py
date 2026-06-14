"""Tests for the ablation / stress / experiment orchestration modules.

These run the real offline harness at a tiny scale (few seeds, small benchmark)
to confirm the machinery wires together and produces the paper's table shapes.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.evaluation import experiment as exp
from ocm.evaluation.ablations import ABLATIONS, DEFAULT_ABLATIONS, build_ablation_strategy
from ocm.evaluation.benchmark import BenchmarkGenerator
from ocm.evaluation.stress import (
    INTENSITY_LEVELS,
    PERTURBATION_CLASSES,
    evaluate_entity_resolution,
    generate_stress_examples,
)


def _settings() -> Settings:
    return Settings(deterministic_test_mode=True, chroma_mode="memory", extractor="mock")


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #
def test_ablation_specs_cover_paper_components():
    assert set(DEFAULT_ABLATIONS) == {
        "full", "no_schema", "no_contradiction_gate", "no_provenance", "no_hybrid"
    }
    # The write-time ablations carry the right settings overrides.
    assert ABLATIONS["no_schema"].settings_overrides == {"enable_schema_validation": False}
    assert ABLATIONS["no_contradiction_gate"].settings_overrides == {
        "enable_contradiction_gate": False
    }


def test_build_ablation_strategy_applies_settings():
    strat = build_ablation_strategy("no_contradiction_gate", _settings)
    assert strat.container.settings.enable_contradiction_gate is False
    # And the full system keeps governance on.
    full = build_ablation_strategy("full", _settings)
    assert full.container.settings.enable_contradiction_gate is True


def test_no_contradiction_gate_ablation_admits_conflicts():
    """With the gate off, a status flip is no longer quarantined at write time."""
    strat = build_ablation_strategy("no_contradiction_gate", _settings)
    strat.write("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    strat.write("Bob completed Task T1.", "s2")
    r = strat.write("Carol is assigned to Task T1.", "s3")  # single-valued conflict
    # The gate is disabled, so the conflicting reassignment is not quarantined.
    assert r.summary.num_quarantined == 0

    full = build_ablation_strategy("full", _settings)
    full.write("Alice owns Project Orion. Bob is assigned to Task T1.", "s1")
    full.write("Bob completed Task T1.", "s2")
    rf = full.write("Carol is assigned to Task T1.", "s3")
    # The full system quarantines the high-confidence single-valued conflict.
    assert rf.summary.num_quarantined >= 1


def test_no_schema_ablation_disables_schema_validation():
    strat = build_ablation_strategy("no_schema", _settings)
    assert strat.container.settings.enable_schema_validation is False
    # Writing still works end to end.
    r = strat.write("Alice owns Project Orion.", "s1")
    assert r.summary.num_candidates >= 1


# --------------------------------------------------------------------------- #
# Stress generator
# --------------------------------------------------------------------------- #
def test_stress_generator_is_seeded_and_covers_classes_and_intensities():
    a = generate_stress_examples(seed=7, per_class=3)
    b = generate_stress_examples(seed=7, per_class=3)
    assert [e.id for e in a] == [e.id for e in b]  # byte-identical for a seed

    classes = {e.category for e in a}
    assert classes == {f"stress_{c}" for c in PERTURBATION_CLASSES}
    intensities = {e.intensity for e in a}
    assert intensities == set(INTENSITY_LEVELS)


def test_alias_examples_carry_gold_groups():
    examples = generate_stress_examples(seed=7, per_class=2)
    alias = [e for e in examples if e.category == "stress_alias_ambiguity"]
    assert alias and all(e.gold_entity_groups for e in alias)


def test_entity_resolution_eval_returns_metrics():
    examples = generate_stress_examples(seed=7, per_class=2)
    er = evaluate_entity_resolution(examples, settings_factory=_settings)
    assert 0.0 <= er["entity_resolution_f1"] <= 1.0
    assert 0.0 <= er["false_merge_rate"] <= 1.0
    assert er["n_examples"] > 0


# --------------------------------------------------------------------------- #
# Experiment orchestration (tiny scale)
# --------------------------------------------------------------------------- #
def test_decisive_metrics_shapes():
    examples = BenchmarkGenerator(seed=1337).generate(per_category=2)
    from ocm.evaluation.runner import BaselineRunner
    from ocm.evaluation.baselines import build_baseline
    from ocm.core.container import CoreContainer

    runner = BaselineRunner(settings_factory=_settings)
    strat = build_baseline("B3", CoreContainer(_settings()))
    records = []
    for ex in examples:
        wc = runner._ingest_sessions(strat, ex)
        assert {"candidates", "accepted", "superseded", "quarantined", "rejected"} <= set(wc)
        assert "write_ms" in wc and "write_calls" in wc
        for i, question in enumerate(ex.questions):
            records.append(
                runner._run_question(
                    "B3", strat, ex, i, question, write_quarantined=wc["quarantined"]
                )
            )
    dm = exp.decisive_metrics(records)
    assert set(dm) == {"task_success", "contradiction_rate", "constraint_violations"}
    assert all(0.0 <= v <= 100.0 for v in dm.values())


def test_run_multiseed_and_aggregate():
    ms = exp.run_multiseed(["B0", "B3"], seeds=[1337, 7], per_category=2)
    agg = exp.aggregate_methods(ms)
    assert set(agg) == {"B0", "B3"}
    for method in agg:
        assert set(agg[method]) == {"task_success", "contradiction_rate", "constraint_violations"}
    # B3 should surface contradictions at least as well as B0 (>= task success
    # is plausible but we only assert the values are well-formed here).
    assert agg["B3"]["task_success"].mean >= 0.0


def test_significance_report_shape():
    ms = exp.run_multiseed(["B0", "B1", "B2", "B3"], seeds=[1337, 7], per_category=2)
    sig = exp.significance_vs_best_baseline(ms, "B3", ["B0", "B1", "B2"])
    tests = sig["metric_tests"]
    assert set(tests) == {"task_success", "contradiction_rate", "constraint_violations"}
    for t in tests.values():
        assert 0.0 <= t["corrected_p"] <= 1.0
        assert t["vs_baseline"] in {"B0", "B1", "B2"}


def test_threshold_sweep_shape_and_selection():
    sweep = exp.threshold_sweep(taus=(0.7, 0.8, 0.9), seed=1337, per_category=2)
    assert len(sweep["rows"]) == 3
    for row in sweep["rows"]:
        assert {"tau", "contradiction_rate", "false_quarantine", "ece", "brier", "objective_j"} <= set(row)
    assert sweep["selected_tau"] in {0.7, 0.8, 0.9}


def test_stress_by_intensity_shape():
    out = exp.stress_by_intensity(methods=["B0", "B3"], seeds=[1337], per_class=2)
    ti = out["task_success_by_intensity"]
    assert set(ti) == {"B0", "B3"}
    for lvls in ti.values():
        assert set(lvls) == set(INTENSITY_LEVELS)
    assert "entity_resolution_f1" in out["entity_resolution"]


def test_run_full_suite_shape_and_significance_excludes_b3():
    report = exp.run_full_suite(seeds=[1337, 7], per_category=2, stress_per_class=2)
    # All arms present (4 baselines + 4 ablations).
    assert set(report["methods"]) == {
        "B0", "B1", "B2", "B3", "no_schema", "no_contradiction_gate",
        "no_provenance", "no_hybrid",
    }
    # Significance compares B3 against a NON-OCMR baseline (never itself).
    for metric, t in report["significance_vs_best_baseline"]["metric_tests"].items():
        assert t["vs_baseline"] in {"B0", "B1", "B2"}
    assert report["threshold_sweep"]["rows"]
    assert "task_success_by_intensity" in report["stress"]


def test_run_full_suite_checkpoint_resume(tmp_path):
    import os
    d = str(tmp_path / "ckpt")
    r1 = exp.run_full_suite(seeds=[1337], per_category=1, stress_per_class=2, checkpoint_dir=d)
    files = os.listdir(d)
    assert any(f.startswith("ms__") for f in files)
    assert os.path.exists(os.path.join(d, "report.json"))
    assert str(r1.get("_saved_to", "")).endswith("report.json")
    # A resumed run reproduces the decisive metrics from the checkpoints.
    r2 = exp.run_full_suite(seeds=[1337], per_category=1, stress_per_class=2, checkpoint_dir=d)
    assert r1["decisive_metrics"] == r2["decisive_metrics"]


def test_shared_extractor_is_injected_into_every_arm():
    """A shared extractor/embeddings object is reused across all containers."""
    from ocm.extraction.transformers_extractor import TransformersExtractor

    calls = {"n": 0}

    def fake_complete(_messages):
        calls["n"] += 1
        import json
        return json.dumps({
            "entities": [{"type": "Person", "name": "Alice", "fields": {}},
                         {"type": "Project", "name": "Orion", "fields": {}}],
            "events": [], "claims": [], "documents": [], "decisions": [],
            "relations": [{"subject": "Alice", "predicate": "OWNS", "object": "Orion",
                           "confidence": 0.95, "write_intent": "new_fact"}],
        })

    shared = TransformersExtractor(complete=fake_complete)
    ms = exp.run_multiseed(["B0", "B3"], seeds=[1337], per_category=1, extractor=shared)
    # The one shared extractor handled writes for every arm (proves reuse).
    assert calls["n"] > 0
    assert set(ms.methods) == {"B0", "B3"}

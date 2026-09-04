"""Evaluation harness.

The experimental arms (baselines B0-B4 and the extended comparison baselines, the
mechanism ablations, and the stress-diagnostic governance arms) live in
:mod:`ocm.evaluation.arms`, which registers all three families in one registry
and builds any of them by name via
:func:`ocm.evaluation.arms.build_arm`. Alongside them: the seeded Benchmark
Generator, the Baseline Runner, and the Metrics Reporter for measuring the
write-time-governance research claim.
"""

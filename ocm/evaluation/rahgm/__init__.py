"""RAHGM evaluation suite — the corpus, metrics, and four experiments of §3–§4.

Modules:

* :mod:`ocm.evaluation.rahgm.corpus` — the 1,500-write, 50-scenario evaluation
  corpus with objective ground truth and the train/dev/canary/test partitions.
* :mod:`ocm.evaluation.rahgm.annotate` — rubric-based annotator simulators and
  Krippendorff's alpha.
* :mod:`ocm.evaluation.rahgm.review_cost` — the explicit reviewer-minutes model.
* :mod:`ocm.evaluation.rahgm.metrics` — eq. (10) and eq. (11).
* :mod:`ocm.evaluation.rahgm.replay` — Experiment 1, the five-arm controlled replay.
* :mod:`ocm.evaluation.rahgm.ablation` — the routing ablation and risk–coverage AUC.
* :mod:`ocm.evaluation.rahgm.analyst` — the simulated analyst.
* :mod:`ocm.evaluation.rahgm.human_study` — Experiment 2 (simulated).
* :mod:`ocm.evaluation.rahgm.adaptation_study` — Experiment 3.
* :mod:`ocm.evaluation.rahgm.end_to_end` — Experiment 4.
* :mod:`ocm.evaluation.rahgm.audit` — the OCMR quarantine audit (§4.1).
* :mod:`ocm.evaluation.rahgm.stats` — random-intercept models and Holm correction.
* :mod:`ocm.evaluation.rahgm.report` — table renderers and the scope note.
* :mod:`ocm.evaluation.rahgm.run_all` — the single-command runner.

Requirements: 9.x, 11.x, 12.x, 13.x, 14.x.
"""

from __future__ import annotations

__all__ = [
    "ablation",
    "adaptation_study",
    "analyst",
    "annotate",
    "audit",
    "corpus",
    "end_to_end",
    "human_study",
    "metrics",
    "replay",
    "report",
    "review_cost",
    "run_all",
    "stats",
]

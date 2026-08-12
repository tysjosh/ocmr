"""RAHGM — Risk-Adaptive Human-Governed Memory.

Adds selective, evidence-based human oversight on top of OCMR's governed write
path. The package is *additive*: it reuses OCMR's constraint checks (W5, C1–C10,
W7) and its ``Commit_Manager``, and inserts a risk-adaptive router at the single
``commit`` seam every write funnels through.

Modules:

* :mod:`ocm.governance.features` — the typed status vector ``f(u)`` and the
  consequence / reversibility / authority rubrics.
* :mod:`ocm.governance.policy` — the transparent escalation score ``z(u)``, the
  monotonic coefficient fit, threshold selection, and the deterministic
  three-tier routing rule ``π(u)``.
* :mod:`ocm.governance.router` — the router and the drop-in
  ``GovernedCommitManager``.
* :mod:`ocm.governance.review_queue` — review items, evidence bundles, the three
  explanation depths, and the review-and-release path OCMR lacked.
* :mod:`ocm.governance.adaptation` — bounded feedback adaptation, the canary
  gate, and policy versioning.
* :mod:`ocm.governance.conditions` — the five experimental governance conditions.

Requirements: 1.x, 2.x, 3.x, 4.x, 5.x, 6.x, 7.x, 8.x, 10.x.
"""

from __future__ import annotations

from ocm.governance.features import (
    FeatureExtractor,
    RiskFeatures,
    Rubric,
    WriteContext,
)
from ocm.governance.policy import (
    MANDATORY_CHECKS,
    EscalationPolicy,
    PolicyParameters,
    RouteGuards,
    ThresholdSelection,
    Tier,
    fit_policy,
    select_thresholds,
)
from ocm.governance.review_queue import (
    Adjudication,
    EvidenceBundle,
    ExplanationDepth,
    ReviewAction,
    ReviewItem,
    ReviewQueue,
    latin_square,
    render_explanation,
)
from ocm.governance.router import (
    GovernedCommitManager,
    RiskAdaptiveRouter,
    RoutingDecision,
)

__all__ = [
    "Adjudication",
    "EscalationPolicy",
    "EvidenceBundle",
    "ExplanationDepth",
    "FeatureExtractor",
    "GovernedCommitManager",
    "MANDATORY_CHECKS",
    "PolicyParameters",
    "ReviewAction",
    "ReviewItem",
    "ReviewQueue",
    "RiskAdaptiveRouter",
    "RiskFeatures",
    "RouteGuards",
    "RoutingDecision",
    "Rubric",
    "ThresholdSelection",
    "Tier",
    "WriteContext",
    "fit_policy",
    "latin_square",
    "render_explanation",
    "select_thresholds",
]

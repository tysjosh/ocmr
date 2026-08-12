"""RAHGM escalation score, coefficient fit, and the three-tier routing rule.

Implements paper §3.3 exactly:

* eq. (3) the transparent escalation score
  ``z(u) = β₀ + βf·f(u) + β_k[k−1]₊ + β_q·q − β_v·v − β_a·a``, ``r(u) = σ(z(u))``;
* eq. (4) the L2-regularized logistic fit over the admissible set ``B``, whose
  sign constraints make the policy monotonic and inspectable;
* eq. (5) threshold selection maximizing ``F₂`` subject to
  ``FN_cons / N_cons ≤ 0.02``;
* eq. (6) the deterministic routing policy ``π(u)``.

Reversibility and authority enter as *displayed discounts*: their coefficients are
constrained nonnegative and subtracted, so a reversible, well-attributed write
needs less confidence to commit autonomously, while an equally uncertain but
irreversible write escalates. Nothing here consults a language model, and the
score is never surfaced as an unexplained confidence value — every
:class:`~ocm.governance.router.RoutingDecision` carries the features and the rule
that fired.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Sequence

from ocm.governance.features import (
    AUTHORITATIVE_FLOOR,
    PASS,
    SCHEMA_CHECK_PREFIX,
    RiskFeatures,
)
from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.ontology.enums import WriteIntent

# --------------------------------------------------------------------------- #
# Immutable governance constants (Req 4.3, 7.4)
# --------------------------------------------------------------------------- #
#: Mandatory constraints. A write failing any of these can never be routed to
#: ``accept``, regardless of its risk score or of any adapted parameter. These
#: identifiers are module constants that the bounded updater has no handle on, so
#: no reachable adaptation can disable them (Req 4.3, 7.4).
MANDATORY_CHECKS = frozenset({"C1", "C2", "C3", "C6", "C9"})

#: Checks whose failure means the write is malformed, prohibited, or otherwise
#: unusable — the ``g(u)`` guard of eq. (6). W5 failures (``schema.*``) are
#: included by prefix.
PROHIBITED_CHECKS = frozenset({"C6", "C9"})

#: Number of feature-vector components (the paper's five typed checks).
N_COMPONENTS = 5

#: Default weight decay ``λ`` for eq. (4).
DEFAULT_LAMBDA = 0.01

#: Missed-consequential-conflict ceiling of eq. (5).
MCR_CEILING = 0.02


class Tier(str, Enum):
    """The bounded routing tiers of eq. (6) — the transition alphabet ``D``."""

    accept = "accept"
    supersede = "supersede"
    review = "review"
    reject = "reject"


#: The four tiers, in the order eq. (6) evaluates them.
TIERS: tuple[Tier, ...] = (Tier.reject, Tier.accept, Tier.supersede, Tier.review)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyParameters:
    """The registered, adaptable parameters of the escalation policy.

    These are the *only* quantities bounded feedback adaptation may change
    (Req 7.1): the eight coefficients of eq. (3) and the two routing thresholds.
    Tier semantics, mandatory constraints, feature encodings, and the rejection
    rule are module constants and are immutable.

    ``beta_v`` and ``beta_a`` are the *displayed discounts*: they are constrained
    nonnegative and enter eq. (3) with a minus sign.
    """

    beta_0: float = -2.2
    beta_f: tuple[float, float, float, float, float] = (1.6, 1.4, 1.2, 1.5, 2.0)
    beta_k: float = 0.9
    beta_q: float = 1.8
    beta_v: float = 1.1
    beta_a: float = 1.3
    tau_l: float = 0.30
    tau_h: float = 0.65
    #: Centering offset for the reversibility discount. ``0.0`` reproduces eq. (3)
    #: exactly and is the default. Setting it to ``0.5`` makes the term
    #: ``−β_v(v − ½)``, so an *irreversible* write gains risk rather than merely
    #: forgoing a discount. Eq. (3) as written is one-sided: because ``β_v ≥ 0``
    #: and ``v ∈ [0,1]``, the term spans ``[−β_v, 0]``, so the least reversible
    #: write is only as risky as a write carrying no reversibility information at
    #: all. This field exists so that asymmetry can be measured rather than
    #: assumed; it is not part of the paper's equation.
    reversibility_center: float = 0.0

    # -- vector interop ----------------------------------------------------
    def coefficient_vector(self) -> tuple[float, ...]:
        """The nine coefficients as a flat vector ``(β₀, βf…, β_k, β_q, β_v, β_a)``."""
        return (self.beta_0, *self.beta_f, self.beta_k, self.beta_q, self.beta_v, self.beta_a)

    @classmethod
    def from_coefficient_vector(
        cls,
        vector: Sequence[float],
        *,
        tau_l: float,
        tau_h: float,
        reversibility_center: float = 0.0,
    ) -> "PolicyParameters":
        """Rebuild parameters from a flat coefficient vector plus thresholds.

        The vector layout is the one :meth:`coefficient_vector` emits:
        ``(β₀, βf₁..βf₅, β_k, β_q, β_v, β_a)``.
        """
        if len(vector) != 10:
            raise ValueError(f"expected 10 coefficients, got {len(vector)}")
        return _params_from_vector(
            vector,
            tau_l=tau_l,
            tau_h=tau_h,
            reversibility_center=reversibility_center,
        )

    def with_thresholds(self, tau_l: float, tau_h: float) -> "PolicyParameters":
        """Return a copy with new thresholds (unprojected)."""
        return replace(self, tau_l=float(tau_l), tau_h=float(tau_h))

    # -- admissible set B (Req 2.3) ---------------------------------------
    def project(self) -> "PolicyParameters":
        """Project onto the admissible set ``B`` of eq. (4).

        Failure, interaction, and consequence coefficients are clamped
        nonnegative, as are the displayed authority and reversibility discounts.
        Thresholds are clamped into ``[0,1]`` with ``τ_l < τ_h`` enforced
        (Req 3.2). The intercept is unconstrained.

        Constraining these signs is what makes the fitted policy monotonic
        (Req 2.5): ``∂r/∂f_i ≥ 0``, ``∂r/∂k ≥ 0``, ``∂r/∂q ≥ 0``,
        ``∂r/∂v ≤ 0``, ``∂r/∂a ≤ 0``.
        """
        tau_l = min(max(self.tau_l, 0.0), 1.0)
        tau_h = min(max(self.tau_h, 0.0), 1.0)
        if tau_h <= tau_l:
            tau_h = min(1.0, tau_l + 1e-3)
            if tau_h <= tau_l:  # tau_l was 1.0
                tau_l = max(0.0, tau_h - 1e-3)
        return PolicyParameters(
            beta_0=float(self.beta_0),
            beta_f=tuple(max(0.0, float(b)) for b in self.beta_f),  # type: ignore[arg-type]
            beta_k=max(0.0, float(self.beta_k)),
            beta_q=max(0.0, float(self.beta_q)),
            beta_v=max(0.0, float(self.beta_v)),
            beta_a=max(0.0, float(self.beta_a)),
            tau_l=tau_l,
            tau_h=tau_h,
            reversibility_center=float(self.reversibility_center),
        )

    @property
    def discount_dominance(self) -> float:
        """How far the displayed discounts can outweigh the consequence term.

        Eq. (3) constrains ``β_v`` and ``β_a`` to be nonnegative but places no
        upper bound on them relative to ``β_q``. When this ratio exceeds 1 a
        well-attributed, cheaply-reversible write can be discounted into
        autonomous commitment even when its consequence is high — which is a
        specification gap rather than a fitting artifact.
        """
        return (self.beta_v + self.beta_a) / self.beta_q if self.beta_q else float("inf")

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view for policy versioning and reports."""
        return {
            "beta_0": self.beta_0,
            "beta_f": list(self.beta_f),
            "beta_k": self.beta_k,
            "beta_q": self.beta_q,
            "beta_v": self.beta_v,
            "beta_a": self.beta_a,
            "tau_l": self.tau_l,
            "tau_h": self.tau_h,
            "reversibility_center": self.reversibility_center,
            "discount_dominance": self.discount_dominance,
        }

    def coefficient_distance(self, other: "PolicyParameters") -> float:
        """Euclidean distance ``‖β − β'‖₂`` over the nine coefficients (Req 7.3)."""
        return math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(self.coefficient_vector(), other.coefficient_vector())
            )
        )


# --------------------------------------------------------------------------- #
# Route guards (eq. 6 predicates)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RouteGuards:
    """The three predicates of eq. (6).

    Attributes:
        g: ``g(u) = 1`` — the write is malformed, prohibited, or unattributed.
        m: ``m(u) = 1`` — a mandatory constraint failed. Immutable (Req 4.3).
        h: ``h(u) = 1`` — an authoritative, temporally resolved, reversible
            correction: authority ≥ 0.90, the temporal relation is resolved, and
            the incumbent assertion remains recoverable (§3.3).
        reasons: Named reasons behind each raised guard, for the review item.
    """

    g: bool = False
    m: bool = False
    h: bool = False
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {"g": self.g, "m": self.m, "h": self.h, "reasons": list(self.reasons)}


def compute_guards(
    candidate: CandidateAssertion,
    vr: ValidationResult,
    features: RiskFeatures,
) -> RouteGuards:
    """Compute ``g(u)``, ``m(u)``, and ``h(u)`` for a candidate write.

    ``g`` covers a W5 structural failure, a prohibited-constraint failure
    (C6 confidence bounds, C9 domain/range), and an unattributed write (blank
    ``source_ref``). ``m`` is the failure of any :data:`MANDATORY_CHECKS` member
    or of W5. ``h`` requires the full authoritative-correction condition of §3.3.
    """
    reasons: list[str] = []
    check = vr.failed_check or ""
    is_w5 = check.startswith(SCHEMA_CHECK_PREFIX)

    unattributed = not (candidate.source_ref or "").strip()
    if unattributed:
        reasons.append("unattributed_write")
    if is_w5:
        reasons.append(f"malformed:{check}")
    if check in PROHIBITED_CHECKS:
        reasons.append(f"prohibited:{check}")
    g = bool(unattributed or is_w5 or check in PROHIBITED_CHECKS)

    m = bool(is_w5 or check in MANDATORY_CHECKS)
    if m and not is_w5:
        reasons.append(f"mandatory:{check}")

    h = bool(
        features.authority >= AUTHORITATIVE_FLOOR
        and features.f_t == PASS
        and features.incumbent_recoverable
        and candidate.write_intent in (WriteIntent.update, WriteIntent.correction)
        and not m
    )
    if h:
        reasons.append("authoritative_correction")

    return RouteGuards(g=g, m=m, h=h, reasons=tuple(reasons))


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
class EscalationPolicy:
    """The transparent escalation score and deterministic routing rule.

    The policy is a pure function of :class:`RiskFeatures` and
    :class:`RouteGuards`; it holds no state beyond its
    :class:`PolicyParameters`, so swapping parameters (as bounded adaptation
    does) cannot change tier semantics.
    """

    def __init__(self, params: PolicyParameters | None = None) -> None:
        """Create a policy, projecting its parameters onto ``B`` (Req 2.3)."""
        self.params = (params or PolicyParameters()).project()

    # -- eq. (3) -----------------------------------------------------------
    def score(self, features: RiskFeatures) -> float:
        """The escalation score ``z(u)`` of eq. (3)."""
        p = self.params
        z = p.beta_0
        for coefficient, component in zip(p.beta_f, features.vector):
            z += coefficient * component
        z += p.beta_k * features.interaction
        z += p.beta_q * features.consequence
        # With the default ``reversibility_center = 0.0`` this is eq. (3) exactly.
        z -= p.beta_v * (features.reversibility - p.reversibility_center)
        z -= p.beta_a * features.authority
        return z

    def risk(self, features: RiskFeatures) -> float:
        """The escalation probability ``r(u) = σ(z(u))``."""
        return _sigmoid(self.score(features))

    # -- eq. (6) -----------------------------------------------------------
    def route(
        self, features: RiskFeatures, guards: RouteGuards
    ) -> tuple[Tier, str, float]:
        """Apply ``π(u)`` and return ``(tier, rule, risk)``.

        The rule string names which clause of eq. (6) fired, so a review item can
        show the decision path rather than a bare score (Req 4.5).
        """
        r = self.risk(features)
        p = self.params
        if guards.g:
            return Tier.reject, "g(u)=1 -> reject", r
        if r < p.tau_l and not guards.m:
            # A write cleared for autonomous commitment that displaces a
            # recoverable incumbent commits as a supersession, not a bare accept.
            # Both are autonomous commitments — the threshold semantics of eq. (6)
            # are unchanged — but on a single-valued predicate "accept" without
            # retiring the incumbent would leave two active values, which is a
            # durable-state violation. Supersession is the transition that commits
            # while retaining the prior assertion and its provenance (§3.3).
            if features.incumbent_recoverable:
                return (
                    Tier.supersede,
                    f"r={r:.3f} < tau_l={p.tau_l:.3f} and m(u)=0 with a recoverable "
                    "incumbent -> supersede (autonomous commit, prior value retained)",
                    r,
                )
            return Tier.accept, f"r={r:.3f} < tau_l={p.tau_l:.3f} and m(u)=0 -> accept", r
        if guards.h and r < p.tau_h:
            return (
                Tier.supersede,
                f"h(u)=1 and r={r:.3f} < tau_h={p.tau_h:.3f} -> supersede",
                r,
            )
        return Tier.review, "otherwise -> review", r

    # -- monotonicity witness (Req 2.5) ------------------------------------
    def is_monotonic(self) -> bool:
        """Whether the parameters satisfy the monotonicity sign constraints."""
        p = self.params
        return (
            all(b >= 0.0 for b in p.beta_f)
            and p.beta_k >= 0.0
            and p.beta_q >= 0.0
            and p.beta_v >= 0.0
            and p.beta_a >= 0.0
        )

    def with_params(self, params: PolicyParameters) -> "EscalationPolicy":
        """Return a new policy with ``params`` (projected onto ``B``)."""
        return EscalationPolicy(params)


# --------------------------------------------------------------------------- #
# Fitting (eq. 4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingSample:
    """One fitting example: features plus the eq. (4) label.

    ``y = 1`` when *autonomous* execution would produce an incorrect transition —
    i.e. when the write genuinely warrants escalation. ``consequential`` and
    ``gold_tier`` are carried through for threshold selection (eq. 5) but are not
    used by the coefficient fit.
    """

    features: RiskFeatures
    label: int
    consequential: bool = False
    gold_tier: Tier | None = None
    weight: float = 1.0


def design_row(features: RiskFeatures) -> tuple[float, ...]:
    """The eq. (3) design row for one write.

    The reversibility and authority entries are **negated** here so the fitted
    coefficient is the displayed nonnegative discount rather than a negative
    weight, keeping the fitted policy directly readable against eq. (3).
    """
    return (
        1.0,
        *features.vector,
        features.interaction,
        features.consequence,
        -features.reversibility,
        -features.authority,
    )


@dataclass(frozen=True)
class FitResult:
    """Outcome of the coefficient fit."""

    params: PolicyParameters
    log_loss: float
    iterations: int
    n_samples: int
    n_positive: int
    monotonic: bool

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "params": self.params.as_dict(),
            "log_loss": self.log_loss,
            "iterations": self.iterations,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "monotonic": self.monotonic,
        }


def fit_policy(
    samples: Sequence[TrainingSample],
    *,
    lam: float = DEFAULT_LAMBDA,
    learning_rate: float = 0.25,
    iterations: int = 4000,
    initial: PolicyParameters | None = None,
    tau_l: float | None = None,
    tau_h: float | None = None,
) -> FitResult:
    """Fit the eq. (3) coefficients by projected-gradient logistic regression.

    Minimizes the L2-regularized negative log-likelihood of eq. (4)

    ``−Σ [y·log r + (1−y)·log(1−r)] + λ‖β‖²``

    subject to ``β ∈ B``: after every gradient step the coefficient vector is
    projected back onto the admissible set, which clamps the failure,
    interaction, consequence, reversibility, and authority coefficients at zero.
    The result is an inspectable monotonic policy (Req 2.3, 2.5).

    The optimizer is deterministic (Req 2.4): full-batch gradients, a fixed
    iteration count, no shuffling, and no dependency beyond the standard library.

    Args:
        samples: Training samples from the training partition.
        lam: Weight decay ``λ``.
        learning_rate: Step size.
        iterations: Number of full-batch steps.
        initial: Optional starting parameters (defaults to the registered prior).
        tau_l: Threshold carried through unchanged (set by :func:`select_thresholds`).
        tau_h: Threshold carried through unchanged.

    Returns:
        A :class:`FitResult` with the fitted, projected parameters.
    """
    samples = list(samples)
    start = (initial or PolicyParameters()).project()
    if not samples:
        return FitResult(
            params=start,
            log_loss=float("nan"),
            iterations=0,
            n_samples=0,
            n_positive=0,
            monotonic=True,
        )

    rows = [design_row(s.features) for s in samples]
    labels = [float(s.label) for s in samples]
    weights = [float(s.weight) for s in samples]
    total_weight = sum(weights) or 1.0

    beta = list(start.coefficient_vector())
    n_features = len(beta)

    for _ in range(iterations):
        gradient = [0.0] * n_features
        for row, y, w in zip(rows, labels, weights):
            z = sum(b * x for b, x in zip(beta, row))
            residual = (_sigmoid(z) - y) * w
            for j in range(n_features):
                gradient[j] += residual * row[j]
        for j in range(n_features):
            gradient[j] = gradient[j] / total_weight
            # Weight decay excludes the intercept.
            if j > 0:
                gradient[j] += 2.0 * lam * beta[j]
            beta[j] -= learning_rate * gradient[j]
        # Projection onto B: every coefficient except the intercept is nonneg.
        for j in range(1, n_features):
            if beta[j] < 0.0:
                beta[j] = 0.0

    params = _params_from_vector(
        beta,
        tau_l=start.tau_l if tau_l is None else tau_l,
        tau_h=start.tau_h if tau_h is None else tau_h,
        reversibility_center=start.reversibility_center,
    ).project()

    policy = EscalationPolicy(params)
    loss = _weighted_log_loss(policy, samples)
    return FitResult(
        params=params,
        log_loss=loss,
        iterations=iterations,
        n_samples=len(samples),
        n_positive=int(sum(labels)),
        monotonic=policy.is_monotonic(),
    )


def _params_from_vector(
    vector: Sequence[float],
    *,
    tau_l: float,
    tau_h: float,
    reversibility_center: float = 0.0,
) -> PolicyParameters:
    """Build :class:`PolicyParameters` from the flat coefficient vector.

    ``reversibility_center`` is carried explicitly because it is a *structural*
    choice, not a fitted coefficient: fitting and bounded adaptation must preserve
    whatever the caller configured rather than silently resetting it to eq. (3).
    """
    return PolicyParameters(
        beta_0=float(vector[0]),
        beta_f=(
            float(vector[1]),
            float(vector[2]),
            float(vector[3]),
            float(vector[4]),
            float(vector[5]),
        ),
        beta_k=float(vector[6]),
        beta_q=float(vector[7]),
        beta_v=float(vector[8]),
        beta_a=float(vector[9]) if len(vector) > 9 else 0.0,
        tau_l=float(tau_l),
        tau_h=float(tau_h),
        reversibility_center=float(reversibility_center),
    )


def _weighted_log_loss(
    policy: EscalationPolicy, samples: Sequence[TrainingSample]
) -> float:
    """Mean weighted logistic loss of ``policy`` on ``samples``."""
    total = 0.0
    weight = 0.0
    for sample in samples:
        r = min(max(policy.risk(sample.features), 1e-12), 1 - 1e-12)
        y = float(sample.label)
        total += -sample.weight * (y * math.log(r) + (1 - y) * math.log(1 - r))
        weight += sample.weight
    return total / (weight or 1.0)


# --------------------------------------------------------------------------- #
# Threshold selection (eq. 5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThresholdSelection:
    """Outcome of the eq. (5) threshold search."""

    tau_l: float
    tau_h: float
    f2: float
    mcr: float
    review_rate: float
    feasible: bool
    n_candidates: int

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "tau_l": self.tau_l,
            "tau_h": self.tau_h,
            "f2": self.f2,
            "mcr": self.mcr,
            "review_rate": self.review_rate,
            "feasible": self.feasible,
            "n_candidates": self.n_candidates,
        }


@dataclass(frozen=True)
class RoutingCase:
    """A development/canary case: features, guards, and ground truth.

    Threshold selection needs the guards because eq. (6) consults ``g``, ``m``,
    and ``h`` before the thresholds, so the review decision is not a bare
    comparison against ``r``.
    """

    features: RiskFeatures
    guards: RouteGuards
    gold_tier: Tier
    consequential: bool = False

    @property
    def review_worthy(self) -> bool:
        """Whether ground truth says a human should have seen this write."""
        return self.gold_tier is Tier.review


def select_thresholds(
    cases: Sequence[RoutingCase],
    params: PolicyParameters,
    *,
    grid: float = 0.01,
    mcr_ceiling: float = MCR_CEILING,
) -> ThresholdSelection:
    """Select ``(τ_l, τ_h)`` per eq. (5) on the development partition.

    Maximizes ``F₂`` of the review decision against gold ``review`` subject to
    ``FN_cons / N_cons ≤ mcr_ceiling``, over the lattice
    ``0 ≤ τ_l < τ_h ≤ 1`` (Req 3.1, 3.2).

    When no lattice point satisfies the constraint the selection falls back to
    the pair minimizing MCR (breaking ties on ``F₂``) and reports
    ``feasible=False`` so the caller can surface the infeasibility rather than
    silently shipping a policy that violates its own safety bound (Req 3.3).
    """
    cases = list(cases)
    if not cases:
        return ThresholdSelection(
            tau_l=params.tau_l,
            tau_h=params.tau_h,
            f2=float("nan"),
            mcr=float("nan"),
            review_rate=float("nan"),
            feasible=False,
            n_candidates=0,
        )

    steps = int(round(1.0 / grid))
    lattice = [round(i * grid, 10) for i in range(steps + 1)]

    best_feasible: tuple[float, float, float, float, float] | None = None
    best_fallback: tuple[float, float, float, float, float] | None = None

    for i, tau_l in enumerate(lattice):
        for tau_h in lattice[i + 1 :]:
            policy = EscalationPolicy(
                PolicyParameters(
                    beta_0=params.beta_0,
                    beta_f=params.beta_f,
                    beta_k=params.beta_k,
                    beta_q=params.beta_q,
                    beta_v=params.beta_v,
                    beta_a=params.beta_a,
                    tau_l=tau_l,
                    tau_h=tau_h,
                )
            )
            f2, mcr, review_rate = _threshold_objective(policy, cases)
            entry = (f2, -mcr, tau_l, tau_h, review_rate)
            if mcr <= mcr_ceiling:
                if best_feasible is None or entry > best_feasible:
                    best_feasible = entry
            fallback_entry = (-mcr, f2, tau_l, tau_h, review_rate)
            if best_fallback is None or fallback_entry > (
                -best_fallback[1],
                best_fallback[0],
                best_fallback[2],
                best_fallback[3],
                best_fallback[4],
            ):
                best_fallback = (f2, mcr, tau_l, tau_h, review_rate)

    if best_feasible is not None:
        f2, neg_mcr, tau_l, tau_h, review_rate = best_feasible
        return ThresholdSelection(
            tau_l=tau_l,
            tau_h=tau_h,
            f2=f2,
            mcr=-neg_mcr,
            review_rate=review_rate,
            feasible=True,
            n_candidates=len(cases),
        )

    assert best_fallback is not None  # non-empty lattice
    f2, mcr, tau_l, tau_h, review_rate = best_fallback
    return ThresholdSelection(
        tau_l=tau_l,
        tau_h=tau_h,
        f2=f2,
        mcr=mcr,
        review_rate=review_rate,
        feasible=False,
        n_candidates=len(cases),
    )


def _threshold_objective(
    policy: EscalationPolicy, cases: Sequence[RoutingCase]
) -> tuple[float, float, float]:
    """Return ``(F₂, MCR, review_rate)`` for a policy over ``cases``."""
    tp = fp = fn = 0
    consequential = 0
    missed_consequential = 0
    escalated = 0

    for case in cases:
        tier, _rule, _r = policy.route(case.features, case.guards)
        escalates = tier is Tier.review
        if escalates:
            escalated += 1
        if case.review_worthy:
            if escalates:
                tp += 1
            else:
                fn += 1
        elif escalates:
            fp += 1

        if case.consequential:
            consequential += 1
            # A miss is a consequential case whose autonomous transition is wrong
            # and which was not escalated (eq. 10 numerator).
            if not escalates and tier is not case.gold_tier:
                missed_consequential += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = 4.0
    denominator = beta2 * precision + recall
    f2 = ((1 + beta2) * precision * recall / denominator) if denominator else 0.0
    mcr = missed_consequential / consequential if consequential else 0.0
    review_rate = escalated / len(cases) if cases else 0.0
    return f2, mcr, review_rate


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def build_training_samples(cases: Iterable[RoutingCase]) -> list[TrainingSample]:
    """Convert routing cases into eq. (4) training samples.

    The label is ``1`` when the *autonomous* transition would be wrong, which the
    corpus encodes as a gold tier of ``review`` (a case a human must adjudicate)
    or ``reject`` (a case autonomous acceptance would corrupt).
    """
    samples: list[TrainingSample] = []
    for case in cases:
        label = 1 if case.gold_tier in (Tier.review, Tier.reject) else 0
        samples.append(
            TrainingSample(
                features=case.features,
                label=label,
                consequential=case.consequential,
                gold_tier=case.gold_tier,
            )
        )
    return samples

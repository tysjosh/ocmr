"""Statistical models for §3.7 — eq. (12) and the Holm correction.

The paper's primary model is a crossed random-intercept logistic regression

``logit Pr(Y_ips = 1) = α + x_ipsᵍ·γ + b_p + c_s``,
``b_p ~ N(0, σ_p²)``, ``c_s ~ N(0, σ_s²)``

with participant and scenario random intercepts. ``statsmodels`` is not a
dependency of this repository, so the model is implemented directly on NumPy and
SciPy:

* :class:`RandomInterceptLogit` maximizes the Laplace approximation to the
  marginal likelihood, with the random-effect mode found by inner Newton
  iterations and the variance components profiled by L-BFGS-B. Standard errors
  come from the profiled observed information.
* :class:`RandomInterceptGaussian` fits the same structure for ``log t`` by exact
  maximum likelihood, using the sparse mixed-model equations rather than forming
  the ``N × N`` covariance.
* :class:`CumulativeLogit` fits a proportional-odds model for ordinal workload
  with cluster-robust (sandwich) standard errors by participant. This is a
  documented simplification: the ordinal outcome uses a cluster-robust fixed-
  effects fit rather than a random-intercept fit.

Requirements: 12.1, 12.2, 12.3, 12.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import optimize, stats

#: Ridge applied to information matrices before inversion, for numerical safety.
_RIDGE = 1e-8


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Coefficient:
    """One fitted fixed effect with inference."""

    name: str
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    ci_low: float
    ci_high: float
    odds_ratio: float | None = None
    odds_ratio_ci: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        out: dict[str, Any] = {
            "name": self.name,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "z": self.z_value,
            "p_value": self.p_value,
            "ci95": [self.ci_low, self.ci_high],
        }
        if self.odds_ratio is not None:
            out["odds_ratio"] = self.odds_ratio
            out["odds_ratio_ci95"] = list(self.odds_ratio_ci or ())
        return out


@dataclass
class ModelFit:
    """A fitted model."""

    model: str
    coefficients: list[Coefficient]
    sigma_participant: float | None = None
    sigma_scenario: float | None = None
    sigma_residual: float | None = None
    log_likelihood: float = float("nan")
    n_obs: int = 0
    n_participants: int = 0
    n_scenarios: int = 0
    converged: bool = True
    notes: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> Coefficient | None:
        """Look up one coefficient."""
        for coefficient in self.coefficients:
            if coefficient.name == name:
                return coefficient
        return None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "model": self.model,
            "coefficients": [c.as_dict() for c in self.coefficients],
            "sigma_participant": self.sigma_participant,
            "sigma_scenario": self.sigma_scenario,
            "sigma_residual": self.sigma_residual,
            "log_likelihood": self.log_likelihood,
            "n_obs": self.n_obs,
            "n_participants": self.n_participants,
            "n_scenarios": self.n_scenarios,
            "converged": self.converged,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Design helpers
# --------------------------------------------------------------------------- #
def _indicator(codes: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    """Map group labels onto integer indices, returning ``(indices, levels)``."""
    levels = sorted({str(c) for c in codes})
    lookup = {level: i for i, level in enumerate(levels)}
    return np.array([lookup[str(c)] for c in codes], dtype=int), levels


def _coefficients(
    names: Sequence[str],
    estimates: np.ndarray,
    covariance: np.ndarray,
    *,
    odds_ratios: bool,
) -> list[Coefficient]:
    """Build coefficient records with Wald inference from a covariance matrix."""
    out: list[Coefficient] = []
    variances = np.clip(np.diag(covariance), 0.0, None)
    for i, name in enumerate(names):
        estimate = float(estimates[i])
        std_error = float(math.sqrt(variances[i])) if variances[i] > 0 else float("nan")
        if std_error and math.isfinite(std_error) and std_error > 0:
            z = estimate / std_error
            p = float(2.0 * stats.norm.sf(abs(z)))
            low = estimate - 1.959963985 * std_error
            high = estimate + 1.959963985 * std_error
        else:
            z, p, low, high = float("nan"), float("nan"), float("nan"), float("nan")
        coefficient = Coefficient(
            name=name,
            estimate=estimate,
            std_error=std_error,
            z_value=z,
            p_value=p,
            ci_low=low,
            ci_high=high,
        )
        if odds_ratios:
            coefficient.odds_ratio = _safe_exp(estimate)
            if math.isfinite(low):
                coefficient.odds_ratio_ci = (_safe_exp(low), _safe_exp(high))
        out.append(coefficient)
    return out


def _safe_exp(value: float) -> float:
    """``exp`` that saturates instead of overflowing.

    A near-separable contrast drives a logit coefficient toward infinity, and the
    corresponding odds ratio is genuinely unbounded. Reporting ``inf`` is the
    honest value; raising ``OverflowError`` would discard an otherwise usable fit.
    """
    if not math.isfinite(value):
        return float("inf") if value > 0 else 0.0
    if value > 700.0:
        return float("inf")
    if value < -700.0:
        return 0.0
    return math.exp(value)


# --------------------------------------------------------------------------- #
# Random-intercept logistic (eq. 12)
# --------------------------------------------------------------------------- #
class RandomInterceptLogit:
    """Crossed random-intercept logistic regression by Laplace approximation.

    The joint penalized log-likelihood in the random effects ``u = (b, c)`` is

    ``ℓ(γ, u) = Σᵢ [yᵢηᵢ − log(1 + e^{ηᵢ})] − ‖b‖²/(2σ_p²) − ‖c‖²/(2σ_s²)``,
    ``ηᵢ = xᵢ·γ + b_{p(i)} + c_{s(i)}``.

    For fixed variance components the mode ``û`` is found by Newton iterations on
    the sparse ``(P + S) × (P + S)`` system, and the Laplace approximation to the
    marginal likelihood is

    ``log L ≈ ℓ(γ, û) − ½ log det H + ½ log det D``,

    with ``H = Z'WZ + D`` and ``D = diag(1/σ_p², 1/σ_s²)``. The outer optimizer
    maximizes this over ``(γ, log σ_p, log σ_s)``.
    """

    def __init__(self, *, max_newton: int = 60, tol: float = 1e-9) -> None:
        """Create a fitter."""
        self.max_newton = max_newton
        self.tol = tol

    def fit(
        self,
        y: Sequence[float],
        X: Sequence[Sequence[float]],
        participant: Sequence[Any],
        scenario: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
    ) -> ModelFit:
        """Fit the model.

        Args:
            y: Binary outcomes.
            X: Fixed-effect design matrix, **including** an intercept column.
            participant: Participant grouping labels.
            scenario: Scenario grouping labels.
            names: Column names for ``X``.

        Returns:
            The :class:`ModelFit`, with odds ratios and 95% CIs (Req 12.4).
        """
        y_arr = np.asarray(y, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n, p = X_arr.shape
        names = list(names or [f"x{i}" for i in range(p)])

        p_idx, p_levels = _indicator(participant)
        s_idx, s_levels = _indicator(scenario)
        n_p, n_s = len(p_levels), len(s_levels)
        n_u = n_p + n_s

        # Z is the concatenation of two indicator blocks; it is never materialized.
        u_index = np.stack([p_idx, n_p + s_idx], axis=1)

        def random_offsets(u: np.ndarray) -> np.ndarray:
            return u[u_index[:, 0]] + u[u_index[:, 1]]

        def inner_mode(
            gamma: np.ndarray, precision: np.ndarray, u0: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """Newton solve for û and the resulting Hessian ``H``."""
            u = u0.copy()
            fixed = X_arr @ gamma
            hessian = np.eye(n_u)
            for _ in range(self.max_newton):
                eta = fixed + random_offsets(u)
                mu = _expit(eta)
                w = mu * (1.0 - mu)
                residual = y_arr - mu

                gradient = np.zeros(n_u)
                np.add.at(gradient, u_index[:, 0], residual)
                np.add.at(gradient, u_index[:, 1], residual)
                gradient -= precision * u

                hessian = np.diag(precision).astype(float)
                # Z'WZ for two crossed indicator blocks: diagonal blocks are group
                # sums of w, off-diagonal entries are per-cell sums.
                np.add.at(hessian, (u_index[:, 0], u_index[:, 0]), w)
                np.add.at(hessian, (u_index[:, 1], u_index[:, 1]), w)
                np.add.at(hessian, (u_index[:, 0], u_index[:, 1]), w)
                np.add.at(hessian, (u_index[:, 1], u_index[:, 0]), w)

                step = np.linalg.solve(hessian + _RIDGE * np.eye(n_u), gradient)
                u = u + step
                if float(np.max(np.abs(step))) < self.tol:
                    break
            return u, hessian

        state: dict[str, Any] = {"u": np.zeros(n_u)}

        def negative_log_marginal(theta: np.ndarray) -> float:
            gamma = theta[:p]
            sigma_p = math.exp(float(theta[p]))
            sigma_s = math.exp(float(theta[p + 1]))
            precision = np.concatenate(
                [np.full(n_p, 1.0 / sigma_p**2), np.full(n_s, 1.0 / sigma_s**2)]
            )
            u, hessian = inner_mode(gamma, precision, state["u"])
            state["u"] = u

            eta = X_arr @ gamma + random_offsets(u)
            log_lik = float(np.sum(y_arr * eta - np.logaddexp(0.0, eta)))
            log_lik -= 0.5 * float(np.sum(precision * u**2))

            sign_h, logdet_h = np.linalg.slogdet(hessian + _RIDGE * np.eye(n_u))
            if sign_h <= 0:
                return 1e12
            logdet_d = float(np.sum(np.log(precision)))
            laplace = log_lik - 0.5 * logdet_h + 0.5 * logdet_d
            if not math.isfinite(laplace):
                return 1e12
            return -laplace

        theta0 = np.concatenate([np.zeros(p), np.log([0.5, 0.5])])
        result = optimize.minimize(
            negative_log_marginal,
            theta0,
            method="L-BFGS-B",
            bounds=[(None, None)] * p + [(math.log(1e-3), math.log(10.0))] * 2,
            options={"maxiter": 500},
        )

        gamma = result.x[:p]
        sigma_p = math.exp(float(result.x[p]))
        sigma_s = math.exp(float(result.x[p + 1]))
        precision = np.concatenate(
            [np.full(n_p, 1.0 / sigma_p**2), np.full(n_s, 1.0 / sigma_s**2)]
        )
        u, hessian = inner_mode(gamma, precision, state["u"])

        covariance = self._profiled_covariance(
            X_arr, y_arr, gamma, u, u_index, hessian, n_u
        )
        coefficients = _coefficients(names, gamma, covariance, odds_ratios=True)

        return ModelFit(
            model="random_intercept_logit",
            coefficients=coefficients,
            sigma_participant=sigma_p,
            sigma_scenario=sigma_s,
            log_likelihood=-float(result.fun),
            n_obs=n,
            n_participants=n_p,
            n_scenarios=n_s,
            converged=bool(result.success),
            notes=[
                "Crossed random intercepts for participant and scenario; Laplace "
                "approximation with inner Newton mode-finding."
            ],
        )

    @staticmethod
    def _profiled_covariance(
        X: np.ndarray,
        y: np.ndarray,
        gamma: np.ndarray,
        u: np.ndarray,
        u_index: np.ndarray,
        hessian: np.ndarray,
        n_u: int,
    ) -> np.ndarray:
        """Covariance of ``γ`` with the random effects profiled out.

        ``Var(γ) ≈ (X'WX − X'WZ·H⁻¹·Z'WX)⁻¹``, the Schur complement of the joint
        observed information.
        """
        eta = X @ gamma + u[u_index[:, 0]] + u[u_index[:, 1]]
        w = _expit(eta) * (1.0 - _expit(eta))

        xwx = X.T @ (X * w[:, None])
        xwz = np.zeros((X.shape[1], n_u))
        np.add.at(xwz.T, u_index[:, 0], X * w[:, None])
        np.add.at(xwz.T, u_index[:, 1], X * w[:, None])

        h_inv = np.linalg.pinv(hessian + _RIDGE * np.eye(n_u))
        schur = xwx - xwz @ h_inv @ xwz.T
        return np.linalg.pinv(schur + _RIDGE * np.eye(X.shape[1]))


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


# --------------------------------------------------------------------------- #
# Random-intercept Gaussian (decision time, Req 12.2)
# --------------------------------------------------------------------------- #
class RandomInterceptGaussian:
    """Crossed random-intercept linear model, fitted by exact maximum likelihood.

    Used for ``log t``. The ``N × N`` covariance ``V = σ²I + σ_p²Z_pZ_p' +
    σ_s²Z_sZ_s'`` is never formed; the likelihood is evaluated through the sparse
    mixed-model equations, so the fit scales with the number of groups rather than
    the number of observations.
    """

    def fit(
        self,
        y: Sequence[float],
        X: Sequence[Sequence[float]],
        participant: Sequence[Any],
        scenario: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
    ) -> ModelFit:
        """Fit the model to a continuous outcome."""
        y_arr = np.asarray(y, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n, p = X_arr.shape
        names = list(names or [f"x{i}" for i in range(p)])

        p_idx, p_levels = _indicator(participant)
        s_idx, s_levels = _indicator(scenario)
        n_p, n_s = len(p_levels), len(s_levels)
        n_u = n_p + n_s
        u_index = np.stack([p_idx, n_p + s_idx], axis=1)

        def zt(vector: np.ndarray) -> np.ndarray:
            """``Z'v`` for the two crossed indicator blocks."""
            out = np.zeros(n_u)
            np.add.at(out, u_index[:, 0], vector)
            np.add.at(out, u_index[:, 1], vector)
            return out

        ztz = np.zeros((n_u, n_u))
        ones = np.ones(n)
        np.add.at(ztz, (u_index[:, 0], u_index[:, 0]), ones)
        np.add.at(ztz, (u_index[:, 1], u_index[:, 1]), ones)
        np.add.at(ztz, (u_index[:, 0], u_index[:, 1]), ones)
        np.add.at(ztz, (u_index[:, 1], u_index[:, 0]), ones)

        ztx = np.zeros((n_u, p))
        np.add.at(ztx, u_index[:, 0], X_arr)
        np.add.at(ztx, u_index[:, 1], X_arr)

        def pieces(log_theta: np.ndarray):
            sigma2 = math.exp(float(log_theta[0]))
            sp2 = math.exp(float(log_theta[1]))
            ss2 = math.exp(float(log_theta[2]))
            precision = np.concatenate(
                [np.full(n_p, 1.0 / sp2), np.full(n_s, 1.0 / ss2)]
            )
            # M = Z'Z/sigma2 + D
            m = ztz / sigma2 + np.diag(precision)
            m_inv = np.linalg.pinv(m + _RIDGE * np.eye(n_u))
            return sigma2, precision, m, m_inv

        def solve_v(vector: np.ndarray, sigma2: float, m_inv: np.ndarray) -> np.ndarray:
            """``V⁻¹v`` via the Woodbury identity."""
            ztv = zt(vector) / sigma2
            correction = m_inv @ ztv
            expanded = correction[u_index[:, 0]] + correction[u_index[:, 1]]
            return (vector - expanded) / sigma2

        def negative_log_likelihood(log_theta: np.ndarray) -> float:
            sigma2, precision, m, m_inv = pieces(log_theta)
            vi_x = np.column_stack(
                [solve_v(X_arr[:, j], sigma2, m_inv) for j in range(p)]
            )
            xtvix = X_arr.T @ vi_x
            vi_y = solve_v(y_arr, sigma2, m_inv)
            xtviy = X_arr.T @ vi_y
            try:
                gamma = np.linalg.solve(xtvix + _RIDGE * np.eye(p), xtviy)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                return 1e12
            residual = y_arr - X_arr @ gamma
            quad = float(residual @ solve_v(residual, sigma2, m_inv))

            sign_m, logdet_m = np.linalg.slogdet(m + _RIDGE * np.eye(n_u))
            if sign_m <= 0:
                return 1e12
            logdet_d = float(np.sum(np.log(precision)))
            logdet_v = n * math.log(sigma2) + logdet_m - logdet_d
            value = 0.5 * (logdet_v + quad + n * math.log(2.0 * math.pi))
            return value if math.isfinite(value) else 1e12

        variance0 = max(float(np.var(y_arr)), 1e-4)
        theta0 = np.log([variance0 * 0.6, variance0 * 0.2, variance0 * 0.2])
        result = optimize.minimize(
            negative_log_likelihood,
            theta0,
            method="L-BFGS-B",
            bounds=[(math.log(1e-8), math.log(1e6))] * 3,
            options={"maxiter": 500},
        )

        sigma2, _precision, m, m_inv = pieces(result.x)
        vi_x = np.column_stack([solve_v(X_arr[:, j], sigma2, m_inv) for j in range(p)])
        xtvix = X_arr.T @ vi_x
        gamma = np.linalg.solve(
            xtvix + _RIDGE * np.eye(p), X_arr.T @ solve_v(y_arr, sigma2, m_inv)
        )
        covariance = np.linalg.pinv(xtvix + _RIDGE * np.eye(p))

        return ModelFit(
            model="random_intercept_gaussian",
            coefficients=_coefficients(names, gamma, covariance, odds_ratios=False),
            sigma_participant=math.sqrt(math.exp(float(result.x[1]))),
            sigma_scenario=math.sqrt(math.exp(float(result.x[2]))),
            sigma_residual=math.sqrt(sigma2),
            log_likelihood=-float(result.fun),
            n_obs=n,
            n_participants=n_p,
            n_scenarios=n_s,
            converged=bool(result.success),
            notes=[
                "Exact ML via Woodbury; outcome is log-transformed decision time."
            ],
        )


# --------------------------------------------------------------------------- #
# Cumulative logit (workload, Req 12.2)
# --------------------------------------------------------------------------- #
class CumulativeLogit:
    """Proportional-odds model with cluster-robust standard errors.

    Fitted for the ordinal NASA-TLX outcome. This is a documented simplification
    of eq. (12): rather than a random intercept, between-participant dependence is
    handled by a cluster-robust (sandwich) covariance estimator clustered on
    participant. The point estimates are consistent and the standard errors
    account for within-participant correlation, which is what the reported
    contrasts require.
    """

    def fit(
        self,
        y: Sequence[int],
        X: Sequence[Sequence[float]],
        cluster: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
    ) -> ModelFit:
        """Fit the model.

        Args:
            y: Ordinal outcomes as integer levels ``0..K−1``.
            X: Design matrix **without** an intercept (the cutpoints absorb it).
            cluster: Clustering labels (participants).
            names: Column names for ``X``.
        """
        y_arr = np.asarray(y, dtype=int)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n, p = X_arr.shape
        names = list(names or [f"x{i}" for i in range(p)])

        levels = sorted(set(int(v) for v in y_arr))
        k = len(levels)
        if k < 2:
            return ModelFit(
                model="cumulative_logit",
                coefficients=[],
                n_obs=n,
                converged=False,
                notes=["outcome has fewer than two levels; model not identified"],
            )
        remap = {level: i for i, level in enumerate(levels)}
        y_idx = np.array([remap[int(v)] for v in y_arr], dtype=int)

        n_cut = k - 1

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Cutpoints are parameterized by increments to keep them ordered."""
            first = theta[0]
            increments = np.exp(theta[1:n_cut]) if n_cut > 1 else np.array([])
            cuts = np.concatenate([[first], first + np.cumsum(increments)])
            beta = theta[n_cut:]
            return cuts, beta

        def negative_log_likelihood(theta: np.ndarray) -> float:
            cuts, beta = unpack(theta)
            eta = X_arr @ beta
            total = 0.0
            for i in range(n):
                level = y_idx[i]
                upper = 1.0 if level == k - 1 else float(_expit(np.array([cuts[level] - eta[i]]))[0])
                lower = 0.0 if level == 0 else float(_expit(np.array([cuts[level - 1] - eta[i]]))[0])
                probability = max(upper - lower, 1e-12)
                total -= math.log(probability)
            return total if math.isfinite(total) else 1e12

        theta0 = np.concatenate([[0.0], np.zeros(max(0, n_cut - 1)), np.zeros(p)])
        result = optimize.minimize(
            negative_log_likelihood, theta0, method="L-BFGS-B", options={"maxiter": 800}
        )
        cuts, beta = unpack(result.x)

        covariance = self._cluster_robust(X_arr, y_idx, cuts, beta, cluster, k)
        return ModelFit(
            model="cumulative_logit",
            coefficients=_coefficients(names, beta, covariance, odds_ratios=True),
            log_likelihood=-float(result.fun),
            n_obs=n,
            n_participants=len(set(str(c) for c in cluster)),
            converged=bool(result.success),
            notes=[
                "Proportional-odds model with cluster-robust (sandwich) standard "
                "errors clustered on participant, in place of a random intercept."
            ],
        )

    @staticmethod
    def _cluster_robust(
        X: np.ndarray,
        y_idx: np.ndarray,
        cuts: np.ndarray,
        beta: np.ndarray,
        cluster: Sequence[Any],
        k: int,
    ) -> np.ndarray:
        """Sandwich covariance for the slope block, clustered on ``cluster``."""
        n, p = X.shape
        eta = X @ beta
        scores = np.zeros((n, p))
        for i in range(n):
            level = y_idx[i]
            upper_cut = None if level == k - 1 else cuts[level]
            lower_cut = None if level == 0 else cuts[level - 1]
            f_upper = 1.0 if upper_cut is None else float(_expit(np.array([upper_cut - eta[i]]))[0])
            f_lower = 0.0 if lower_cut is None else float(_expit(np.array([lower_cut - eta[i]]))[0])
            probability = max(f_upper - f_lower, 1e-12)
            d_upper = 0.0 if upper_cut is None else f_upper * (1 - f_upper)
            d_lower = 0.0 if lower_cut is None else f_lower * (1 - f_lower)
            # d/d(eta) of the cell probability is -(d_upper - d_lower).
            scores[i] = (-(d_upper - d_lower) / probability) * X[i]

        bread = scores.T @ scores + _RIDGE * np.eye(p)
        groups: dict[str, np.ndarray] = {}
        for i, label in enumerate(cluster):
            key = str(label)
            groups[key] = groups.get(key, np.zeros(p)) + scores[i]
        meat = np.zeros((p, p))
        for total in groups.values():
            meat += np.outer(total, total)

        bread_inv = np.linalg.pinv(bread)
        return bread_inv @ meat @ bread_inv


# --------------------------------------------------------------------------- #
# Multiplicity and resampling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HolmResult:
    """One Holm-corrected test."""

    name: str
    p_value: float
    adjusted_p: float
    rejected: bool

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "name": self.name,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
            "rejected": self.rejected,
        }


def holm(
    tests: Sequence[tuple[str, float]], *, alpha: float = 0.05
) -> list[HolmResult]:
    """Holm–Bonferroni correction over the primary contrasts (Req 12.3).

    Args:
        tests: ``(name, p_value)`` pairs.
        alpha: Familywise error rate.

    Returns:
        Results in the input order, with monotone adjusted p-values.
    """
    indexed = sorted(
        ((p if math.isfinite(p) else 1.0, name, i) for i, (name, p) in enumerate(tests)),
    )
    m = len(indexed)
    adjusted: list[float] = [0.0] * m
    running = 0.0
    for rank, (p, _name, _i) in enumerate(indexed):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[rank] = running

    out: list[HolmResult | None] = [None] * m
    for rank, (p, name, original_index) in enumerate(indexed):
        out[original_index] = HolmResult(
            name=name,
            p_value=p,
            adjusted_p=adjusted[rank],
            rejected=adjusted[rank] <= alpha,
        )
    return [r for r in out if r is not None]


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    iterations: int = 2000,
    seed: int = 1337,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a statistic.

    Returns ``(point_estimate, ci_low, ci_high)``. An empty input yields ``nan``s.
    """
    data = np.asarray(list(values), dtype=float)
    statistic = statistic or (lambda v: float(np.mean(v)))
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(data))
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for i in range(iterations):
        sample = data[rng.integers(0, data.size, data.size)]
        draws[i] = statistic(sample)
    tail = (1.0 - level) / 2.0
    return point, float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[Any],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    iterations: int = 2000,
    seed: int = 1337,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI, resampling whole clusters with replacement.

    Appropriate when observations within a participant or scenario are correlated,
    which they are throughout this evaluation.
    """
    statistic = statistic or (lambda v: float(np.mean(v)))
    grouped: dict[str, list[float]] = {}
    for value, label in zip(values, clusters):
        grouped.setdefault(str(label), []).append(float(value))
    keys = sorted(grouped)
    if not keys:
        return float("nan"), float("nan"), float("nan")

    flat = [v for key in keys for v in grouped[key]]
    point = float(statistic(flat))
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for i in range(iterations):
        chosen = rng.integers(0, len(keys), len(keys))
        sample = [v for index in chosen for v in grouped[keys[index]]]
        draws[i] = statistic(sample) if sample else float("nan")
    tail = (1.0 - level) / 2.0
    return point, float(np.nanquantile(draws, tail)), float(np.nanquantile(draws, 1.0 - tail))


def two_proportion_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int
) -> tuple[float, float]:
    """Two-sided score test for a difference in proportions.

    Returns ``(difference, p_value)``; ``(nan, nan)`` when either group is empty.
    """
    if n_a <= 0 or n_b <= 0:
        return float("nan"), float("nan")
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if standard_error == 0:
        return p_a - p_b, 1.0
    z = (p_a - p_b) / standard_error
    return p_a - p_b, float(2.0 * stats.norm.sf(abs(z)))

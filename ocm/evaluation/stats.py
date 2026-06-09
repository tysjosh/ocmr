"""Statistical primitives for the evaluation harness (paper §IV-B).

Implements the inference and calibration machinery the paper reports without a
SciPy dependency (only the stdlib ``math`` + ``numpy``):

* :func:`mean_ci` — mean with a Student-t 95% confidence interval.
* :func:`paired_t_test` — two-sided paired t-test (t statistic, df, p-value).
* :func:`wilcoxon_signed_rank` — paired Wilcoxon signed-rank (normal approx.).
* :func:`cohens_d_paired` / :func:`rank_biserial_paired` — paired effect sizes.
* :func:`holm_bonferroni` — Holm-Bonferroni multiple-comparison correction.
* :func:`expected_calibration_error` / :func:`brier_score` — calibration.

The t- and normal-distribution tail probabilities are computed from the
regularized incomplete beta function and ``math.erf`` respectively, so the
results match SciPy to within numerical tolerance for the sample sizes used in
the study (a handful of seeds per method).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


# --------------------------------------------------------------------------- #
# Special functions (regularized incomplete beta -> Student-t CDF)
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT = 200
    EPS = 3.0e-12
    FPMIN = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """CDF of the Student-t distribution with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    ib = reg_incomplete_beta(df / 2.0, 0.5, x)
    if t > 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided tail probability ``P(|T| >= |t|)`` for Student-t."""
    cdf = student_t_cdf(abs(t), df)
    return 2.0 * (1.0 - cdf)


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of Student-t via bisection on :func:`student_t_cdf`."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_cdf(z: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_sf_two_sided(z: float) -> float:
    """Two-sided standard-normal tail probability ``P(|Z| >= |z|)``."""
    return 2.0 * (1.0 - normal_cdf(abs(z)))


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeanCI:
    """A mean with a symmetric confidence interval."""

    mean: float
    low: float
    high: float
    half_width: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.mean, self.low, self.high)


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> MeanCI:
    """Mean and a Student-t confidence interval for ``values``.

    With a single value the interval collapses to the point (half-width 0); with
    none it is all zeros. Uses the t critical value for ``n - 1`` df so small
    multi-seed samples are handled correctly.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return MeanCI(0.0, 0.0, 0.0, 0.0, 0)
    mean = sum(vals) / n
    if n == 1:
        return MeanCI(mean, mean, mean, 0.0, 1)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    tcrit = student_t_ppf(0.5 + confidence / 2.0, n - 1)
    half = tcrit * se
    return MeanCI(mean, mean - half, mean + half, half, n)


# --------------------------------------------------------------------------- #
# Paired significance tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TestResult:
    """The outcome of a significance test."""

    statistic: float
    p_value: float
    df: float | None = None
    test: str = ""
    effect_size: float | None = None
    effect_name: str = ""


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Two-sided paired t-test of ``a`` vs ``b`` (equal length, n >= 2).

    Returns the t statistic, degrees of freedom ``n - 1``, the two-sided
    p-value, and Cohen's d for paired samples as the effect size.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) != len(b):
        raise ValueError("paired_t_test requires equal-length samples")
    n = len(a)
    if n < 2:
        return TestResult(0.0, 1.0, df=max(n - 1, 0), test="paired_t_test")
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d)
    if sd_d == 0.0:
        # No variance: a perfect (or null) separation.
        p = 0.0 if mean_d != 0.0 else 1.0
        d_eff = float("inf") if mean_d != 0.0 else 0.0
        return TestResult(
            float("inf") if mean_d != 0.0 else 0.0,
            p,
            df=n - 1,
            test="paired_t_test",
            effect_size=d_eff,
            effect_name="cohen_d",
        )
    se = sd_d / math.sqrt(n)
    t = mean_d / se
    p = student_t_sf_two_sided(t, n - 1)
    return TestResult(
        t, p, df=n - 1, test="paired_t_test",
        effect_size=mean_d / sd_d, effect_name="cohen_d",
    )


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples (mean difference / sd of differences)."""
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    n = len(a)
    if n < 2:
        return 0.0
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d)
    if sd_d == 0.0:
        return 0.0 if mean_d == 0.0 else float("inf")
    return mean_d / sd_d


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Paired Wilcoxon signed-rank test (normal approximation w/ continuity).

    Zero differences are dropped (Wilcoxon convention). Ties share averaged
    ranks. The effect size is the rank-biserial correlation. For the small
    samples used here the normal approximation is adequate; exact tables are not
    required for the study's reporting.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) != len(b):
        raise ValueError("wilcoxon_signed_rank requires equal-length samples")
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0.0]
    n = len(diffs)
    if n == 0:
        return TestResult(0.0, 1.0, test="wilcoxon", effect_size=0.0, effect_name="rank_biserial")

    # Rank the absolute differences with average ranks for ties.
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average of 1-based ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = min(w_plus, w_minus)
    total = n * (n + 1) / 2.0
    mean_w = total / 2.0
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_w == 0.0:
        return TestResult(w, 1.0, test="wilcoxon", effect_size=0.0, effect_name="rank_biserial")
    # Continuity correction.
    z = (w - mean_w + 0.5) / sd_w if w < mean_w else (w - mean_w - 0.5) / sd_w
    p = normal_sf_two_sided(z)
    rank_biserial = (w_plus - w_minus) / total
    return TestResult(
        w, min(p, 1.0), test="wilcoxon",
        effect_size=rank_biserial, effect_name="rank_biserial",
    )


def is_approximately_normal(values: Sequence[float]) -> bool:
    """Cheap normality heuristic to choose t-test vs Wilcoxon.

    With very small samples (n < 8) we cannot test normality meaningfully, so we
    default to assuming normality (use the t-test). Otherwise we apply a simple
    skew/kurtosis sanity bound. This mirrors the paper's "paired t-test when
    normality holds; Wilcoxon otherwise" rule pragmatically.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 8:
        return True
    mean = sum(vals) / n
    m2 = sum((v - mean) ** 2 for v in vals) / n
    if m2 == 0.0:
        return True
    m3 = sum((v - mean) ** 3 for v in vals) / n
    m4 = sum((v - mean) ** 4 for v in vals) / n
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0
    return abs(skew) < 2.0 and abs(kurt) < 7.0


def paired_test_auto(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Pick a paired test by normality of the differences (paper §IV-B rule)."""
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    if is_approximately_normal(diffs):
        return paired_t_test(a, b)
    return wilcoxon_signed_rank(a, b)


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #
def holm_bonferroni(
    p_values: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, float | bool]]:
    """Holm-Bonferroni step-down correction over a labelled set of p-values.

    Returns, per label, the ``corrected_p`` (monotone, capped at 1.0) and a
    ``reject`` flag at family-wise ``alpha``.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running_max = 0.0
    for rank, (label, p) in enumerate(items):
        corrected = (m - rank) * p
        corrected = min(corrected, 1.0)
        running_max = max(running_max, corrected)  # enforce monotonicity
        out[label] = {
            "raw_p": p,
            "corrected_p": running_max,
            "reject": running_max < alpha,
        }
    return out


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width confidence bins."""
    conf = [float(c) for c in confidences]
    corr = [1.0 if c else 0.0 for c in correct]
    n = len(conf)
    if n == 0 or n != len(corr):
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Include the right edge in the final bin.
        idx = [
            i for i in range(n)
            if (conf[i] > lo or (b == 0 and conf[i] >= lo)) and (conf[i] <= hi)
        ]
        if not idx:
            continue
        avg_conf = sum(conf[i] for i in idx) / len(idx)
        avg_acc = sum(corr[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(avg_conf - avg_acc)
    return ece


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error between confidence and correctness (Brier score)."""
    conf = [float(c) for c in confidences]
    corr = [1.0 if c else 0.0 for c in correct]
    n = len(conf)
    if n == 0 or n != len(corr):
        return 0.0
    return sum((conf[i] - corr[i]) ** 2 for i in range(n)) / n

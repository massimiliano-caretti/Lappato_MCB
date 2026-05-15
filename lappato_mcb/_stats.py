"""Statistical helpers used by the v1.6 *insight detectors*.

Strategy 3 (hybrid) — every helper has a pure-stdlib implementation
that runs under any Python ≥ 3.9 with no external dependencies, and a
SciPy-accelerated fallback that is used transparently when SciPy is
importable.  This preserves LAPPATO_MCB's "stdlib-only by default"
promise (``dependencies = []`` in ``pyproject.toml``) while letting
power users opt into faster / more numerically robust implementations
by simply ``pip install scipy``.

Naming convention: every public helper here returns a *p-value*
(two-sided unless explicitly noted) so detector code can compose them
freely without worrying about effect-size sign conventions.

References for the pure-stdlib implementations:
    Press, Teukolsky, Vetterling, Flannery — *Numerical Recipes in C*,
    2nd ed. (1992).  §6.2 (incomplete gamma), §6.4 (incomplete beta),
    §6.14 (statistical-test wrappers).
    Beasley & Springer — *Algorithm AS 111: The Percentage Points of
    the Normal Distribution*, Applied Statistics 26(1), 1977.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# ── scipy detection (one-time, module-import-time) ──────────────────────
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY: bool = True
except Exception:                                                          # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False


# ── Helpers (stdlib special functions, all needed below) ────────────────

def _lgamma(x: float) -> float:
    """``log Γ(x)`` — wrapper around stdlib ``math.lgamma`` so the rest
    of the file can swap to a pure-Python series if a future stdlib
    drops ``lgamma`` (it has been there since Python 2.7, but the wrap
    makes the dependency explicit)."""
    return math.lgamma(x)


def _gammaln_diff(a: float, b: float) -> float:
    """``log Γ(a) − log Γ(b)`` — used to keep the binomial coefficient
    in log-space inside Fisher's exact test (avoids overflow on large
    factorials)."""
    return _lgamma(a) - _lgamma(b)


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF Φ(x) using ``math.erf`` — exact identity
    Φ(x) = ½(1 + erf(x / √2))."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_inv_cdf(p: float) -> float:
    """Inverse standard-normal CDF (a.k.a. probit), Beasley-Springer
    1977 algorithm with Wichura 1988 tail-region refinement.

    Numerically accurate to ~1e-9 for p ∈ (1e-15, 1 − 1e-15).
    Used by the sample-size helper below.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
                (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def _gammp(a: float, x: float, *, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Lower regularised incomplete gamma function P(a, x).

    Combines a series expansion (small x) with a continued-fraction
    expansion (large x), matching Numerical Recipes §6.2 and the
    reference implementation in ``cephes`` / ``scipy.special.gammainc``.
    """
    if x < 0 or a <= 0:
        raise ValueError(f"invalid args to gammp: a={a}, x={x}")
    if x == 0.0:
        return 0.0
    gln = _lgamma(a)
    if x < a + 1.0:
        # Series representation (NR eq. 6.2.5).
        ap = a
        s = 1.0 / a
        term = s
        for _ in range(max_iter):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * eps:
                return s * math.exp(-x + a * math.log(x) - gln)
        return s * math.exp(-x + a * math.log(x) - gln)
    # Continued-fraction representation (NR eq. 6.2.7), then complement.
    b = x + 1.0 - a
    c = 1.0 / 1.0e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = b + an / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def _betacf(a: float, b: float, x: float, *,
            max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued-fraction expansion for the regularised incomplete
    beta function (NR eq. 6.4.5).  Used by Student's t CDF below."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1.0e-30:
        d = 1.0e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) (NR eq. 6.4.1)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(_lgamma(a + b) - _lgamma(a) - _lgamma(b)
                   + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


# ── Public API: tests + sample-size helpers ─────────────────────────────

def chi2_p_value(observed: Sequence[Sequence[float]]) -> float:
    """Two-tailed χ² p-value for a contingency table.

    ``observed`` is an iterable of rows; each row is an iterable of
    nonneg counts.  Uses ``scipy.stats.chi2_contingency`` when SciPy
    is available, otherwise computes χ² with the Pearson formula and
    converts via the regularised lower-incomplete gamma function.

    Returns 1.0 (i.e. "no evidence") on degenerate inputs (empty
    table, single row/column, zero margins) — fail-closed convention
    consistent with the rest of LAPPATO's detectors.
    """
    table = [[float(v) for v in row] for row in observed]
    if not table or not table[0]:
        return 1.0
    n_rows = len(table)
    n_cols = len(table[0])
    if any(len(r) != n_cols for r in table):
        return 1.0
    if n_rows < 2 or n_cols < 2:
        return 1.0
    if _HAS_SCIPY:
        try:
            chi2, p, dof, _ = _scipy_stats.chi2_contingency(table)
            return float(p)
        except Exception:
            pass
    # Pure-stdlib fallback.
    row_totals = [sum(r) for r in table]
    col_totals = [sum(table[r][c] for r in range(n_rows)) for c in range(n_cols)]
    n_total = sum(row_totals)
    if n_total <= 0.0:
        return 1.0
    chi2 = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            expected = row_totals[r] * col_totals[c] / n_total
            if expected <= 0.0:
                continue
            chi2 += (table[r][c] - expected) ** 2 / expected
    dof = (n_rows - 1) * (n_cols - 1)
    if dof <= 0:
        return 1.0
    # P(X ≤ chi2) = P(a=dof/2, x=chi2/2) → upper tail = 1 − P.
    return 1.0 - _gammp(dof / 2.0, chi2 / 2.0)


def fisher_exact_2x2(table: Sequence[Sequence[float]]) -> float:
    """Two-tailed p-value for a 2×2 contingency table.  Uses
    ``scipy.stats.fisher_exact`` when SciPy is available, otherwise
    sums the hypergeometric tail in pure Python.
    """
    if (len(table) != 2 or len(table[0]) != 2 or len(table[1]) != 2):
        raise ValueError("fisher_exact_2x2 requires a 2x2 table")
    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError("fisher_exact_2x2 requires non-negative counts")
    if _HAS_SCIPY:
        try:
            _, p = _scipy_stats.fisher_exact([[a, b], [c, d]],
                                              alternative="two-sided")
            return float(p)
        except Exception:
            pass
    # Pure stdlib: hypergeometric tail sum.
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    if n == 0 or r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        return 1.0
    # Log-pmf of cell (k, r1-k, c1-k, r2-c1+k) given the margins.
    log_n_fact = _lgamma(n + 1)
    log_r1 = _lgamma(r1 + 1)
    log_r2 = _lgamma(r2 + 1)
    log_c1 = _lgamma(c1 + 1)
    log_c2 = _lgamma(c2 + 1)
    log_const = log_r1 + log_r2 + log_c1 + log_c2 - log_n_fact

    def log_pmf(k: int) -> float:
        if k < 0 or k > c1 or (r1 - k) < 0 or (r2 - c1 + k) < 0:
            return float("-inf")
        return log_const - (
            _lgamma(k + 1) + _lgamma(r1 - k + 1)
            + _lgamma(c1 - k + 1) + _lgamma(r2 - c1 + k + 1)
        )

    # Two-sided: sum every cell whose pmf ≤ pmf(observed).
    obs = log_pmf(a)
    k_min = max(0, c1 - r2)
    k_max = min(c1, r1)
    p_total = 0.0
    for k in range(k_min, k_max + 1):
        lpk = log_pmf(k)
        if lpk <= obs + 1e-9:
            p_total += math.exp(lpk)
    return min(1.0, p_total)


def t_test_p_value(slope_t: float, dof: int) -> float:
    """Two-tailed p-value for a Student-t test statistic ``slope_t``
    with ``dof`` degrees of freedom.  Uses the regularised incomplete
    beta function (pure Python).  SciPy ``stats.t.sf(|t|, dof) * 2``
    is used when present.
    """
    if dof <= 0:
        return 1.0
    if _HAS_SCIPY:
        try:
            return float(2.0 * _scipy_stats.t.sf(abs(slope_t), df=dof))
        except Exception:
            pass
    x = dof / (dof + slope_t * slope_t)
    return _betai(dof / 2.0, 0.5, x)


def linear_regression_slope_p(
    xs: Sequence[float], ys: Sequence[float],
) -> tuple[float, float, float]:
    """Closed-form OLS on ``ys ~ a + b·xs``.

    Returns ``(slope, intercept, two-sided p-value of slope=0)``.
    Pure stdlib (no SciPy needed), ~10 lines.
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0, (ys[0] if ys else 0.0), 1.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    if sxx <= 0.0:
        return 0.0, mean_y, 1.0
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
    sse = sum(r * r for r in residuals)
    if n - 2 <= 0:
        return slope, intercept, 1.0
    if sse <= 0.0:
        # Perfect linear fit (zero residual variance).  The slope is
        # perfectly determined → the t-test for slope ≠ 0 trivially
        # rejects H₀.  Return p = 0.0 (matches scipy.stats.linregress
        # behaviour on perfect-fit data) when slope is non-zero, and
        # p = 1.0 when slope is exactly zero (no signal).
        return slope, intercept, (0.0 if abs(slope) > 1e-12 else 1.0)
    s_e2 = sse / (n - 2)
    se_slope = math.sqrt(s_e2 / sxx)
    if se_slope <= 0.0:
        return slope, intercept, (0.0 if abs(slope) > 1e-12 else 1.0)
    t = slope / se_slope
    return slope, intercept, t_test_p_value(t, n - 2)


def required_sample_size_for_proportion(
    observed_p: float, target_p: float,
    *, alpha: float = 0.05, power: float = 0.8,
) -> int:
    """Compute the sample size N required for a one-sided proportion
    test (H₀: p ≤ target_p, H₁: p > target_p) to achieve ``power`` at
    significance level ``alpha``.

    Returns:
        0  — when ``observed_p ≤ target_p`` (the alternative hypothesis
             is not supported by the observation; no sample size will
             demonstrate it without changing the underlying distribution).
        N  — minimum number of samples (Casella & Berger 2002 eq. 8.3.21,
             standard normal-approximation formula) when
             ``observed_p > target_p``.

    Useful inside detector cards to translate "your metric is X%, target
    is Y%" into a concrete "you need at least N samples to demonstrate
    Y% with power=0.8 at alpha=0.05".
    """
    if not (0.0 < observed_p < 1.0) or not (0.0 < target_p < 1.0):
        return 0
    if observed_p <= target_p:
        # The observation does NOT exceed the target — no finite sample
        # size will reject H₀.  Return 0 (= "not applicable") so callers
        # can render a different message rather than an absurd number.
        return 0
    z_alpha = _normal_inv_cdf(1.0 - alpha)
    z_beta  = _normal_inv_cdf(power)
    p0 = target_p
    p1 = observed_p
    num = (z_alpha * math.sqrt(p0 * (1.0 - p0))
           + z_beta  * math.sqrt(p1 * (1.0 - p1))) ** 2
    delta = p1 - p0
    return int(math.ceil(num / (delta * delta)))


__all__ = [
    "chi2_p_value",
    "fisher_exact_2x2",
    "t_test_p_value",
    "linear_regression_slope_p",
    "required_sample_size_for_proportion",
    "_HAS_SCIPY",
]

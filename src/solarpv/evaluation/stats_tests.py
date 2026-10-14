"""Diebold-Mariano test (stage 8, decision D-14, PROJECT_MASTER.md §6.9).

Loss differential d_t = e_a,t^2 - e_b,t^2 (positive mean: model a is worse than b).
Long-run variance: Newey-West with Bartlett weights 1 - k/(L+1), k = 1..L, L = 24, and
gamma_0 with ddof=1; this "raw" statistic with a normal p-value is exactly the formula behind the
old dm_tests.csv (finding F-04).
Corrected statistic (what the report describes): Harvey-Leybourne-Newbold small-sample factor
sqrt((n + 1 - 2h + h(h-1)/n) / n) with h = 24, p-value from Student t with n-1 degrees of freedom,
and Bonferroni over the comparisons made within one target x horizon.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from solarpv.config.constants import DM_NW_LAGS, SIGNIFICANCE_LEVEL


@dataclass(frozen=True)
class DMResult:
    n: int
    mean_diff: float
    dm_stat_raw: float
    p_value_raw: float
    hln_factor: float
    dm_stat: float
    p_value: float


def newey_west_variance(d, lags: int = DM_NW_LAGS) -> float:
    """Bartlett-weighted long-run variance of d (gamma_0 with ddof=1, gamma_k averaged)."""
    d = np.asarray(d, dtype=float)
    dbar = d.mean()
    var = d.var(ddof=1)
    for k in range(1, lags + 1):
        gamma_k = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * (1.0 - k / (lags + 1)) * gamma_k
    return float(var)


def hln_factor(n: int, h: int = DM_NW_LAGS) -> float:
    return float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))


def diebold_mariano(
    y_true, pred_a, pred_b, lags: int = DM_NW_LAGS, h: int = DM_NW_LAGS
) -> DMResult:
    """DM test of model a against model b on the same rows (squared-error loss)."""
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    if not (y.shape == a.shape == b.shape):
        raise ValueError("y_true, pred_a and pred_b must have the same shape")
    d = (y - a) ** 2 - (y - b) ** 2
    n = d.size
    if n <= 2 * h:
        raise ValueError(f"too few pairs ({n}) for h={h}")
    var = max(newey_west_variance(d, lags), 1e-12)
    dm_raw = d.mean() / np.sqrt(var / n)
    factor = hln_factor(n, h)
    dm = dm_raw * factor
    return DMResult(
        n=n,
        mean_diff=float(d.mean()),
        dm_stat_raw=float(dm_raw),
        p_value_raw=float(2 * stats.norm.sf(abs(dm_raw))),
        hln_factor=factor,
        dm_stat=float(dm),
        p_value=float(2 * stats.t.sf(abs(dm), df=n - 1)),
    )


def effective_n(d, max_lags: int = 50) -> float:
    """Autocorrelation-adjusted sample size n / (1 + 2 * sum of positive-truncated ACF),
    as in the original detectable_effect.py (lags 1 .. min(50, n/4))."""
    d = np.asarray(d, dtype=float)
    n = d.size
    acf = [np.corrcoef(d[:-k], d[k:])[0, 1] for k in range(1, min(max_lags, n // 4))]
    return float(max(n / (1 + 2 * max(np.nansum(acf), 0.0)), 1.0))


def power_analysis(d, alpha: float = SIGNIFICANCE_LEVEL, power: float = 0.80) -> dict:
    """Observed standardised effect |mean d| / sd(d) against the minimum detectable effect."""
    d = np.asarray(d, dtype=float)
    n_eff = effective_n(d)
    mde = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n_eff)
    sd = d.std(ddof=1)
    obs = abs(d.mean()) / sd if sd > 0 else 0.0
    return {"n_eff": n_eff, "obs_std_effect": float(obs), "min_detectable_effect": float(mde),
            "pct_of_mde": float(obs / mde * 100)}


def bonferroni(p_values, alpha: float = SIGNIFICANCE_LEVEL) -> tuple[float, np.ndarray]:
    """Adjusted alpha and the significance flags for a family of tests."""
    p = np.asarray(p_values, dtype=float)
    alpha_adj = alpha / max(p.size, 1)
    return alpha_adj, p < alpha_adj

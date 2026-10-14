"""T-03: Diebold-Mariano against independent hand computations."""

import numpy as np
import pytest
from scipy import stats

from solarpv.evaluation.stats_tests import (
    bonferroni,
    diebold_mariano,
    effective_n,
    hln_factor,
    newey_west_variance,
    power_analysis,
)


def _manual_nw(d, lags):
    n = len(d)
    dbar = sum(d) / n
    var = sum((x - dbar) ** 2 for x in d) / (n - 1)
    for k in range(1, lags + 1):
        gk = sum((d[t] - dbar) * (d[t - k] - dbar) for t in range(k, n)) / (n - k)
        var += 2 * (1 - k / (lags + 1)) * gk
    return var


def test_newey_west_matches_manual_loop():
    d = list(np.random.default_rng(3).normal(0.1, 1.0, 200))
    for lags in (0, 1, 5, 24):
        assert newey_west_variance(d, lags) == pytest.approx(_manual_nw(d, lags), rel=1e-12)


def test_raw_dm_without_lags_is_the_classic_t_ratio():
    rng = np.random.default_rng(4)
    y = rng.random(300)
    a = y + rng.normal(0, 0.3, 300)
    b = y + rng.normal(0, 0.2, 300)
    d = (y - a) ** 2 - (y - b) ** 2
    res = diebold_mariano(y, a, b, lags=0, h=1)
    assert res.dm_stat_raw == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
    assert res.p_value_raw == pytest.approx(2 * stats.norm.sf(abs(res.dm_stat_raw)))


def test_hln_correction_and_t_pvalue():
    rng = np.random.default_rng(5)
    y = rng.random(1000)
    a = y + rng.normal(0, 0.3, 1000)
    b = y + rng.normal(0, 0.25, 1000)
    res = diebold_mariano(y, a, b)
    n, h = 1000, 24
    assert res.hln_factor == pytest.approx(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    assert res.hln_factor == pytest.approx(hln_factor(n))
    assert res.dm_stat == pytest.approx(res.dm_stat_raw * res.hln_factor)
    assert res.p_value == pytest.approx(2 * stats.t.sf(abs(res.dm_stat), df=n - 1))
    assert res.dm_stat > 0  # model a has the larger error variance


def test_identical_forecasts_give_zero_mean_difference():
    y = np.linspace(0, 1, 100)
    res = diebold_mariano(y, y + 0.1, y + 0.1)
    assert res.mean_diff == 0.0


def test_effective_n_white_noise_close_to_n():
    d = np.random.default_rng(6).normal(size=2000)
    assert 0.8 * 2000 < effective_n(d) <= 2000 * 1.25


def test_effective_n_shrinks_with_autocorrelation():
    rng = np.random.default_rng(7)
    e = rng.normal(size=3000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, e.size):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    assert effective_n(ar) < 0.3 * ar.size


def test_power_analysis_formula():
    d = np.random.default_rng(8).normal(0.05, 1.0, 1500)
    out = power_analysis(d)
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / np.sqrt(out["n_eff"])
    assert out["min_detectable_effect"] == pytest.approx(mde)
    assert out["pct_of_mde"] == pytest.approx(out["obs_std_effect"] / mde * 100)


def test_bonferroni():
    alpha_adj, sig = bonferroni([0.001, 0.02, 0.2], alpha=0.05)
    assert alpha_adj == pytest.approx(0.05 / 3)
    assert sig.tolist() == [True, False, False]

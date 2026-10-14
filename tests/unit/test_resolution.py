"""Polled-source detection and reconstruction (F-18)."""

import numpy as np
import pandas as pd
import pytest

from solarpv.data.resolution import (
    POLL_PERIOD_MIN,
    PolledEra,
    dehold,
    hourly_source,
    polled_era,
    source_mask,
)


def smooth_signal(t):
    """A clear-day-like bump over ~10 h plus a slow ripple (t in minutes)."""
    return 800 * np.clip(np.sin(np.pi * (t - 60) / 600), 0, None) + 30 * np.sin(t / 37)


def polled_minutes(n=720, phase=6.3):
    """1-minute averages of a signal polled every 15 min and held (blend in the poll minute)."""
    polls = np.arange(phase, n + POLL_PERIOD_MIN, POLL_PERIOD_MIN)
    values = smooth_signal(polls)

    def held(t):
        k = np.searchsorted(polls, t, side="right") - 1
        return np.where(k >= 0, values[np.clip(k, 0, None)], values[0])

    fine = np.arange(n * 60) / 60.0  # 1-s steps
    return held(fine).reshape(n, 60).mean(axis=1)


def best_shift(x, ref, max_lag=12):
    lags = np.arange(-max_lag, max_lag + 1)
    core = slice(max_lag, len(x) - max_lag)
    sse = [((x[core] - np.roll(ref, lag)[core]) ** 2).sum() for lag in lags]
    return lags[int(np.argmin(sse))]


def test_dehold_removes_the_hold_lag():
    n = 720
    raw = pd.Series(polled_minutes(n))
    truth = smooth_signal(np.arange(n) + 0.5)
    assert best_shift(raw.to_numpy(), truth) >= 6  # held data lag by ~7.5 min
    rebuilt = dehold(raw).to_numpy()
    assert best_shift(rebuilt, truth) == 0
    inner = slice(30, n - 30)
    assert np.abs(rebuilt[inner] - truth[inner]).max() < 15  # interpolation error only
    assert np.abs(raw.to_numpy()[inner] - truth[inner]).max() > 40


def test_dehold_keeps_constants_and_nan_blocks():
    s = pd.Series([0.0] * 50 + [np.nan] * 5 + [3.0] * 40)
    out = dehold(s)
    assert out.iloc[:50].eq(0).all()
    assert out.iloc[50:55].isna().all()
    assert out.iloc[55:].eq(3.0).all()


def test_hourly_source_classifies_minute_polled_frozen():
    idx = pd.date_range("2025-03-01 06:00", periods=180, freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "irradiance": np.r_[rng.normal(500, 50, 60), polled_minutes(240)[120:180],
                            np.full(60, 7.0)],
    }, index=idx)
    states = hourly_source(df, np.ones(len(df), dtype=bool))["irradiance"].tolist()
    assert states == ["minute", "polled", "other"]


def _hourly(states_by_day: dict[str, str]) -> pd.DataFrame:
    rows = []
    for day, state in states_by_day.items():
        for h in range(8, 14):
            rows.append((pd.Timestamp(f"{day} {h:02d}:00", tz="UTC"), state))
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame({c: [r[1] for r in rows] for c in
                         ("irradiance", "s01_inv1_DC_kW", "s01_inv2_DC_kW", "s01_inv1_AC_kW",
                          "s01_inv2_AC_kW")}, index=idx)


def test_polled_era_ignores_isolated_hours_and_finds_transition():
    days = pd.date_range("2025-01-01", "2025-03-31", freq="D").strftime("%Y-%m-%d")
    states = dict.fromkeys(days, "minute")
    states["2025-01-10"] = "polled"  # isolated: not part of the era
    for d in ("2025-02-26", "2025-02-28", "2025-03-01"):  # transition with a 1-day gap
        states[d] = "polled"
    for d in pd.date_range("2025-03-03", "2025-03-31").strftime("%Y-%m-%d"):
        states[d] = "polled"
    era = polled_era(_hourly(states))
    assert era == PolledEra(pd.Timestamp("2025-02-26"), pd.Timestamp("2025-03-03"))


def test_polled_era_none_without_sustained_polling():
    days = pd.date_range("2025-01-01", "2025-01-31", freq="D").strftime("%Y-%m-%d")
    states = dict.fromkeys(days, "minute")
    states["2025-01-10"] = "polled"
    assert polled_era(_hourly(states)) is None


@pytest.mark.parametrize("era", [None])
def test_source_mask_without_era_is_all_minute(era):
    idx = pd.date_range("2025-03-01", periods=10, freq="min", tz="UTC")
    df = pd.DataFrame({"irradiance": 1.0, "s01_inv1_AC_kW": 2.0}, index=idx)
    mask = source_mask(df, pd.DataFrame(), era)
    assert mask.eq("minute").all().all()

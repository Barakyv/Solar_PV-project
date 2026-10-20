"""Stage-6 feature construction on synthetic data (docs/stage6_plan.md).

T-09  past-day features are horizon-safe: changing any source value from T0 on changes no
      feature of the forecast made at T0, for D+1, D+2 and D+3
T-18  the same across both DST transitions; the 25th hour of the fall-back day gets NaN where a
      24-hour shift would reach into the forecast day
T-34  imputation uses only the same UTC hour on earlier days; never backward fill
T-35  D-22 masks: per-inverter flags remove that inverter's AC, op-limited removes the total,
      weather is not masked
T-36  calendar: the Israeli weekend, solar noon from the transit, hourly means of minutes
"""

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import HORIZONS, TIMEZONE_LOCAL
from solarpv.features import build as B
from solarpv.features.contract import LAG_SOURCES, features_for_horizon

# a fall-back day (2022-10-30, 25 h) and a spring-forward day (2022-03-25, 23 h) in Israel;
# the windows leave 16 days of history before the first tested day and test the DST day itself
WINDOWS = {"fall back": ("2022-10-05", "2022-11-08"), "spring forward": ("2022-03-01",
                                                                         "2022-04-05")}
DST_DAYS = {"fall back": pd.Timestamp("2022-10-30"), "spring forward": pd.Timestamp("2022-03-25")}


def _sources(start: str, end: str, seed: int = 0, holes: float = 0.05) -> dict[str, pd.Series]:
    """Random hourly sources with a share of missing hours, so that the imputation is exercised
    (a source without gaps would never reach it; the mutation check showed that)."""
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    out = {}
    for c in LAG_SOURCES:
        values = rng.uniform(0, 1000, len(hours))
        values[rng.random(len(hours)) < holes] = np.nan
        out[c] = pd.Series(values, index=hours)
    return out


def _local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()


# T-09 / T-18 ---------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("window", list(WINDOWS))
@pytest.mark.parametrize("horizon", HORIZONS)
def test_nothing_after_issuance_reaches_the_forecast(window, horizon):
    sources = _sources(*WINDOWS[window])
    features, _, _, t0 = B.past_day_features(sources, horizon)
    days = sorted(set(_local_day(features.index)))[16:-1]  # skip the warm-up of the window
    assert DST_DAYS[window] in days
    for day in days:
        rows = _local_day(features.index) == day
        issue = t0[rows][0]
        changed = {c: s.where(s.index < issue, s + 12345.0) for c, s in sources.items()}
        again, _, _, _ = B.past_day_features(changed, horizon)
        pd.testing.assert_frame_equal(features[rows], again[rows], check_exact=True,
                                      obj=f"{window} D+{horizon} {day.date()}")


def test_the_25th_hour_of_the_fall_back_day_is_guarded():
    sources = _sources(*WINDOWS["fall back"])
    hours = sources[LAG_SOURCES[0]].index
    t0 = B.issuance_utc(hours, 1)
    lag1 = B.past_value(sources[LAG_SOURCES[0]], 1, t0)
    fall_back = _local_day(hours) == pd.Timestamp("2022-10-30")
    assert fall_back.sum() == 25
    last = np.flatnonzero(fall_back)[-1]
    assert np.isnan(lag1[last])  # a plain shift would read 21:00Z on the 29th = T0's own hour
    assert np.isfinite(lag1[np.flatnonzero(fall_back)[:-1]]).all()


def test_issuance_is_local_midnight_h_minus_1_days_before():
    hours = pd.date_range("2022-07-01 00:00", periods=48, freq="h", tz="UTC")
    for h in HORIZONS:
        t0 = B.issuance_utc(hours, h)
        local = t0.tz_convert(TIMEZONE_LOCAL)
        assert (local.hour == 0).all() and (local.minute == 0).all()
        gap = _local_day(hours) - local.tz_localize(None).normalize()
        assert (gap == pd.Timedelta(days=h - 1)).all()
        assert (t0 <= hours).all()


def test_rolling_uses_days_h_to_h_plus_n_minus_1():
    sources = _sources("2022-06-01", "2022-06-20", holes=0.0)
    s = sources["site_total_AC_kW"]
    t0 = B.issuance_utc(s.index, 2)
    out = B.rolling_features(sources, 2, t0)
    i = 24 * 15 + 10  # an hour well inside the window
    expected = np.mean([s.iloc[i - 24 * k] for k in (2, 3, 4)])
    assert out["site_total_rolling_mean_3d_same_tod"][i] == pytest.approx(expected)
    expected_std = np.std([s.iloc[i - 24 * k] for k in (2, 3, 4)], ddof=1)
    assert out["site_total_rolling_std_3d_same_tod"][i] == pytest.approx(expected_std)


def test_feature_names_match_the_contract():
    sources = _sources("2022-06-01", "2022-06-10")
    for h in HORIZONS:
        features, _, _, _ = B.past_day_features(sources, h)
        expected = {f for f in features_for_horizon(h) if "_lag_" in f or "rolling" in f}
        assert set(features.columns) == expected


# T-34 imputation -----------------------------------------------------------

def test_imputation_uses_only_the_same_hour_on_earlier_days():
    hours = pd.date_range("2022-01-01", periods=24 * 40, freq="h", tz="UTC")
    values = pd.Series(np.arange(len(hours), dtype=float), index=hours)
    frame = pd.DataFrame({"x": values})
    frame.iloc[24 * 35 + 7, 0] = np.nan  # day 35, 07 UTC
    frame.iloc[5, 0] = np.nan  # day 0, 05 UTC: no earlier value at that hour
    filled, shares = B.impute_same_hour(frame)
    same_hour_before = values.iloc[[24 * d + 7 for d in range(5, 35)]]  # 30 earlier days
    assert filled.iloc[24 * 35 + 7, 0] == pytest.approx(same_hour_before.median())
    assert np.isnan(filled.iloc[5, 0])  # never filled from the future (no backward fill)
    assert shares["x"] == pytest.approx(2 / len(hours))


def test_imputation_ignores_later_values():
    hours = pd.date_range("2022-01-01", periods=24 * 10, freq="h", tz="UTC")
    a = pd.DataFrame({"x": np.ones(len(hours))}, index=hours)
    a.iloc[24 * 5 + 3, 0] = np.nan
    b = a.copy()
    b.iloc[24 * 5 + 4:, 0] = 999.0  # change everything after the gap
    fa, _ = B.impute_same_hour(a)
    fb, _ = B.impute_same_hour(b)
    assert fa.iloc[24 * 5 + 3, 0] == fb.iloc[24 * 5 + 3, 0] == 1.0


# T-35 D-22 masks -----------------------------------------------------------

def test_masks_follow_each_inverters_own_flags():
    minutes = pd.date_range("2022-05-01", periods=3 * 60, freq="min", tz="UTC")
    flags = pd.DataFrame(False, index=minutes, columns=sorted(
        {f for v in B.INVERTER_FLAGS.values() for f in v}))
    flags.iloc[0:10, flags.columns.get_loc("derate_inv1_flag")] = True  # hour 0: inv1, 17 %
    flags.iloc[60:62, flags.columns.get_loc("downtime_inv2_flag")] = True  # hour 1: 3 %, below
    flags.iloc[120:180, flags.columns.get_loc("curtail_flag")] = True  # hour 2: both
    hours = minutes[::60]
    hourly = pd.DataFrame({c: [100.0, 200.0, 300.0] for c in LAG_SOURCES}, index=hours)
    hourly["op_limited_flag"] = [True, False, True]
    sources, counts = B.masked_sources(hourly, flags, share=0.05)
    assert sources["s01_inv1_AC_kW"].isna().tolist() == [True, False, True]
    assert sources["s01_inv2_AC_kW"].isna().tolist() == [False, False, True]
    assert sources["site_total_AC_kW"].isna().tolist() == [True, False, True]
    for weather in ("irradiance", "mod_temp1", "mod_temp2", "ambient_temp"):
        assert sources[weather].notna().all()
    assert counts == {"s01_inv1_AC_kW": 2, "s01_inv2_AC_kW": 1, "site_total_AC_kW": 2}


# T-36 calendar and geometry --------------------------------------------------

def test_israeli_weekend_and_local_calendar():
    # 2022-07-07 is a Thursday; hours chosen at 12 local (09 UTC in summer)
    hours = pd.DatetimeIndex([pd.Timestamp(f"2022-07-{d:02d} 09:00", tz="UTC")
                              for d in (7, 8, 9, 10)])
    sun = pd.DataFrame({"sunrise": hours - pd.Timedelta(hours=6),
                        "sunset": hours + pd.Timedelta(hours=8),
                        "transit": hours + pd.Timedelta(minutes=30)}, index=hours)
    cal = B.calendar_features(hours, sun, max_day_length=14.0)
    assert cal["is_weekend"].tolist() == [0, 1, 1, 0]  # Thu, Fri, Sat, Sun
    assert cal["hour_local"].tolist() == [12] * 4
    assert cal["time_to_solar_noon"].tolist() == [0.0] * 4  # transit at the middle of the hour
    assert cal["day_length"].tolist() == [14.0] * 4
    assert cal["daylight_fraction"].tolist() == [1.0] * 4


def test_hourly_geometry_is_the_mean_of_minute_values():
    from solarpv.config.site import load_site
    from solarpv.features.geometry import hourly_geometry, load_panel_plane, minute_geometry
    site, plane = load_site(), load_panel_plane()
    hours = pd.date_range("2022-12-21 05:00", periods=3, freq="h", tz="UTC")
    hourly = hourly_geometry(hours, site, plane)
    minutes = minute_geometry(pd.date_range(hours[1], periods=60, freq="min"), site, plane)
    assert hourly.loc[hours[1], "poa_global_clearsky"] == pytest.approx(
        minutes["poa_global_clearsky"].mean())
    assert hourly.loc[hours[1], "sun_elevation"] == pytest.approx(minutes["sun_elevation"].mean())

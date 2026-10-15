"""Stage-5 cleaning rules on synthetic data (docs/stage5_test_plan.md, T-22a..j).

Thresholds come from the real configs/cleaning_policy.yaml, so a change of policy that breaks a
documented behaviour fails here."""

import numpy as np
import pandas as pd
import pytest

from solarpv.data.clean import (
    CURTAIL_FLAG,
    OP_FLAG,
    apply_value_rules,
    clean_minutes,
    load_policy,
    to_hourly,
)

POLICY = load_policy()
DAY, NIGHT, DEEP_NIGHT = 30.0, -2.0, -30.0
NAN = np.nan


def rules(column, values, elev=DAY, source=0):
    v = np.asarray(values, dtype=float)
    e = np.broadcast_to(np.asarray(elev, dtype=float), v.shape).copy()
    s = np.broadcast_to(np.asarray(source), v.shape).copy()
    return apply_value_rules(column, v, e, s, POLICY)


def wiggle(n, base=500.0, step=0.37):
    """A series that never repeats a value."""
    return base + np.arange(n) * step


# T-22a range ---------------------------------------------------------------

def test_fault_code_and_its_blend_neighbours_become_nan():
    out, code = rules("ambient_temp", [20.1, 20.2, -270.0, 20.4, 20.5, 20.6])
    assert np.isnan(out[1:4]).all() and (code[1:4] == 1).all()
    assert out[0] == 20.1 and out[4] == 20.5 and (code[[0, 4, 5]] == 0).all()


@pytest.mark.parametrize("spike", [40825.0, 32766.0, 8302.0])
def test_power_spikes_become_nan_and_the_limit_is_kept(spike):
    v = wiggle(11)
    v[5] = spike
    v[9] = 1897.0  # exactly at the limit
    out, code = rules("s01_inv1_AC_kW", v)
    assert np.isnan(out[4:7]).all() and (code[4:7] == 1).all()
    assert out[9] == 1897.0 and code[9] == 0
    assert (code[[0, 1, 2, 3, 7, 8, 10]] == 0).all()


# T-22b margin by source -----------------------------------------------------

def test_margin_is_one_poll_period_on_the_polled_source():
    v = np.full(41, 25.0) + np.arange(41) * 0.01
    v[20] = -270.0
    _, code_minute = rules("mod_temp1", v, source=0)
    _, code_polled = rules("mod_temp1", v, source=1)
    assert np.flatnonzero(code_minute == 1).tolist() == [19, 20, 21]
    assert np.flatnonzero(code_polled == 1).tolist() == list(range(5, 36))


# T-22c offset ---------------------------------------------------------------

def test_small_negative_irradiance_is_offset_large_is_fault():
    out, code = rules("irradiance", [-7.8, 1.0, 2.0, 3.0, -25.0, 4.0, 5.0, 6.0], elev=NIGHT)
    assert out[0] == 0.0 and code[0] == 2
    assert np.isnan(out[3:6]).all() and (code[3:6] == 1).all()
    assert out[7] == 6.0 and code[7] == 0


# T-22d frozen ---------------------------------------------------------------

def _with_run(n_run, value, base_len=100, start=30, base=500.0, step=0.37):
    v = wiggle(base_len, base, step)
    v[start:start + n_run] = value
    return v


def test_frozen_daytime_run_of_30_is_nan_29_is_kept():
    out, code = rules("s01_inv1_AC_kW", _with_run(30, 1296.05))
    assert np.isnan(out[30:60]).all() and (code[29:61] == 3).all() and (code[:29] == 0).all()
    _, code = rules("s01_inv1_AC_kW", _with_run(29, 1296.05))
    assert (code == 0).all()


def test_frozen_margin_covers_the_blend_minute_after_the_run():
    v = _with_run(30, 1296.05)
    v[60] = 247.4  # communication resumes: stale value blended with the real one
    _, code = rules("s01_inv1_AC_kW", v)
    assert (code[29:61] == 3).all() and code[28] == 0 and code[61] == 0
    _, code = rules("s01_inv1_AC_kW", _with_run(30, 1296.05), source=1)
    assert (code[15:75] == 3).all() and code[14] == 0 and code[75] == 0


def test_zero_run_is_an_inverter_stop_not_a_frozen_sensor():
    out, code = rules("s01_inv1_AC_kW", _with_run(120, 0.0))
    assert (code == 0).all() and (out[30:150] == 0).all()


def test_frozen_run_that_never_reaches_daytime_is_kept():
    _, code = rules("irradiance", _with_run(60, 50.0, base=10.0), elev=DEEP_NIGHT)
    assert (code != 3).all()


def test_ambient_temperature_needs_60_minutes():
    _, code = rules("ambient_temp", _with_run(59, 31.25, base=30.0, step=0.01))
    assert (code == 0).all()
    _, code = rules("ambient_temp", _with_run(60, 31.25, base=30.0, step=0.01))
    assert (code[29:91] == 3).all()


# T-22e/f night --------------------------------------------------------------

def test_night_rule_zeroes_noise_only():
    elev = np.array([NIGHT, NIGHT, NIGHT, DAY, DEEP_NIGHT])
    out, code = rules("s01_inv1_AC_kW", [3.0, -4.0, 20.0, 3.0, 0.0], elev=elev)
    assert out.tolist()[:2] == [0.0, 0.0] and code.tolist()[:2] == [4, 4]
    assert out[2] == 20.0 and code[2] == 0  # real twilight production is kept
    assert out[3] == 3.0 and code[3] == 0  # daytime is never zeroed (T-16)
    assert code[4] == 0  # an exact zero is not a change


def test_power_with_the_sun_far_below_the_horizon_is_stale():
    out, code = rules("s01_inv1_AC_kW", [1296.0, 2.0], elev=DEEP_NIGHT)
    assert np.isnan(out[0]) and code[0] == 5
    assert out[1] == 0.0 and code[1] == 4


# T-22g/h flags --------------------------------------------------------------

def _frame(n=6, **cols):
    idx = pd.date_range("2024-06-01 08:00", periods=n, freq="min", tz="UTC")
    base = {"s01_inv1_AC_kW": wiggle(n), "s01_inv2_AC_kW": wiggle(n, 510.0),
            "irradiance": wiggle(n, 700.0), "iec_sp_w": np.full(n, 100.0)}
    base.update({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    return pd.DataFrame(base, index=idx)


def _clean(df, elev=DAY, poa_cs=900.0):
    e = np.full(len(df), elev, dtype=float)
    return clean_minutes(df, e, np.full(len(df), poa_cs), POLICY)


def test_downtime_is_flagged_and_values_are_kept():
    df = _frame(s01_inv1_AC_kW=[0.0, 0.0, 600.0, 0.0, NAN, 0.0],
                irradiance=[800.0, 100.0, 800.0, NAN, 800.0, 800.0])
    out = _clean(df)
    assert out["downtime_inv1_flag"].tolist() == [True, False, False, True, False, True]
    assert not out["downtime_inv2_flag"].any()
    np.testing.assert_array_equal(out["s01_inv1_AC_kW"], df["s01_inv1_AC_kW"])


def test_curtailment_is_flagged_by_setpoint_in_daytime_only():
    df = _frame(iec_sp_w=[50.0, 100.0, 0.0, 75.0, 100.0, 99.0])
    assert _clean(df)[CURTAIL_FLAG].tolist() == [True, False, True, True, False, True]
    assert not _clean(df, elev=-10.0)[CURTAIL_FLAG].any()
    out = _clean(df)
    assert out[OP_FLAG].tolist() == out[CURTAIL_FLAG].tolist()
    np.testing.assert_array_equal(out["s01_inv1_AC_kW"], df["s01_inv1_AC_kW"])


# T-22i precedence and idempotence -------------------------------------------

def test_first_rule_wins():
    v = _with_run(60, -270.0, base=25.0, step=0.01)  # a fault that is also frozen in daytime
    _, code = rules("mod_temp1", v)
    assert (code[29:91] == 1).all()


def test_cleaning_twice_equals_cleaning_once():
    n = 200
    df = _frame(n, s01_inv1_AC_kW=_with_run(40, 1296.05, base_len=n),
                ambient_temp=np.r_[np.full(10, -270.0), wiggle(n - 10, 25.0, 0.01)],
                irradiance=np.r_[np.full(5, -3.0), wiggle(n - 5, 700.0)])
    elev = np.r_[np.full(5, NIGHT), np.full(n - 5, DAY)]
    once = clean_minutes(df, elev, np.full(n, 900.0), POLICY)
    twice = clean_minutes(once[df.columns], elev, np.full(n, 900.0), POLICY)
    pd.testing.assert_frame_equal(once[df.columns], twice[df.columns])
    assert all((twice[c] == 0).all() for c in twice if c.startswith("rule_"))


# T-22j hourly flags ---------------------------------------------------------

def test_op_limited_hour_needs_5_percent_of_minutes():
    n = 180
    sp = np.full(n, 100.0)
    sp[:3] = 50.0  # hour 1: 3 of 60 minutes -> 0.05
    sp[60:62] = 50.0  # hour 2: 2 of 60 minutes
    df = _frame(n, iec_sp_w=sp)
    df["src_meteo"] = np.r_[np.zeros(120), np.ones(60)].astype("int8")
    df["src_inv1"] = np.r_[np.zeros(150), np.ones(30)].astype("int8")
    df["src_inv2"] = df["src_inv1"]
    out = _clean(df)
    values = [c for c in out.columns if c in df.columns and not c.startswith("src_")]
    hourly = to_hourly(out, [*values, "site_total_AC_kW"], POLICY)
    assert hourly["frac_curtail_flag"].round(4).tolist() == [0.05, 0.0333, 0.0]
    assert hourly[OP_FLAG].tolist() == [True, False, False]
    assert hourly["hour_source"].tolist() == ["minute", "minute", "mixed"]


# T-22l scale fault of a redundant sensor --------------------------------------

def test_pyranometer_scale_fault_removes_the_whole_local_day():
    idx = pd.date_range("2023-06-01 21:00", periods=2 * 1440, freq="min", tz="UTC")  # 2 days
    hour = idx.tz_convert("Asia/Jerusalem").hour
    sunny = (hour >= 9) & (hour < 15)
    irr = np.where(sunny, 800.0, 0.0) + np.arange(len(idx)) * 1e-4
    pyr = irr.copy()
    first_day = np.arange(len(idx)) < 1440
    pyr[first_day] *= 0.75  # day 1: calibration-constant fault
    df = _frame(len(idx), irradiance=irr, pyranometer=pyr).set_axis(idx)
    elev = np.where(sunny, 50.0, -20.0)
    out = clean_minutes(df, elev, np.full(len(idx), 900.0), POLICY)
    assert out["pyranometer"][first_day].isna().all()
    assert (out["rule_pyranometer"][first_day] == 7).all()
    np.testing.assert_array_equal(out["pyranometer"][~first_day], pyr[~first_day])
    np.testing.assert_array_equal(out["irradiance"], irr)  # the reference is untouched


# T-22k dead DC sensor ---------------------------------------------------------

def test_dead_dc_sensor_is_nan_and_ac_is_untouched():
    dc = np.array([0.0, 0.0, 810.0, 0.0])
    ac = np.array([800.0, 50.0, 790.0, 0.0])
    out, code = apply_value_rules("s01_inv1_DC_kW", dc, np.full(4, DAY), np.zeros(4), POLICY, ac)
    assert np.isnan(out[0]) and code[0] == 6
    assert out[1:].tolist() == [0.0, 810.0, 0.0] and (code[1:] == 0).all()
    df = _frame(4, s01_inv1_AC_kW=ac)
    df["s01_inv1_DC_kW"] = dc
    cleaned = _clean(df)
    assert np.isnan(cleaned["s01_inv1_DC_kW"].iloc[0])
    np.testing.assert_array_equal(cleaned["s01_inv1_AC_kW"], ac)

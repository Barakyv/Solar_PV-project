"""Clock-offset estimation helpers and correction (F-15, D-19)."""

import numpy as np
import pandas as pd
import pytest

from solarpv.data.clock import (
    apply_correction,
    clock_group,
    coarse_steps,
    refine_step,
    shift_fit,
    smooth_within_segments,
)


def test_clock_groups():
    assert clock_group("s01_inv1_AC_kW") == "inverter"
    assert clock_group("iec_sp_w") == "inverter"
    assert clock_group("irradiance") == "meteo"
    assert clock_group("mod_temp1") == "meteo"


def test_shift_fit_recovers_a_known_offset():
    t = np.arange(1440, dtype=float)
    model = np.clip(np.sin(np.pi * (t - 360) / 720), 0, None) * 1000
    true_offset = 5.3  # stamped later than true time
    meas = 0.9 * np.interp(t - true_offset, t, model)
    tau, rel_rmse = shift_fit(meas, model, 420, 1020)
    assert tau == pytest.approx(true_offset, abs=0.1)
    assert rel_rmse < 0.01


def _days(start, n):
    return pd.Series(pd.date_range(start, periods=n, freq="D"))


def test_steps_are_found_and_refined_to_the_day():
    rng = np.random.default_rng(1)
    days = _days("2023-03-01", 140)
    values = pd.Series(np.where(days < pd.Timestamp("2023-05-09"), -8.0, 0.0)
                       + rng.normal(0, 0.4, len(days)))
    coarse = coarse_steps(days, values, threshold=3.0)
    assert len(coarse) == 1
    assert refine_step(days, values, coarse[0]) == pd.Timestamp("2023-05-09")


def test_smoothing_does_not_cross_steps():
    days = _days("2023-01-01", 100)
    step = pd.Timestamp("2023-02-20")
    values = pd.Series(np.where(days < step, -9.0, 0.0))
    out = smooth_within_segments(days, values, [step], pd.DatetimeIndex(days))
    assert (out[out.index < step] == -9.0).all()
    assert (out[out.index >= step] == 0.0).all()


def _offsets(days, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(days))


def test_apply_correction_restamps_each_group_and_source():
    # two full local days (UTC+3)
    idx = pd.date_range("2024-06-01 21:00", "2024-06-03 20:59", freq="min", tz="UTC")
    minute = np.arange(len(idx), dtype=float)
    df = pd.DataFrame({"irradiance": minute, "s01_inv1_AC_kW": minute}, index=idx)
    offsets = _offsets(["2024-06-02", "2024-06-03"],
                       minute_meteo=[-10.0, -10.0], minute_inverter=[3.0, 3.0],
                       polled_meteo=[2.0, 2.0], polled_inverter=[2.0, 2.0])
    out = apply_correction(df, offsets)
    # meteo stamped 10 min early: the value stamped at minute m belongs at m + 10
    assert out["irradiance"].iloc[600] == 590
    assert out["s01_inv1_AC_kW"].iloc[600] == 603
    # a day-constant shift loses nothing in the middle of the day
    assert out.iloc[60:2800].notna().all().all()

    source = pd.DataFrame("minute", index=idx, columns=df.columns)
    source.iloc[1440:, 0] = "polled"
    out2 = apply_correction(df, offsets, source)
    assert out2["irradiance"].iloc[2000] == 2002


def test_apply_correction_requires_offsets_for_every_day():
    idx = pd.date_range("2024-06-01 21:00", periods=1440 * 2, freq="min", tz="UTC")
    df = pd.DataFrame({"irradiance": 1.0}, index=idx)
    offsets = _offsets(["2024-06-02"], minute_meteo=[0.0], minute_inverter=[0.0])
    with pytest.raises(ValueError, match="missing clock offset"):
        apply_correction(df, offsets)

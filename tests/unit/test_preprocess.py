"""Hourly aggregation rule (PROJECT_MASTER.md §6.4, assumption A-09)."""

import numpy as np
import pandas as pd

from solarpv.data.preprocess import hourly_mean


def test_hourly_mean_is_left_labelled_with_45_minute_coverage():
    idx = pd.date_range("2024-06-01 08:00", periods=180, freq="min", tz="UTC")
    v = np.arange(180, dtype=float)
    v[60:76] = np.nan  # hour 2: 44 valid minutes -> NaN
    v[120:135] = np.nan  # hour 3: 45 valid minutes -> mean of the valid ones
    means, counts = hourly_mean(pd.DataFrame({"x": v}, index=idx))
    assert list(means.index) == list(pd.date_range("2024-06-01 08:00", periods=3, freq="h",
                                                   tz="UTC"))
    assert means["x"].iloc[0] == np.mean(np.arange(60))
    assert np.isnan(means["x"].iloc[1])
    assert means["x"].iloc[2] == np.mean(np.arange(135, 180))
    assert counts["x"].tolist() == [60, 44, 45]

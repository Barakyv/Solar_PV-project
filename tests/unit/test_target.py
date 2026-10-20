"""Stage-6 targets on synthetic data (PROJECT_MASTER.md §6.5; T-11, T-13)."""

import numpy as np
import pandas as pd
import pytest

from solarpv.features.target import clear_sky_index, suspect, theoretical_power, to_kw


def _hours(n):
    return pd.date_range("2022-06-01", periods=n, freq="h", tz="UTC")


def test_k_inverts_exactly_to_ac():  # T-11
    idx = _hours(5)
    theo = theoretical_power(pd.Series([0.0, 10.0, 300.0, 800.0, 1000.0], index=idx))
    ac = pd.Series([0.0, 5.0, 900.0, 2500.0, 2900.0], index=idx)
    k, valid = clear_sky_index(ac, theo)
    assert valid.tolist() == [False, False, True, True, True]  # theo > 50 kW only
    np.testing.assert_allclose(to_kw(k[valid], theo[valid], clip=False), ac[valid], rtol=0,
                               atol=1e-9)
    assert (k[~valid] == 0).all()


def test_k_is_never_clipped():  # T-13
    idx = _hours(3)
    theo = pd.Series([1000.0, 1000.0, 1000.0], index=idx)
    ac = pd.Series([1300.0, 1200.0, 1600.0], index=idx)
    k, _ = clear_sky_index(ac, theo)
    assert k.tolist() == [1.3, 1.2, 1.6]
    assert suspect(k).tolist() == [False, False, True]  # flagged, not cut


def test_missing_ac_gives_missing_k():
    idx = _hours(2)
    k, valid = clear_sky_index(pd.Series([np.nan, 500.0], index=idx),
                               pd.Series([1000.0, 1000.0], index=idx))
    assert np.isnan(k.iloc[0]) and not valid.iloc[0]
    assert k.iloc[1] == 0.5


def test_forecast_conversion_is_clipped_to_the_plant():
    idx = _hours(3)
    kw = to_kw(pd.Series([-0.1, 0.5, 2.0], index=idx), pd.Series([1000.0, 1000.0, 3000.0],
                                                                  index=idx))
    assert kw.tolist() == [0.0, 500.0, pytest.approx(3449.0)]

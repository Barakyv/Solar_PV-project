"""T-07: the evaluation daytime mask reproduces the original sample sizes (finding F-02)."""

import pandas as pd
import pytest

from solarpv.config import constants as C
from solarpv.config.site import load_site
from solarpv.evaluation.daytime import daytime_mask

pytestmark = pytest.mark.needs_private_site


def test_test_window_has_2580_daytime_hours():
    idx = pd.date_range(C.TEST_START, "2025-05-31 23:00", freq="h", tz="UTC")
    assert int(daytime_mask(idx, load_site()).sum()) == 2580


@pytest.mark.parametrize(("horizon", "expected"), [(1, 2510), (2, 2512), (3, 2514)])
def test_lstm_fixture_daytime_counts(lstm_fixture, horizon, expected):
    df = lstm_fixture
    sub = df[(df["target"] == "indirect") & (df["horizon"] == horizon)]
    idx = pd.DatetimeIndex(sub["timestamp"])
    assert int(daytime_mask(idx, load_site()).sum()) == expected


def test_rejects_naive_index():
    with pytest.raises(ValueError):
        daytime_mask(pd.date_range("2024-10-01", periods=3, freq="h"))

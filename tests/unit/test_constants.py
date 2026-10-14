import pandas as pd
import pytest

from solarpv.config import constants as C


def test_splits_are_ordered_and_contiguous():
    train_end = pd.Timestamp(C.TRAIN_END, tz="UTC")
    val_start = pd.Timestamp(C.VAL_START, tz="UTC")
    val_end = pd.Timestamp(C.VAL_END, tz="UTC")
    test_start = pd.Timestamp(C.TEST_START, tz="UTC")
    assert train_end < val_start < val_end < test_start
    assert val_start - train_end == pd.Timedelta(minutes=1)
    assert test_start - val_end == pd.Timedelta(minutes=1)


def test_expected_sizes_match_data_range():
    start = pd.Timestamp(C.DATA_START_UTC, tz="UTC")
    end = pd.Timestamp(C.DATA_END_UTC, tz="UTC")
    minutes = int((end - start) / pd.Timedelta(minutes=1)) + 1
    assert minutes == C.N_MINUTES_EXPECTED
    assert C.N_MINUTES_EXPECTED == C.N_HOURS_EXPECTED * 60


def test_data_range_is_local_midnight_to_midnight():
    tz = C.TIMEZONE_LOCAL
    assert pd.Timestamp(C.DATA_START_UTC, tz="UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M") == (
        "2021-09-01 00:00"
    )
    assert pd.Timestamp(C.DATA_END_UTC, tz="UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M") == (
        "2025-05-31 23:59"
    )


def test_test_window_has_5832_hours():
    idx = pd.date_range(C.TEST_START, "2025-05-31 23:00", freq="h", tz="UTC")
    assert len(idx) == 5832


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (1, ["a_lag_1d", "a_lag_2d", "a_lag_14d", "b"]),
        (2, ["a_lag_2d", "a_lag_14d", "b"]),
        (3, ["a_lag_14d", "b"]),
    ],
)
def test_get_features_drops_exact_lag_tokens(horizon, expected):
    feats = ["a_lag_1d", "a_lag_2d", "a_lag_14d", "b"]
    assert C.get_features(feats, horizon) == expected


def test_get_features_rejects_unknown_horizon():
    with pytest.raises(ValueError):
        C.get_features(["x"], 4)


def test_norm_factor():
    assert C.norm_factor("direct") == C.AC_CAPACITY_KW
    assert C.norm_factor("indirect", mean_k=0.5) == 0.5
    with pytest.raises(ValueError):
        C.norm_factor("other")

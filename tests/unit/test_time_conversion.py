"""T-05: local Israel time with DST -> UTC (fix for finding F-01)."""

import pandas as pd
import pytest

from solarpv.data.ingest import to_utc


def test_winter_and_summer_offsets():
    local = pd.to_datetime(["2024-01-15 12:00", "2024-07-15 12:00"])
    utc = to_utc(local)
    assert list(utc.strftime("%Y-%m-%d %H:%M")) == ["2024-01-15 10:00", "2024-07-15 09:00"]


def test_fall_back_hour_recorded_twice_in_order():
    # 2024-10-27: local 01:00-01:59 occurs twice (IDT, then IST), as in the raw files
    first = pd.date_range("2024-10-27 00:58", "2024-10-27 01:59", freq="min")
    second = pd.date_range("2024-10-27 01:00", "2024-10-27 02:01", freq="min")
    utc = to_utc(first.append(second))
    assert utc.is_monotonic_increasing and utc.is_unique
    assert (utc[1:] - utc[:-1] == pd.Timedelta(minutes=1)).all()
    assert utc[2] == pd.Timestamp("2024-10-26 22:00", tz="UTC")  # first 01:00 = IDT
    assert utc[62] == pd.Timestamp("2024-10-26 23:00", tz="UTC")  # second 01:00 = IST


def test_spring_forward_gap_is_skipped_by_the_raw_files():
    # 2025-03-28: local 02:00-02:59 does not exist; the raw files jump from 01:59 to 03:00
    local = pd.to_datetime(["2025-03-28 01:59", "2025-03-28 03:00"])
    utc = to_utc(local)
    assert utc[1] - utc[0] == pd.Timedelta(minutes=1)


def test_nonexistent_local_time_raises():
    with pytest.raises(Exception, match="2025-03-28 02:30"):
        to_utc(pd.to_datetime(["2025-03-28 02:30"]))


def test_rejects_tz_aware_input():
    with pytest.raises(ValueError):
        to_utc(pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))

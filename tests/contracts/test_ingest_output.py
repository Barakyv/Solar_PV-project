"""Contract of the stage-2 output (data/processed/01_ingest) and test T-08.

Skipped when the stage has not been run (the parquet is not in git).
"""

import json

import pandas as pd
import pytest

from solarpv.config import constants as C
from solarpv.config.paths import PROCESSED_DIR, RAW_DIR
from solarpv.data.ingest import TAG_MAP, raw_files

STAGE = PROCESSED_DIR / "01_ingest"
PARQUET = STAGE / "scada_1min_utc.parquet"

needs_output = pytest.mark.skipif(not PARQUET.exists(), reason="stage 2 output not present")


@pytest.mark.needs_raw
def test_tag_map_matches_raw_header():
    header = pd.read_csv(raw_files()[0], nrows=0).columns
    assert set(header) == {"t_stamp", *TAG_MAP}


@pytest.mark.needs_raw
def test_45_monthly_files():
    assert len(raw_files(RAW_DIR)) == 45


@pytest.fixture(scope="module")
def data():
    return pd.read_parquet(PARQUET)


@needs_output
def test_time_axis_is_full_utc_minute_grid(data):
    idx = data.index
    assert str(idx.tz) == "UTC"
    assert len(idx) == C.N_MINUTES_EXPECTED
    assert idx[0] == pd.Timestamp(C.DATA_START_UTC, tz="UTC")
    assert idx[-1] == pd.Timestamp(C.DATA_END_UTC, tz="UTC")
    assert idx.is_unique and idx.is_monotonic_increasing
    assert (idx[1:] - idx[:-1] == pd.Timedelta(minutes=1)).all()


@needs_output
def test_columns_are_canonical(data):
    assert list(data.columns) == list(TAG_MAP.values())


@needs_output
def test_repeated_october_hour_holds_two_distinct_readings(data):
    """T-05 on real data: local 2024-10-27 01:30 occurs at 22:30 and 23:30 UTC."""
    a = data.loc[pd.Timestamp("2024-10-26 22:30", tz="UTC")]
    b = data.loc[pd.Timestamp("2024-10-26 23:30", tz="UTC")]
    assert not a.equals(b)


@needs_output
def test_manifest_consistent(data):
    manifest = json.loads((STAGE / "ingest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == len(data)
    assert manifest["raw_files_verified_against_checksums"] == 45


@needs_output
@pytest.mark.slow
def test_production_centred_on_solar_noon(data):
    """T-08: clear-day DC centroid within +-30 min of solar noon in every month (F-01).

    The drift of up to ~13 min is a logger-clock effect (F-15), corrected in stage 4.
    """
    from solarpv.data.qa import production_centroid_offsets

    off = production_centroid_offsets(data)
    off["month"] = [f"{d.year}-{d.month:02d}" for d in off["day"]]
    monthly = off.groupby("month")["offset_min"].median()
    assert len(monthly) == 45
    assert monthly.abs().max() < 30

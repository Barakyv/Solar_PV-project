"""Contract of the stage-4 output (data/processed/04_preprocess) and the tightened T-08.

Skipped when the stage has not been run (the parquet files are not in git).
"""

import json

import numpy as np
import pandas as pd
import pytest

from solarpv.config import constants as C
from solarpv.config.paths import CONFIGS_DIR, PROCESSED_DIR
from solarpv.config.provenance import sha256, text_sha256
from solarpv.data.preprocess import AC_COLUMNS, COUNTED, TARGET, hourly_mean

STAGE = PROCESSED_DIR / "04_preprocess"
HOURLY = STAGE / "scada_1H_utc.parquet"
MINUTE = STAGE / "scada_1min_true_utc.parquet"
T08_TOLERANCE_MIN = 2.0  # D-19 acceptance

needs_output = pytest.mark.skipif(not HOURLY.exists(), reason="stage 4 output not present")


@pytest.fixture(scope="module")
def hourly():
    return pd.read_parquet(HOURLY)


@pytest.fixture(scope="module")
def minute():
    return pd.read_parquet(MINUTE)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((STAGE / "preprocess_manifest.json").read_text(encoding="utf-8"))


@needs_output
def test_hourly_axis(hourly):
    idx = hourly.index
    assert str(idx.tz) == "UTC"
    assert len(idx) == C.N_HOURS_EXPECTED
    assert idx[0] == pd.Timestamp(C.DATA_START_UTC, tz="UTC")
    assert (idx[1:] - idx[:-1] == pd.Timedelta(hours=1)).all()


@needs_output
def test_columns(hourly):
    assert TARGET in hourly.columns
    assert {f"n_{c}" for c in COUNTED} <= set(hourly.columns)


@needs_output
def test_hourly_values_are_means_of_the_corrected_minutes(hourly, minute):
    means, _ = hourly_mean(minute[[TARGET, "irradiance"]])
    pd.testing.assert_frame_equal(means, hourly[[TARGET, "irradiance"]], check_names=False,
                                  check_freq=False)


@needs_output
def test_site_total_is_sum_of_inverters(minute):
    both = minute[list(AC_COLUMNS)].notna().all(axis=1)
    np.testing.assert_allclose(minute.loc[both, TARGET],
                               minute.loc[both, list(AC_COLUMNS)].sum(axis=1))
    assert minute.loc[~both, TARGET].isna().all()


@needs_output
def test_manifest_matches_inputs_and_outputs(manifest, hourly):
    assert manifest["rows_1H"] == len(hourly)
    assert manifest["clock_offsets"]["sha256"] == text_sha256(CONFIGS_DIR / "clock_offsets.csv")
    assert manifest["outputs"]["scada_1H_utc.parquet"] == sha256(HOURLY)


@needs_output
def test_no_daytime_hours_lost_to_a_time_shift(manifest):
    """Stage-4 acceptance vs F-01c: the old pipeline had 339 'zero' daytime hours, all an
    artefact of local time read as UTC. Zero hours left now are real stops (stage 5)."""
    s = manifest["site_total_AC_kW"]
    assert s["daytime_hours_zero"] < 0.01 * s["daytime_hours"]


def _monthly_centroid(minute, columns):
    from solarpv.data.qa import production_centroid_offsets

    off = production_centroid_offsets(minute.ffill(limit=5), columns=columns)
    off["month"] = [f"{d.year}-{d.month:02d}" for d in off["day"]]
    return off.groupby("month")["offset_min"].median()


@needs_output
@pytest.mark.needs_private_site  # compares with the true site's sun (RH-01)
@pytest.mark.slow
def test_meteo_clock_centred_on_solar_noon(minute):
    """T-08 (tightened by D-19), meteo clock: clear-day centroid of the in-plane irradiance
    sensor within +-2 min of solar noon in every month."""
    monthly = _monthly_centroid(minute, ("irradiance",))
    assert len(monthly) == 45
    assert monthly.abs().max() <= T08_TOLERANCE_MIN, monthly[monthly.abs() > T08_TOLERANCE_MIN]


@needs_output
@pytest.mark.needs_private_site  # compares with the true site's sun (RH-01)
@pytest.mark.slow
def test_inverter_clock_matches_meteo_clock(minute):
    """T-08 (tightened by D-19), inverter clock: on the minute source, the DC-irradiance lag
    from short-term fluctuations (delta) is within +-1 min in every month. This is the clock
    test for the inverters; the DC centroid itself sits 1-3 min after solar noon for physical
    reasons (array vs sensor orientation, see docs/deviations.md)."""
    from solarpv.config.site import load_site
    from solarpv.data.clock import Orientation, clear_sky_components, daily_estimates

    site = load_site()
    comp = clear_sky_components(minute.index, site)
    on_minute = (minute["src_meteo"].eq(0) & minute["src_inv1"].eq(0)
                 & minute["src_inv2"].eq(0)).to_numpy()
    usable = pd.DataFrame({"irradiance": on_minute, "dc": on_minute}, index=minute.index)
    est = daily_estimates(minute, comp, Orientation(site.surface_tilt, site.surface_azimuth),
                          usable)
    d = est[(est["delta_corr"] >= 0.7) & (est["variability"] >= 0.005)
            & (est["delta"].abs() < 29)]
    monthly = d.groupby(d["day"].dt.to_period("M"))["delta"].median()
    assert len(monthly) >= 40
    assert monthly.abs().max() <= 1.0, monthly[monthly.abs() > 1.0]


@needs_output
@pytest.mark.needs_private_site  # compares with the true site's sun (RH-01)
@pytest.mark.slow
def test_dc_centroid_near_solar_noon(minute):
    """Coarse guard on the inverter group: DC centroid within +-3 min (physical asymmetry of up
    to ~2.7 min in summer, documented in docs/deviations.md)."""
    monthly = _monthly_centroid(minute, ("s01_inv1_DC_kW", "s01_inv2_DC_kW"))
    assert len(monthly) == 45
    assert monthly.abs().max() <= 3.0, monthly[monthly.abs() > 3.0]

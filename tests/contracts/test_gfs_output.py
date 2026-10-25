"""T-17 and T-44: the real GFS tables of stage 6b (docs/stage6b_plan.md §5).

Skipped when stage 6b has not been run; the parquet files are not in git.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from solarpv.config.constants import HORIZONS, N_HOURS_EXPECTED, NWP_MIN_AVAILABILITY_HOURS
from solarpv.config.paths import PROCESSED_DIR
from solarpv.features.build import issuance_utc
from solarpv.nwp import check as K
from solarpv.nwp.gfs import PREFIX, REPORT_DIR, STAGE_DIR

FILES = {h: STAGE_DIR / f"gfs_features_D{h}.parquet" for h in HORIZONS}
pytestmark = pytest.mark.skipif(not FILES[1].exists(), reason="stage 6b output not present")


@pytest.fixture(scope="module")
def tables():
    return {h: pd.read_parquet(p) for h, p in FILES.items()}


@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_row_uses_a_run_published_before_issuance(tables, horizon):  # T-17
    t = tables[horizon].dropna(subset=["gfs_init"])
    t0 = issuance_utc(t.index, horizon)
    assert (pd.DatetimeIndex(t["t0_utc"]) == t0).all()
    assert (t["gfs_init"] + pd.Timedelta(hours=NWP_MIN_AVAILABILITY_HOURS) <= t["t0_utc"]).all()
    assert (t["gfs_uploaded_max"] <= t["t0_utc"]).all()
    assert (t["gfs_run"] == 12).mean() > 0.99  # fallbacks are rare and flagged
    assert (t.loc[t["gfs_run"] != 12, "gfs_fallback"]).all()


def test_coverage_and_axis(tables):
    for h, t in tables.items():
        assert len(t) == N_HOURS_EXPECTED and str(t.index.tz) == "UTC"
        assert t[f"{PREFIX}ghi"].notna().mean() > 0.99, h


def test_the_same_target_hour_differs_between_horizons(tables):  # N2
    common = tables[1].index[tables[1][f"{PREFIX}ghi"].gt(100)
                             & tables[3][f"{PREFIX}ghi"].notna()]
    diff = (tables[1].loc[common, f"{PREFIX}ghi"] - tables[3].loc[common, f"{PREFIX}ghi"]).abs()
    assert (diff > 1e-6).mean() > 0.95


def test_no_negative_irradiance_and_percentages_in_range(tables):
    for t in tables.values():
        assert (t[f"{PREFIX}ghi"].dropna() >= 0).all()
        for c in ("tcc", "lcc", "mcc", "hcc", "rh2m"):
            assert t[f"{PREFIX}{c}"].dropna().between(0, 100).all(), c


def test_sanity_record():
    sanity = json.loads((REPORT_DIR / "sanity.json").read_text(encoding="utf-8"))
    assert sanity["best_lag_h"] == 0  # N3: no clock shift between GFS and the plant
    assert not sanity["red_flag_too_good_at_D3"]
    assert not sanity["red_flag_skill_rises_with_horizon"]


def test_decision_follows_the_rule():
    """The recorded D-12 decision is what the rule gives on the recorded value check."""
    values = pd.read_csv(REPORT_DIR / "value_check.csv")
    record = yaml.safe_load(K.DECISION_FILE.read_text(encoding="utf-8"))
    assert record["admitted"] == K.decide(values)["admitted"]
    assert set(values["features"]) == {"contract", "contract+gfs"}
    assert np.isfinite(values["nrmse_kw_mean"]).all()


def test_the_check_reads_no_test_row():
    t = K.load(1)
    assert set(t["split"]) <= {"train", "val"}
    assert (PROCESSED_DIR / "06_features" / "features_1H_D1.parquet").exists()

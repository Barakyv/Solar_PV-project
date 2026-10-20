"""Contract of the stage-6 output (docs/stage6_plan.md): T-33, and T-01 / T-09 / T-11 / T-13 /
T-18 / T-35 on the real data. Skipped when stage 6 has not been run (the parquet files are not
in git).
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from solarpv.config import constants as C
from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from solarpv.features import build as B
from solarpv.features.contract import features_for_horizon
from solarpv.features.leakage import is_leaky

STAGE = PROCESSED_DIR / "06_features"
FILES = {h: STAGE / f"features_1H_D{h}.parquet" for h in C.HORIZONS}
HOURLY = PROCESSED_DIR / "05_clean" / "scada_1H_clean.parquet"
pytestmark = pytest.mark.skipif(not FILES[1].exists(), reason="stage 6 output not present")


@pytest.fixture(scope="module")
def tables():
    return {h: pd.read_parquet(p) for h, p in FILES.items()}


@pytest.fixture(scope="module")
def hourly():
    return pd.read_parquet(HOURLY)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((STAGE / "features_manifest.json").read_text(encoding="utf-8"))


# T-33 ---------------------------------------------------------------------------

def test_rows_axis_and_feature_sets(tables):
    for h, t in tables.items():
        assert len(t) == C.N_HOURS_EXPECTED and t.index.is_unique
        assert str(t.index.tz) == "UTC" and t.index[0] == pd.Timestamp(C.DATA_START_UTC,
                                                                       tz="UTC")
        feats = features_for_horizon(h)
        assert len(feats) == 74 - 7 * (h - 1)
        assert all(f in t.columns for f in feats)
        dropped = {f for f in features_for_horizon(1) if f not in feats}
        assert not dropped & set(t.columns), f"D+{h} carries dropped lags"


def test_no_leaky_column_among_the_features(tables):  # T-01 at matrix level
    for h, t in tables.items():
        leaky = [f for f in features_for_horizon(h) if f in t.columns and is_leaky(f)]
        assert not leaky, leaky


def test_metadata_agrees_with_stage5(tables, hourly):
    t = tables[1]
    assert (t["op_limited_flag"] == hourly["op_limited_flag"].astype(bool)).all()
    assert (t["hour_source"] == hourly["hour_source"]).all()
    np.testing.assert_array_equal(t["site_total_AC_kW"].to_numpy(),
                                  hourly["site_total_AC_kW"].to_numpy())
    counts = t["split"].value_counts().to_dict()
    assert counts == {"train": 21195, "val": 5832, "test": 5829}
    assert int(t["daytime"].sum()) == 15180
    assert int(t["warmup"].sum()) == B.WARMUP_DAYS * 24


def test_issuance_times(tables):
    for h, t in tables.items():
        local = t["t0_utc"].dt.tz_convert(C.TIMEZONE_LOCAL)
        assert ((local.dt.hour == 0) & (local.dt.minute == 0)).all()
        assert (t["t0_utc"] <= t.index).all()
        assert (t.index - t["t0_utc"] < pd.Timedelta(hours=24 * h + 1)).all()


def test_decided_target_columns(tables):
    decision = yaml.safe_load((CONFIGS_DIR / "target.yaml").read_text(encoding="utf-8"))
    den = decision["k_denominator"]
    t = tables[1]
    assert (t["theo_kw"] == t[f"theo_{den}_kw"]).all()
    np.testing.assert_array_equal(t["target_k"].to_numpy(), t[f"k_{den}"].to_numpy())
    derived = yaml.safe_load(C.DERIVED_CONSTANTS_FILE.read_text(encoding="utf-8"))
    m = t["k_valid"] & t["daytime"] & (t["split"] == "train")
    assert derived["mean_daytime_k"] == pytest.approx(float(t.loc[m, "target_k"].mean()),
                                                      abs=1e-6)
    assert C.mean_daytime_k() == derived["mean_daytime_k"]


# T-11 / T-13 ------------------------------------------------------------------------

def test_k_inverts_exactly(tables):  # T-11
    t = tables[1]
    v = t["k_valid"]
    err = (t.loc[v, "target_k"] * t.loc[v, "theo_kw"] - t.loc[v, "site_total_AC_kW"]).abs()
    assert err.max() < 1e-6


def test_k_is_not_clipped(tables):  # T-13
    t = tables[1]
    k = t.loc[t["k_valid"], "target_k"]
    assert (k >= 0).all()
    assert int((k == 1.2).sum()) == 0  # the original piled 16.8 % of test hours at exactly 1.2
    assert (k > 1.2).any()  # values above 1.2 exist and are kept
    assert t.loc[t["k_suspect"], "target_k"].gt(C.K_SUSPECT_MAX).all()


# T-09 / T-18 / T-35 on real data -------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("horizon", C.HORIZONS)
def test_real_features_do_not_see_past_issuance(horizon, hourly):
    """Perturb the stage-5 values from T0 on, around the 2022 fall-back day, and rebuild."""
    window = hourly.loc["2022-10-01":"2022-11-05"]
    flags = pd.DataFrame(False, index=pd.date_range(window.index[0], window.index[-1]
                                                    + pd.Timedelta(minutes=59), freq="min"),
                         columns=sorted({f for v in B.INVERTER_FLAGS.values() for f in v}))
    sources, _ = B.masked_sources(window, flags, share=0.05)
    base, _, _, t0 = B.past_day_features(sources, horizon)
    local_day = base.index.tz_convert(C.TIMEZONE_LOCAL).tz_localize(None).normalize()
    for day in (pd.Timestamp("2022-10-29"), pd.Timestamp("2022-10-30"),
                pd.Timestamp("2022-10-31")):
        rows = local_day == day
        issue = t0[rows][0]
        changed = {c: s.where(s.index < issue, -s - 1.0) for c, s in sources.items()}
        again, _, _, _ = B.past_day_features(changed, horizon)
        pd.testing.assert_frame_equal(base[rows], again[rows], check_exact=True)


def test_masked_power_never_feeds_a_past_day_feature(tables, hourly):  # T-35
    """Where the site was op-limited at hour s, no site-total lag of any later row equals the
    raw value of hour s (it was replaced by NaN, then imputed from other hours)."""
    t = tables[1]
    limited = hourly["op_limited_flag"].astype(bool) & hourly["site_total_AC_kW"].gt(100)
    raw = hourly["site_total_AC_kW"]
    for d in (1, 2, 3, 7, 14):
        lag = t[f"site_total_AC_kW_lag_{d}d"]
        source_limited = limited.shift(24 * d, fill_value=False).to_numpy()
        same = np.isclose(lag.to_numpy(), raw.shift(24 * d).to_numpy())
        assert not (source_limited & same).any(), d


def test_manifest_and_dictionary(manifest, tables):
    assert manifest["rows_per_horizon"] == {f"D{h}": C.N_HOURS_EXPECTED for h in C.HORIZONS}
    assert manifest["dst_guard_removed_values"] == {"D1": 28, "D2": 56, "D3": 84}
    dictionary = pd.read_csv(OUTPUTS_DIR / "06_features" / "feature_dictionary.csv")
    assert set(dictionary["feature"]) == set(features_for_horizon(1))
    assert dictionary["available_at_T0"].notna().all()
    assert dictionary["definition"].is_unique  # one definition per feature, not per group
    assert dictionary["unit"].isin(["W/m2", "deg", "kW", "degC", "h", "-", "0/1"]).all()

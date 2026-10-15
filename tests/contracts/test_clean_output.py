"""Contract of the stage-5 output (docs/stage5_test_plan.md: T-23..T-28, T-16, T-01).

The rule conditions are re-derived here with independent code (pandas group-bys instead of the
module's numpy helpers), so a bug in solarpv.data.clean cannot hide behind itself.
Skipped when stage 5 has not been run (the parquet files are not in git).
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from solarpv.config import constants as C
from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from solarpv.config.provenance import sha256, text_sha256
from solarpv.data.preprocess import AC_COLUMNS, TARGET, hourly_mean
from solarpv.features.leakage import is_leaky

STAGE = PROCESSED_DIR / "05_clean"
MINUTE = STAGE / "scada_1min_clean.parquet"
HOURLY = STAGE / "scada_1H_clean.parquet"
BEFORE = PROCESSED_DIR / "04_preprocess" / "scada_1min_true_utc.parquet"
POLICY = yaml.safe_load((CONFIGS_DIR / "cleaning_policy.yaml").read_text(encoding="utf-8"))
NAN_RULES, ZERO_RULES = {1, 3, 5, 6, 7}, {2, 4}
PRECEDENCE = [1, 3, 6, 7, 5, 2, 4]

needs_output = pytest.mark.skipif(not HOURLY.exists(), reason="stage 5 output not present")
pytestmark = needs_output


def cleaned_columns():
    return sorted(set(POLICY["range"]) | set(POLICY["negative_to_zero"])
                  | set(POLICY["frozen_min_minutes"]) | set(POLICY["night"]["columns"])
                  | set(POLICY["dc_dead"]["pairs"]) | set(POLICY["scale_fault"]["pairs"]))


@pytest.fixture(scope="module")
def after():
    return pd.read_parquet(MINUTE)


@pytest.fixture(scope="module")
def before():
    return pd.read_parquet(BEFORE)


@pytest.fixture(scope="module")
def hourly():
    return pd.read_parquet(HOURLY)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((STAGE / "cleaning_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sun(after):
    from solarpv.config.site import load_site
    from solarpv.data.clock import Orientation, clear_sky_components, poa_from_components

    site = load_site()
    comp = clear_sky_components(after.index, site)
    poa = poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth))
    return comp["elev"].to_numpy(), poa


def _source_of(col, frame):
    ref = ("src_inv2" if col.startswith("s01_inv2_") else
           "src_inv1" if col.startswith(("s01_", "iec_")) else "src_meteo")
    return frame[ref].to_numpy()


def _runs(values: pd.Series, daytime: np.ndarray):
    """Run length and 'reaches daytime' per minute, by pandas group-by (NaN breaks runs)."""
    run = values.ne(values.shift()).cumsum()
    g = pd.DataFrame({"run": run.to_numpy(), "day": daytime}).groupby("run")["day"]
    return g.transform("size").to_numpy(), g.transform("max").to_numpy()


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Distance in minutes of every minute to the nearest True of `mask` (inf if none)."""
    hits = np.flatnonzero(mask)
    pos = np.arange(len(mask))
    if len(hits) == 0:
        return np.full(len(mask), np.inf)
    k = np.searchsorted(hits, pos)
    right = hits[np.minimum(k, len(hits) - 1)]
    left = hits[np.maximum(k - 1, 0)]
    return np.minimum(np.abs(pos - right), np.abs(pos - left)).astype(float)


def _conditions(col, before, sun, after=None):
    """Minutes where each rule's condition holds on the stage-4 values (independent code).
    R7 compares against the cleaned reference sensor, taken from `after`."""
    elev, _ = sun
    v = before[col]
    a = v.to_numpy(dtype=float)
    finite = np.isfinite(a)
    cond = {}
    if col in POLICY["range"]:
        lo, hi = POLICY["range"][col]
        bad_pos = np.flatnonzero(finite & ((a < lo) | (a > hi)))
        margin = np.where(_source_of(col, before) == 1, POLICY["range_margin_minutes"]["polled"],
                          POLICY["range_margin_minutes"]["minute"])
        cond[1] = finite & (_distance_to(np.isin(np.arange(len(a)), bad_pos)) <= margin)
        if col in POLICY["negative_to_zero"]:
            cond[2] = finite & (a < 0) & (a >= lo)
    if col in POLICY["frozen_min_minutes"]:
        length, reaches = _runs(v, elev > POLICY["sun"]["daytime_min_elevation"])
        core = finite & (length >= POLICY["frozen_min_minutes"][col]) & reaches
        if col in POLICY["frozen_ignore_zero"]:
            core &= a != 0
        fm = POLICY["frozen_margin_minutes"]
        margin = np.where(_source_of(col, before) == 1, fm["polled"], fm["minute"])
        cond[3] = finite & (_distance_to(core) <= margin)
    if col in POLICY["night"]["columns"]:
        n = POLICY["night"]
        cond[4] = finite & (elev < n["max_elevation"]) & (np.abs(a) <= n["noise_kw"]) & (a != 0)
        cond[5] = finite & (elev < n["stale_below_elevation"]) & (a > n["noise_kw"])
    if col in POLICY["dc_dead"]["pairs"]:
        ac = before[POLICY["dc_dead"]["pairs"][col]].to_numpy(dtype=float)
        cond[6] = finite & (a == 0) & (ac > POLICY["dc_dead"]["min_ac_kw"])
    if col in POLICY["scale_fault"]["pairs"] and after is not None:
        p = POLICY["scale_fault"]
        ref = after[p["pairs"][col]]
        day = pd.Series(v.index.tz_convert(C.TIMEZONE_LOCAL).date, index=v.index)
        use = v.notna() & ref.notna() & (elev > p["min_elevation"]) & (ref > p["min_reference"])
        ratio = (v[use] / ref[use]).groupby(day[use]).median()
        bad = ratio.index[(ratio < p["ratio_range"][0]) | (ratio > p["ratio_range"][1])]
        cond[7] = finite & day.isin(set(bad)).to_numpy()
    return cond


# T-23 reconciliation ---------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("col", cleaned_columns())
def test_every_change_is_explained_and_every_rule_hit_is_justified(col, before, after, sun):
    a = before[col].to_numpy(dtype=float)
    b = after[col].to_numpy(dtype=float)
    code = after[f"rule_{col}"].to_numpy()
    changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    assert np.array_equal(changed, code != 0), "change without a rule, or a rule without change"
    assert np.isnan(b[np.isin(code, list(NAN_RULES))]).all()
    assert (b[np.isin(code, list(ZERO_RULES))] == 0).all()
    cond = _conditions(col, before, sun, after)
    for r in np.unique(code[code != 0]):
        hit = code == r
        assert cond[r][hit].all(), f"rule {r}: {int((hit & ~cond[r]).sum())} unjustified minutes"
    for r, c in cond.items():  # completeness: a condition met is coded r or an earlier rule
        allowed = PRECEDENCE[:PRECEDENCE.index(r) + 1]
        missed = c & ~np.isin(code, allowed)
        assert not missed.any(), f"rule {r}: {int(missed.sum())} minutes not cleaned"


# T-16 real production never removed -------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("col", AC_COLUMNS)
def test_real_daytime_production_is_never_removed(col, before, after, sun):
    elev, _ = sun
    a = before[col].to_numpy(dtype=float)
    b = after[col].to_numpy(dtype=float)
    code = after[f"rule_{col}"].to_numpy()
    producing = (elev > 0) & (a > POLICY["night"]["noise_kw"])
    assert not (producing & (b == 0)).any()
    assert set(np.unique(code[producing & np.isnan(b)])) <= {1, 3}


# T-24 physical ranges -----------------------------------------------------------

def test_values_are_inside_the_policy_ranges(after):
    for col, (lo, hi) in POLICY["range"].items():
        s = after[col].dropna()
        assert s.min() >= lo and s.max() <= hi, col


@pytest.mark.slow
def test_no_power_far_below_the_horizon_and_no_dead_dc(after, sun):
    elev, _ = sun
    for col in POLICY["night"]["columns"]:
        assert not ((elev < POLICY["night"]["stale_below_elevation"])
                    & (after[col].to_numpy() > POLICY["night"]["noise_kw"])).any(), col
    for dc, ac in POLICY["dc_dead"]["pairs"].items():
        assert not ((after[dc] == 0) & (after[ac] > POLICY["dc_dead"]["min_ac_kw"])).any(), dc


# T-25 flags never change values --------------------------------------------------

def test_flags_never_change_values(before, after):
    flagged = after["op_limited_flag"].to_numpy()
    for col in AC_COLUMNS:
        untouched = flagged & (after[f"rule_{col}"].to_numpy() == 0)
        a = before[col].to_numpy()[untouched]
        b = after[col].to_numpy()[untouched]
        assert ((a == b) | (np.isnan(a) & np.isnan(b))).all(), col


@pytest.mark.slow
def test_flags_match_their_definition(after, sun):
    elev, poa = sun
    d, c = POLICY["downtime"], POLICY["curtailment"]
    irr = after["irradiance"].to_numpy()
    bright = np.where(np.isfinite(irr), irr > d["min_irradiance"],
                      poa > d["fallback_min_poa_clearsky"])
    for i in (1, 2):
        ac = after[f"s01_inv{i}_AC_kW"].to_numpy()
        expected = (elev > d["min_elevation"]) & bright & np.isfinite(ac) & (ac < d["max_ac_kw"])
        assert np.array_equal(after[f"downtime_inv{i}_flag"].to_numpy(), expected)
    sp = after[c["setpoint_column"]].to_numpy()
    expected = (elev > c["min_elevation"]) & (sp < c["below"])
    assert np.array_equal(after["curtail_flag"].to_numpy(), expected)
    op = after[["downtime_inv1_flag", "downtime_inv2_flag", "curtail_flag"]].any(axis=1)
    assert np.array_equal(after["op_limited_flag"].to_numpy(), op.to_numpy())


# T-26 hourly output -----------------------------------------------------------------

def test_hourly_axis_and_values(hourly, after):
    idx = hourly.index
    assert len(idx) == C.N_HOURS_EXPECTED and str(idx.tz) == "UTC"
    assert idx[0] == pd.Timestamp(C.DATA_START_UTC, tz="UTC")
    cols = [TARGET, *AC_COLUMNS, "irradiance", "mod_temp1", "ambient_temp"]
    means, _ = hourly_mean(after[cols])
    pd.testing.assert_frame_equal(means, hourly[cols], check_names=False, check_freq=False)


def test_site_total_is_sum_of_cleaned_inverters(after):
    both = after[list(AC_COLUMNS)].notna().all(axis=1)
    np.testing.assert_allclose(after.loc[both, TARGET],
                               after.loc[both, list(AC_COLUMNS)].sum(axis=1))
    assert after.loc[~both, TARGET].isna().all()


def test_hourly_flags_and_source(hourly):
    for c in [c for c in hourly.columns if c.startswith("frac_")]:
        assert hourly[c].between(0, 1).all(), c
    share = POLICY["hourly"]["op_limited_min_share"]
    assert (hourly["op_limited_flag"] == (hourly["frac_op_limited_flag"] >= share)).all()
    assert set(hourly["hour_source"].unique()) <= {"minute", "polled", "mixed"}
    stage4 = json.loads((PROCESSED_DIR / "04_preprocess" / "preprocess_manifest.json")
                        .read_text(encoding="utf-8"))
    first = pd.Timestamp(stage4["polled_source"]["first_day"])
    local_day = hourly.index.tz_convert(C.TIMEZONE_LOCAL).tz_localize(None).normalize()
    assert (local_day[hourly["hour_source"] != "minute"] >= first).all()


# T-27 energy on fault-free days ------------------------------------------------------

@pytest.mark.slow
def test_energy_unchanged_on_fault_free_days(before, after, sun):
    elev, _ = sun
    faults = np.zeros(len(after), dtype=bool)
    for col in AC_COLUMNS:
        faults |= np.isin(after[f"rule_{col}"].to_numpy(), [1, 3, 5])
    day = pd.DatetimeIndex(after.index.tz_convert(C.TIMEZONE_LOCAL).date)
    bad_days = set(day[faults])
    up = elev > 0
    e_before = before[TARGET].where(up).groupby(day).sum()
    e_after = after[TARGET].where(up).groupby(day).sum()
    ok = ~e_before.index.isin(list(bad_days)) & (e_before > 0)
    assert ok.sum() > 1300
    ratio = e_after[ok] / e_before[ok]
    assert ratio.between(0.999, 1.001).all(), ratio[~ratio.between(0.999, 1.001)]


# T-28 log and manifest ----------------------------------------------------------------

def test_log_and_manifest_match_the_output(after, manifest):
    log = pd.read_csv(OUTPUTS_DIR / "05_cleaning" / "cleaning_log.csv")
    names = {1: "R1_range", 2: "R2_offset", 3: "R3_frozen", 4: "R4_night_zero",
             5: "R5_night_stale", 6: "R6_dc_dead", 7: "R7_scale_fault"}
    recomputed = {}
    for col in cleaned_columns():
        counts = after[f"rule_{col}"].value_counts()
        for r, n in counts.items():
            if r != 0:
                recomputed[f"{names[int(r)]}:{col}"] = int(n)
    from_log = {f"{r.rule}:{r.column}": int(r.minutes) for r in log.itertuples()}
    assert recomputed == from_log == manifest["rule_minutes"]
    assert manifest["flag_minutes"] == {f: int(after[f].sum()) for f in manifest["flag_minutes"]}


def test_manifest_hashes(manifest):
    policy_sha = text_sha256(CONFIGS_DIR / "cleaning_policy.yaml")
    assert manifest["policy"]["sha256"] == policy_sha
    assert text_sha256(OUTPUTS_DIR / "05_cleaning" / "cleaning_policy_applied.yaml") == policy_sha
    assert manifest["input"]["sha256"] == sha256(BEFORE)
    assert manifest["outputs"]["scada_1H_clean.parquet"] == sha256(HOURLY)
    assert manifest["outputs"]["scada_1min_clean.parquet"] == sha256(MINUTE)


# T-01 quality columns can never be model inputs -----------------------------------------

def test_quality_columns_are_blocked_as_features(hourly, after):
    quality = [c for c in [*hourly.columns, *after.columns]
               if c.startswith(("n_", "frac_", "rule_", "src_")) or c.endswith("_flag")
               or c == "hour_source"]
    assert quality
    assert all(is_leaky(c) for c in quality), [c for c in quality if not is_leaky(c)]

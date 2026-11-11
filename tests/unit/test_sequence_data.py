"""The datasets of the sequence models, on synthetic tables in the stage-5/6/6b layout
(docs/stage9_plan.md §17, W3).

T-66  the step -> horizon mapping: T0 at local midnight; decoders of 71 / 72 / 73 steps around
      2024-03-29 and 2024-10-27; every D+h block is its local day; issuance_utc agrees (X20); a
      mapping by step number is wrong on these days (M14)
T-67  origins and coverage per scope (the real calendar counts; every hour of the scope once per
      horizon); no row at or after TEST_START read in development (X21); training decoders end
      before VAL_START; test windows only with allow_test; the loss mask is fit_mask (X23)
T-68  the encoder ends at T0; values at or after T0 change no input; the D-22 masks and the fill
      flags; the encoder series equals the stage-6 lag (X22)
T-69  A-14: the scaler from train hours only; targets not transformed
T-70  the writer: the contract of §6.10, the evaluation's alignment and coverage checks, and the
      refusal of duplicates, gaps and missing forecasts (M18)
T-71  the inputs (D-26-D-28): the known part of model_inputs.json, deterministic inputs equal
      across the tables, GFS per step from the step's horizon table and run, PatchTST target only
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from solarpv.config.constants import TEST_START, TIMEZONE_LOCAL, VAL_START
from solarpv.data.clean import split_of
from solarpv.evaluation import evaluate as E
from solarpv.features.build import impute_same_hour, issuance_utc, masked_sources, past_value
from solarpv.features.contract import LAG_SOURCES
from solarpv.models import sequence_data as S
from solarpv.models.xgb_sota import fit_mask
from solarpv.pipeline.config import local_days_window, split_window
from solarpv.selection.common import PREVDAY

HOUR = pd.Timedelta(hours=1)
FLAGS = ("curtail_flag", "derate_inv1_flag", "derate_inv2_flag", "downtime_inv1_flag",
         "downtime_inv2_flag")
GFS_COLUMNS = ("fcst_gfs_ghi", "fcst_gfs_kt", "fcst_gfs_tcc", PREVDAY)
INPUTS = {
    "indirect_D1": ["ghi_clearsky", "aoi", "fcst_gfs_kt", "irradiance_lag_3d", "fcst_gfs_tcc",
                    PREVDAY],
    "indirect_D2": ["ghi_clearsky", "doy_sin", "fcst_gfs_kt", "ambient_temp_lag_2d", PREVDAY],
    "indirect_D3": ["aoi", "hour_cos", "fcst_gfs_tcc", "s01_inv2_rolling_mean_3d_same_tod"],
    "direct_D1": ["sun_elevation", "poa_global_clearsky", "fcst_gfs_ghi",
                  "site_total_AC_kW_lag_1d", "day_of_week"],
    "direct_D2": ["sun_elevation", "fcst_gfs_ghi", "fcst_gfs_kt", "site_total_AC_kW_lag_2d"],
    "direct_D3": ["poa_global_clearsky", "fcst_gfs_ghi", "irradiance_clear_index_prevday"],
}


def utc(text):
    return pd.Timestamp(text, tz="UTC")


# ── synthetic inputs in the stage-5 / 6 / 6b layout ────────────────────────────


def build_inputs(seed: int = 0) -> dict:
    """2023-09-01 to 2024-10-07 local: train with the fall-back day 2023-10-29, validation with
    the spring-forward day 2024-03-29, and one week of test rows (never read in development)."""
    start, stop = S.t0_of(["2023-09-01", "2024-10-08"])
    idx = pd.date_range(start, stop, freq="h", inclusive="left", name="timestamp_utc")
    n = len(idx)
    rng = np.random.default_rng(seed)
    local_hour = idx.tz_convert(TIMEZONE_LOCAL).hour.to_numpy()
    sun = np.clip(np.sin(np.pi * (local_hour - 6) / 12), 0, None)
    cloud = np.clip(rng.beta(2, 5, n), 0, 1)
    hourly = pd.DataFrame(index=idx)
    hourly["irradiance"] = 900 * sun * (1 - cloud)
    hourly["s01_inv1_AC_kW"] = 1500 * sun * (1 - cloud) + rng.normal(0, 5, n) * (sun > 0)
    hourly["s01_inv2_AC_kW"] = 1480 * sun * (1 - cloud) + rng.normal(0, 5, n) * (sun > 0)
    hourly["site_total_AC_kW"] = hourly["s01_inv1_AC_kW"] + hourly["s01_inv2_AC_kW"]
    for col, base in (("mod_temp1", 25), ("mod_temp2", 25), ("ambient_temp", 20)):
        hourly[col] = base + 15 * sun + rng.normal(0, 1, n)
    for col in LAG_SOURCES:
        hourly.loc[rng.random(n) < 0.01, col] = np.nan
    hourly["op_limited_flag"] = rng.random(n) < 0.03
    hourly["hour_source"] = "minute"
    minutes = pd.date_range(idx[0], idx[-1] + pd.Timedelta(minutes=59), freq="min",
                            name="timestamp_utc")
    share = {"downtime_inv1_flag": 0.02, "downtime_inv2_flag": 0.01, "curtail_flag": 0.01,
             "derate_inv1_flag": 0.01, "derate_inv2_flag": 0.01}
    flags = pd.DataFrame({f: np.repeat(rng.random(n) < share[f], 60) for f in FLAGS},
                         index=minutes)
    theo = 3449 * sun
    ac = hourly["site_total_AC_kW"].to_numpy()
    k = np.where(theo > 50, ac / np.where(theo > 50, theo, 1), 0.0)
    det = {c: rng.normal(size=n) for c in S.DETERMINISTIC}
    first_day = idx[0].tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    local = idx.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    warmup = local < first_day + pd.Timedelta(days=14)
    base = pd.DataFrame({"split": split_of(idx), "warmup": warmup,
                         "daytime": sun > 0.1, "op_limited_flag": hourly["op_limited_flag"],
                         "theo_kw": theo, "target_k": k, "k_valid": (theo > 50) & np.isfinite(ac),
                         "site_total_AC_kW": ac, **det}, index=idx)
    sources, _ = masked_sources(hourly, flags, 0.05)
    t0_1 = issuance_utc(idx, 1)
    lags = pd.DataFrame({f"{c}_lag_1d": past_value(sources[c], 1, t0_1) for c in LAG_SOURCES},
                        index=idx)
    lags, _ = impute_same_hour(lags)
    tables, gfs = {}, {}
    for h in (1, 2, 3):
        t0 = issuance_utc(idx, h)
        table = base.copy()
        table.insert(0, "t0_utc", t0)
        if h == 1:
            table = table.join(lags)
        tables[h] = table
        g = pd.DataFrame({"t0_utc": t0, "gfs_init": t0.floor("D") + pd.Timedelta(hours=12),
                          "gfs_run": "12Z"}, index=idx)
        for col in GFS_COLUMNS:
            g[col] = rng.normal(10 * h, 1, n)  # each horizon its own values
        g.loc[rng.random(n) < 0.01, PREVDAY] = np.nan
        gfs[h] = g
    return {"tables": tables, "gfs": gfs, "hourly": hourly, "flags": flags, "share": 0.05}


def write_inputs(data: dict, folder) -> S.Paths:
    folder.mkdir(parents=True, exist_ok=True)
    feats = {h: folder / f"features_1H_D{h}.parquet" for h in (1, 2, 3)}
    gfs = {h: folder / f"gfs_features_D{h}.parquet" for h in (1, 2, 3)}
    for h in (1, 2, 3):
        data["tables"][h].to_parquet(feats[h])
        data["gfs"][h].to_parquet(gfs[h])
    data["hourly"].to_parquet(folder / "scada_1H_clean.parquet")
    data["flags"].to_parquet(folder / "scada_1min_clean.parquet")
    (folder / "cleaning_policy.yaml").write_text(
        yaml.safe_dump({"hourly": {"op_limited_min_share": data["share"]}}), encoding="utf-8")
    return S.Paths(feats, gfs, folder / "scada_1H_clean.parquet",
                   folder / "scada_1min_clean.parquet", folder / "cleaning_policy.yaml")


@pytest.fixture(scope="module")
def data():
    return build_inputs()


@pytest.fixture(scope="module")
def paths(data, tmp_path_factory):
    return write_inputs(data, tmp_path_factory.mktemp("seq"))


@pytest.fixture(scope="module")
def sources(paths):
    return S.load_sources(paths=paths)


@pytest.fixture(scope="module")
def scaler(sources):
    return S.Scaler.fit(sources)


@pytest.fixture(scope="module")
def val_scope(sources):
    return S.evaluation_scope(sources, split_window("val"))


@pytest.fixture(scope="module")
def train_scope(sources):
    return S.training_scope(sources, "dev")


def assemble_dev(tables, gfs, hourly, flags, share):
    """S.assemble on the rows before TEST_START, as the loader reads them in development."""
    cut = utc(TEST_START)
    return S.assemble({h: t[t.index < cut] for h, t in tables.items()},
                      {h: g[g.index < cut] for h, g in gfs.items()}, hourly[hourly.index < cut],
                      flags[flags.index < cut], share)


def make(sources, scaler, scope, model="bilstm", target="indirect", **kw):
    hours = kw.pop("encoder_hours", None) or S.MODELS[model] or 168
    return S.build(sources, scaler, model, target, scope, inputs=INPUTS, encoder_hours=hours,
                   **kw)


# ── T-66: the mapping ──────────────────────────────────────────────────────────


def test_t0_is_local_midnight_in_utc():
    got = S.t0_of(["2024-03-29", "2024-07-01", "2024-12-01", "2024-10-27"])
    assert list(got) == [utc("2024-03-28 22:00"), utc("2024-06-30 21:00"),
                         utc("2024-11-30 22:00"), utc("2024-10-26 21:00")]


@pytest.mark.parametrize("first,last,dst_day", [("2024-03-24", "2024-03-31", "2024-03-29"),
                                                ("2024-10-22", "2024-10-29", "2024-10-27")])
def test_decoders_cover_the_three_local_days(first, last, dst_day):
    days = pd.date_range(first, last, freq="D")
    t0, valid, horizon, n_steps = S.decoder_layout(days)
    long = 23 if dst_day.endswith("03-29") else 25
    for i, day in enumerate(days):
        lengths = []
        for h in (1, 2, 3):
            d = day + pd.Timedelta(days=h - 1)
            local = pd.date_range(d, d + pd.Timedelta(days=1), freq="h", inclusive="left")
            expected = pd.date_range(*S.t0_of([d, d + pd.Timedelta(days=1)]), freq="h",
                                     inclusive="left")
            block = pd.DatetimeIndex(valid[i][horizon[i] == h], tz="UTC")
            assert np.array_equal(S._ns(block), S._ns(expected))  # its local day, hour by hour
            assert (block.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize() == d).all()
            lengths.append(len(block))
            assert len(local) == 24  # the naive clock would always say 24
        dst_inside = 0 <= (pd.Timestamp(dst_day) - day).days <= 2
        assert n_steps[i] == (72 + (long - 24) if dst_inside else 72)
        assert sum(lengths) == n_steps[i]
        assert (horizon[i][n_steps[i]:] == 0).all() and (valid[i][n_steps[i]:] == S.NAT).all()
    by_day = dict(zip(days.strftime("%Y-%m-%d"), range(len(days)), strict=True))
    i = by_day[dst_day]
    assert [int((horizon[i] == h).sum()) for h in (1, 2, 3)] == [long, 24, 24]
    j = by_day[(pd.Timestamp(dst_day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")]
    assert [int((horizon[j] == h).sum()) for h in (1, 2, 3)] == [24, 24, long]


def test_every_step_agrees_with_issuance_utc():
    days = pd.date_range("2024-03-20", "2024-11-05", freq="D")
    t0, valid, horizon, _ = S.decoder_layout(days)
    for h in (1, 2, 3):
        sel = horizon == h
        steps = pd.DatetimeIndex(valid[sel], tz="UTC")
        origin = np.broadcast_to(t0[:, None], sel.shape)[sel]
        assert (S._ns(issuance_utc(steps, h)) == origin).all()


def test_a_mapping_by_step_number_is_wrong_on_dst_days():
    days = pd.DatetimeIndex(["2024-03-29", "2024-10-27", "2024-06-10"])
    _, valid, horizon, n_steps = S.decoder_layout(days)
    naive = (np.arange(S.MAX_STEPS) // 24 + 1)[None, :]
    inside = horizon > 0
    wrong = (naive != horizon) & inside
    assert wrong[0].any() and wrong[1].any()  # "steps 1-24 = D+1" fails on both DST days
    assert not wrong[2].any()


# ── T-67: origins and coverage ─────────────────────────────────────────────────


def _hours(window):
    return pd.date_range(window.start_ts, window.end_ts, freq="h", inclusive="left")


def test_origins_of_the_real_scopes():
    val = S.origin_days(_hours(split_window("val")))
    assert (len(val), str(val[0].date()), str(val[-1].date())) == (246, "2024-01-30",
                                                                    "2024-10-01")
    smoke = S.origin_days(_hours(local_days_window("2024-03-15", "2024-04-11")))
    assert (len(smoke), str(smoke[0].date())) == (30, "2024-03-13")
    # the data end at 2025-05-31 20:59Z (23:59 local), before the end of the test window
    test = S.origin_days(pd.date_range(utc(TEST_START), utc("2025-05-31 20:00"), freq="h"))
    assert (len(test), str(test[0].date()), str(test[-1].date())) == (245, "2024-09-29",
                                                                      "2025-05-31")
    first_loss_hour = S.t0_of(["2021-09-15"])[0]
    train = S.origin_days(pd.date_range(first_loss_hour, utc("2024-01-31 23:00"), freq="h"))
    assert (len(train), str(train[0].date()), str(train[-1].date())) == (872, "2021-09-13",
                                                                         "2024-02-01")


@pytest.mark.parametrize("target", ["indirect", "direct"])
@pytest.mark.parametrize("which", ["train", "val"])
def test_every_hour_of_the_scope_once_per_horizon(sources, scaler, val_scope, train_scope,
                                                  target, which):
    scope = train_scope if which == "train" else val_scope
    ds = make(sources, scaler, scope, target=target)
    checks = S.check_dataset(ds, sources)
    assert all(v for v in checks.values() if isinstance(v, bool)), checks
    for h in (1, 2, 3):
        assert ds.info["kept_steps_per_horizon"][f"D{h}"] == len(scope.hours)
    if which == "val":
        assert checks["block_lengths"].get("23", 0) >= 1  # 2024-03-29
    else:
        assert checks["block_lengths"].get("25", 0) >= 1  # 2023-10-29


def test_development_reads_no_test_row(paths, sources, scaler, val_scope, monkeypatch):
    calls = []
    real = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append(kwargs.get("filters"))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(S.pd, "read_parquet", spy)
    loaded = S.load_sources(paths=paths)
    assert len(calls) == 8  # 3 feature tables, 3 GFS tables, hourly, minute flags
    assert all(c == [(S.INDEX, "<", utc(TEST_START))] for c in calls)
    assert loaded.frame.index.max() < utc(TEST_START)
    ds = make(sources, scaler, val_scope)
    assert (ds.enc_time < S._ns([utc(TEST_START)])[0]).all()
    assert (ds.dec_time[ds.dec_exists] < S._ns([utc(TEST_START)])[0]).all()
    assert (ds.dec_time[~ds.dec_exists] == S.NAT).all()  # cut steps carry no time


def test_training_decoders_end_before_validation(sources, scaler, train_scope):
    ds = make(sources, scaler, train_scope, model="tft168")
    assert (ds.dec_time[ds.dec_exists] < S._ns([utc(VAL_START)])[0]).all()
    kept = pd.DatetimeIndex(ds.dec_time[ds.loss_mask], tz="UTC")
    assert (sources.frame.loc[kept, "split"] == "train").all()
    assert not sources.frame.loc[kept, "warmup"].any()
    assert ds.info["n_origins"] == len(S.origin_days(train_scope.hours))
    smoke = S.training_scope(sources, "smoke")
    assert len(S.origin_days(smoke.hours)) == 58


def test_test_windows_need_allow_test(paths, sources):
    with pytest.raises(PermissionError):
        S.evaluation_scope(sources, split_window("test"))
    full = S.load_sources(paths=paths, allow_test=True)
    assert full.frame.index.max() >= utc(TEST_START)
    scope = S.evaluation_scope(full, split_window("test"))
    ds = S.build(full, S.Scaler.fit(full), "bilstm", "direct", scope, inputs=INPUTS)
    assert S.check_dataset(ds, full)["every_scope_hour_once_per_horizon"]


def test_the_loss_mask_is_fit_mask(sources, scaler, val_scope):
    for target in ("indirect", "direct"):
        ds = make(sources, scaler, val_scope, target=target)
        assert not (ds.loss_mask & ~ds.dec_keep).any()
        for h in (1, 2, 3):
            sel = ds.loss_mask & (ds.dec_horizon == h)
            steps = pd.DatetimeIndex(ds.dec_time[sel], tz="UTC")
            assert fit_mask(sources.frame, target).loc[steps].all()
            rows = fit_mask(sources.frame, target) & sources.frame.index.isin(val_scope.hours)
            rows &= sources.frame[S.TARGET_COLUMN[target]].notna()
            assert int(sel.sum()) == int(rows.sum())


# ── T-68: the encoder ──────────────────────────────────────────────────────────


def test_the_encoder_ends_at_t0(sources, scaler, val_scope):
    for model in ("bilstm", "tft96", "tft168", "patchtst"):
        ds = make(sources, scaler, val_scope, model=model)
        assert (ds.enc_time[:, -1] + S.HOUR_NS == S._ns(ds.origins)).all()
        assert ds.enc_time.shape[1] == ds.encoder_hours


def test_values_at_or_after_t0_change_no_input(data, sources, scaler, val_scope):
    ds = make(sources, scaler, val_scope)
    i = 120
    t0 = ds.origins[i]
    changed = {"tables": {h: t.copy() for h, t in data["tables"].items()},
               "gfs": data["gfs"], "hourly": data["hourly"].copy(),
               "flags": data["flags"].copy()}
    after = changed["hourly"].index >= t0
    for col in LAG_SOURCES:
        values = changed["hourly"][col].to_numpy(dtype=float).copy()
        values[after] = values[after] * 3 + 7
        changed["hourly"][col] = values
    for h in (1, 2, 3):
        for col in ("target_k", "site_total_AC_kW"):
            values = changed["tables"][h][col].to_numpy(dtype=float).copy()
            values[after] = values[after] + 1.0
            changed["tables"][h][col] = values
    flip = changed["flags"].index >= t0
    for col in FLAGS:
        values = changed["flags"][col].to_numpy().copy()
        values[flip] = ~values[flip]
        changed["flags"][col] = values
    other = assemble_dev(changed["tables"], changed["gfs"], changed["hourly"],
                         changed["flags"], data["share"])
    ds2 = make(other, scaler, val_scope)
    earlier = S._ns(ds.origins) <= S._ns([t0])[0]
    assert np.array_equal(ds.enc_x[earlier], ds2.enc_x[earlier])
    assert np.array_equal(ds.dec_x[earlier], ds2.dec_x[earlier])
    assert not np.allclose(np.nan_to_num(ds.y_true[i]), np.nan_to_num(ds2.y_true[i]))


def test_masked_hours_are_nan_before_the_fill_and_flagged(data, sources, scaler, val_scope):
    flags = data["flags"]["downtime_inv1_flag"].resample("1h").mean() >= 0.05
    raw = data["hourly"]["s01_inv1_AC_kW"]
    hit = flags & raw.notna() & (raw.index >= utc(VAL_START))
    assert hit.any()
    hour = hit[hit].index[0]
    assert np.isnan(sources.series.loc[hour, "s01_inv1_AC_kW"])
    assert np.isfinite(sources.filled.loc[hour, "s01_inv1_AC_kW"])  # filled from older days
    ds = make(sources, scaler, val_scope)
    c = ds.enc_channels.index("s01_inv1_AC_kW" + S.FLAG_SUFFIX)
    where = ds.enc_time == S._ns([hour])[0]
    assert where.any() and (ds.enc_x[..., c][where] == 1).all()
    unflagged = ~sources.series["s01_inv1_AC_kW"].isna()
    assert (sources.filled["s01_inv1_AC_kW"][unflagged]
            == sources.series["s01_inv1_AC_kW"][unflagged]).all()


def test_the_encoder_series_equals_the_stage6_lag(data, sources):
    assert S.lag_consistency(sources) == {c: 0 for c in LAG_SOURCES}
    raw = data["hourly"].loc[sources.frame.index, list(LAG_SOURCES)].copy()
    unmasked = replace(sources, series=raw)
    bad = S.lag_consistency(unmasked)  # the unmasked AC is not what the lags were built from
    assert bad["s01_inv1_AC_kW"] > 0 and bad["site_total_AC_kW"] > 0 and bad["irradiance"] == 0


def test_the_fill_uses_older_hours_only(data, sources):
    series = sources.series.copy()
    cut = utc("2024-05-01")
    later = series.index >= cut
    noisy = series.copy()
    for col in noisy.columns:
        values = noisy[col].to_numpy(dtype=float).copy()
        values[later] = 1e6
        noisy[col] = values
    a, _ = impute_same_hour(series)
    b, _ = impute_same_hour(noisy)
    assert a[~later].equals(b[~later])


# ── T-69: scalers ─────────────────────────────────────────────────────────────


def test_the_scaler_comes_from_train_hours_only(sources, scaler, train_scope):
    checks = S.scaler_checks(sources, scaler)
    assert checks["recomputed_from_train"] and checks["unchanged_when_other_splits_change"]
    rows = (sources.frame["split"] == "train") & ~sources.frame["warmup"]
    assert checks["fit_rows"] == int(rows.sum())
    noisy_frame = sources.frame.copy()
    outside = (noisy_frame["split"] != "train").to_numpy()
    for col in S.DETERMINISTIC:
        values = noisy_frame[col].to_numpy(dtype=float).copy()
        values[outside] = -99.0
        noisy_frame[col] = values
    series = sources.series.copy()
    for col in series.columns:
        values = series[col].to_numpy(dtype=float).copy()
        values[outside] = -99.0
        series[col] = values
    filled, _ = impute_same_hour(series)
    noisy = replace(sources, frame=noisy_frame, series=series, filled=filled)
    a = make(sources, scaler, train_scope)
    b = make(noisy, S.Scaler.fit(noisy), train_scope)
    assert np.array_equal(a.enc_x, b.enc_x) and np.array_equal(a.dec_x, b.dec_x)


def test_targets_are_not_transformed(sources, scaler, val_scope):
    k = make(sources, scaler, val_scope, target="indirect")
    sel = k.dec_exists & np.isfinite(k.y_true)
    steps = pd.DatetimeIndex(k.dec_time[sel], tz="UTC")
    assert np.array_equal(k.y_true[sel], sources.frame.loc[steps, "target_k"].to_numpy())
    assert np.allclose(k.y[sel], k.y_true[sel], rtol=1e-6, atol=1e-7)
    ac = make(sources, scaler, val_scope, target="direct")
    sel = ac.dec_exists & np.isfinite(ac.y_true)
    assert ac.target_scale == 3449.0
    assert np.allclose(ac.y[sel] * 3449.0, ac.y_true[sel], rtol=1e-6, atol=1e-3)


# ── T-70: the writer ───────────────────────────────────────────────────────────


def _truth(data):
    t = data["tables"][1]
    return pd.DataFrame({"indirect": t["target_k"], "direct": t["site_total_AC_kW"],
                         "split": t["split"], "daytime": t["daytime"]}, index=t.index)


@pytest.mark.parametrize("target", ["indirect", "direct"])
def test_the_writer_passes_the_evaluation_checks(data, sources, scaler, val_scope, target,
                                                 tmp_path):
    ds = make(sources, scaler, val_scope, target=target)
    preds = S.predictions_frame(ds, np.nan_to_num(ds.y, nan=0.0), model_name="lstm_sota",
                                seed=42)
    assert len(preds) == 3 * len(val_scope.hours)
    path = tmp_path / "predictions_lstm_sota_seed42.csv"
    preds.to_csv(path, index=False)
    loaded = E.load_predictions(path)  # the contract of §6.10
    truth = _truth(data)
    assert E.check_alignment(loaded, truth, "val")["ok"].all()
    assert int(E.check_coverage(loaded, truth, "val")["missing"].sum()) == 0
    local = pd.DatetimeIndex(loaded["timestamp"]).tz_convert(TIMEZONE_LOCAL).normalize()
    d1 = loaded[(loaded["horizon"] == 1) & (local.strftime("%Y-%m-%d") == "2024-03-29")]
    assert len(d1) == 23  # the 23-hour day keeps its 23 hours at D+1


def test_the_writer_refuses_duplicates_gaps_and_missing_forecasts(sources, scaler, val_scope):
    ds = make(sources, scaler, val_scope)
    y = np.zeros(ds.dec_keep.shape)
    with pytest.raises(ValueError):
        S.predictions_frame(ds, y[:, :10], model_name="m", seed=1)
    gap = replace(ds, dec_keep=ds.dec_keep.copy())
    gap.dec_keep[5, np.argmax(gap.dec_keep[5])] = False
    with pytest.raises(S.SequenceDataError, match="without a forecast"):
        S.predictions_frame(gap, y, model_name="m", seed=1)
    twice = replace(ds, dec_keep=np.vstack([ds.dec_keep, ds.dec_keep[:1]]),
                    dec_time=np.vstack([ds.dec_time, ds.dec_time[:1]]),
                    dec_horizon=np.vstack([ds.dec_horizon, ds.dec_horizon[:1]]),
                    y_true=np.vstack([ds.y_true, ds.y_true[:1]]))
    with pytest.raises(S.SequenceDataError, match="duplicated"):
        S.predictions_frame(twice, np.zeros(twice.dec_keep.shape), model_name="m", seed=1)
    holes = y.copy()
    holes[ds.dec_keep] = np.nan
    with pytest.raises(S.SequenceDataError, match="finite"):
        S.predictions_frame(ds, holes, model_name="m", seed=1)


# ── T-71: the inputs ───────────────────────────────────────────────────────────


def test_known_inputs_are_the_union_known_at_t0():
    det, gfs = S.known_inputs("indirect", INPUTS)
    assert det == ["ghi_clearsky", "aoi", "doy_sin", "hour_cos"]
    assert gfs == ["fcst_gfs_kt", "fcst_gfs_tcc", PREVDAY]
    det, gfs = S.known_inputs("direct", INPUTS)
    assert det == ["sun_elevation", "poa_global_clearsky", "day_of_week"]
    assert gfs == ["fcst_gfs_ghi", "fcst_gfs_kt", PREVDAY]
    for bad in ("irradiance", "theoretical_power", "foo"):
        wrong = {**INPUTS, "indirect_D2": INPUTS["indirect_D2"] + [bad]}
        with pytest.raises(ValueError):
            S.known_inputs("indirect", wrong)


def test_the_real_stage7_inputs():
    counts = {}
    for target in ("indirect", "direct"):
        det, gfs = S.known_inputs(target)
        assert not set(det + gfs) & S.PAST
        counts[target] = (len(det), len(gfs))
    assert counts == {"indirect": (10, 20), "direct": (27, 22)}


def test_deterministic_inputs_are_equal_across_tables(data):
    tables = {h: t.copy() for h, t in data["tables"].items()}
    values = tables[2]["aoi"].to_numpy().copy()
    values[100] += 1.0
    tables[2]["aoi"] = values
    with pytest.raises(S.SequenceDataError, match="aoi differs"):
        assemble_dev(tables, data["gfs"], data["hourly"], data["flags"], data["share"])
    with pytest.raises(S.SequenceDataError, match="test period"):  # the guard of development
        S.assemble(data["tables"], data["gfs"], data["hourly"], data["flags"], data["share"])


def test_gfs_of_each_step_comes_from_its_horizon_table(data, sources, scaler, val_scope):
    ds = make(sources, scaler, val_scope)
    for col in ("fcst_gfs_kt", "fcst_gfs_tcc"):
        c = ds.dec_channels.index(col)
        raw = ds.dec_x[..., c] * scaler.std[col] + scaler.mean[col]
        for h in (1, 2, 3):
            sel = ds.dec_exists & (ds.dec_horizon == h)
            steps = pd.DatetimeIndex(ds.dec_time[sel], tz="UTC")
            assert np.allclose(raw[sel], data["gfs"][h].loc[steps, col], atol=1e-4)
        e = ds.enc_channels.index(col)
        enc_raw = ds.enc_x[..., e] * scaler.std[col] + scaler.mean[col]
        hours = pd.DatetimeIndex(ds.enc_time.ravel(), tz="UTC")
        assert np.allclose(enc_raw.ravel(), data["gfs"][1].loc[hours, col], atol=1e-4)


@pytest.mark.parametrize("column,shift", [("t0_utc", pd.Timedelta(days=1)),
                                          ("gfs_init", pd.Timedelta(hours=8))])
def test_a_step_of_another_issuance_or_a_late_run_is_refused(data, scaler, val_scope, column,
                                                             shift):
    gfs = {h: g.copy() for h, g in data["gfs"].items()}
    times = gfs[2][column].copy()
    row = gfs[2].index.get_loc(utc("2024-05-02 10:00"))
    times.iloc[row] = times.iloc[row] + shift
    gfs[2][column] = times
    other = assemble_dev(data["tables"], gfs, data["hourly"], data["flags"], data["share"])
    with pytest.raises(S.SequenceDataError, match="D\\+2"):
        make(other, scaler, val_scope)


def test_patchtst_sees_the_target_only(sources, scaler, val_scope):
    # X33 (W4): the input size is set in one place, the neuralforecast input size of §18.4
    from solarpv.models.neural_job import FIXED
    assert S.MODELS["patchtst"] == FIXED["patchtst"]["input_size"] == 168
    ds = S.build(sources, scaler, "patchtst", "indirect", val_scope, inputs=INPUTS)
    assert ds.encoder_hours == 168 and ds.enc_x.shape[1] == 168
    for target, own in (("indirect", "target_k"), ("direct", "site_total_AC_kW")):
        ds = make(sources, scaler, val_scope, model="patchtst", target=target,
                  encoder_hours=96)
        assert ds.enc_channels == [own] and ds.dec_channels == []
        assert ds.enc_x.shape[2] == 1 and ds.dec_x.shape[2] == 0
        hours = pd.DatetimeIndex(ds.enc_time.ravel(), tz="UTC")
        expected = sources.filled.loc[hours, own].to_numpy() / ds.target_scale
        assert np.allclose(ds.enc_x[..., 0].ravel(), np.nan_to_num(expected), atol=1e-5)


def test_the_command_checks_pass_on_synthetic_inputs(paths, tmp_path):
    report = S.run("dev", out_dir=tmp_path, paths=paths, inputs=INPUTS)
    assert report["all_checks_pass"], json.dumps(report["datasets"], default=str)[:2000]
    assert (tmp_path / "dataset_report_dev.json").exists()
    assert (tmp_path / "checks_dev.csv").exists()
    logs = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.log"))
    assert len(logs) == 1 and logs[0].startswith("runs/") and logs[0].count("/") == 1  # X24
    assert report["run_log"].endswith(logs[0].split("/")[-1])
    with pytest.raises(ValueError):
        S.run("final", out_dir=tmp_path, paths=paths)

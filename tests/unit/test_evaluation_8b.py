"""Stage 8B on synthetic data (docs/stage8b_plan.md §5).

T-38  references: persistence with the DST guard and gap filling from earlier days only;
      climatology from normal-operation train rows only; clear sky x MEAN_K formulas
T-39  evaluation rules: split guard, alignment, coverage, zero-output hours kept (E3), paired
      route comparison (E2), clear-sky route advantage = 0
T-40  metrics: CLIPER, nRMSE by the mean, kW conversion, seed table, DM CI, views
T-54  (part) fixes of the stage-9 self-review: duplicates across files, no stale manifest after a
      failure, no glob in main(), the truth without the test split
T-65  Q-19: DM and skill in kW for both routes (equal to the target-unit ones for the direct
      target, a hand computation for the indirect one); none in the fixture mode; main() reads
      no test row for a validation evaluation (X17, X18 of docs/stage9_plan.md §16)
"""

import json

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import AC_CAPACITY_KW, HORIZONS, TIMEZONE_LOCAL
from solarpv.config.site import EXAMPLE_FILE, load_site
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.daytime import daytime_mask
from solarpv.evaluation.metrics import bootstrap_mean_ci, cliper_rmse, rmse
from solarpv.evaluation.stats_tests import diebold_mariano
from solarpv.evaluation.truth import load_truth
from solarpv.features.build import issuance_utc
from solarpv.models.baselines import clearsky_k, climatology, persistence, reference_predictions

SITE = load_site(EXAMPLE_FILE)  # anonymised coordinates: enough for synthetic checks
MEAN_K = 0.8


def _truth(start="2022-10-01", end="2022-11-20", seed=0) -> pd.DataFrame:
    """A synthetic truth table in the layout of solarpv.evaluation.truth."""
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    day = daytime_mask(idx, SITE)
    theo = np.where(day, rng.uniform(100, 3000, len(idx)), 10.0)
    k = np.where(theo > 50, rng.uniform(0.2, 1.1, len(idx)), 0.0)
    # train on days 1-20 of each month, so every (month, hour) cell has training data
    split = np.where(idx.day <= 20, "train", np.where(idx.day <= 25, "val", "test"))
    return pd.DataFrame({"indirect": k, "direct": k * theo, "theo_kw": theo, "k_valid": theo > 50,
                         "daytime": day, "split": split,
                         "op_limited_flag": rng.random(len(idx)) < 0.05,
                         "hour_source": np.where(rng.random(len(idx)) < 0.3, "polled", "minute")},
                        index=idx)


def _model_preds(truth, split="val", model="xgb_sota", seed=42, noise=0.1, targets=None):
    rng = np.random.default_rng(seed)
    rows = truth.index[truth["split"] == split]
    frames = []
    for target in targets or ("indirect", "direct"):
        y = truth.loc[rows, target].to_numpy()
        for h in HORIZONS:
            frames.append(pd.DataFrame({
                "timestamp": rows, "y_true": y, "y_pred": y * (1 + noise * rng.standard_normal(
                    len(y))), "target": target, "horizon": h, "model": model, "seed": seed,
                "split": split}))
    return pd.concat(frames, ignore_index=True)


def _write(frame, path):
    frame.to_csv(path, index=False)
    return path


# ── T-38 references ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("horizon", HORIZONS)
def test_persistence_sees_nothing_after_issuance(horizon):
    """Around the 2022 fall-back day: changing the truth from T0 on changes no forecast of
    that day (the 25th hour would otherwise read the hour starting at T0)."""
    s = _truth()["direct"]
    s[s.sample(frac=0.05, random_state=1).index] = np.nan  # gaps exercise the filling
    base = persistence(s, horizon)
    local_day = s.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    t0 = issuance_utc(s.index, horizon)
    for day in pd.date_range("2022-10-28", "2022-11-01"):
        rows = local_day == day
        issue = t0[rows][0]
        changed = s.where(s.index < issue, s + 12345.0)
        again = persistence(changed, horizon)
        pd.testing.assert_series_equal(base[rows], again[rows], check_exact=True)


def test_persistence_guards_the_25th_hour():
    s = pd.Series(np.arange(24 * 40, dtype=float),
                  index=pd.date_range("2022-10-01", periods=24 * 40, freq="h", tz="UTC"))
    last = pd.Timestamp("2022-10-30 21:00", tz="UTC")  # 23:00 local, the 25th hour
    plain = s.shift(24)[last]  # = 2022-10-29 21:00 UTC, the hour starting at T0
    assert persistence(s, 1)[last] != plain


def test_climatology_uses_normal_train_rows_only():
    t = _truth()
    train = (t["split"] == "train") & ~t["op_limited_flag"]
    base = climatology(t["direct"], train)
    changed = t["direct"].where(train, t["direct"] + 999.0)  # change val, test, op-limited
    pd.testing.assert_series_equal(base, climatology(changed, train))
    ts = t.index[(t["split"] == "val") & t["daytime"]][0]
    same = train & (t.index.month == ts.month) & (t.index.hour == ts.hour)
    assert base[ts] == pytest.approx(t.loc[same, "direct"].mean())


def test_clearsky_reference():
    theo = pd.Series([10.0, 60.0, 1000.0, 5000.0])
    assert clearsky_k(theo, MEAN_K, "indirect").tolist() == [0.0, MEAN_K, MEAN_K, MEAN_K]
    assert clearsky_k(theo, MEAN_K, "direct").tolist() == pytest.approx(
        [8.0, 48.0, 800.0, AC_CAPACITY_KW])


def test_reference_predictions_follow_the_contract():
    t = _truth()
    refs = reference_predictions(t, "val", MEAN_K)
    n_val = int((t["split"] == "val").sum())
    assert set(refs["model"]) == {"persistence", "climatology", "clearsky_k"}
    assert len(refs) == 3 * 2 * len(HORIZONS) * n_val
    assert (refs["split"] == "val").all()
    assert set(refs.columns) >= set(E.REQUIRED_COLUMNS)


# ── T-39 evaluation rules ──────────────────────────────────────────────────

def test_test_split_needs_explicit_permission(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "TEST_ACCESS_LOG", tmp_path / "log.csv")
    with pytest.raises(PermissionError):
        E.run([], tmp_path / "out", MEAN_K, site=SITE, truth=_truth(), split="test")
    assert not (tmp_path / "log.csv").exists()


def test_alignment_catches_a_shift_and_a_swap():
    t = _truth()
    good = _model_preds(t)
    table = E.check_alignment(good, t, "val")
    assert table["ok"].all() and (table["compared_rows"] == table["rows"]).all()
    # R-G1-13: a file whose truth is withheld (phase 1) compares nothing, and the table says so
    vacuous = E.check_alignment(E.withhold_test_truth(_model_preds(t, split="test")), t, "test")
    assert (vacuous["compared_rows"] == 0).all() and vacuous["ok"].all()
    shifted = good.copy()
    shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=1)  # the v6 BiLSTM bug
    assert not E.check_alignment(shifted, t, "val")["ok"].all()
    swapped = good.copy()
    swapped["target"] = swapped["target"].map({"indirect": "direct", "direct": "indirect"})
    assert not E.check_alignment(swapped, t, "val")["ok"].any()
    wrong_split = _model_preds(t, split="test")
    wrong_split["split"] = "val"
    assert not E.check_alignment(wrong_split, t, "val")["ok"].any()


def test_misaligned_file_fails_the_run(tmp_path):
    t = _truth()
    shifted = _model_preds(t)
    shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=1)
    path = _write(shifted, tmp_path / "predictions_xgb_sota.csv")
    with pytest.raises(E.EvaluationError, match="misaligned"):
        E.run([path], tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False)


def test_coverage_catches_a_missing_daytime_hour(tmp_path):
    t = _truth()
    preds = _model_preds(t)
    hole = t.index[(t["split"] == "val") & t["daytime"]][5]
    preds = preds[~((preds["timestamp"] == hole) & (preds["horizon"] == 2))]
    cov = E.check_coverage(preds, t, "val")
    assert cov.loc[cov["horizon"] == 2, "missing"].tolist() == [1, 1]
    assert cov.loc[cov["horizon"] != 2, "missing"].eq(0).all()
    path = _write(preds, tmp_path / "predictions_xgb_sota.csv")
    with pytest.raises(E.EvaluationError, match="incomplete"):
        E.run([path], tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False)


def test_zero_output_daytime_hours_are_kept():
    """E3: daytime comes from geometry; an hour with y_true = 0 (a stop) stays in."""
    t = _truth()
    ts = t.index[(t["split"] == "val") & t["daytime"]][:3]
    t.loc[ts, ["indirect", "direct"]] = 0.0
    t.loc[ts[1:], ["indirect", "direct"]] = np.nan  # missing truth drops out
    preds = _model_preds(t)
    out = E.attach_truth(preds[preds["timestamp"].isin(ts)], t)
    assert set(out["timestamp"]) == {ts[0]}


def test_route_comparison_is_paired():
    t = _truth()
    preds = E.attach_truth(_model_preds(t), t)
    preds["daytime"] = True
    extra = preds[(preds["target"] == "indirect")].copy()
    extra["timestamp"] = extra["timestamp"] + pd.Timedelta(days=200)  # rows only one route has
    t_long = pd.concat([t, t.set_axis(t.index + pd.Timedelta(days=200))])
    out = E.route_comparison(pd.concat([preds, extra]), t_long["theo_kw"])
    n = out.groupby(["model", "horizon"])["n"].nunique()
    assert (n == 1).all()  # both routes on the same rows
    assert out["n"].iloc[0] == (t["split"] == "val").sum()


def test_clearsky_route_advantage_is_zero(tmp_path):
    t = _truth()
    tables = E.run([], tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False)
    route = tables["route_comparison"]
    clear = route[(route["model"] == "clearsky_k") & (route["route"] == "direct")]
    assert clear["advantage_pp"].abs().max() < 1e-9


# ── T-40 metrics and tables ──────────────────────────────────────────────

def test_cliper_matches_theory_on_an_ar1_series():
    phi, n_days = 0.7, 3000
    rng = np.random.default_rng(3)
    y = np.zeros(24 * n_days)
    for i in range(24, len(y)):
        y[i] = phi * y[i - 24] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
    s = pd.Series(y, index=pd.date_range("2000-01-01", periods=len(y), freq="h", tz="UTC"))
    use = pd.Series(True, index=s.index)
    assert cliper_rmse(s, use, 24) == pytest.approx(np.sqrt(1 - phi ** 2), rel=0.03)


def test_kw_metrics_and_conversion():
    assert E.to_kw("indirect", [0.5, 2.0, -0.1], [1000.0, 3000.0, 500.0]).tolist() == [
        500.0, AC_CAPACITY_KW, 0.0]
    assert E.to_kw("direct", [5.0], [1.0]).tolist() == [5.0]
    truth = pd.DataFrame({"direct": [100.0, 300.0], "theo_kw": [200.0, 400.0]},
                         index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))
    g = pd.DataFrame({"timestamp": truth.index, "y_pred": [0.5, 0.5]})  # 100 and 200 kW
    met = E._kw_metrics(g, "indirect", truth)
    rmse_kw = np.sqrt((0.0 ** 2 + 100.0 ** 2) / 2)
    assert met["nRMSE_kW_pct"] == pytest.approx(rmse_kw / AC_CAPACITY_KW * 100)
    assert met["nRMSE_mean_pct"] == pytest.approx(rmse_kw / 200.0 * 100)


def test_seed_table_excludes_references():
    bench = pd.DataFrame({"model": ["xgb_sota"] * 3 + ["persistence"],
                          "seed": [42, 123, 777, -1], "target": "direct", "horizon": 1,
                          "nRMSE_pct": [10.0, 10.2, 10.1, 15.0]})
    out = E.seed_variability_table(bench)
    assert out["model"].tolist() == ["xgb_sota"]
    assert out["n_seeds"].iloc[0] == 3
    assert out["nRMSE_range"].iloc[0] == pytest.approx(0.2)
    assert out["nRMSE_std"].iloc[0] == pytest.approx(np.std([10.0, 10.2, 10.1], ddof=1))


def test_dm_mean_ci_is_reproducible_and_contains_the_mean():
    d = np.random.default_rng(5).normal(0.3, 1.0, 2000)
    lo, hi = bootstrap_mean_ci(d)
    assert lo < d.mean() < hi
    assert (lo, hi) == bootstrap_mean_ci(d)


def test_views_follow_the_flags(tmp_path):
    t = _truth()
    tables = E.run([_write(_model_preds(t), tmp_path / "predictions_xgb_sota.csv")],
                   tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False)
    views = tables["benchmark_views"]
    day = t[(t["split"] == "val") & t["daytime"]]
    row = views[(views["model"] == "xgb_sota") & (views["target"] == "direct")
                & (views["horizon"] == 1)].set_index("view")
    assert row.loc["all", "n"] == len(day)
    assert row.loc["normal_operation", "n"] == int((~day["op_limited_flag"]).sum())
    assert row.loc["source_polled", "n"] == int((day["hour_source"] == "polled").sum())
    bench = tables["benchmark_summary"].set_index(["model", "target", "horizon"])
    assert pd.isna(bench.loc[("persistence", "direct", 1), "skill_score"])  # not vs itself
    assert bench.loc[("xgb_sota", "direct", 1), "n_skill"] == len(day)  # A1: full reference


def test_persistence_does_not_beat_the_cliper_reference(tmp_path):
    """On its own sample the CLIPER blend is never worse than persistence (2(1-g) >= 1-g^2).
    A reference from train statistics broke this on validation (docs/stage8b_plan.md §7)."""
    tables = E.run([], tmp_path / "out", MEAN_K, site=SITE, truth=_truth(), ci=False)
    bench = tables["benchmark_summary"]
    assert (bench.loc[bench["model"] == "persistence", "skill_cliper"] <= 0.02).all()


# ── T-54 fixes found by the stage-9 self-review (docs/stage9_plan.md §10.2) ──

def test_duplicates_across_prediction_files_are_refused(tmp_path):
    """X2: two files of the same model and seed doubled n (54 -> 108) before the fix."""
    t = _truth()
    a = _write(_model_preds(t), tmp_path / "predictions_xgb_sota.csv")
    b = _write(_model_preds(t), tmp_path / "predictions_xgb_sota_copy.csv")
    with pytest.raises(E.EvaluationError, match="more than once"):
        E.run([a, b], tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False)
    refs = reference_predictions(t, "val", MEAN_K)
    c = _write(refs[refs["model"] == "persistence"], tmp_path / "predictions_persistence.csv")
    with pytest.raises(E.EvaluationError, match="with_references=False"):
        E.run([c], tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False)
    tables = E.run([c], tmp_path / "out", MEAN_K, site=SITE, truth=t, ci=False,
                   with_references=False)
    assert set(tables["benchmark_summary"]["model"]) == {"persistence"}


def test_a_failed_evaluation_leaves_no_manifest(tmp_path, monkeypatch):
    """X4: new tables next to an old manifest looked like a complete run."""
    t = _truth()
    out = tmp_path / "out"
    E.run([], out, MEAN_K, site=SITE, truth=t, ci=False)
    assert (out / "manifest.json").exists()
    real, calls = pd.DataFrame.to_csv, []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("disk full")
        return real(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_csv", failing)
    with pytest.raises(OSError):
        E.run([], out, MEAN_K, site=SITE, truth=t, ci=False)
    assert not (out / "manifest.json").exists()


def test_main_never_globs_for_prediction_files(tmp_path, monkeypatch):
    """X1: a stale file under outputs/06_* was scored by the glob of evaluate.main."""
    stale = tmp_path / "outputs" / "06_xgb_sota" / "predictions_xgb_sota.csv"
    stale.parent.mkdir(parents=True)
    _write(_model_preds(_truth()), stale)
    site = load_site()  # main() uses the default site
    idx = pd.date_range("2022-10-01", "2022-11-20", freq="h", tz="UTC", inclusive="left")
    truth = _truth().assign(daytime=daytime_mask(idx, site))
    monkeypatch.setattr(E, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(E, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(E, "FEATURES_FILE", stale)  # any file: only its hash is recorded
    calls = []
    monkeypatch.setattr(E, "load_truth", lambda **kw: calls.append(kw) or truth)
    monkeypatch.setattr(E, "mean_daytime_k", lambda: MEAN_K)
    assert E.main(["--no-ci", "--out", str(tmp_path / "out")]) == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"] == [] and manifest["references"] == "computed from the truth"
    assert calls == [{"exclude_splits": ("test",)}]  # X17: validation reads no test row


def test_load_truth_can_exclude_the_test_split(tmp_path):
    """X3: the test rows are dropped inside the parquet reader."""
    t = _truth()
    table = t.rename(columns={"indirect": "target_k", "direct": "site_total_AC_kW"})
    table = table.sort_index()
    table.index.name = "timestamp_utc"
    # the synthetic splits interleave by day; keep a test tail so the grid stays complete
    table["split"] = np.where(table.index < table.index[-24 * 5], "val", "test")
    path = tmp_path / "features.parquet"
    table.to_parquet(path)
    full = load_truth(path)
    part = load_truth(path, exclude_splits=("test",))
    assert set(part["split"]) == {"val"}
    pd.testing.assert_frame_equal(part, full[full["split"] != "test"])


# ── T-65 Q-19: DM and skill in kW for both routes (docs/stage9_plan.md §16) ──

def _one(frame, model, target, h=1):
    g = frame[(frame["model"] == model) & (frame["target"] == target) & (frame["horizon"] == h)]
    return g.set_index("timestamp")


def test_dm_and_skill_in_kw_for_both_routes(tmp_path):
    t = _truth()
    model = _model_preds(t)
    tables = E.run([_write(model, tmp_path / "predictions_xgb_sota.csv")], tmp_path / "out",
                   MEAN_K, site=SITE, truth=t, ci=False)
    assert (tmp_path / "out" / "dm_tests_kw.csv").exists()
    dm, dm_kw = tables["dm_tests"], tables["dm_tests_kw"]
    assert list(dm_kw.columns) == list(dm.columns) + ["unit"] and set(dm_kw["unit"]) == {"kW"}
    key = ["target", "horizon", "model_a", "model_b"]
    direct = dm[dm["target"] == "direct"].set_index(key).sort_index()
    pd.testing.assert_frame_equal(  # X18: the direct target is already in kW
        direct, dm_kw[dm_kw["target"] == "direct"].set_index(key).sort_index()[direct.columns])
    # the indirect route by hand: actual AC as truth, both forecasts through to_kw
    refs = reference_predictions(t, "val", MEAN_K)
    clim, xgb = _one(refs, "climatology", "indirect"), _one(model, "xgb_sota", "indirect")
    day = t.index[(t["split"] == "val") & t["daytime"]]
    common = day.intersection(clim.index).intersection(xgb.index).sort_values()
    theo, ac = t.loc[common, "theo_kw"], t.loc[common, "direct"].to_numpy()
    res = diebold_mariano(ac, E.to_kw("indirect", clim.loc[common, "y_pred"], theo),
                          E.to_kw("indirect", xgb.loc[common, "y_pred"], theo))
    row = dm_kw[(dm_kw["target"] == "indirect") & (dm_kw["horizon"] == 1)
                & (dm_kw["model_a"] == "climatology")].iloc[0]
    assert row["model_b"] == "xgb_sota" and row["n_pairs"] == res.n
    assert row["dm_stat_hln"] == pytest.approx(res.dm_stat, rel=1e-12)
    k_row = dm[(dm["target"] == "indirect") & (dm["horizon"] == 1)
               & (dm["model_a"] == "climatology")].iloc[0]
    assert k_row["dm_stat_hln"] != pytest.approx(row["dm_stat_hln"])  # another unit
    bench = tables["benchmark_summary"].set_index(["model", "target", "horizon"])
    d = bench.xs("direct", level="target")
    for k_col, kw_col in (("skill_score", "skill_score_kw"), ("skill_cliper", "skill_cliper_kw")):
        np.testing.assert_allclose(d[kw_col], d[k_col], rtol=0, atol=1e-12, equal_nan=True)
    pers = _one(refs, "persistence", "indirect")
    both = day.intersection(pers.index).intersection(xgb.index)
    theo_b, ac_b = t.loc[both, "theo_kw"], t.loc[both, "direct"]
    expected = 1 - rmse(ac_b, E.to_kw("indirect", xgb.loc[both, "y_pred"], theo_b)) / rmse(
        ac_b, E.to_kw("indirect", pers.loc[both, "y_pred"], theo_b))
    assert bench.loc[("xgb_sota", "indirect", 1), "skill_score_kw"] == pytest.approx(expected)
    rmse_kw = bench.loc[("xgb_sota", "indirect", 1), "nRMSE_kW_pct"] * AC_CAPACITY_KW / 100
    cliper_ac = E.cliper_references(t, "val")[("direct", 1)]
    assert bench.loc[("xgb_sota", "indirect", 1), "skill_cliper_kw"] == pytest.approx(
        1 - rmse_kw / cliper_ac, rel=1e-9)


def test_the_fixture_mode_has_no_kw_tables(tmp_path):
    preds = _model_preds(_truth(), split="train")  # 40 days: enough pairs for the DM
    tables = E.run([_write(preds, tmp_path / "predictions_xgb_sota.csv")],
                   tmp_path / "out", MEAN_K, site=SITE, ci=False)  # no truth table: stage 8A
    assert "dm_tests_kw" not in tables
    assert "skill_score_kw" not in tables["benchmark_summary"].columns

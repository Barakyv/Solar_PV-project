"""Stage 11, part A (docs/stage11_plan.md, OP-66): the analyses of the frozen final run.

T-169 to T-176. The synthetic tests check the arithmetic of `solarpv.figures.analyses11`; the output
tests check that the written tables are the ones the module computes from the frozen files, and
that nothing here can score the test split by a path other than reading the saved forecasts.
"""

import json

import numpy as np
import pandas as pd
import pytest

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import sha256
from solarpv.figures import analyses11 as A

OUT = REPO_ROOT / A.OUT


def _frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    """A synthetic frame in the layout of `load_frame`: two splits, two seeds of one model."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, offset in (("val", "2024-03-01"), ("test", "2024-11-01")):
        ts = pd.date_range(offset, periods=n, freq="h", tz="UTC")
        y = rng.uniform(0, 3000, n)
        op = rng.random(n) < 0.08
        hour, month, season = A.local_season(ts)
        for s in (42, 123):
            f = y + rng.normal(30, 250, n)
            rows.append(pd.DataFrame({
                "model": "xgb_sota", "seed": s, "target": "direct", "horizon": 1, "split": split,
                "timestamp": ts, "y": y, "f": f, "op_limited": op, "hour": hour, "month": month,
                "season": season}))
    return pd.concat(rows, ignore_index=True)


def test_t169_decomposition_identity_and_seed_split():
    dec, ens, worst = A.decomposition(_frame())
    assert worst < 1e-12
    r = dec[(dec["view"] == "all") & (dec["split"] == "test") & (dec["seed"] == 42)].iloc[0]
    assert r["bias2_share"] + r["error_variance_share"] == pytest.approx(1.0, abs=1e-12)
    assert 0 <= r["bias2_share"] < 0.1  # the mean error is small against the noise
    e = ens[(ens["view"] == "all") & (ens["split"] == "test")].iloc[0]
    # MSE of a seed = bias² of the seed mean + variance between seeds
    assert e["mse_seed_mean_kw2"] == pytest.approx(
        e["bias2_of_seed_mean_kw2"] + e["seed_variance_kw2"], rel=1e-9)
    assert e["n_seeds"] == 2


def test_t170_quintile_edges_come_from_validation_and_the_bins_partition():
    f = _frame()
    q = A.quintiles(f)
    for _, g in q.groupby(["seed", "split", "binned_by"]):
        n = int(f[(f["seed"] == g["seed"].iloc[0]) & (f["split"] == g["split"].iloc[0])].shape[0])
        assert g["bin"].tolist() == [1, 2, 3, 4, 5] and g["n"].sum() == n
    val = f[(f["split"] == "val") & (f["seed"] == 42)]
    edges = q[(q["seed"] == 42) & (q["split"] == "val") & (q["binned_by"] == "observed")
              ]["edge_hi_kw"].to_numpy()[:4]
    assert edges == pytest.approx(np.quantile(val["y"], [0.2, 0.4, 0.6, 0.8]))
    # the test hours never move the edges (the edges are fixed before the test is read)
    g = f.copy()
    g.loc[g["split"] == "test", "y"] *= 2.0
    again = A.quintiles(g)
    cols = ["edge_lo_kw", "edge_hi_kw"]
    pd.testing.assert_frame_equal(q[cols].reset_index(drop=True),
                                  again[cols].reset_index(drop=True))


def test_t171_binning_by_the_outcome_makes_the_pattern_without_conditional_bias():
    """M45: a forecast that is unbiased given the forecast shows positive errors in the lowest and
    negative errors in the highest bin of the observed power."""
    rng = np.random.default_rng(3)
    n = 6000
    y = rng.uniform(0, 3000, n)
    f = y + rng.normal(0, 400, n)
    ts = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({"model": "xgb_sota", "seed": 42, "target": "direct", "horizon": 1,
                          "split": np.where(np.arange(n) < n // 2, "val", "test"),
                          "timestamp": ts, "y": y, "f": f, "op_limited": False, "hour": 12,
                          "month": 3, "season": "Spring"})
    ctl = A.quintile_control(frame)
    t = ctl[ctl["split"] == "test"].set_index("bin")["me_calibrated_kw"]
    assert t.loc[1] > 0 > t.loc[5]


def test_t172_local_hour_and_season_follow_the_local_clock_with_dst():
    ts = pd.DatetimeIndex(["2024-03-28 10:00", "2024-03-30 10:00", "2024-02-29 22:30",
                           "2024-12-01 05:00"], tz="UTC")
    hour, month, season = A.local_season(ts)
    # Israel moved to summer time on 2024-03-29: UTC+2 before, UTC+3 after
    assert hour.tolist() == [12, 13, 0, 7]
    assert month.tolist() == [3, 3, 3, 12] and season.tolist() == ["Spring", "Spring", "Spring",
                                                                   "Winter"]


def test_t173_written_tables_match_the_manifest_and_the_frozen_files():
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert man["run"] == A.RUN and set(man["rows"]) == set(
        p.stem for p in OUT.glob("*.csv"))
    for rel, digest in man["outputs_sha256"].items():
        assert sha256(REPO_ROOT / rel) == digest, f"{rel} changed after the analyses were written"
    assert A.verify_inputs() == man["inputs_sha256"]  # every input is the file its step recorded
    assert man["max_abs_diff_vs_report_nRMSE_kW_pct"] < 5e-6
    assert man["max_rel_identity_error_bias2_plus_variance"] < 1e-9
    metrics = pd.read_csv(OUT / "metrics_kw.csv")
    assert set(metrics["split"]) == {"val", "test"} and metrics["n"].min() > 2000 - 1


def test_t174_the_analyses_read_saved_forecasts_and_never_run_the_test_step():
    text = (REPO_ROOT / "src/solarpv/figures/analyses11.py").read_text(encoding="utf-8")
    for forbidden in ("allow_test=True", "--allow-test", "solarpv.pipeline", "run_step"):
        assert forbidden not in text, f"analyses11 must not use {forbidden}"
    nb = (REPO_ROOT / "scripts/make_stage11_part_a_notebook.py").read_text(encoding="utf-8")
    assert "allow_test=True" not in nb


def test_t175_the_literature_numbers_are_those_of_the_review():
    review = (REPO_ROOT / "docs" / "literature_review.md").read_text(encoding="utf-8")
    for paper, _, value in A.LITERATURE:
        assert value in review, f"{paper}: {value} is not in docs/literature_review.md"
    assert "normalised by mean daytime power" in review or "normalised by the mean" in review


def test_t177_the_gate_audit_does_not_have_to_cover_a_final_run(tmp_path, monkeypatch):
    """X106: after stage 10 the G-1 checklist (S09-34) read the final run as an unaudited run and
    reported the item 'test split touched no decision' as waiting. A final run is authorised by
    D-32; any other run the audit did not see still fails the check."""
    from solarpv.figures import s09_w6

    runs = tmp_path / "outputs" / "runs"
    for name in ("20260101T000000Z_dev", "20261114T083930Z_final"):
        (runs / name).mkdir(parents=True)
        (runs / name / "predictions_x.csv").write_text("a\n", encoding="utf-8")
    monkeypatch.setattr(s09_w6, "REPO_ROOT", tmp_path)
    audit = {"per_run": [{"run": "20260101T000000Z_dev"}], "prediction_files": 1}
    assert s09_w6._access_covers_disk(audit)
    (runs / "20260202T000000Z_dev").mkdir()
    assert not s09_w6._access_covers_disk(audit)  # a run the audit did not see


def test_t176_every_read_of_the_test_rows_is_in_the_access_log():
    log = (REPO_ROOT / "outputs" / "07_evaluation" / "test_access_log.csv").read_text(
        encoding="utf-8")
    assert "python -m solarpv.figures.analyses11" in log

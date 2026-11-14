"""Stage 11, part A: analyses of the frozen final run (docs/stage11_plan.md; OP-66).

    python -m solarpv.figures.analyses11

Reads only files of the run `20261114T083930Z_final` (the prediction files its steps wrote, checked
against the SHA-256 of their step records) and the stage-6 truth table, never re-runs a step and
never trains. The truth of the evaluation comes from the feature table, as in the step `evaluate`
(E8); every row is scored in kW against the actual AC (Q-19). Before anything is written, the
nRMSE of every model, seed, route, horizon and split is compared with the run's own report tables
(X100). The reads of the test rows are appended to `outputs/07_evaluation/test_access_log.csv`
(X101). Nothing here selects, tunes or decides: the output is a set of tables in
`outputs/11_analyses/`, read by the figures `S11-xx` and the notebook.
"""

import json
import sys

import numpy as np
import pandas as pd

from solarpv.config.cli import no_options
from solarpv.config.constants import AC_CAPACITY_KW, TIMEZONE_LOCAL
from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import git_commit, sha256
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.metrics import bootstrap_mean_ci, bootstrap_nrmse_ci
from solarpv.evaluation.stats_tests import bonferroni, diebold_mariano, power_analysis
from solarpv.evaluation.truth import load_truth

RUN = "outputs/runs/20261114T083930Z_final"
OUT = "outputs/11_analyses"
SPLITS = ("val", "test")
VIEWS = ("all", "normal_operation")
CELLS = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]
MIN_HOURS = 20  # a cell of the hour x season table with fewer hours is masked in the figure (M47)
SCATTER_BINS = 25

# (step folder, model, seeds); the file names are those of the run
INPUTS = (
    ("baselines", "clearsky_k", (-1,)), ("baselines", "climatology", (-1,)),
    ("baselines", "persistence", (-1,)),
    ("xgb", "xgb_sota", (42, 123, 777)), ("bilstm", "lstm_sota", (42, 123, 777)),
    ("tft96", "tft_96", (42,)), ("tft168", "tft_168", (42,)), ("tftbase", "tft", (42,)),
    ("patchtst", "patchtst", (42,)),
)
REFERENCES = ("clearsky_k", "climatology", "persistence")
ENSEMBLE_MODELS = ("xgb_sota", "lstm_sota")  # the models with three seeds
SEED_ORDER_FIRST = {"xgb_sota": 42, "lstm_sota": 42}

# What the papers of docs/literature_review.md report, normalised by the mean daytime power
# (C1: Markovics et al. 2026; C2: Mayer et al. 2025). The text of the review is checked by a test.
LITERATURE = (
    ("C1", "2-day persistence, mean of 7 Hungarian plants", "81.6"),
    ("C1", "physical model chain", "48.7"),
    ("C1", "best tuned models (RNN, Pyraformer)", "42.7"),
    ("C1", "worst tuned models (XGBoost, LightGBM)", "43.5"),
    ("C2", "best workflow, ensemble NWP with power bias correction", "37.4"),
)


def prediction_files() -> list[tuple[str, str, int, str]]:
    """(step, model, seed, repo-relative path) of every file the analyses read."""
    out = []
    for step, model, seeds in INPUTS:
        for seed in seeds:
            name = f"predictions_{model}.csv" if seed == -1 else \
                f"predictions_{model}_seed{seed}.csv"
            out.append((step, model, seed, f"{RUN}/{step}/{name}"))
    return out


def verify_inputs() -> dict[str, str]:
    """SHA-256 of every prediction file, equal to the one its step record wrote."""
    shas = {}
    for step, _, _, rel in prediction_files():
        record = json.loads((REPO_ROOT / RUN / step / "step.json").read_text(encoding="utf-8"))
        name = rel.rsplit("/", 1)[1]
        digest = sha256(REPO_ROOT / rel)
        assert record["outputs"][name] == digest, f"{rel} differs from the record of its step"
        shas[rel] = digest
    return shas


def local_season(ts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local hour, local month and season of UTC timestamps: the local clock with DST (F-01,
    X102)."""
    local = pd.DatetimeIndex(ts).tz_convert(TIMEZONE_LOCAL)
    month = local.month.to_numpy()
    return local.hour.to_numpy(), month, np.array([E.SEASON_OF_MONTH[m] for m in month])


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every scored row (daytime, with truth, validation and test): truth y and forecast f in kW."""
    truth = load_truth()
    parts = []
    for _, _, _, rel in prediction_files():
        p = E.load_predictions(REPO_ROOT / rel)
        p = p[p["split"].isin(SPLITS)]
        p = E.in_kw(p, truth)
        ts = pd.DatetimeIndex(p["timestamp"])
        keep = truth["daytime"].reindex(ts).to_numpy(dtype=bool)
        p = p[keep].copy()
        ts = pd.DatetimeIndex(p["timestamp"])
        hour, month, season = local_season(ts)
        parts.append(pd.DataFrame({
            "model": p["model"].to_numpy(), "seed": p["seed"].to_numpy(),
            "target": p["target"].to_numpy(), "horizon": p["horizon"].to_numpy(),
            "split": p["split"].to_numpy(), "timestamp": ts, "y": p["y_true"].to_numpy(float),
            "f": p["y_pred"].to_numpy(float),
            "op_limited": truth["op_limited_flag"].reindex(ts).to_numpy(dtype=bool),
            "hour": hour, "month": month, "season": season}))
    return pd.concat(parts, ignore_index=True), truth


def _cells(frame: pd.DataFrame, keys=("model", "seed", "target", "horizon", "split")):
    return frame.groupby(list(keys), sort=True)


def _view(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    return frame if view == "all" else frame[~frame["op_limited"]]


# ── the metrics, and their check against the run's report (X100) ────────────

def metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in VIEWS:
        for key, g in _cells(_view(frame, view)):
            e = g["f"].to_numpy() - g["y"].to_numpy()
            rmse = float(np.sqrt(np.mean(e ** 2)))
            rows.append({**dict(zip(("model", "seed", "target", "horizon", "split"), key,
                                    strict=True)), "view": view, "n": len(g),
                         "rmse_kw": rmse, "nRMSE_kW_pct": rmse / AC_CAPACITY_KW * 100.0,
                         "nRMSE_mean_pct": rmse / float(g["y"].mean()) * 100.0,
                         "me_kw": float(e.mean())})
    return pd.DataFrame(rows)


def check_against_report(metrics: pd.DataFrame) -> float:
    """The largest difference between this module's nRMSE in kW and the report tables of the run."""
    worst = 0.0
    for split in SPLITS:
        rep = pd.read_csv(REPO_ROOT / f"{RUN}/report/report_benchmark_{split}.csv")
        mine = metrics[(metrics["split"] == split) & (metrics["view"] == "all")]
        key = ["model", "seed", "target", "horizon"]
        both = rep.merge(mine, on=key, suffixes=("_report", "_mine"))
        assert len(both) == len(rep) == len(mine), f"{split}: the rows of the tables differ"
        assert (both["n_report"] == both["n_mine"]).all(), f"{split}: n differs"
        worst = max(worst, float((both["nRMSE_kW_pct_report"]
                                  - both["nRMSE_kW_pct_mine"]).abs().max()))
    assert worst < 5e-6, f"the nRMSE in kW differs from the report by {worst}"
    return worst


# ── S-39 and L-15: bias² + variance, and the distribution diagnostics ──────

def decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """MSE = ME² + Var(e) per forecast (the definition of the submitted §12.3), and the seed split
    for the models with three seeds: MSE of a seed = bias² of the seed mean + variance across seeds
    (seeds enter with equal weight, ddof 0)."""
    rows, worst = [], 0.0
    for view in VIEWS:
        for key, g in _cells(_view(frame, view)):
            y, f = g["y"].to_numpy(), g["f"].to_numpy()
            e = f - y
            mse, me = float(np.mean(e ** 2)), float(e.mean())
            var_e = float(np.mean((e - me) ** 2))
            worst = max(worst, abs(mse - (me ** 2 + var_e)) / mse)
            rows.append({**dict(zip(("model", "seed", "target", "horizon", "split"), key,
                                    strict=True)), "view": view, "n": len(g), "mse_kw2": mse,
                         "me_kw": me, "bias2_share": me ** 2 / mse,
                         "error_variance_share": var_e / mse,
                         "corr_forecast_observed": float(np.corrcoef(f, y)[0, 1]),
                         "variance_ratio": float(np.var(f) / np.var(y))})
    seeds = []
    for view in VIEWS:
        part = _view(frame, view)
        for (model, target, h, split), g in part[part["model"].isin(ENSEMBLE_MODELS)].groupby(
                ["model", "target", "horizon", "split"], sort=True):
            wide = g.pivot(index="timestamp", columns="seed", values="f").sort_index()
            assert not wide.isna().any().any(), "the seeds do not cover the same hours"
            y = g[g["seed"] == wide.columns[0]].set_index("timestamp")["y"].reindex(wide.index)
            f = wide.to_numpy()
            mse_seed = np.mean((f - y.to_numpy()[:, None]) ** 2, axis=0)
            bias2 = float(np.mean((f.mean(axis=1) - y.to_numpy()) ** 2))
            seed_var = float(np.mean(f.var(axis=1)))
            assert abs(mse_seed.mean() - (bias2 + seed_var)) / mse_seed.mean() < 1e-9
            seeds.append({"model": model, "target": target, "horizon": h, "split": split,
                          "view": view, "n_seeds": f.shape[1], "n": len(wide),
                          "mse_seed_mean_kw2": float(mse_seed.mean()),
                          "bias2_of_seed_mean_kw2": bias2, "seed_variance_kw2": seed_var,
                          "seed_variance_share": seed_var / float(mse_seed.mean())})
    assert worst < 1e-9, f"the identity MSE = ME² + Var(e) fails by {worst}"
    return pd.DataFrame(rows), pd.DataFrame(seeds), worst


# ── S-40: production quintiles, with the control of M45 ─────────────────────

def quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Quintile bins of the observed power (as in the submitted §12.4) and of the forecast itself,
    with edges fixed on the validation hours of the same cell and used on both splits."""
    rows = []
    for key, g in frame[frame["split"] == "val"].groupby(
            ["model", "seed", "target", "horizon"], sort=True):
        model, seed, target, h = key
        edges = {"observed": np.quantile(g["y"], [0.2, 0.4, 0.6, 0.8]),
                 "forecast": np.quantile(g["f"], [0.2, 0.4, 0.6, 0.8])}
        for split in SPLITS:
            s = frame[(frame["model"] == model) & (frame["seed"] == seed)
                      & (frame["target"] == target) & (frame["horizon"] == h)
                      & (frame["split"] == split)]
            for by, col in (("observed", "y"), ("forecast", "f")):
                b = np.searchsorted(edges[by], s[col].to_numpy(), side="right")
                for q in range(5):
                    m = b == q
                    d = s[m]
                    e = d["f"].to_numpy() - d["y"].to_numpy()
                    lo = -np.inf if q == 0 else edges[by][q - 1]
                    hi = np.inf if q == 4 else edges[by][q]
                    rows.append({"model": model, "seed": seed, "target": target, "horizon": h,
                                 "split": split, "binned_by": by, "bin": q + 1, "edge_lo_kw": lo,
                                 "edge_hi_kw": hi, "n": int(m.sum()),
                                 "mean_observed_kw": float(d["y"].mean()) if m.any() else np.nan,
                                 "mean_forecast_kw": float(d["f"].mean()) if m.any() else np.nan,
                                 "me_kw": float(e.mean()) if m.any() else np.nan,
                                 "nRMSE_kW_pct": float(np.sqrt(np.mean(e ** 2))
                                                       / AC_CAPACITY_KW * 100) if m.any()
                                 else np.nan,
                                 "op_limited_share": float(d["op_limited"].mean()) if m.any()
                                 else np.nan})
    return pd.DataFrame(rows)


def quintile_control(frame: pd.DataFrame) -> pd.DataFrame:
    """A control for M45. The XGB forecast of seed 42 is made conditionally unbiased on validation
    (isotonic regression of the observed power on the forecast, a control and not a model), and
    its mean error is read in the bins of the observed power, with the validation edges. If the
    pattern of the submitted §12.4 appears for a forecast that has no conditional bias by
    construction, binning by the outcome produces it."""
    from sklearn.isotonic import IsotonicRegression
    rows = []
    for (target, h), g in frame[(frame["model"] == "xgb_sota") & (frame["seed"] == 42)].groupby(
            ["target", "horizon"], sort=True):
        val = g[g["split"] == "val"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(val["f"], val["y"])
        edges = np.quantile(val["y"], [0.2, 0.4, 0.6, 0.8])
        for split in SPLITS:
            s = g[g["split"] == split]
            e = iso.predict(s["f"]) - s["y"].to_numpy()
            b = np.searchsorted(edges, s["y"].to_numpy(), side="right")
            for q in range(5):
                rows.append({"model": "xgb_sota", "seed": 42, "target": target, "horizon": h,
                             "split": split, "bin": q + 1, "n": int((b == q).sum()),
                             "me_calibrated_kw": float(e[b == q].mean()),
                             "me_all_hours_calibrated_kw": float(e.mean())})
    return pd.DataFrame(rows)


# ── S-30, S-29, L-20: error by local hour and season ────────────────────────

def _months(g: pd.DataFrame) -> str:
    ym = sorted({(t.tz_convert(TIMEZONE_LOCAL).strftime("%Y-%m")) for t in g["timestamp"]})
    return f"{ym[0]}..{ym[-1]}"


def hour_season(frame: pd.DataFrame) -> pd.DataFrame:
    """RMSE and mean error by local hour and season; per seed first, then the mean over seeds."""
    keys = ["model", "seed", "target", "horizon", "split", "season", "hour"]
    outs = []
    for view in VIEWS:
        part = _view(frame, view)
        per_seed = part.assign(e=part["f"] - part["y"]).groupby(keys, sort=True).agg(
            n=("e", "size"), mse=("e", lambda e: float(np.mean(e ** 2))), me=("e", "mean")
        ).reset_index()
        per_seed["rmse_kw"] = np.sqrt(per_seed["mse"])
        out = per_seed.groupby([k for k in keys if k != "seed"], sort=True).agg(
            n=("n", "first"), rmse_kw=("rmse_kw", "mean"), me_kw=("me", "mean")).reset_index()
        out["nRMSE_kW_pct"] = out["rmse_kw"] / AC_CAPACITY_KW * 100.0
        outs.append(out.assign(view=view))
    return pd.concat(outs, ignore_index=True)


def seasonal(frame: pd.DataFrame) -> pd.DataFrame:
    """nRMSE in kW and by the mean observed power of the group, per season, in the units of every
    model; the months of the season that the split holds are named (X104)."""
    rows = []
    months = {}
    for (split, season), g in frame.groupby(["split", "season"], sort=True):
        months[(split, season)] = _months(g)
    for key, g in frame.groupby(["model", "seed", "target", "horizon", "split", "season"],
                                sort=True):
        e = g["f"].to_numpy() - g["y"].to_numpy()
        rmse = float(np.sqrt(np.mean(e ** 2)))
        model, seed, target, h, split, season = key
        rows.append({"model": model, "seed": seed, "target": target, "horizon": h, "split": split,
                     "season": season, "months": months[(split, season)], "n": len(g),
                     "rmse_kw": rmse, "nRMSE_kW_pct": rmse / AC_CAPACITY_KW * 100.0,
                     "nRMSE_mean_pct": rmse / float(g["y"].mean()) * 100.0,
                     "me_kw": float(e.mean()), "op_limited_share": float(g["op_limited"].mean())})
    df = pd.DataFrame(rows)
    return df.groupby(["model", "target", "horizon", "split", "season", "months"],
                      sort=True).agg(n=("n", "first"), nRMSE_kW_pct=("nRMSE_kW_pct", "mean"),
                                     nRMSE_mean_pct=("nRMSE_mean_pct", "mean"),
                                     me_kw=("me_kw", "mean"),
                                     op_limited_share=("op_limited_share", "first")
                                     ).reset_index()


# ── S-34 and S-35: two paired comparisons ───────────────────────────────────

def _pair(frame: pd.DataFrame, a: tuple[str, int], b: tuple[str, int], split: str, target: str,
          h: int) -> dict:
    def pick(m):
        g = frame[(frame["model"] == m[0]) & (frame["seed"] == m[1]) & (frame["target"] == target)
                  & (frame["horizon"] == h) & (frame["split"] == split)]
        return g.set_index("timestamp").sort_index()

    ga, gb = pick(a), pick(b)
    common = ga.index.intersection(gb.index)
    assert len(common) == len(ga) == len(gb), "the two models do not cover the same hours"
    y = ga.loc[common, "y"].to_numpy()
    pa, pb = ga.loc[common, "f"].to_numpy(), gb.loc[common, "f"].to_numpy()
    res = diebold_mariano(y, pa, pb)
    d = (y - pa) ** 2 - (y - pb) ** 2
    lo, hi = bootstrap_mean_ci(d)
    nr = lambda p: float(np.sqrt(np.mean((y - p) ** 2)) / AC_CAPACITY_KW * 100)  # noqa: E731
    return {"model_a": a[0], "seed_a": a[1], "model_b": b[0], "seed_b": b[1], "split": split,
            "target": target, "horizon": h, "n": res.n, "nRMSE_a_kW_pct": nr(pa),
            "nRMSE_b_kW_pct": nr(pb), "diff_a_minus_b_pp": nr(pa) - nr(pb),
            "mean_diff_kw2": res.mean_diff, "mean_diff_ci_lo": lo, "mean_diff_ci_hi": hi,
            "p_value_hln": res.p_value, **power_analysis(d)}


def encoder_length(frame: pd.DataFrame) -> pd.DataFrame:
    """TFT-168 against TFT-96 (a = 168, b = 96): a positive mean difference means 168 has the
    larger squared error. Bonferroni over the six cells of a split."""
    rows = [_pair(frame, ("tft_168", 42), ("tft_96", 42), s, t, h)
            for s in SPLITS for t, h in CELLS]
    df = pd.DataFrame(rows)
    for _, g in df.groupby("split"):
        alpha, sig = bonferroni(g["p_value_hln"].to_numpy(), 0.05)
        df.loc[g.index, "alpha_bonferroni"] = alpha
        df.loc[g.index, "significant"] = sig
    return df


def patchtst_gap(frame: pd.DataFrame) -> pd.DataFrame:
    """PatchTST against XGB (seed 42) and against climatology (a = PatchTST)."""
    rows = []
    for s in SPLITS:
        for t, h in CELLS:
            for ref in (("xgb_sota", 42), ("climatology", -1)):
                rows.append(_pair(frame, ("patchtst", 42), ref, s, t, h))
    df = pd.DataFrame(rows)
    for _, g in df.groupby(["split", "model_b"]):
        alpha, sig = bonferroni(g["p_value_hln"].to_numpy(), 0.05)
        df.loc[g.index, "alpha_bonferroni"] = alpha
        df.loc[g.index, "significant"] = sig
    return df


# ── S-23, S-24: the benchmark with intervals in kW, and the scatter ────────

def benchmark_ci(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in _cells(frame):
        lo, hi = bootstrap_nrmse_ci(g["y"], g["f"], AC_CAPACITY_KW)
        e = g["f"].to_numpy() - g["y"].to_numpy()
        rows.append({**dict(zip(("model", "seed", "target", "horizon", "split"), key,
                                strict=True)), "n": len(g),
                     "nRMSE_kW_pct": float(np.sqrt(np.mean(e ** 2)) / AC_CAPACITY_KW * 100),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def scatter_hist(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts of the hours in a grid of observed against forecast power, D+1, lowest seed."""
    edges = np.linspace(0, AC_CAPACITY_KW, SCATTER_BINS + 1)
    rows = []
    for (model, target, split), g in frame[(frame["horizon"] == 1)].groupby(
            ["model", "target", "split"], sort=True):
        g = g[g["seed"] == g["seed"].min()]
        h, _, _ = np.histogram2d(g["y"].clip(0, AC_CAPACITY_KW - 1e-9),
                                 g["f"].clip(0, AC_CAPACITY_KW - 1e-9), bins=[edges, edges])
        for i, j in zip(*np.nonzero(h), strict=True):
            rows.append({"model": model, "seed": int(g["seed"].iloc[0]), "target": target,
                         "split": split, "obs_bin": int(i), "fc_bin": int(j), "count": int(h[i, j]),
                         "n": len(g)})
    return pd.DataFrame(rows)


# ── S-26 and L-20: horizon robustness and the literature alignment ──────────

def horizon_robustness(metrics: pd.DataFrame) -> pd.DataFrame:
    """The seed-mean nRMSE of a model on D+1, D+2 and D+3 in the same view, and D+3 minus D+1."""
    m = metrics[metrics["view"] == "all"]
    mean = m.groupby(["model", "target", "horizon", "split"])["nRMSE_kW_pct"].mean().unstack(
        "horizon")
    mean.columns = [f"D{c}" for c in mean.columns]
    mean["D3_minus_D1_pp"] = mean["D3"] - mean["D1"]
    return mean.reset_index()


def literature_alignment(metrics: pd.DataFrame) -> pd.DataFrame:
    """Our D+1 nRMSE normalised by the mean observed daytime power (L-16) next to the numbers of
    the papers, which are normalised the same way. No ranking is made (M50)."""
    m = metrics[(metrics["view"] == "all") & (metrics["horizon"] == 1)]
    ours = m.groupby(["model", "target", "split"])["nRMSE_mean_pct"].mean().reset_index()
    ours.insert(0, "source", "this rebuild (D+1, hourly, GFS, one plant)")
    ours = ours.rename(columns={"nRMSE_mean_pct": "nRMSE_mean_pct"})
    lit = pd.DataFrame([{"source": f"{c} ({what})", "model": "literature", "target": "",
                         "split": "test year", "nRMSE_mean_pct": float(v)}
                        for c, what, v in LITERATURE])
    return pd.concat([ours, lit], ignore_index=True)


# ── run ─────────────────────────────────────────────────────────────────────

def run(out: str = OUT) -> dict[str, pd.DataFrame]:
    shas = verify_inputs()
    frame, _ = load_frame()
    metrics = metrics_table(frame)
    worst_report = check_against_report(metrics)
    dec, ens, worst_identity = decomposition(frame)
    quint = quintiles(frame)
    for _, g in quint.groupby(["model", "seed", "target", "horizon", "split", "binned_by"]):
        assert g["bin"].tolist() == [1, 2, 3, 4, 5] and g["n"].sum() > 0  # X103
    n_rows = frame.groupby(["model", "seed", "target", "horizon", "split"]).size()
    got = quint.groupby(["model", "seed", "target", "horizon", "split", "binned_by"])["n"].sum()
    assert (got.droplevel("binned_by").groupby(level=[0, 1, 2, 3, 4]).first() == n_rows).all()
    tables = {
        "metrics_kw": metrics, "decomposition": dec, "decomposition_seeds": ens,
        "quintiles": quint, "quintile_control": quintile_control(frame),
        "hour_season": hour_season(frame), "seasonal_kw": seasonal(frame),
        "encoder_length": encoder_length(frame), "patchtst_gap": patchtst_gap(frame),
        "benchmark_ci_kw": benchmark_ci(frame), "scatter_hist": scatter_hist(frame),
        "horizon_robustness": horizon_robustness(metrics),
        "literature_alignment": literature_alignment(metrics),
    }
    folder = REPO_ROOT / out
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").unlink(missing_ok=True)
    written = {}
    for name, table in tables.items():
        path = folder / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.9g", lineterminator="\n")
        written[f"{out}/{name}.csv"] = sha256(path)
    command = "python -m solarpv.figures.analyses11 [stage 11 part A: tables of the frozen run]"
    E._log_test_access([REPO_ROOT / r for r in shas], folder, command)
    manifest = {
        "run": RUN, "git_commit": git_commit(), "inputs_sha256": shas,
        "max_abs_diff_vs_report_nRMSE_kW_pct": worst_report,
        "max_rel_identity_error_bias2_plus_variance": worst_identity,
        "rows": {k: len(v) for k, v in tables.items()}, "outputs_sha256": written,
        "min_hours_per_cell_masked": MIN_HOURS,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8",
                                          newline="\n")
    return tables


def main(argv: list[str] | None = None) -> int:
    no_options("python -m solarpv.figures.analyses11", __doc__, argv)  # X62: before any work
    return 0 if run() else 1


if __name__ == "__main__":
    sys.exit(main())

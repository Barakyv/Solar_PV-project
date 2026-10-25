"""Authoritative evaluation (stages 8A and 8B; PROJECT_MASTER.md §6.9-§6.10, docs/stage8b_plan.md).

    python -m solarpv.evaluation.evaluate                          # validation (default)
    python -m solarpv.evaluation.evaluate --split test --allow-test  # final evaluation only
    python -m solarpv.evaluation.evaluate --fixture                # old LSTM predictions (T-15)

Rules:
  * predictions follow the file contract (§6.10): one row per model/seed/target/horizon/ts;
  * the truth comes from the stage-6 feature table, never from the prediction files (E8). The
    y_true written in a file is only an alignment check: a shifted or swapped file fails (E9);
  * coverage: every model must predict every daytime hour of the split that has truth, so all
    models are scored on the same n (E9). Strict mode is the default outside the fixture;
  * daytime = sun elevation at the middle of the hour > 5 deg (F-02), from geometry only, never
    from the truth (E3). It must equal the feature table's `daytime`;
  * validation is the default split. The test split needs --allow-test, and every such run is
    appended to outputs/07_evaluation/test_access_log.csv (E10);
  * the references (persistence, climatology, clear sky x MEAN_K) are computed from the truth
    table by one code path (models.baselines). Persistence (h days, DST-safe) is the skill
    reference;
  * route comparison: actual AC is the truth for both routes, on the identical row set
    (C-16, E2). advantage_pp = direct - indirect nRMSE, so a positive value favours the
    indirect route;
  * DM: raw (old formula) and HLN-corrected with Bonferroni per target x horizon (D-14).
"""

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import (
    AC_CAPACITY_KW,
    HORIZONS,
    MEAN_DAYTIME_K_LEGACY,
    SIGNIFICANCE_LEVEL,
    TIMEZONE_LOCAL,
    mean_daytime_k,
)
from solarpv.config.paths import LSTM_FIXTURE, OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import git_commit, sha256
from solarpv.config.runlog import RunLog
from solarpv.config.site import Site, load_site
from solarpv.evaluation.daytime import daytime_mask
from solarpv.evaluation.metrics import (
    bootstrap_mean_ci,
    bootstrap_nrmse_ci,
    cliper_rmse,
    point_metrics,
    rmse,
)
from solarpv.evaluation.stats_tests import bonferroni, diebold_mariano, power_analysis
from solarpv.evaluation.truth import FEATURES_FILE, load_truth
from solarpv.models.baselines import (
    REFERENCES,
    persistence,
    reference_predictions,
    to_prediction_frame,
)

REQUIRED_COLUMNS = ("timestamp", "y_true", "y_pred", "target", "horizon", "model")
KEY = ["model", "seed", "target", "horizon"]
TARGETS = ("indirect", "direct")
SPLITS = ("train", "val", "test")
PERSISTENCE = "persistence"
NO_SEED = -1
EVAL_DIR = OUTPUTS_DIR / "07_evaluation"
TEST_ACCESS_LOG = EVAL_DIR / "test_access_log.csv"
ALIGN_TOL = {"indirect": 1e-6, "direct": 1e-3}  # k, kW
# v6 identifiers in the prediction files -> names in the report (docs/scope_register.md)
MODEL_LABELS = {"xgb_sota": "XGB-Opt", "lstm_sota": "BiLSTM-Opt", "tft_96": "TFT-96",
                "tft_168": "TFT-168", "tft": "TFT baseline", "patchtst": "PatchTST",
                "persistence": "Persistence", "climatology": "Climatology",
                "clearsky_k": "Clear sky x MEAN_K"}
VIEWS = ("all", "normal_operation", "source_minute", "source_polled", "source_mixed", "no_polled")
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


class EvaluationError(ValueError):
    """A prediction file breaks a rule of the evaluation (alignment, coverage, contract)."""


# ── loading and checks ─────────────────────────────────────────────────────


def load_predictions(path: Path) -> pd.DataFrame:
    """Read one prediction file and validate it against the contract (§6.10)."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise EvaluationError(f"{path.name}: missing columns {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "seed" not in df.columns:
        df["seed"] = NO_SEED
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(NO_SEED).astype(int)
    df["horizon"] = df["horizon"].astype(int)
    bad_target = set(df["target"]) - set(TARGETS)
    bad_horizon = set(df["horizon"]) - set(HORIZONS)
    if bad_target or bad_horizon:
        raise EvaluationError(f"{path.name}: unknown target {sorted(bad_target)} or horizon "
                              f"{sorted(bad_horizon)}")
    bad_split = set(df["split"]) - set(SPLITS) if "split" in df.columns else set()
    if bad_split:
        raise EvaluationError(f"{path.name}: unknown split {sorted(bad_split)}")
    if not np.isfinite(df["y_pred"].to_numpy(dtype=float)).all():
        raise EvaluationError(f"{path.name}: non-finite y_pred")
    ts = pd.DatetimeIndex(df["timestamp"])
    if ((ts.minute != 0) | (ts.second != 0)).any():
        raise EvaluationError(f"{path.name}: timestamps off the hourly grid")
    dup = df.duplicated(subset=KEY + ["timestamp"])
    if dup.any():
        raise EvaluationError(f"{path.name}: {int(dup.sum())} duplicated (model, seed, target, "
                              "horizon, timestamp) rows")
    return df


def check_alignment(preds: pd.DataFrame, truth: pd.DataFrame, split: str) -> pd.DataFrame:
    """Largest |y_true in the file - truth| and the split labels, per model/seed/target/horizon.

    A model that wrote its truth one hour late (the v6 BiLSTM bug), swapped the targets, or
    mislabelled the split shows here. Returns one row per key with `ok`."""
    rows = []
    for key, g in preds.groupby(KEY, sort=True):
        target = key[2]
        ref = truth[target].reindex(g["timestamp"]).to_numpy()
        own = g["y_true"].to_numpy(dtype=float)
        both = np.isfinite(ref) & np.isfinite(own)
        diff = float(np.max(np.abs(own[both] - ref[both]))) if both.any() else 0.0
        in_split = truth["split"].reindex(g["timestamp"]).to_numpy() == split
        rows.append({**dict(zip(KEY, key, strict=True)), "rows": len(g),
                     "max_abs_diff": diff, "rows_outside_split": int((~in_split).sum()),
                     "ok": diff <= ALIGN_TOL[target] and bool(in_split.all())})
    return pd.DataFrame(rows, columns=[*KEY, "rows", "max_abs_diff", "rows_outside_split", "ok"])


def attach_truth(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Replace y_true with the authoritative truth (columns 'indirect', 'direct', UTC index);
    rows without truth are dropped for every model alike."""
    out = preds.copy()
    values = np.full(len(out), np.nan)
    for target in TARGETS:
        rows = out["target"].to_numpy() == target
        values[rows] = truth[target].reindex(out.loc[rows, "timestamp"]).to_numpy()
    out["y_true_file"] = out["y_true"]
    out["y_true"] = values
    return out.dropna(subset=["y_true"])


def check_coverage(preds: pd.DataFrame, truth: pd.DataFrame, split: str) -> pd.DataFrame:
    """Daytime hours of the split with truth that each model/seed/target/horizon misses."""
    rows = []
    for key, g in preds.groupby(KEY, sort=True):
        target = key[2]
        need = truth.index[(truth["split"] == split) & truth["daytime"]
                           & truth[target].notna()]
        have = pd.DatetimeIndex(g["timestamp"])
        rows.append({**dict(zip(KEY, key, strict=True)), "required": len(need),
                     "missing": len(need.difference(have))})
    return pd.DataFrame(rows, columns=[*KEY, "required", "missing"])


def check_truth_consistency(preds: pd.DataFrame, atol: float = 1e-6) -> None:
    """A1.1 / T-04: one truth per (target, timestamp) across models, seeds and horizons."""
    agg = preds.groupby(["target", "timestamp"])["y_true"].agg(["max", "min"])
    spread = agg["max"] - agg["min"]
    bad = spread[spread > atol]
    if not bad.empty:
        raise ValueError(f"y_true differs between models/horizons at {len(bad)} timestamps, "
                         f"max spread {bad.max():.6g}")


def add_daytime(preds: pd.DataFrame, site: Site, truth: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """The F-02 mask from solar geometry. With a truth table, it must equal its `daytime`."""
    ts = pd.DatetimeIndex(preds["timestamp"].unique()).sort_values()
    mask = pd.Series(daytime_mask(ts, site), index=ts)
    if truth is not None and not (truth["daytime"].reindex(ts).to_numpy() == mask.to_numpy()).all():
        raise EvaluationError("the daytime mask differs from the feature table's daytime (F-02)")
    out = preds.copy()
    out["daytime"] = mask.reindex(out["timestamp"]).to_numpy()
    return out


def add_persistence(preds: pd.DataFrame) -> pd.DataFrame:
    """Fixture mode only: persistence built from the prediction file's own truth."""
    frames = [preds]
    for target, g in preds.groupby("target"):
        truth = g.groupby("timestamp")["y_true"].first().sort_index().asfreq("h")
        for h in sorted(g["horizon"].unique()):
            pred = persistence(truth, int(h))
            frames.append(to_prediction_frame(pred, truth.dropna(), target, int(h), PERSISTENCE,
                                              seed=NO_SEED))
    out = pd.concat(frames, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    return out


# ── helpers ────────────────────────────────────────────────────────────────


def _denominator(target: str, mean_k: float) -> float:
    return AC_CAPACITY_KW if target == "direct" else mean_k


def to_kw(target: str, y_pred, theo_kw) -> np.ndarray:
    """kW forecast: the direct forecast as is; the indirect one as clip(k * theo, 0, cap)."""
    y_pred = np.asarray(y_pred, dtype=float)
    if target == "direct":
        return y_pred
    return np.clip(y_pred * np.asarray(theo_kw, dtype=float), 0.0, AC_CAPACITY_KW)


def cliper_references(truth: pd.DataFrame) -> dict[tuple[str, int], float]:
    """RMSE_cp per target and horizon from the train daytime series (L-13)."""
    train_day = (truth["split"] == "train") & truth["daytime"]
    return {(target, h): cliper_rmse(truth[target], train_day, 24 * h)
            for target in TARGETS for h in HORIZONS}


def _kw_metrics(g: pd.DataFrame, target: str, truth: pd.DataFrame) -> dict[str, float]:
    ts = pd.DatetimeIndex(g["timestamp"])
    actual = truth["direct"].reindex(ts).to_numpy()
    pred = to_kw(target, g["y_pred"].to_numpy(), truth["theo_kw"].reindex(ts).to_numpy())
    err = rmse(actual, pred)
    return {"nRMSE_kW_pct": err / AC_CAPACITY_KW * 100.0,
            "nRMSE_mean_pct": err / float(np.mean(actual)) * 100.0}


# ── tables ─────────────────────────────────────────────────────────────────


def benchmark_table(preds: pd.DataFrame, mean_k: float, ci: bool = True,
                    truth: pd.DataFrame | None = None,
                    cliper: dict[tuple[str, int], float] | None = None) -> pd.DataFrame:
    day = preds[preds["daytime"]]
    pers = (day[day["model"] == PERSISTENCE].set_index(["target", "horizon", "timestamp"])
            .sort_index())
    pers_keys = set(pers.index.droplevel(2))
    rows = []
    for (model, seed, target, h), g in day.groupby(KEY, sort=True):
        g = g.sort_values("timestamp")
        denom = _denominator(target, mean_k)
        row = {"model": model, "label": MODEL_LABELS.get(model, model), "seed": seed,
               "target": target, "horizon": h,
               **point_metrics(g["y_true"], g["y_pred"], denom)}
        if ci:
            row["nRMSE_CI_lo"], row["nRMSE_CI_hi"] = bootstrap_nrmse_ci(
                g["y_true"], g["y_pred"], denom)
        if truth is not None:
            row.update(_kw_metrics(g, target, truth))
        row["skill_score"] = np.nan
        if model != PERSISTENCE and (target, h) in pers_keys:
            p = pers.loc[(target, h)]
            common = pd.DatetimeIndex(g["timestamp"]).intersection(p.index)
            if len(common) > 0:
                gm = g.set_index("timestamp").loc[common]
                row["skill_score"] = 1.0 - rmse(gm["y_true"], gm["y_pred"]) / rmse(
                    p.loc[common, "y_true"], p.loc[common, "y_pred"])
                row["n_skill"] = len(common)
        if cliper is not None:
            row["skill_cliper"] = 1.0 - row["RMSE"] / cliper[(target, h)]
        rows.append(row)
    return pd.DataFrame(rows)


def view_masks(truth: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks over the truth index for the secondary views (D-21, D-22)."""
    src = truth["hour_source"]
    return {"all": pd.Series(True, index=truth.index),
            "normal_operation": ~truth["op_limited_flag"].astype(bool),
            "source_minute": src == "minute", "source_polled": src == "polled",
            "source_mixed": src == "mixed", "no_polled": src == "minute"}


def views_table(preds: pd.DataFrame, mean_k: float, truth: pd.DataFrame) -> pd.DataFrame:
    """Main metric on all daytime hours, and the same on the secondary views."""
    day = preds[preds["daytime"]]
    masks = view_masks(truth)
    rows = []
    for (model, seed, target, h), g in day.groupby(KEY, sort=True):
        for view in VIEWS:
            keep = masks[view].reindex(g["timestamp"]).to_numpy()
            sub = g[keep]
            if sub.empty:
                continue
            met = point_metrics(sub["y_true"], sub["y_pred"], _denominator(target, mean_k))
            rows.append({"model": model, "seed": seed, "target": target, "horizon": h,
                         "view": view, "n": met["n"], "nRMSE_pct": met["nRMSE_pct"],
                         **_kw_metrics(sub, target, truth)})
    return pd.DataFrame(rows)


def _one_seed_per_model(g: pd.DataFrame) -> pd.DataFrame:
    """DM and common-mask use one run per model: its lowest seed (42 = development seed)."""
    first = g.groupby("model")["seed"].transform("min")
    return g[g["seed"] == first]


def dm_table(preds: pd.DataFrame, reference: str,
             alpha: float = SIGNIFICANCE_LEVEL) -> pd.DataFrame:
    day = _one_seed_per_model(preds[preds["daytime"]])
    rows = []
    for (target, h), g in day.groupby(["target", "horizon"], sort=True):
        ref = g[g["model"] == reference].set_index("timestamp")
        if ref.empty:
            continue
        family = []
        for model, m in g[g["model"] != reference].groupby("model", sort=True):
            m = m.set_index("timestamp")
            common = ref.index.intersection(m.index).sort_values()
            y = ref.loc[common, "y_true"].to_numpy()
            pa, pb = m.loc[common, "y_pred"].to_numpy(), ref.loc[common, "y_pred"].to_numpy()
            res = diebold_mariano(y, pa, pb)
            d = (y - pa) ** 2 - (y - pb) ** 2
            ci_lo, ci_hi = bootstrap_mean_ci(d)
            family.append({"target": target, "horizon": h, "model_a": model,
                           "model_b": reference, "n_pairs": res.n, "mean_diff": res.mean_diff,
                           "mean_diff_ci_lo": ci_lo, "mean_diff_ci_hi": ci_hi,
                           "dm_stat_raw": res.dm_stat_raw, "p_value_raw": res.p_value_raw,
                           "dm_stat_hln": res.dm_stat, "p_value_hln": res.p_value,
                           **power_analysis(d)})
        if family:
            alpha_adj, sig = bonferroni([r["p_value_hln"] for r in family], alpha)
            for r, s in zip(family, sig, strict=True):
                r.update({"alpha_bonferroni": alpha_adj, "significant": bool(s),
                          "nw_lags": 24})
            rows += family
    return pd.DataFrame(rows)


def route_comparison(preds: pd.DataFrame, theo_kw: pd.Series) -> pd.DataFrame:
    """Indirect (k * theo) vs direct, both scored against actual AC on the identical rows.

    The original scored each route on its own rows, filtered by its own truth (E2, E3, C-34).
    Here the row set is the intersection of both routes, theo and the AC truth."""
    day = preds[preds["daytime"]]
    truth_kw = day[day["target"] == "direct"].groupby("timestamp")["y_true"].first()
    rows = []
    for (model, seed, h), g in day.groupby(["model", "seed", "horizon"], sort=True):
        ind = g[g["target"] == "indirect"].set_index("timestamp")
        dirr = g[g["target"] == "direct"].set_index("timestamp")
        common = (ind.index.intersection(dirr.index).intersection(theo_kw.index)
                  .intersection(truth_kw.index).sort_values())
        if len(common) == 0:
            continue
        y = truth_kw.loc[common].to_numpy()
        ind_kw = to_kw("indirect", ind.loc[common, "y_pred"], theo_kw.loc[common])
        dir_kw = dirr.loc[common, "y_pred"].to_numpy()
        m_ind = point_metrics(y, ind_kw, AC_CAPACITY_KW)
        m_dir = point_metrics(y, dir_kw, AC_CAPACITY_KW)
        for route, met in (("indirect_to_power", m_ind), ("direct", m_dir)):
            rows.append({"model": model, "seed": seed, "horizon": h, "route": route,
                         **{f"{k}_kW" if k in ("MAE", "RMSE") else k: v
                            for k, v in met.items()}})
        rows[-1]["advantage_pp"] = rows[-1]["nRMSE_pct"] - rows[-2]["nRMSE_pct"]
    return pd.DataFrame(rows)


def seasonal_table(preds: pd.DataFrame, mean_k: float) -> pd.DataFrame:
    day = preds[preds["daytime"]].copy()
    local_month = pd.DatetimeIndex(day["timestamp"]).tz_convert(TIMEZONE_LOCAL).month
    day["season"] = [SEASON_OF_MONTH[m] for m in local_month]
    rows = []
    for (model, seed, target, h, season), g in day.groupby(KEY + ["season"], sort=True):
        met = point_metrics(g["y_true"], g["y_pred"], _denominator(target, mean_k))
        rows.append({"model": model, "seed": seed, "target": target, "horizon": h,
                     "season": season, "n": met["n"], "nRMSE_pct": met["nRMSE_pct"]})
    return pd.DataFrame(rows)


def common_mask_tables(preds: pd.DataFrame, mean_k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics on the timestamps shared by all models, and how many rows each model keeps
    (the original common_mask_summary and common_mask_sensitivity)."""
    day = _one_seed_per_model(preds[preds["daytime"]])
    summary, sensitivity = [], []
    for (target, h), g in day.groupby(["target", "horizon"], sort=True):
        sets = {model: set(m["timestamp"]) for model, m in g.groupby("model")}
        if len(sets) < 2:
            continue
        shared = set.intersection(*sets.values())
        for model, m in g.groupby("model", sort=True):
            kept = m[m["timestamp"].isin(shared)]
            met = point_metrics(kept["y_true"], kept["y_pred"], _denominator(target, mean_k))
            summary.append({"model": model, "target": target, "horizon": h, "n": met["n"],
                            "nRMSE_pct_shared": met["nRMSE_pct"], "R2_shared": met["R2"]})
            sensitivity.append({"model": model, "target": target, "horizon": h,
                                "n_model": len(m), "n_shared": len(kept),
                                "pct_retained": len(kept) / len(m) * 100.0})
    return pd.DataFrame(summary), pd.DataFrame(sensitivity)


def common_mask_table(preds: pd.DataFrame, mean_k: float) -> pd.DataFrame:
    return common_mask_tables(preds, mean_k)[0]


def seed_variability_table(bench: pd.DataFrame) -> pd.DataFrame:
    """Spread across seeds of the evaluated nRMSE, per model/target/horizon (E5, C-10)."""
    models = bench[~bench["model"].isin(REFERENCES) & (bench["seed"] != NO_SEED)]
    if models.empty:
        return pd.DataFrame(columns=["model", "target", "horizon", "n_seeds", "nRMSE_mean",
                                     "nRMSE_std", "nRMSE_min", "nRMSE_max", "nRMSE_range"])
    out = (models.groupby(["model", "target", "horizon"])["nRMSE_pct"]
           .agg(n_seeds="count", nRMSE_mean="mean", nRMSE_std="std", nRMSE_min="min",
                nRMSE_max="max").reset_index())
    out["nRMSE_range"] = out["nRMSE_max"] - out["nRMSE_min"]
    return out


# ── run ────────────────────────────────────────────────────────────────────


def _log_test_access(files: list[Path], out_dir: Path, command: str) -> None:
    TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not TEST_ACCESS_LOG.exists()
    with TEST_ACCESS_LOG.open("a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("utc_time,git_commit,command,out_dir,prediction_files\n")
        rel = ";".join(p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT)
                       else p.as_posix() for p in files)
        f.write(f"{datetime.now(UTC).isoformat(timespec='seconds')},{git_commit()},"
                f"\"{command}\",{out_dir.relative_to(REPO_ROOT).as_posix()},\"{rel}\"\n")


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in REQUIRED_COLUMNS}).assign(
        seed=pd.Series(dtype=int), timestamp=pd.Series(dtype="datetime64[ns, UTC]"))


def run(
    files: Iterable[Path],
    out_dir: Path,
    mean_k: float,
    site: Site | None = None,
    truth: pd.DataFrame | None = None,
    split: str = "val",
    allow_test: bool = False,
    strict: bool = True,
    reference: str | None = None,
    with_persistence: bool = True,
    ci: bool = True,
    log: RunLog | None = None,
    command: str = "",
) -> dict[str, pd.DataFrame]:
    """Evaluate prediction files. With `truth` (the feature table) this is the stage-8B
    evaluation on `split`; without it, the fixture mode of stage 8A (file truth, old n)."""
    files = [Path(f) for f in files]
    site = site or load_site()
    note = log.note if log else (lambda _: None)
    preds = (pd.concat([load_predictions(f) for f in files], ignore_index=True) if files
             else _empty_predictions())
    checks: dict[str, object] = {}

    if truth is None:  # fixture mode (T-15)
        check_truth_consistency(preds)
        if with_persistence:
            preds = add_persistence(preds)
        preds = add_daytime(preds, site)
    else:
        if split not in ("val", "test"):
            raise ValueError("split must be 'val' or 'test'")
        if split == "test" and not allow_test:
            raise PermissionError("the test split is evaluated only with allow_test=True "
                                  "(final evaluation, after g1-frozen)")
        if split == "test":
            _log_test_access(files, out_dir, command)
            if log:
                log.warn("TEST SPLIT EVALUATED; recorded in outputs/07_evaluation/"
                         "test_access_log.csv")
        if "split" in preds.columns:
            preds = preds[preds["split"].isna() | (preds["split"] == split)]
        align = check_alignment(preds, truth, split)
        checks["alignment"] = align.to_dict(orient="records")
        if strict and not align["ok"].all():
            raise EvaluationError(f"misaligned prediction files:\n{align[~align['ok']]}")
        preds = attach_truth(preds, truth)
        coverage = check_coverage(preds, truth, split)
        checks["coverage"] = coverage.to_dict(orient="records")
        if strict and (coverage["missing"] > 0).any():
            raise EvaluationError(f"incomplete predictions:\n{coverage[coverage['missing'] > 0]}")
        refs = attach_truth(reference_predictions(truth, split, mean_k), truth)
        preds = refs if preds.empty else pd.concat([preds, refs], ignore_index=True)
        preds["seed"] = preds["seed"].astype(int)
        preds["horizon"] = preds["horizon"].astype(int)
        check_truth_consistency(preds)
        preds = add_daytime(preds, site, truth)
        note(f"{split}: {preds['model'].nunique()} models "
             f"({', '.join(sorted(preds['model'].unique()))}); daytime rows per model/target/"
             f"horizon: {int(preds[preds['daytime']].groupby(KEY).size().iloc[0])}")
    models = set(preds["model"])
    reference = reference or ("xgb_sota" if "xgb_sota" in models else PERSISTENCE)
    cliper = cliper_references(truth) if truth is not None else None

    bench = benchmark_table(preds, mean_k, ci=ci, truth=truth, cliper=cliper)
    mask_summary, mask_sensitivity = common_mask_tables(preds, mean_k)
    tables = {
        "benchmark_summary": bench,
        "dm_tests": dm_table(preds, reference),
        "seasonal_metrics": seasonal_table(preds, mean_k),
        "common_mask_summary": mask_summary,
    }
    if truth is not None:
        tables.update({
            "benchmark_views": views_table(preds, mean_k, truth),
            "common_mask_sensitivity": mask_sensitivity,
            "route_comparison": route_comparison(preds, truth["theo_kw"]),
            "seed_variability_summary": seed_variability_table(bench),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
        written.append(path)
    manifest = {
        "inputs": [f.relative_to(REPO_ROOT).as_posix() if f.is_relative_to(REPO_ROOT)
                   else f.as_posix() for f in files],
        "inputs_sha256": {f.name: sha256(f) for f in files},
        "git_commit": git_commit(),
        "mode": "feature table" if truth is not None else "fixture (prediction-file truth)",
        "split": split if truth is not None else "test (fixture)",
        "mean_daytime_k": mean_k,
        "dm_reference": reference,
        "truth": FEATURES_FILE.relative_to(REPO_ROOT).as_posix() if truth is not None
        else "prediction files",
        "truth_sha256": sha256(FEATURES_FILE) if truth is not None else None,
        "cliper_rmse": {f"{t}_D{h}": v for (t, h), v in cliper.items()} if cliper else None,
        "site_private": site.is_private,
        "strict": strict,
        "checks": checks,
        "rows": {name: len(t) for name, t in tables.items()},
        "run_log": log.relative_path if log else None,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8",
                    newline="\n")
    if log:
        log.outputs(written + [path])
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate forecasts (PROJECT_MASTER.md §6.9).")
    parser.add_argument("--fixture", action="store_true",
                        help="evaluate the surviving old LSTM predictions (test T-15)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true",
                        help="required for --split test (final evaluation only)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-ci", action="store_true", help="skip bootstrap CIs (faster)")
    args = parser.parse_args(argv)
    command = "python -m solarpv.evaluation.evaluate " + " ".join(argv or sys.argv[1:])

    if args.fixture:
        out = args.out or OUTPUTS_DIR / "07_evaluation_fixture"
        tables = run([LSTM_FIXTURE], out, MEAN_DAYTIME_K_LEGACY, reference=PERSISTENCE,
                     ci=not args.no_ci)
    else:
        out = args.out or EVAL_DIR / args.split
        files = sorted(OUTPUTS_DIR.glob("06_*/predictions_*.csv"))
        with RunLog("07_evaluation", command.strip(), out_dir=EVAL_DIR) as log:
            mean_k = mean_daytime_k()
            log.note(f"split {args.split}; MEAN_DAYTIME_K {mean_k}; {len(files)} prediction "
                     "file(s); references computed from the truth table")
            log.inputs([FEATURES_FILE, *files])
            with log.step("load truth"):
                truth = load_truth()
            with log.step("evaluate"):
                tables = run(files, out, mean_k, truth=truth, split=args.split,
                             allow_test=args.allow_test, ci=not args.no_ci, log=log,
                             command=command.strip())
            for name, t in tables.items():
                log.value(f"rows_{name}", len(t))
    print(tables["benchmark_summary"].to_string(index=False))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

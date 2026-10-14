"""Authoritative evaluation (stage 8, PROJECT_MASTER.md §6.9-§6.10).

    python -m solarpv.evaluation.evaluate --fixture   # old LSTM predictions, old denominator
    python -m solarpv.evaluation.evaluate             # outputs/06_*/predictions_*.csv (stage 9+)

Rules:
  * predictions follow the file contract (all test hours; one row per model/seed/target/horizon/ts);
  * y_true comes from the feature file (truth); the fixture mode keeps the file's own y_true;
  * daytime = sun elevation at the middle of the hour > 5 deg (F-02), applied here only;
  * route comparison uses actual AC as the truth for both routes, on shared timestamps (C-16);
  * DM: raw (old formula) and HLN-corrected with Bonferroni per target x horizon (D-14).
"""

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import (
    AC_CAPACITY_KW,
    MEAN_DAYTIME_K_LEGACY,
    SIGNIFICANCE_LEVEL,
    TIMEZONE_LOCAL,
    mean_daytime_k,
)
from solarpv.config.paths import LSTM_FIXTURE, OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import git_commit
from solarpv.config.site import Site, load_site
from solarpv.evaluation.daytime import daytime_mask
from solarpv.evaluation.metrics import bootstrap_nrmse_ci, point_metrics, rmse
from solarpv.evaluation.stats_tests import bonferroni, diebold_mariano, power_analysis
from solarpv.models.baselines import persistence, to_prediction_frame

REQUIRED_COLUMNS = ("timestamp", "y_true", "y_pred", "target", "horizon", "model")
KEY = ["model", "seed", "target", "horizon"]
PERSISTENCE = "persistence"
NO_SEED = -1
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


# ── loading ────────────────────────────────────────────────────────────────


def load_predictions(path: Path) -> pd.DataFrame:
    """Read one prediction file and validate it against the contract."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "seed" not in df.columns:
        df["seed"] = NO_SEED
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(NO_SEED).astype(int)
    df["horizon"] = df["horizon"].astype(int)
    dup = df.duplicated(subset=KEY + ["timestamp"])
    if dup.any():
        raise ValueError(f"{path.name}: {int(dup.sum())} duplicated (model, seed, target, "
                         "horizon, timestamp) rows")
    return df


def attach_truth(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Replace y_true with the authoritative truth (columns 'indirect', 'direct', UTC index)."""
    out = preds.copy()
    values = np.full(len(out), np.nan)
    for target in ("indirect", "direct"):
        rows = out["target"].to_numpy() == target
        values[rows] = truth[target].reindex(out.loc[rows, "timestamp"]).to_numpy()
    out["y_true_file"] = out["y_true"]
    out["y_true"] = values
    return out.dropna(subset=["y_true"])


def check_truth_consistency(preds: pd.DataFrame, atol: float = 1e-6) -> None:
    """A1.1 / T-04: one truth per (target, timestamp) across models, seeds and horizons."""
    agg = preds.groupby(["target", "timestamp"])["y_true"].agg(["max", "min"])
    spread = agg["max"] - agg["min"]
    bad = spread[spread > atol]
    if not bad.empty:
        raise ValueError(f"y_true differs between models/horizons at {len(bad)} timestamps, "
                         f"max spread {bad.max():.6g}")


def add_daytime(preds: pd.DataFrame, site: Site) -> pd.DataFrame:
    ts = pd.DatetimeIndex(preds["timestamp"].unique()).sort_values()
    mask = pd.Series(daytime_mask(ts, site), index=ts)
    out = preds.copy()
    out["daytime"] = mask.reindex(out["timestamp"]).to_numpy()
    return out


def add_persistence(preds: pd.DataFrame) -> pd.DataFrame:
    """Append horizon-matched persistence forecasts built from the truth itself."""
    frames = [preds]
    for target, g in preds.groupby("target"):
        truth = g.groupby("timestamp")["y_true"].first().sort_index()
        truth = truth.asfreq("h")
        for h in sorted(g["horizon"].unique()):
            pred = persistence(truth, int(h))
            frames.append(to_prediction_frame(pred, truth.dropna(), target, int(h), PERSISTENCE,
                                              seed=NO_SEED))
    out = pd.concat(frames, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    return out


# ── tables ─────────────────────────────────────────────────────────────────


def _denominator(target: str, mean_k: float) -> float:
    return AC_CAPACITY_KW if target == "direct" else mean_k


def benchmark_table(preds: pd.DataFrame, mean_k: float, ci: bool = True) -> pd.DataFrame:
    day = preds[preds["daytime"]]
    pers = day[day["model"] == PERSISTENCE].set_index(["target", "horizon", "timestamp"])
    rows = []
    for (model, seed, target, h), g in day.groupby(KEY, sort=True):
        g = g.sort_values("timestamp")
        denom = _denominator(target, mean_k)
        row = {"model": model, "seed": seed, "target": target, "horizon": h,
               **point_metrics(g["y_true"], g["y_pred"], denom)}
        if ci:
            row["nRMSE_CI_lo"], row["nRMSE_CI_hi"] = bootstrap_nrmse_ci(
                g["y_true"], g["y_pred"], denom)
        row["skill_score"] = np.nan
        if model != PERSISTENCE and (target, h) in pers.index.droplevel(2):
            p = pers.loc[(target, h)]
            common = pd.DatetimeIndex(g["timestamp"]).intersection(p.index)
            if len(common) > 0:
                gm = g.set_index("timestamp").loc[common]
                row["skill_score"] = 1.0 - rmse(gm["y_true"], gm["y_pred"]) / rmse(
                    p.loc[common, "y_true"], p.loc[common, "y_pred"])
                row["n_skill"] = len(common)
        rows.append(row)
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
            family.append({"target": target, "horizon": h, "model_a": model,
                           "model_b": reference, "n_pairs": res.n, "mean_diff": res.mean_diff,
                           "dm_stat_raw": res.dm_stat_raw, "p_value_raw": res.p_value_raw,
                           "dm_stat_hln": res.dm_stat, "p_value_hln": res.p_value,
                           **power_analysis((y - pa) ** 2 - (y - pb) ** 2)})
        if family:
            alpha_adj, sig = bonferroni([r["p_value_hln"] for r in family], alpha)
            for r, s in zip(family, sig, strict=True):
                r.update({"alpha_bonferroni": alpha_adj, "significant": bool(s),
                          "nw_lags": 24})
            rows += family
    return pd.DataFrame(rows)


def route_comparison(preds: pd.DataFrame, theo_kw: pd.Series) -> pd.DataFrame:
    """Indirect (k * theo) vs direct, both scored against actual AC on the same rows (C-16)."""
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
        ind_kw = np.clip(ind.loc[common, "y_pred"].to_numpy() * theo_kw.loc[common].to_numpy(),
                         0.0, AC_CAPACITY_KW)
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


def common_mask_table(preds: pd.DataFrame, mean_k: float) -> pd.DataFrame:
    day = _one_seed_per_model(preds[preds["daytime"] & (preds["model"] != PERSISTENCE)])
    rows = []
    for (target, h), g in day.groupby(["target", "horizon"], sort=True):
        sets = [set(m["timestamp"]) for _, m in g.groupby("model")]
        if len(sets) < 2:
            continue
        shared = set.intersection(*sets)
        for model, m in g.groupby("model", sort=True):
            m = m[m["timestamp"].isin(shared)]
            met = point_metrics(m["y_true"], m["y_pred"], _denominator(target, mean_k))
            rows.append({"model": model, "target": target, "horizon": h, "n": met["n"],
                         "nRMSE_pct_shared": met["nRMSE_pct"], "R2_shared": met["R2"]})
    return pd.DataFrame(rows)


# ── run ────────────────────────────────────────────────────────────────────


def run(
    files: Iterable[Path],
    out_dir: Path,
    mean_k: float,
    site: Site | None = None,
    truth: pd.DataFrame | None = None,
    theo_kw: pd.Series | None = None,
    reference: str = "xgb_sota",
    with_persistence: bool = True,
    ci: bool = True,
) -> dict[str, pd.DataFrame]:
    files = [Path(f) for f in files]
    site = site or load_site()
    preds = pd.concat([load_predictions(f) for f in files], ignore_index=True)
    if truth is not None:
        preds = attach_truth(preds, truth)
    check_truth_consistency(preds)
    if with_persistence:
        preds = add_persistence(preds)
    preds = add_daytime(preds, site)

    tables = {
        "benchmark_summary": benchmark_table(preds, mean_k, ci=ci),
        "dm_tests": dm_table(preds, reference),
        "seasonal_metrics": seasonal_table(preds, mean_k),
        "common_mask_summary": common_mask_table(preds, mean_k),
    }
    if theo_kw is not None:
        tables["route_comparison"] = route_comparison(preds, theo_kw)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6f")
    manifest = {
        "inputs": [f.relative_to(REPO_ROOT).as_posix() if f.is_relative_to(REPO_ROOT)
                   else f.as_posix() for f in files],
        "git_commit": git_commit(),
        "mean_daytime_k": mean_k,
        "dm_reference": reference,
        "truth": "feature file" if truth is not None else "prediction files (fixture mode)",
        "site_private": site.is_private,
        "rows": {name: len(t) for name, t in tables.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate forecasts (PROJECT_MASTER.md §6.9).")
    parser.add_argument("--fixture", action="store_true",
                        help="evaluate the surviving old LSTM predictions (test T-15)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-ci", action="store_true", help="skip bootstrap CIs (faster)")
    args = parser.parse_args(argv)

    if args.fixture:
        out = args.out or OUTPUTS_DIR / "07_evaluation_fixture"
        tables = run([LSTM_FIXTURE], out, MEAN_DAYTIME_K_LEGACY, reference=PERSISTENCE,
                     ci=not args.no_ci)
    else:
        files = sorted(OUTPUTS_DIR.glob("06_*/predictions_*.csv"))
        if not files:
            print("no predictions found under outputs/06_*/ (stage 9 produces them)")
            return 1
        mean_daytime_k()  # raises until stage 6 has written configs/derived_constants.yaml
        raise NotImplementedError("wire the stage-6 feature file (truth, clear-sky power) here")
    print(tables["benchmark_summary"].to_string(index=False))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

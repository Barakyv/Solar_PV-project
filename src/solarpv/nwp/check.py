"""Stage 6b, S4-S5: sanity checks of the GFS tables, and the value check of decision D-12
(docs/stage6b_plan.md).

    python -m solarpv.nwp.check

**Train and validation only.** Every statistic below is computed on rows whose split is train
or val. The code asserts that no test row enters (plan §1, never use the test split).

S4, sanity:
- **Coverage:** rows with GFS per horizon; fallback runs.
- **Horizon skill:** GFS kt_poa against the measured irradiance / POA_cs, on the daytime hours
  that every horizon covers, hourly and daily. Red flags of the original's problem: daily r
  above 0.95 at D+3, or skill rising from D+1 to D+3 by more than 0.05.
- **Phase:** the lag of the best hourly correlation of GFS POA and the measurement on the
  daytime hours of train. Must be 0.
- **Bias by season:** mean GFS kt_poa minus measured.

S5, value (D-12, criterion 2; the rule was fixed in the plan before the run):
- The fixed XGBoost of D-11 (decide_k.XGB_PARAMS; seeds 42/123/777) on each target and
  horizon, trained on train (no warm-up, normal operation, valid target; for k also
  theo > 50 kW), scored on validation daytime hours with truth: nRMSE in kW.
- Compared: the stage-6 contract of the horizon, against the same plus the GFS candidates.
- Admitted if, for both targets, the mean gain at D+1 exceeds max(0.05 pp, 2 x pooled seed
  std), and at D+2 and D+3 the result is not worse by more than that margin.

Writes `outputs/06b_gfs/sanity.json`, `horizon_skill.csv`, `value_check.csv`, `value_check.png`,
`configs/nwp_decision.yaml` and a run log.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from solarpv.config.cli import no_options
from solarpv.config.constants import AC_CAPACITY_KW, HORIZONS, RANDOM_SEEDS, TIMEZONE_LOCAL
from solarpv.config.paths import CONFIGS_DIR, PROCESSED_DIR
from solarpv.config.provenance import git_commit
from solarpv.config.runlog import RunLog
from solarpv.features.contract import features_for_horizon
from solarpv.features.decide_k import XGB_PARAMS
from solarpv.nwp.gfs import HOURLY_FILE, PREFIX, REPORT_DIR, STAGE_DIR

FEATURES = PROCESSED_DIR / "06_features"
DECISION_FILE = CONFIGS_DIR / "nwp_decision.yaml"
ALLOWED_SPLITS = ("train", "val")
MIN_GAIN_PP = 0.05
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


def load(h: int) -> pd.DataFrame:
    """The stage-6 table of horizon h joined with its GFS table, train and validation only."""
    base = pd.read_parquet(FEATURES / f"features_1H_D{h}.parquet")
    gfs = pd.read_parquet(STAGE_DIR / f"gfs_features_D{h}.parquet")
    table = base.join(gfs.drop(columns=["t0_utc"]), how="left")
    table = table[table["split"].isin(ALLOWED_SPLITS)]
    assert not (table["split"] == "test").any()  # the test split is never read further
    return table


def gfs_candidates(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith(PREFIX)] + ["irradiance_clear_index_prevday"]


# ── S4 ─────────────────────────────────────────────────────────────────────


def horizon_skill(tables: dict[int, pd.DataFrame], measured: pd.Series) -> pd.DataFrame:
    """kt_poa of each horizon against the measured clear-sky index on the common daytime hours."""
    common = None
    for t in tables.values():
        use = t.index[t["daytime"] & (t["sun_elevation"] > 10) & t[f"{PREFIX}kt_poa"].notna()
                      & (t["poa_global_clearsky"] > 50)]
        common = use if common is None else common.intersection(use)
    meas = (measured.reindex(common) / tables[1]["poa_global_clearsky"].reindex(common))
    common = common[meas.notna().to_numpy()]
    meas = meas.dropna()
    local = common.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    rows = []
    for h, t in tables.items():
        g = t.loc[common, f"{PREFIX}kt_poa"]
        daily = pd.DataFrame({"g": g.to_numpy(), "m": meas.to_numpy()}, index=local).groupby(
            level=0).mean()
        rows.append({"horizon": h, "hours": len(common), "days": len(daily),
                     "corr_hourly": float(g.corr(meas)),
                     "corr_daily": float(daily["g"].corr(daily["m"])),
                     "bias": float((g - meas).mean()), "mae": float((g - meas).abs().mean())})
    return pd.DataFrame(rows)


def lag_check(t1: pd.DataFrame, measured: pd.Series) -> dict[int, float]:
    """Correlation of GFS POA (D+1) with the measurement shifted by -3..3 h, train daytime."""
    rows = t1.index[(t1["split"] == "train") & t1["daytime"]]
    g = t1[f"{PREFIX}poa"]
    out = {}
    for lag in range(-3, 4):
        m = measured.shift(lag).reindex(rows)
        out[lag] = float(g.reindex(rows).corr(m))
    return out


def bias_by_season(t1: pd.DataFrame, measured: pd.Series) -> dict[str, float]:
    use = t1["daytime"] & (t1["sun_elevation"] > 10) & (t1["poa_global_clearsky"] > 50)
    diff = (t1.loc[use, f"{PREFIX}kt_poa"]
            - measured.reindex(t1.index[use]) / t1.loc[use, "poa_global_clearsky"]).dropna()
    season = [SEASONS[m] for m in diff.index.tz_convert(TIMEZONE_LOCAL).month]
    return diff.groupby(season).mean().round(4).to_dict()


# ── S5 ─────────────────────────────────────────────────────────────────────


def fit_score(table: pd.DataFrame, feats: list[str], target: str, seed: int) -> float:
    """nRMSE in kW on validation daytime hours of the fixed XGBoost (decide_k protocol)."""
    train = (table["split"] == "train") & ~table["warmup"] & ~table["op_limited_flag"]
    if target == "indirect":
        train &= table["k_valid"] & (table["theo_kw"] > 50)
        y = table["target_k"]
    else:
        train &= table["site_total_AC_kW"].notna()
        y = table["site_total_AC_kW"]
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
    model.fit(table.loc[train, feats], y[train])
    val = (table["split"] == "val") & table["daytime"] & table["site_total_AC_kW"].notna()
    pred = model.predict(table.loc[val, feats])
    if target == "indirect":
        pred_kw = np.clip(np.clip(pred, 0, None) * table.loc[val, "theo_kw"].to_numpy(), 0,
                          AC_CAPACITY_KW)
    else:
        pred_kw = pred
    err = pred_kw - table.loc[val, "site_total_AC_kW"].to_numpy()
    return float(np.sqrt(np.mean(err ** 2)) / AC_CAPACITY_KW * 100)


def value_check(tables: dict[int, pd.DataFrame], log=None) -> pd.DataFrame:
    rows = []
    for h, t in tables.items():
        base = features_for_horizon(h)
        for target in ("indirect", "direct"):
            for name, feats in (("contract", base), ("contract+gfs", base + gfs_candidates(t))):
                scores = [fit_score(t, feats, target, s) for s in RANDOM_SEEDS]
                rows.append({"horizon": h, "target": target, "features": name,
                             "n_features": len(feats), "nrmse_kw_mean": float(np.mean(scores)),
                             "nrmse_kw_std": float(np.std(scores, ddof=1)),
                             **{f"seed_{s}": v for s, v in zip(RANDOM_SEEDS, scores,
                                                               strict=True)}})
                if log:
                    log.note(f"D+{h} {target} {name}: {np.mean(scores):.3f} +/- "
                             f"{np.std(scores, ddof=1):.3f} % (3 seeds)")
    return pd.DataFrame(rows)


def decide(values: pd.DataFrame) -> dict:
    """The D-12 rule of the plan (S5)."""
    out, admitted = {}, True
    for target in ("indirect", "direct"):
        for h in HORIZONS:
            v = values[(values["target"] == target) & (values["horizon"] == h)].set_index(
                "features")
            gain = v.loc["contract", "nrmse_kw_mean"] - v.loc["contract+gfs", "nrmse_kw_mean"]
            noise = 2 * float(np.sqrt(np.mean(np.square(v["nrmse_kw_std"]))))
            margin = max(MIN_GAIN_PP, noise)
            ok = gain > margin if h == 1 else gain > -margin
            admitted &= ok
            out[f"{target}_D{h}"] = {"gain_pp": round(float(gain), 4),
                                     "margin_pp": round(margin, 4), "passes": bool(ok)}
    return {"admitted": bool(admitted), "by_target_horizon": out}


def run() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with RunLog("06b_gfs", "python -m solarpv.nwp.check", out_dir=REPORT_DIR) as log:
        files = [FEATURES / f"features_1H_D{h}.parquet" for h in HORIZONS] + \
                [STAGE_DIR / f"gfs_features_D{h}.parquet" for h in HORIZONS] + [HOURLY_FILE]
        log.inputs(files)
        tables = {h: load(h) for h in HORIZONS}
        measured = pd.read_parquet(HOURLY_FILE, columns=["irradiance"])["irradiance"]
        with log.step("S4 sanity (train and validation)"):
            coverage = {f"D{h}": float(t[f"{PREFIX}ghi"].notna().mean()) for h, t in
                        tables.items()}
            skill = horizon_skill(tables, measured)
            lags = lag_check(tables[1], measured)
            best_lag = max(lags, key=lags.get)
            sk = skill.set_index("horizon")
            sanity = {
                "coverage_train_val": coverage,
                "fallback_rows_D1": int(tables[1]["gfs_fallback"].fillna(False).sum()),
                "horizon_skill": skill.round(4).to_dict(orient="records"),
                "red_flag_too_good_at_D3": bool(sk.loc[3, "corr_daily"] > 0.95),
                "red_flag_skill_rises_with_horizon": bool(
                    sk.loc[3, "corr_daily"] > sk.loc[1, "corr_daily"] + 0.05),
                "lag_correlations_train": {str(k): round(v, 4) for k, v in lags.items()},
                "best_lag_h": int(best_lag),
                "bias_kt_poa_by_season": bias_by_season(tables[1], measured),
            }
            for key in ("coverage_train_val", "red_flag_too_good_at_D3",
                        "red_flag_skill_rises_with_horizon", "best_lag_h"):
                log.value(key, sanity[key])
            if best_lag != 0:
                log.warn(f"the best lag is {best_lag} h, not 0: check the time alignment")
        with log.step("S5 value check (fixed XGBoost, 3 seeds, validation)"):
            values = value_check(tables, log)
            decision = decide(values)
            log.value("admitted", decision["admitted"])
        skill.to_csv(REPORT_DIR / "horizon_skill.csv", index=False, lineterminator="\n")
        values.to_csv(REPORT_DIR / "value_check.csv", index=False, lineterminator="\n")
        (REPORT_DIR / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8",
                                                newline="\n")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for target, ax in zip(("indirect", "direct"), axes, strict=True):
            v = values[values["target"] == target]
            for name, style in (("contract", "o--"), ("contract+gfs", "s-")):
                w = v[v["features"] == name]
                ax.errorbar(w["horizon"], w["nrmse_kw_mean"], yerr=w["nrmse_kw_std"],
                            fmt=style, capsize=3, label=name)
            ax.set_xticks(list(HORIZONS))
            ax.set_xticklabels([f"D+{h}" for h in HORIZONS])
            ax.set_ylabel("validation nRMSE [% of 3,449 kW]")
            ax.set_title(f"{target} target: fixed XGBoost, 3 seeds", fontsize=9)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "value_check.png", dpi=110)
        plt.close(fig)
        record = {"decision": "D-12", "admitted": decision["admitted"],
                  "rule": "fixed XGBoost of D-11, 3 seeds, validation nRMSE in kW; gain at D+1 "
                          "> max(0.05 pp, 2 x pooled seed std) for both targets, and not worse "
                          "by more than that margin at D+2 and D+3",
                  "results": decision["by_target_horizon"],
                  "candidates": gfs_candidates(tables[1]), "git_commit": git_commit(),
                  "run_log": log.relative_path}
        DECISION_FILE.write_text("# D-12 value check; written by python -m solarpv.nwp.check\n"
                                 + yaml.safe_dump(record, sort_keys=False), encoding="utf-8",
                                 newline="\n")
        log.outputs([REPORT_DIR / f for f in ("horizon_skill.csv", "value_check.csv",
                                              "sanity.json", "value_check.png")]
                    + [DECISION_FILE])
    return {"sanity": sanity, "decision": decision}


def main(argv: list[str] | None = None) -> int:
    no_options("python -m solarpv.nwp.check", __doc__, argv)  # X62: --help runs nothing
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 6: decision D-11 (the denominator of k) and the low-sun analysis L-04.

    python -m solarpv.features.build        # first: both k variants
    python -m solarpv.features.decide_k     # then: criteria, decision, configs/target.yaml
    python -m solarpv.features.build        # again: target_k, theo_kw, MEAN_DAYTIME_K

D-11 compares GHI_cs and POA_cs (panel plane) as the clear-sky reference of k on three criteria
fixed in PROJECT_MASTER.md before any result:
  1. the share of valid daytime train hours with k > 1 (lower is better: k should be a fraction
     of clear-sky output);
  2. the flatness of clear-day k across the train months: spread (max - min) of the monthly
     median k on clear days, sun above 30 deg (lower is better: a good reference removes the
     season);
  3. nRMSE in kW on validation of one fixed, untuned XGBoost per denominator (D+1, same
     features, same seed), its k forecast converted with that denominator's theo (lower is
     better). This is the only use of validation in stage 6.
Decision rule, written before criteria 2 and 3 were computed: the denominator that wins the
majority of the three criteria.

L-04 (literature, C2 and stage 3b Q22): k by sun-elevation band on train, and the validation
nRMSE in kW when hours with small theo are left out of the training loss (theo > 50, 100,
150 kW). The evaluation mask (F-02) does not change.
Adoption rule. The first rule (a gain of >= 0.05 pp on one fit) turned out to be below the
fit-to-fit noise: on the first run 100 kW was 0.18 pp worse and 150 kW 0.05 pp better than
50 kW, a non-monotonic pattern of noise (run journal R6-03). The rule now used: three seeds
per threshold (RANDOM_SEEDS); a higher threshold is adopted only if its mean nRMSE is lower
than at 50 kW by more than max(0.05 pp, twice the pooled seed standard deviation). The same
three seeds are reported for D-11 criterion 3.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
import yaml  # noqa: E402

from solarpv.config.cli import no_options  # noqa: E402
from solarpv.config.constants import (  # noqa: E402
    AC_CAPACITY_KW,
    DEV_SEED,
    RANDOM_SEEDS,
    TIMEZONE_LOCAL,
    XGB_THREADS,
)
from solarpv.config.paths import PROCESSED_DIR, output_dir  # noqa: E402
from solarpv.config.provenance import git_commit  # noqa: E402
from solarpv.config.runlog import RunLog  # noqa: E402
from solarpv.features.build import TARGET_FILE  # noqa: E402
from solarpv.features.contract import features_for_horizon  # noqa: E402
from solarpv.features.target import DENOMINATORS, clear_days, share_above, to_kw  # noqa: E402

FEATURES_D1 = PROCESSED_DIR / "06_features" / "features_1H_D1.parquet"
HOURLY_FILE = PROCESSED_DIR / "05_clean" / "scada_1H_clean.parquet"
XGB_PARAMS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 5, "tree_method": "hist",
              "random_state": DEV_SEED, "n_jobs": XGB_THREADS}
THEO_THRESHOLDS = (50.0, 100.0, 150.0)
MIN_GAIN_PP = 0.05
ELEVATION_BANDS = [0, 5, 10, 15, 30, 90]


def nrmse_kw(pred_kw: pd.Series, true_kw: pd.Series) -> float:
    return float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)) / AC_CAPACITY_KW * 100)


def fit_predict(table: pd.DataFrame, den: str, min_theo: float = 50.0,
                seed: int = DEV_SEED) -> pd.Series:
    """Fixed XGBoost on train (no warm-up, valid k, normal operation, theo > min_theo);
    returns the kW forecast for every validation hour."""
    feats = features_for_horizon(1)
    theo = table[f"theo_{den}_kw"]
    train = ((table["split"] == "train") & ~table["warmup"] & table[f"k_valid_{den}"]
             & ~table["op_limited_flag"] & (theo > min_theo))
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
    model.fit(table.loc[train, feats], table.loc[train, f"k_{den}"])
    val = table["split"] == "val"
    k_hat = pd.Series(np.clip(model.predict(table.loc[val, feats]), 0, None),
                      index=table.index[val])
    return to_kw(k_hat, theo[val])


def run() -> dict:
    report = output_dir("06_features")
    with RunLog("06_features", "python -m solarpv.features.decide_k", out_dir=report) as log:
        log.inputs([FEATURES_D1, HOURLY_FILE])
        t = pd.read_parquet(FEATURES_D1)
        hourly = pd.read_parquet(HOURLY_FILE, columns=["irradiance"])
        day = t["daytime"].to_numpy()
        train = (t["split"] == "train").to_numpy()
        val = (t["split"] == "val").to_numpy()
        crit = {}
        with log.step("criterion 1: share of k > 1 (train, daytime, valid)"):
            for den in DENOMINATORS:
                s = share_above(t[f"k_{den}"], t[f"k_valid_{den}"].to_numpy() & day & train)
                crit.setdefault(den, {})["share_k_above_1"] = s["k>1.0"]
                crit[den]["share_k_above_1.2"] = s["k>1.2"]
                log.note(f"{den}: {s}")
        with log.step("criterion 2: clear-day k flatness across train months"):
            local_day = t.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
            sun30 = (t["sun_elevation"] > 30).to_numpy()
            days = clear_days(hourly["irradiance"][train], t["poa_global_clearsky"][train],
                              np.asarray(local_day[train]), min_elevation_mask=sun30[train])
            on_clear = np.isin(np.asarray(local_day), days) & sun30 & train
            log.value("clear_days", len(days))
            monthly = {}
            for den in DENOMINATORS:
                m = on_clear & t[f"k_valid_{den}"].to_numpy()
                med = t.loc[m, f"k_{den}"].groupby(t.index[m].tz_convert(TIMEZONE_LOCAL)
                                                   .month).median()
                monthly[den] = med
                crit[den]["clear_day_k_spread"] = round(float(med.max() - med.min()), 4)
                crit[den]["clear_day_k_monthly_std"] = round(float(med.std()), 4)
                log.note(f"{den}: monthly median clear-day k {med.min():.3f}..{med.max():.3f} "
                         f"(spread {med.max() - med.min():.3f})")
        with log.step("criterion 3: nRMSE in kW on validation (fixed XGBoost, D+1)"):
            truth = t.loc[val, "site_total_AC_kW"]
            main = val & day & t["site_total_AC_kW"].notna().to_numpy()
            main_idx = t.index[main]
            for den in DENOMINATORS:
                scores = [nrmse_kw(fit_predict(t, den, seed=s)[main_idx], truth[main_idx])
                          for s in RANDOM_SEEDS]
                crit[den]["val_nrmse_kw_pct"] = round(float(np.mean(scores)), 4)
                crit[den]["val_nrmse_kw_seed_std"] = round(float(np.std(scores, ddof=1)), 4)
                log.note(f"{den}: validation nRMSE {np.mean(scores):.3f} +/- "
                         f"{np.std(scores, ddof=1):.3f} % of 3,449 kW over seeds {RANDOM_SEEDS} "
                         f"({len(main_idx)} daytime hours)")
        wins = {"ghi": 0, "poa": 0}
        for key in ("share_k_above_1", "clear_day_k_spread", "val_nrmse_kw_pct"):
            winner = min(DENOMINATORS, key=lambda d: crit[d][key])
            wins[winner] += 1
            log.note(f"{key}: {winner} wins ({crit['ghi'][key]} vs {crit['poa'][key]})")
        chosen = max(wins, key=wins.get)
        log.value("d11_decision", chosen)
        log.value("d11_wins", wins)

        with log.step("L-04: k by sun elevation, and the training threshold on theo"):
            band = pd.cut(t["sun_elevation"], ELEVATION_BANDS)
            rows = []
            for b, g in t[train & day].groupby(band, observed=True):
                g = g[g[f"k_valid_{chosen}"]]
                k = g[f"k_{chosen}"]
                rows.append({"sun_elevation": str(b), "hours": len(g),
                             "k_median": round(float(k.median()), 4),
                             "k_p99": round(float(k.quantile(0.99)), 4),
                             "share_k_above_1.2": round(float((k > 1.2).mean()), 4),
                             "share_k_above_1.5": round(float((k > 1.5).mean()), 4)})
            bands = pd.DataFrame(rows)
            thresholds, spread = {}, {}
            for thr in THEO_THRESHOLDS:
                scores = [nrmse_kw(fit_predict(t, chosen, min_theo=thr, seed=s)[main_idx],
                                   truth[main_idx]) for s in RANDOM_SEEDS]
                thresholds[thr] = round(float(np.mean(scores)), 4)
                spread[thr] = round(float(np.std(scores, ddof=1)), 4)
                log.note(f"training threshold theo > {thr:.0f} kW: validation nRMSE "
                         f"{thresholds[thr]:.3f} +/- {spread[thr]:.3f} % (3 seeds)")
            noise = 2 * float(np.sqrt(np.mean(np.square(list(spread.values())))))
            needed = max(MIN_GAIN_PP, noise)
            best = min(thresholds, key=thresholds.get)
            gain = thresholds[50.0] - thresholds[best]
            min_theo = best if gain > needed else 50.0
            log.value("l04_training_min_theo_kw", min_theo)
            log.note(f"best threshold {best:.0f} kW gains {gain:.3f} pp; needed > "
                     f"{needed:.3f} pp (max of {MIN_GAIN_PP} and twice the pooled seed std): "
                     f"training threshold {min_theo:.0f} kW")

        criteria = pd.DataFrame(crit).T.reset_index(names="denominator")
        criteria.to_csv(report / "d11_criteria.csv", index=False, lineterminator="\n")
        bands.to_csv(report / "l04_k_by_elevation.csv", index=False, lineterminator="\n")
        pd.DataFrame({"min_theo_kw": list(thresholds),
                      "val_nrmse_kw_pct_mean": list(thresholds.values()),
                      "val_nrmse_kw_pct_seed_std": list(spread.values())}).to_csv(
            report / "l04_training_threshold.csv", index=False, lineterminator="\n")
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
        for den, med in monthly.items():
            a.plot(med.index, med.to_numpy(), "o-", label=f"{den} (spread "
                   f"{crit[den]['clear_day_k_spread']:.3f})")
        a.set_xlabel("month (train)")
        a.set_ylabel("median k on clear days, sun > 30 deg")
        a.set_title("D-11 criterion 2: does the reference remove the season?", fontsize=9)
        a.legend(fontsize=8)
        b.bar(bands["sun_elevation"], bands["share_k_above_1.2"], color="#455a64")
        b.set_xlabel("sun elevation band [deg]")
        b.set_ylabel(f"share of hours with k > 1.2 ({chosen})")
        b.set_title("L-04: where k is unstable", fontsize=9)
        fig.tight_layout()
        fig.savefig(report / "d11_l04.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        decision = {
            "k_denominator": chosen,
            "training_min_theo_kw": float(min_theo),
            "evaluation_min_theo_kw": 50.0,
            "decided_on": "2026-10-20",
            "rule": "majority of the three D-11 criteria (fixed before criteria 2 and 3)",
            "wins": wins,
            "criteria": crit,
            "l04_rule": "mean over 3 seeds lower than at 50 kW by more than max(0.05 pp, "
                        "2 x pooled seed std)",
            "l04_needed_gain_pp": round(needed, 4),
            "l04_validation_nrmse_by_threshold": {str(k): v for k, v in thresholds.items()},
            "l04_seed_std_by_threshold": {str(k): v for k, v in spread.items()},
            "git_commit": git_commit(),
        }
        TARGET_FILE.write_text(
            "# D-11 decision and the L-04 training threshold; written by\n"
            "# python -m solarpv.features.decide_k (stage 6). Evidence: outputs/06_features/\n"
            + yaml.safe_dump(decision, sort_keys=False), encoding="utf-8", newline="\n")
        log.outputs([report / "d11_criteria.csv", report / "l04_k_by_elevation.csv",
                     report / "l04_training_threshold.csv", report / "d11_l04.png",
                     TARGET_FILE])
        print(json.dumps(decision, indent=2, default=str))
    return decision


def main(argv: list[str] | None = None) -> int:
    no_options("python -m solarpv.features.decide_k", __doc__, argv)  # X62: --help runs nothing
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

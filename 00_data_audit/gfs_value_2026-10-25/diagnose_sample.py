"""Is "D+3 better than D+1 with GFS" a property of the validation sample? (stage 6b, R6b-07)

The mapping of horizons to runs is correct (leads 10-34 / 34-58 / 58-82 h). GFS kt_poa loses skill
with the horizon on train (r 0.64 / 0.63 / 0.56 with the measured clear-sky index), but on
validation alone D+3 is as good as D+1 (0.63 / 0.57 / 0.62). Two checks, without the test split:

1. **Paired block bootstrap on validation** (24-row circular blocks, 2,000 resamples) of
   RMSE(D+3) - RMSE(D+1) for the GFS models: does the CI exclude 0?
2. **Inner validation inside train:** fit on train up to 2023-04-30, score on 2023-05-01 to
   2024-01-31, a period disjoint from the validation split. If D+1 < D+2 < D+3 there, the
   validation order is a sample effect.

    python 00_data_audit/gfs_value_2026-10-25/diagnose_sample.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from solarpv.config.constants import AC_CAPACITY_KW, HORIZONS, RANDOM_SEEDS
from solarpv.config.runlog import RunLog
from solarpv.evaluation.metrics import circular_block_indices
from solarpv.features.contract import features_for_horizon
from solarpv.features.decide_k import XGB_PARAMS
from solarpv.nwp import check as K

HERE = Path(__file__).resolve().parent
INNER_TRAIN_END = pd.Timestamp("2023-04-30 23:59", tz="UTC")


def predict_kw(t: pd.DataFrame, feats, train_rows, eval_rows, target, seed) -> np.ndarray:
    train = train_rows & ~t["warmup"] & ~t["op_limited_flag"]
    if target == "indirect":
        train &= t["k_valid"] & (t["theo_kw"] > 50)
        y = t["target_k"]
    else:
        train &= t["site_total_AC_kW"].notna()
        y = t["site_total_AC_kW"]
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
    model.fit(t.loc[train, feats], y[train])
    pred = model.predict(t.loc[eval_rows, feats])
    if target == "indirect":
        pred = np.clip(np.clip(pred, 0, None) * t.loc[eval_rows, "theo_kw"].to_numpy(), 0,
                       AC_CAPACITY_KW)
    return pred


def main() -> int:
    with RunLog("06b_gfs", "python 00_data_audit/gfs_value_2026-10-25/diagnose_sample.py",
                out_dir=HERE) as log:
        tables = {h: K.load(h) for h in HORIZONS}
        t1 = tables[1]
        val = (t1["split"] == "val") & t1["daytime"] & t1["site_total_AC_kW"].notna()
        inner_train = (t1["split"] == "train") & (t1.index <= INNER_TRAIN_END)
        inner_eval = (t1["split"] == "train") & (t1.index > INNER_TRAIN_END) & t1["daytime"] \
            & t1["site_total_AC_kW"].notna()
        rows, boot = [], []
        for target in ("indirect", "direct"):
            sq = {}
            for h, t in tables.items():
                feats = K.gfs_candidates(t) + features_for_horizon(h)
                actual_val = t.loc[val, "site_total_AC_kW"].to_numpy()
                actual_in = t.loc[inner_eval, "site_total_AC_kW"].to_numpy()
                p_val = np.mean([predict_kw(t, feats, t["split"] == "train", val, target, s)
                                 for s in RANDOM_SEEDS], axis=0)
                p_in = np.mean([predict_kw(t, feats, inner_train, inner_eval, target, s)
                                for s in RANDOM_SEEDS], axis=0)
                sq[h] = (p_val - actual_val) ** 2
                for name, p, a in (("validation", p_val, actual_val),
                                   ("inner (train, after 2023-04)", p_in, actual_in)):
                    rows.append({"target": target, "horizon": h, "sample": name, "n": len(a),
                                 "nrmse_kw": float(np.sqrt(np.mean((p - a) ** 2))
                                                   / AC_CAPACITY_KW * 100)})
            idx = circular_block_indices(len(sq[1]), 24, 2000, np.random.default_rng(42))
            diff = (np.sqrt(sq[3][idx].mean(axis=1)) - np.sqrt(sq[1][idx].mean(axis=1))) \
                / AC_CAPACITY_KW * 100
            lo, hi = np.percentile(diff, [2.5, 97.5])
            boot.append({"target": target, "diff_D3_minus_D1_pp": float(
                (np.sqrt(sq[3].mean()) - np.sqrt(sq[1].mean())) / AC_CAPACITY_KW * 100),
                "ci_lo": float(lo), "ci_hi": float(hi)})
            d31 = boot[-1]["diff_D3_minus_D1_pp"]
            log.note(f"{target}: validation RMSE(D+3) - RMSE(D+1) = {d31:.3f} pp, "
                     f"95 % CI [{lo:.3f}, {hi:.3f}]")
        out = pd.DataFrame(rows)
        out.to_csv(HERE / "diagnose_sample.csv", index=False, lineterminator="\n")
        pd.DataFrame(boot).to_csv(HERE / "diagnose_sample_bootstrap.csv", index=False,
                                  lineterminator="\n")
        log.outputs([HERE / "diagnose_sample.csv", HERE / "diagnose_sample_bootstrap.csv"])
        print(out.pivot_table(index=["target", "sample"], columns="horizon",
                              values="nrmse_kw").round(3).to_string())
        print(pd.DataFrame(boot).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

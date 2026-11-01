"""Why is the 05e set worse than the full pool on validation for direct D+2/D+3? (R7-03)

    python 00_data_audit/selection_2026-11-01/diagnose_gap.py

The S8 look found the direct 05e sets 0.33 and 0.43 pp worse than the full pool at D+2 and D+3
(indirect D+3: 0.09 pp), above the margins of the D-12 rule. Two hypotheses, each with its own
check; nothing here changes a selection.

1. **Method:** the small selection model undervalues features the fixed XGBoost of D-11 uses.
   Then the gap appears inside train too: the fixed model, 3 seeds, on (a) the 5 date folds and
   (b) an inner hold-out (fit to 2023-04-30, scored 2023-05 to 2024-01, as in R6b-07).
2. **Sample:** the validation period differs (curtailment is frequent in spring 2024; op-limited
   hours are scored but never trained on). Then the gap concentrates in op-limited hours and in
   some months: the S8 models are refitted with the same seeds and their validation errors split
   by operation and month (a diagnosis of the same comparison, not a new decision).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, HORIZONS, RANDOM_SEEDS
from solarpv.config.paths import OUTPUTS_DIR
from solarpv.config.runlog import RunLog
from solarpv.features.decide_k import XGB_PARAMS as FIXED_XGB
from solarpv.nwp import check as nwp_check
from solarpv.selection import common as C

HERE = Path(__file__).resolve().parent
OUT = OUTPUTS_DIR / "07_feature_selection"
INNER_TRAIN_END = pd.Timestamp("2023-04-30 23:59", tz="UTC")


def sets(target: str, h: int) -> dict[str, list[str]]:
    e = json.loads((OUT / f"final_features_{target}_D{h}.json").read_text(encoding="utf-8"))
    t = C.train_table(h)
    return {"05e": e["recommended"]["features"], "full pool": C.pool(t, h)}


def nrmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)) / AC_CAPACITY_KW * 100)


def main() -> int:
    with RunLog("audit", "python 00_data_audit/selection_2026-11-01/diagnose_gap.py",
                out_dir=HERE) as log:
        inner, split_rows = [], []
        for h in HORIZONS:
            train = C.train_table(h)
            both = nwp_check.load(h)  # train and validation; asserts no test row
            for target in C.TARGETS:
                s = sets(target, h)
                p = C.problem(train, target)
                holdout = [(pd.Series(train.index <= INNER_TRAIN_END, index=train.index),
                            pd.Series(train.index > INNER_TRAIN_END, index=train.index))]
                for sample, folds in (("train folds", None), ("inner hold-out", holdout)):
                    for name, feats in s.items():
                        v = [C.cv_score(p, feats, FIXED_XGB, seed, with_shap=False,
                                        folds=folds)[0] for seed in RANDOM_SEEDS]
                        inner.append({"target": target, "horizon": h, "sample": sample,
                                      "features": name, "nrmse_kw": float(np.mean(v))})
                if not (target == "direct" or h == 3):
                    continue
                pv = C.problem(both, target)
                fit = (both["split"] == "train") & pv.fit
                rows = (both["split"] == "val") & pv.score
                for name, feats in s.items():
                    errs = []
                    for seed in RANDOM_SEEDS:
                        m = C.model(FIXED_XGB, seed)
                        m.fit(both.loc[fit, feats], pv.y[fit])
                        pred = C.to_kw(pv, m.predict(both.loc[rows, feats]), rows)
                        errs.append(pred - pv.ac[rows].to_numpy())
                    err = np.mean(errs, axis=0)  # seed-averaged prediction error
                    frame = pd.DataFrame({"err": err,
                                          "op": both.loc[rows, "op_limited_flag"].to_numpy(),
                                          "month": both.index[rows].strftime("%Y-%m")})
                    split_rows.append({"target": target, "horizon": h, "features": name,
                                       "view": "all", "n": len(frame),
                                       "nrmse_kw": nrmse(frame["err"].to_numpy())})
                    for view, sub in (("normal operation", frame[~frame["op"]]),
                                      ("op-limited", frame[frame["op"]])):
                        split_rows.append({"target": target, "horizon": h, "features": name,
                                           "view": view, "n": len(sub),
                                           "nrmse_kw": nrmse(sub["err"].to_numpy())})
                    for month, sub in frame.groupby("month"):
                        split_rows.append({"target": target, "horizon": h, "features": name,
                                           "view": month, "n": len(sub),
                                           "nrmse_kw": nrmse(sub["err"].to_numpy())})
                log.note(f"{target} D+{h} done")
        a = pd.DataFrame(inner).pivot_table(index=["target", "horizon", "sample"],
                                            columns="features", values="nrmse_kw").reset_index()
        a["gap_05e_minus_full"] = a["05e"] - a["full pool"]
        a.round(4).to_csv(HERE / "gap_inner_train.csv", index=False, lineterminator="\n")
        b = pd.DataFrame(split_rows).pivot_table(index=["target", "horizon", "view", "n"],
                                                 columns="features",
                                                 values="nrmse_kw").reset_index()
        b["gap_05e_minus_full"] = b["05e"] - b["full pool"]
        b.round(4).to_csv(HERE / "gap_validation_views.csv", index=False, lineterminator="\n")
        for _, r in a.iterrows():
            log.note(f"inner {r['target']} D+{r['horizon']} {r['sample']}: gap "
                     f"{r['gap_05e_minus_full']:+.3f} pp")
        for _, r in b[b["view"].isin(["all", "normal operation", "op-limited"])].iterrows():
            log.note(f"val {r['target']} D+{r['horizon']} {r['view']} (n={r['n']}): gap "
                     f"{r['gap_05e_minus_full']:+.3f} pp")
        log.outputs([HERE / "gap_inner_train.csv", HERE / "gap_validation_views.csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

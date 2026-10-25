"""Why is the D+3 model with GFS better than the D+1 model on validation? (stage 6b, R6b-07)

The value check gave, for the indirect target with GFS: D+1 6.853 %, D+2 7.310 %, D+3 6.715 %.
GFS itself loses skill with the horizon (daily r 0.82 / 0.79 / 0.74), so the D+3 model should
not beat D+1. The plan treats skill rising with the horizon as a red flag.

The script trains the same fixed XGBoost (3 seeds, train -> validation, nRMSE in kW) on:
1. GFS candidates + the stage-6 features without past-day power and weather (no lags, no
   rolling): if the order becomes D+1 < D+2 < D+3, the past-day features cause it;
2. GFS candidates only;
3. the full set, with the horizon-specific lag and rolling columns of each horizon swapped in.

Train and validation only.

    python 00_data_audit/gfs_value_2026-10-25/diagnose_horizons.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import HORIZONS, RANDOM_SEEDS
from solarpv.config.runlog import RunLog
from solarpv.features.contract import features_for_horizon
from solarpv.nwp import check as K

HERE = Path(__file__).resolve().parent


def is_past_day(name: str) -> bool:
    return "_lag_" in name or "rolling" in name


def main() -> int:
    with RunLog("06b_gfs", "python 00_data_audit/gfs_value_2026-10-25/diagnose_horizons.py",
                out_dir=HERE) as log:
        tables = {h: K.load(h) for h in HORIZONS}
        rows = []
        for h, t in tables.items():
            gfs = K.gfs_candidates(t)
            static = [f for f in features_for_horizon(h) if not is_past_day(f)]
            past = [f for f in features_for_horizon(h) if is_past_day(f)]
            sets = {"gfs+static": gfs + static, "gfs_only": gfs,
                    "static_only": static, "gfs+static+past": gfs + static + past}
            for name, feats in sets.items():
                for target in ("indirect", "direct"):
                    scores = [K.fit_score(t, feats, target, s) for s in RANDOM_SEEDS]
                    rows.append({"horizon": h, "target": target, "features": name,
                                 "n_features": len(feats), "nrmse_mean": np.mean(scores),
                                 "nrmse_std": np.std(scores, ddof=1)})
                    log.note(f"D+{h} {target} {name} ({len(feats)}): {np.mean(scores):.3f} "
                             f"+/- {np.std(scores, ddof=1):.3f}")
        out = pd.DataFrame(rows)
        out.to_csv(HERE / "diagnose_horizons.csv", index=False, lineterminator="\n")
        log.outputs([HERE / "diagnose_horizons.csv"])
        print(out.pivot_table(index=["target", "features"], columns="horizon",
                              values="nrmse_mean").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

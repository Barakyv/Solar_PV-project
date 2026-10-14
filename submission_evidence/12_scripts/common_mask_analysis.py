"""
common_mask_analysis.py - Phase H: Common Evaluation Protocol
=============================================================
Computes metrics on a shared timestamp set (intersection of all model predictions)
under a unified daytime mask and compares with target-specific evaluation.

Outputs:
  outputs/07_evaluation/common_mask_summary.csv
  outputs/07_evaluation/common_mask_sensitivity.csv
"""
import pathlib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "production" / "00_config"))
from constants import AC_CAPACITY_KW, MEAN_DAYTIME_K, SUN_ELEV_MIN, HORIZONS

OUTPUTS = ROOT / "outputs" / "07_evaluation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

PRED_FILES = {
    "xgb_sota":   ROOT / "outputs/06_xgb_sota/predictions_xgb_sota.csv",
    "lstm_sota":  ROOT / "outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv",
    "tft":        ROOT / "outputs/06_tft/predictions_tft_aligned.csv",
}

NORM = {"indirect": MEAN_DAYTIME_K, "direct": AC_CAPACITY_KW}


def load_preds():
    frames = {}
    for model, path in PRED_FILES.items():
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames[model] = df
        print(f"  Loaded {model}: {len(df):,} rows")
    return frames


def compute_nrmse(y_true, y_pred, denom):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / max(denom, 1e-9) * 100.0


def run():
    print("[H] Loading predictions...")
    frames = load_preds()
    if len(frames) < 2:
        print("  Need at least 2 models. Exiting.")
        return

    rows_shared, rows_sensitivity = [], []

    for tgt in ["indirect", "direct"]:
        denom = NORM[tgt]

        # Daytime threshold per target
        day_thresh = 0.01 if tgt == "indirect" else 10.0

        for h in HORIZONS:
            # ── Per-model slices ──────────────────────────────────────────────
            slices = {}
            for model, df in frames.items():
                sub = df[(df["target"] == tgt) & (df["horizon"] == h)].copy()
                slices[model] = sub

            if not slices:
                continue

            # ── Build shared timestamp set ────────────────────────────────────
            ts_sets = [set(s["timestamp"].astype(str)) for s in slices.values()]
            shared_ts = ts_sets[0].intersection(*ts_sets[1:])
            n_shared = len(shared_ts)

            # Report per-model N before and after intersection
            for model, sub in slices.items():
                n_total  = len(sub)
                n_target_day = (sub["y_true"] > day_thresh).sum()
                n_shared_day = sub[sub["timestamp"].astype(str).isin(shared_ts) &
                                   (sub["y_true"] > day_thresh)].shape[0]
                rows_sensitivity.append({
                    "model": model, "target": tgt, "horizon": h,
                    "n_target_specific": n_target_day,
                    "n_shared_set": n_shared_day,
                    "pct_retained": round(n_shared_day / max(n_target_day, 1) * 100, 1),
                })

            # ── Metrics on shared mask ────────────────────────────────────────
            ref_model = "xgb_sota"
            if ref_model not in slices:
                continue
            ref = slices[ref_model]
            ref_shared = ref[ref["timestamp"].astype(str).isin(shared_ts) &
                             (ref["y_true"] > day_thresh)]

            for model, sub in slices.items():
                sub_shared = sub[sub["timestamp"].astype(str).isin(shared_ts) &
                                 (sub["y_true"] > day_thresh)]
                # Align on exact timestamps
                merged = ref_shared[["timestamp", "y_true"]].merge(
                    sub_shared[["timestamp", "y_pred"]], on="timestamp")
                if len(merged) < 10:
                    continue

                y_true = merged["y_true"].values
                y_pred = merged["y_pred"].values
                nrmse  = compute_nrmse(y_true, y_pred, denom)
                r2     = r2_score(y_true, y_pred)

                rows_shared.append({
                    "model": model, "target": tgt, "horizon": h,
                    "n": len(merged),
                    "nRMSE_pct_shared": round(nrmse, 4),
                    "R2_shared": round(r2, 4),
                })

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_shared = pd.DataFrame(rows_shared)
    df_sens   = pd.DataFrame(rows_sensitivity)

    df_shared.to_csv(OUTPUTS / "common_mask_summary.csv", index=False)
    df_sens.to_csv(OUTPUTS / "common_mask_sensitivity.csv", index=False)

    print("\n[H] Common mask results (shared timestamps, daytime):")
    print(df_shared[df_shared["model"].isin(["xgb_sota","lstm_sota","tft"]) &
                    (df_shared["horizon"]==1)].to_string(index=False))

    print("\n[H] Sensitivity (rows retained per model):")
    print(df_sens[df_sens["horizon"]==1].to_string(index=False))

    # ── Ranking check ─────────────────────────────────────────────────────────
    print("\n[H] Ranking under shared mask vs target-specific mask:")
    for tgt in ["indirect", "direct"]:
        sub = df_shared[(df_shared["target"]==tgt) & (df_shared["horizon"]==1)]
        sub = sub.sort_values("nRMSE_pct_shared")
        print(f"  {tgt.upper()} D+1 ranking: " +
              ", ".join(f"{r.model}={r.nRMSE_pct_shared:.2f}%" for _, r in sub.iterrows()))

    print("\n[H] Saved common_mask_summary.csv + common_mask_sensitivity.csv")


if __name__ == "__main__":
    run()

"""
compare_routes.py - Phase C1-C4: Indirect vs Direct Route Evaluation
=====================================================================
Compares two forecasting routes at the same AC-power scale (kW):
  Route 1 (indirect): predict clear-sky performance index k, then reconstruct
                      AC_kW = pred_k * GHI_clearsky * AC_CAPACITY_KW / 1000
  Route 2 (direct):   predict AC power directly

Both routes are evaluated with the same metric (nRMSE denominator = AC_CAPACITY_KW).
This is the operationally fair comparison: two routes to the same target.

Outputs:
  outputs/07_evaluation/route_comparison.csv
  outputs/07_evaluation/figures/route_comparison.png
"""
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "production" / "00_config"))
from constants import AC_CAPACITY_KW, SUN_ELEV_MIN, HORIZONS, MEAN_DAYTIME_K

OUTPUTS    = ROOT / "outputs" / "07_evaluation"
FIG_DIR    = OUTPUTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PARQUET = (ROOT / "04_feature_engineering" / "data" / "processed" /
                    "04_features_target_k" / "features_1H_target_k.parquet")

INDIRECT_PREDS = {
    "xgb_sota":  ROOT / "outputs/06_xgb_sota/predictions_xgb_sota.csv",
    "lstm_sota": ROOT / "outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv",
}
DIRECT_PREDS = {
    "xgb_sota":  ROOT / "outputs/06_xgb_sota/predictions_xgb_sota.csv",
    "lstm_sota": ROOT / "outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv",
}


def k_to_power(pred_k: np.ndarray, ghi_clearsky: np.ndarray) -> np.ndarray:
    """Reconstruct AC power from clear-sky performance index predictions.
    Exact inverse of: target_k = AC_kW / (GHI_clearsky * AC_CAPACITY_KW / 1000)
    """
    p = pred_k * ghi_clearsky * AC_CAPACITY_KW / 1000.0
    return np.clip(p, 0, AC_CAPACITY_KW)


def compute_metrics(y_true, y_pred):
    denom = AC_CAPACITY_KW
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    return {
        "nRMSE_kW_pct": round(rmse / denom * 100, 4),
        "MAE_kW":       round(mae, 2),
        "R2":           round(r2, 4),
        "RMSE_kW":      round(rmse, 2),
    }


def run():
    print("[C] Loading feature matrix for GHI_clearsky...")
    feat = pd.read_parquet(FEATURES_PARQUET, columns=["ghi_clearsky", "sun_elevation"])
    feat.index = pd.to_datetime(feat.index, utc=True)
    print(f"  Feature matrix: {feat.shape}")

    rows = []

    for model in ["xgb_sota", "lstm_sota"]:
        ind_path = INDIRECT_PREDS.get(model)
        dir_path = DIRECT_PREDS.get(model)

        if not ind_path or not ind_path.exists():
            print(f"  [SKIP] {model} indirect predictions not found")
            continue

        ind_df = pd.read_csv(ind_path)
        ind_df["timestamp"] = pd.to_datetime(ind_df["timestamp"], utc=True)

        dir_df = pd.read_csv(dir_path)
        dir_df["timestamp"] = pd.to_datetime(dir_df["timestamp"], utc=True)

        for h in HORIZONS:
            # ── Route 1: indirect k → power ───────────────────────────────────
            ind_h = ind_df[(ind_df["target"] == "indirect") &
                           (ind_df["horizon"] == h)].copy()
            # Join GHI clearsky
            ind_h = ind_h.join(feat[["ghi_clearsky"]], on="timestamp", how="inner")
            ind_h = ind_h.dropna(subset=["ghi_clearsky"])

            # Reconstruct AC power
            ind_h["y_pred_kw"] = k_to_power(ind_h["y_pred"].values,
                                             ind_h["ghi_clearsky"].values)
            # y_true in indirect is target_k; reconstruct true AC power too
            ind_h["y_true_kw"] = k_to_power(ind_h["y_true"].values,
                                             ind_h["ghi_clearsky"].values)

            # Daytime filter (sun_elevation > threshold via GHI proxy)
            ind_h = ind_h[ind_h["y_true"] > 0.01]

            if len(ind_h) < 10:
                continue

            m_ind = compute_metrics(ind_h["y_true_kw"].values, ind_h["y_pred_kw"].values)
            rows.append({"model": model, "horizon": h, "route": "indirect_to_power",
                         "n": len(ind_h), **m_ind})

            # ── Route 2: direct AC power ──────────────────────────────────────
            dir_h = dir_df[(dir_df["target"] == "direct") &
                           (dir_df["horizon"] == h)].copy()
            dir_h = dir_h[dir_h["y_true"] > 10]  # daytime filter for direct

            if len(dir_h) < 10:
                continue

            m_dir = compute_metrics(dir_h["y_true"].values, dir_h["y_pred"].values)
            rows.append({"model": model, "horizon": h, "route": "direct",
                         "n": len(dir_h), **m_dir})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "route_comparison.csv", index=False)
    print("\n[C] Route comparison results:")
    print(df.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────────
    models   = df["model"].unique()
    horizons = sorted(df["horizon"].unique())
    routes   = ["direct", "indirect_to_power"]
    colors   = {"direct": "#2196F3", "indirect_to_power": "#FF9800"}
    markers  = {"direct": "o", "indirect_to_power": "s"}

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        for route in routes:
            sub = df[(df["model"] == model) & (df["route"] == route)]
            if sub.empty:
                continue
            sub = sub.sort_values("horizon")
            ax.plot(sub["horizon"], sub["nRMSE_kW_pct"],
                    color=colors[route], marker=markers[route],
                    linewidth=2, markersize=7, label=route)

        ax.set_title(model.replace("_", " ").upper(), fontsize=11)
        ax.set_xlabel("Horizon")
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"D+{h}" for h in horizons])
        ax.set_ylabel("nRMSE [% of AC capacity]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Route Comparison: Indirect k→Power vs Direct AC Power\n"
        f"(denominator = AC capacity {AC_CAPACITY_KW:.0f} kW, daytime only)",
        fontsize=11,
    )
    plt.tight_layout()
    out_fig = FIG_DIR / "route_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[C] Saved route_comparison.png")
    print(f"[C] Saved route_comparison.csv")


if __name__ == "__main__":
    run()

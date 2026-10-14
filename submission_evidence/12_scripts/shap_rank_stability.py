"""
shap_rank_stability.py - Phase E1/E2: SHAP Rank Stability & Feature Grouping
=============================================================================
Computes Spearman rank correlation of SHAP feature importances:
  - Across horizons D+1/D+2/D+3 (horizon stability)
  - Across targets (indirect vs direct)

Groups features into families: clearsky, rolling_stats, time_encodings,
lag_power, and other. Reports SHAP importance by group.

Outputs:
  outputs/07_evaluation/shap_rank_stability.csv   (feature-level correlations)
  outputs/07_evaluation/shap_group_importance.csv  (group-level aggregates)
  outputs/07_evaluation/shap_horizon_corr.csv      (Spearman corr matrix)
"""
import pathlib
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).parent.parent
SHAP_DIR = ROOT / "outputs" / "06_xgb_sota" / "shap"
OUTPUTS = ROOT / "outputs" / "07_evaluation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3]
TARGETS = ["indirect", "direct"]

# Feature family groupings (keywords / prefixes)
GROUPS = {
    "clearsky":       ["ghi_clearsky", "dhi_clearsky", "dni_clearsky",
                       "poa_global_clearsky", "poa_direct_clearsky", "poa_diffuse_clearsky",
                       "sun_elevation", "clearsky"],
    "rolling_stats":  ["rolling_mean", "rolling_std", "rolling_sum", "rolling_max",
                       "ewm", "diff_", "change_"],
    "time_encodings": ["sin_", "cos_", "hour", "month", "dayofyear", "dayofweek",
                       "is_weekend", "season"],
    "lag_power":      ["_lag", "lag_"],
    "nwp_forecast":   ["fcst_", "poa_global_forecast", "poa_direct_forecast",
                       "poa_diffuse_forecast"],
}


def assign_group(feature: str) -> str:
    f = feature.lower()
    for group, keywords in GROUPS.items():
        if any(kw in f for kw in keywords):
            return group
    return "other"


def load_shap(target: str, horizon: int) -> pd.Series:
    path = SHAP_DIR / f"shap_{target}_D{horizon}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path).set_index("feature")["shap_importance"]
    return df


def run():
    # ── Load all SHAP importance series ──────────────────────────────────────
    all_shap = {}
    for tgt in TARGETS:
        for h in HORIZONS:
            s = load_shap(tgt, h)
            if s.empty:
                print(f"  [SKIP] {tgt} D+{h} — file not found")
                continue
            all_shap[(tgt, h)] = s

    if not all_shap:
        print("No SHAP files found. Exiting.")
        return

    # ── Build unified feature index ───────────────────────────────────────────
    all_features = sorted(set().union(*[s.index for s in all_shap.values()]))
    df_all = pd.DataFrame(
        {key: s.reindex(all_features, fill_value=0.0) for key, s in all_shap.items()},
        index=all_features,
    )
    df_all.columns = [f"{t}_D{h}" for t, h in df_all.columns]

    # ── Horizon Spearman correlation (within target) ──────────────────────────
    corr_rows = []
    for tgt in TARGETS:
        cols = [f"{tgt}_D{h}" for h in HORIZONS if f"{tgt}_D{h}" in df_all.columns]
        if len(cols) < 2:
            continue
        sub = df_all[cols]
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                rho, pval = spearmanr(sub[c1], sub[c2])
                corr_rows.append({
                    "target": tgt, "compare": f"{c1} vs {c2}",
                    "spearman_rho": round(rho, 4), "p_value": round(pval, 4),
                })

    # Cross-target correlation (D+1 only, representative)
    if "indirect_D1" in df_all.columns and "direct_D1" in df_all.columns:
        rho, pval = spearmanr(df_all["indirect_D1"], df_all["direct_D1"])
        corr_rows.append({
            "target": "cross-target", "compare": "indirect_D1 vs direct_D1",
            "spearman_rho": round(rho, 4), "p_value": round(pval, 4),
        })

    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(OUTPUTS / "shap_horizon_corr.csv", index=False)

    print("\n[E1] SHAP Rank Stability (Spearman correlation):")
    print(df_corr.to_string(index=False))

    # ── Group-level importance ────────────────────────────────────────────────
    df_all["group"] = [assign_group(f) for f in df_all.index]

    group_rows = []
    for tgt in TARGETS:
        for h in HORIZONS:
            col = f"{tgt}_D{h}"
            if col not in df_all.columns:
                continue
            grp = df_all.groupby("group")[col].sum().reset_index()
            grp.columns = ["group", "shap_total"]
            grp["pct"] = (grp["shap_total"] / grp["shap_total"].sum() * 100).round(1)
            grp["target"] = tgt
            grp["horizon"] = h
            group_rows.append(grp)

    df_grp = pd.concat(group_rows, ignore_index=True)
    df_grp = df_grp[["target", "horizon", "group", "shap_total", "pct"]]
    df_grp = df_grp.sort_values(["target", "horizon", "pct"], ascending=[True, True, False])
    df_grp.to_csv(OUTPUTS / "shap_group_importance.csv", index=False)

    print("\n[E2] Feature Group Importance by Target & Horizon:")
    for tgt in TARGETS:
        for h in HORIZONS:
            sub = df_grp[(df_grp["target"] == tgt) & (df_grp["horizon"] == h)]
            print(f"\n  {tgt.upper()} D+{h}:")
            for _, row in sub.iterrows():
                print(f"    {row['group']:20s}: {row['pct']:5.1f}%")

    # ── Top-10 feature rank comparison (indirect D+1 vs D+2 vs D+3) ──────────
    print("\n[E1] Top-10 features by SHAP rank:")
    for tgt in TARGETS:
        cols = [f"{tgt}_D{h}" for h in HORIZONS if f"{tgt}_D{h}" in df_all.columns]
        if not cols:
            continue
        top = df_all[cols].mean(axis=1).nlargest(10).index.tolist()
        print(f"\n  {tgt.upper()} (avg rank across horizons):")
        for rank, feat in enumerate(top, 1):
            vals = "  ".join(f"D+{h}={df_all.get(f'{tgt}_D{h}', pd.Series()).__getitem__(feat) if feat in df_all.get(f'{tgt}_D{h}', pd.Series()).index else 0:.4f}" for h in HORIZONS if f"{tgt}_D{h}" in df_all.columns)
            print(f"    {rank:2d}. {feat:50s}  {vals}")

    # ── Per-feature rank stability CSV ────────────────────────────────────────
    df_feat = df_all.drop(columns=["group"]).copy()
    df_feat["group"] = [assign_group(f) for f in df_feat.index]
    df_feat.to_csv(OUTPUTS / "shap_rank_stability.csv")

    print(f"\n[E] Saved:")
    print(f"  shap_rank_stability.csv")
    print(f"  shap_group_importance.csv")
    print(f"  shap_horizon_corr.csv")


if __name__ == "__main__":
    run()

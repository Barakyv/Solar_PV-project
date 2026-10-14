"""
compile_seed_variability.py - Phase B3: Seed Variability Summary
================================================================
Compiles nRMSE results from XGB and LSTM ×3 seeds (42, 123, 777)
into a summary table with mean ± std per (model, target, horizon).

Also compiles ablation seed variability (mean ± std of dNRMSE).

Inputs:
  outputs/06_xgb_sota/metrics_xgb_sota.csv
  outputs/06_xgb_sota/metrics_xgb_sota_seed123.csv
  outputs/06_xgb_sota/metrics_xgb_sota_seed777.csv
  outputs/06_lstm_sota/metrics_lstm_sota.csv
  outputs/06_lstm_sota/metrics_lstm_sota_seed123.csv
  outputs/06_lstm_sota/metrics_lstm_sota_seed777.csv
  outputs/09_cv_ablation/ablation_results.csv (seed=42)
  outputs/09_cv_ablation/ablation_results_seed123.csv
  outputs/09_cv_ablation/ablation_results_seed777.csv

Outputs:
  outputs/07_evaluation/seed_variability.csv
  outputs/07_evaluation/seed_variability_summary.csv
  outputs/07_evaluation/ablation_seed_variability.csv
"""
import pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs" / "07_evaluation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 777]

MODEL_METRICS = {
    "xgb_sota": {
        42:  ROOT / "outputs/06_xgb_sota/metrics_xgb_sota.csv",
        123: ROOT / "outputs/06_xgb_sota/metrics_xgb_sota_seed123.csv",
        777: ROOT / "outputs/06_xgb_sota/metrics_xgb_sota_seed777.csv",
    },
    "lstm_sota": {
        42:  ROOT / "outputs/06_lstm_sota/metrics_lstm_sota.csv",
        123: ROOT / "outputs/06_lstm_sota/metrics_lstm_sota_seed123.csv",
        777: ROOT / "outputs/06_lstm_sota/metrics_lstm_sota_seed777.csv",
    },
}

ABLATION_FILES = {
    42:  ROOT / "outputs/09_cv_ablation/ablation_results.csv",
    123: ROOT / "outputs/09_cv_ablation/ablation_results_seed123.csv",
    777: ROOT / "outputs/09_cv_ablation/ablation_results_seed777.csv",
}


def load_model_seeds():
    rows = []
    for model, seed_files in MODEL_METRICS.items():
        for seed, path in seed_files.items():
            if not path.exists():
                print(f"  [MISSING] {path.name}")
                continue
            df = pd.read_csv(path)
            df["seed"] = seed
            df["model"] = model
            rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_ablation_seeds():
    rows = []
    for seed, path in ABLATION_FILES.items():
        if not path.exists():
            print(f"  [MISSING] {path.name}")
            continue
        df = pd.read_csv(path)
        if "seed" not in df.columns:
            df["seed"] = seed
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run():
    print("[B3] Compiling seed variability...")

    # ── Model seed variability ────────────────────────────────────────────────
    df_all = load_model_seeds()
    if df_all.empty:
        print("  No model metrics found.")
    else:
        # Per-seed table
        df_all.to_csv(OUTPUTS / "seed_variability.csv", index=False)
        print(f"  Saved seed_variability.csv ({len(df_all)} rows)")

        # Summary: mean ± std
        grp = df_all.groupby(["model", "target", "horizon"])
        summary = grp["nRMSE_pct"].agg(
            nRMSE_mean="mean",
            nRMSE_std="std",
            nRMSE_min="min",
            nRMSE_max="max",
            n_seeds="count",
        ).reset_index()
        summary["nRMSE_mean"] = summary["nRMSE_mean"].round(3)
        summary["nRMSE_std"]  = summary["nRMSE_std"].round(3)
        summary["nRMSE_min"]  = summary["nRMSE_min"].round(3)
        summary["nRMSE_max"]  = summary["nRMSE_max"].round(3)

        summary.to_csv(OUTPUTS / "seed_variability_summary.csv", index=False)
        print(f"\n[B3] Seed variability summary (nRMSE_pct):")
        print(summary.to_string(index=False))

        # Check ranking stability
        print("\n[B3] Ranking check per (target, horizon):")
        for tgt in ["indirect", "direct"]:
            for h in [1, 2, 3]:
                sub = summary[(summary["target"] == tgt) & (summary["horizon"] == h)]
                if sub.empty:
                    continue
                sub = sub.sort_values("nRMSE_mean")
                rank_str = " < ".join(
                    f"{r.model}({r.nRMSE_mean:.3f}+/-{r.nRMSE_std:.3f})"
                    for _, r in sub.iterrows()
                )
                print(f"  {tgt.upper()} D+{h}: {rank_str}")

    # ── Ablation seed variability ─────────────────────────────────────────────
    df_abl = load_ablation_seeds()
    if df_abl.empty:
        print("\n  No ablation seed files found.")
    else:
        # Compute dNRMSE vs baseline per (target, horizon, seed)
        delta_rows = []
        for (tgt, h, seed), grp in df_abl.groupby(["target", "horizon", "seed"]):
            base = grp[grp["mode"] == "baseline"]["nRMSE_pct"].values
            if len(base) == 0:
                continue
            base_val = base[0]
            for _, row in grp.iterrows():
                if row["mode"] == "baseline":
                    continue
                delta_rows.append({
                    "target": tgt, "horizon": h, "seed": seed,
                    "mode": row["mode"],
                    "nRMSE_pct": round(row["nRMSE_pct"], 4),
                    "delta_nRMSE": round(row["nRMSE_pct"] - base_val, 4),
                })

        df_delta = pd.DataFrame(delta_rows)

        # Summary: mean ± std of delta across seeds
        abl_summary = df_delta.groupby(["target", "horizon", "mode"])["delta_nRMSE"].agg(
            delta_mean="mean",
            delta_std="std",
        ).reset_index()
        abl_summary["delta_mean"] = abl_summary["delta_mean"].round(3)
        abl_summary["delta_std"]  = abl_summary["delta_std"].fillna(0).round(3)

        abl_summary.to_csv(OUTPUTS / "ablation_seed_variability.csv", index=False)
        print(f"\n[E3] Ablation seed variability (dNRMSE mean+/-std across seeds):")
        # Show averaged across horizons
        abl_avg = abl_summary.groupby(["target", "mode"])[["delta_mean", "delta_std"]].mean().round(3).reset_index()
        for tgt in ["indirect", "direct"]:
            sub = abl_avg[abl_avg["target"] == tgt].sort_values("delta_mean", ascending=False)
            print(f"\n  {tgt.upper()}:")
            for _, r in sub.iterrows():
                sign = "+" if r["delta_mean"] >= 0 else ""
                print(f"    {r['mode']:22s}: {sign}{r['delta_mean']:.3f} +/- {r['delta_std']:.3f} pp")

        print(f"\n[E3] Saved ablation_seed_variability.csv")


if __name__ == "__main__":
    run()

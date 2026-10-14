"""
detectable_effect.py - Phase A2/A3: Detectable Effect & Extended DM Reporting
==============================================================================
For each (target, horizon, model_pair) in dm_tests.csv:
  1. Compute effective sample size from loss-differential ACF
  2. Compute minimum detectable standardised effect at 80% power, 5% significance
  3. Compare observed effect size against detectable threshold
  4. Extend dm_tests.csv with CI columns and effect-size fields

Outputs:
  outputs/07_evaluation/dm_tests_extended.csv
  outputs/07_evaluation/detectable_effect_summary.csv
"""
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "production" / "00_config"))
from constants import AC_CAPACITY_KW, MEAN_DAYTIME_K, HORIZONS

OUTPUTS = ROOT / "outputs" / "07_evaluation"

PRED_FILES = {
    "xgb_sota":  ROOT / "outputs/06_xgb_sota/predictions_xgb_sota.csv",
    "lstm_sota": ROOT / "outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv",
    "tft":       ROOT / "outputs/06_tft/predictions_tft_aligned.csv",
}
NORM = {"indirect": MEAN_DAYTIME_K, "direct": AC_CAPACITY_KW}
BLOCK_SIZE = 24


def block_bootstrap_ci(d, n_boot=2000, block_size=BLOCK_SIZE, seed=42):
    """Block bootstrap CI for mean of loss differential d."""
    rng = np.random.default_rng(seed)
    n = len(d)
    starts = np.arange(0, n - block_size + 1)
    boot_means = []
    for _ in range(n_boot):
        samp = []
        while len(samp) < n:
            s = rng.choice(starts)
            samp.extend(d[s: s + block_size])
        boot_means.append(np.mean(samp[:n]))
    boot_means = np.sort(np.array(boot_means))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def effective_n(d, max_lags=50):
    """Effective sample size accounting for autocorrelation."""
    n = len(d)
    acf_vals = [np.corrcoef(d[:-k], d[k:])[0, 1] if k > 0 else 1.0 for k in range(min(max_lags, n // 4))]
    rho_sum = sum(acf_vals[1:])  # skip lag 0
    n_eff = n / (1 + 2 * max(rho_sum, 0))
    return max(n_eff, 1.0)


def min_detectable_effect(n_eff, alpha=0.05, power=0.80):
    """Minimum detectable standardised effect (Cohen's d-style) under given n_eff."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    return (z_alpha + z_beta) / np.sqrt(n_eff)


def run():
    print("[A2] Loading prediction files...")
    frames = {}
    for model, path in PRED_FILES.items():
        if path.exists():
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames[model] = df
            print(f"  {model}: {len(df):,} rows")

    ref_model = "xgb_sota"
    if ref_model not in frames:
        print("  XGB SOTA predictions not found. Exiting.")
        return

    dm_extended_rows = []
    effect_rows = []

    for tgt in ["indirect", "direct"]:
        for h in HORIZONS:
            ref = frames[ref_model]
            ref_h = ref[(ref["target"] == tgt) & (ref["horizon"] == h)].set_index("timestamp")

            for model, df in frames.items():
                if model == ref_model:
                    continue

                cmp_h = df[(df["target"] == tgt) & (df["horizon"] == h)].set_index("timestamp")
                common = ref_h.index.intersection(cmp_h.index)
                if len(common) < 30:
                    continue

                y_true   = ref_h.loc[common, "y_true"].values
                pred_ref = ref_h.loc[common, "y_pred"].values
                pred_cmp = cmp_h.loc[common, "y_pred"].values

                # Loss differential (squared error)
                d = (y_true - pred_cmp)**2 - (y_true - pred_ref)**2

                # Bootstrap CI
                ci_lo, ci_hi = block_bootstrap_ci(d)

                # Effective N and detectable effect
                n_eff  = effective_n(d)
                mde    = min_detectable_effect(n_eff)
                obs_std_effect = abs(d.mean()) / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
                pct_of_mde = obs_std_effect / mde * 100 if mde > 0 else 0.0

                # DM stat (Newey-West h=24, same as evaluate.py)
                d_bar = d.mean()
                n = len(d)
                gamma0 = np.var(d, ddof=1)
                nw_var = gamma0
                for k in range(1, 25):
                    gk = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
                    nw_var += 2 * (1 - k / 25) * gk
                dm_stat = d_bar / np.sqrt(max(nw_var, 1e-12) / n)
                p_val   = float(2 * stats.norm.cdf(-abs(dm_stat)))

                dm_extended_rows.append({
                    "target":                tgt,
                    "horizon":               h,
                    "model_a":               model,
                    "model_b":               ref_model,
                    "n_pairs":               len(common),
                    "n_eff":                 round(n_eff, 1),
                    "mean_diff":             round(float(d_bar), 6),
                    "ci_lo_95":              round(ci_lo, 6),
                    "ci_hi_95":              round(ci_hi, 6),
                    "dm_stat":               round(dm_stat, 4),
                    "p_value":               round(p_val, 4),
                    "significant_5pct":      p_val < 0.05,
                    "nw_lags":               24,
                    "block_length":          BLOCK_SIZE,
                    "obs_std_effect":        round(obs_std_effect, 4),
                    "min_detectable_effect": round(mde, 4),
                    "pct_of_mde":            round(pct_of_mde, 1),
                })

                effect_rows.append({
                    "target":      tgt, "horizon": h,
                    "model_a":     model, "model_b": ref_model,
                    "n_pairs":     len(common), "n_eff": round(n_eff, 1),
                    "obs_std_eff": round(obs_std_effect, 4),
                    "mde_80pct":   round(mde, 4),
                    "pct_of_mde":  round(pct_of_mde, 1),
                    "conclusion":  "DETECTABLE" if pct_of_mde >= 100 else "UNDERPOWERED",
                })

    df_ext  = pd.DataFrame(dm_extended_rows)
    df_eff  = pd.DataFrame(effect_rows)

    df_ext.to_csv(OUTPUTS / "dm_tests_extended.csv", index=False)
    df_eff.to_csv(OUTPUTS / "detectable_effect_summary.csv", index=False)

    print("\n[A2] Detectable-effect summary:")
    print(df_eff.to_string(index=False))

    print("\n[A2] Extended DM table (key columns):")
    cols = ["target","horizon","model_a","n_pairs","n_eff","mean_diff",
            "ci_lo_95","ci_hi_95","p_value","pct_of_mde"]
    print(df_ext[cols].to_string(index=False))

    print(f"\n[A2] Saved dm_tests_extended.csv + detectable_effect_summary.csv")


if __name__ == "__main__":
    run()

"""
run_ablation_study.py
=====================
Systematic ablation study for XGB SOTA: measures how much each feature
group contributes to performance by removing one group at a time and
re-training on the official train split (≤ 2024-01-31).

Experiments:
  baseline        : all safe features (should ≈ global_metrics.csv)
  no_rolling7d    : remove all rolling_mean_7d_same_tod features
  no_rolling3d    : remove all rolling_mean_3d_same_tod features
  no_all_rolling  : remove all rolling_mean_* features
  no_lags         : remove all lag_* features
  no_lags_rolling : remove all lag_* AND rolling_mean_* features
  no_nwp          : remove all fcst_* NWP columns
  no_clearsky     : remove all clearsky & forecast_clearsky columns
  no_time         : remove cyclical time features (hour_cos, hour_sin, etc.)
  minimal         : keep ONLY rolling_mean_7d_same_tod + fcst_dni + sun_elevation

Outputs:
  outputs/09_cv_ablation/ablation_results.csv
  reports/figures_academic/fig14_ablation_impact.png
  reports/figures_academic/fig14_ablation_impact.pdf

Run:
  python scripts/run_ablation_study.py [--quick]
"""
import sys, pathlib, warnings, time, argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true',
                    help='Run indirect D+1 only for smoke test')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for XGB (default 42)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
FE_PARQ = ROOT / '04_feature_engineering' / 'data' / 'processed' / '04_features_target_k' / 'features_1H_target_k.parquet'
XGB_MODEL_DIR = ROOT / 'outputs' / '06_xgb_sota' / 'models'
OUT_DIR = ROOT / 'outputs' / '09_cv_ablation'
FIG_DIR = ROOT / 'reports' / 'figures_academic'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Official split boundaries (must match fix_and_retrain_all.py)
TRAIN_END  = pd.Timestamp('2024-01-31 23:59', tz='UTC')
TEST_START = pd.Timestamp('2024-10-01 00:00', tz='UTC')
TEST_END   = pd.Timestamp('2025-05-31 23:59', tz='UTC')

AC_CAPACITY_KW = 3449.0
SUN_ELEV_MIN   = 5.0

TARGETS = {'indirect': 'target_k', 'direct': 'site_total_AC_kW'} \
          if not args.quick else {'indirect': 'target_k'}
HORIZONS = [1, 2, 3] if not args.quick else [1]

LEAKY_EXCL = [
    'target_k', 'site_total_AC_kW',
    's01_inv1_AC_kW', 's01_inv2_AC_kW',
    's01_inv1_DC_kW', 's01_inv2_DC_kW',
    '_daylight',
    'actual_power_kW', 'sum_inv_ac_kW', 'performance_ratio',
    'theoretical_power', 'site_loading',
    'inv1_efficiency', 'inv2_efficiency', 'site_avg_efficiency',
    'site_power_per_irr', 'site_total_DC_kW',
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('=' * 70)
print('ABLATION STUDY: XGB SOTA feature group removal')
print('=' * 70)

df = pd.read_parquet(FE_PARQ)
print(f'  Shape: {df.shape}')

# Daylight mask
if 'sun_elevation' in df.columns:
    daylight = df['sun_elevation'] > SUN_ELEV_MIN
elif '_daylight' in df.columns:
    daylight = df['_daylight'].astype(bool)
else:
    daylight = pd.Series(True, index=df.index)

# Base safe feature list
NON_FEAT = [c for c in df.columns if df[c].dtype == 'object' or c.startswith('_')]
EXCL_BASE = list(set(LEAKY_EXCL + NON_FEAT))
ALL_SAFE = sorted([
    c for c in df.columns
    if c not in EXCL_BASE and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
])
print(f'  Base safe features: {len(ALL_SAFE)}')

# Feature group classifiers
def is_rolling7d(c):  return 'rolling_mean_7d' in c
def is_rolling3d(c):  return 'rolling_mean_3d' in c
def is_rolling14d(c): return 'rolling_mean_14d' in c
def is_any_rolling(c): return 'rolling_mean' in c
def is_lag(c):        return 'lag_' in c
def is_nwp(c):        return c.startswith('fcst_')
def is_clearsky(c):   return 'clearsky' in c or 'clear_sky' in c
def is_timecyclic(c): return any(x in c for x in ['hour_cos','hour_sin','month_cos','month_sin',
                                                    'dow_cos','dow_sin','doy_cos','doy_sin'])

# Minimal feature set: best rolling7d + NWP + solar geometry
MINIMAL_KEEP = {'rolling_mean_7d_same_tod', 'fcst_dni', 'fcst_ghi', 'sun_elevation', 'hour_cos', 'dhi_clearsky'}

def get_ablation_feats(mode, base_feats):
    """Return list of features for a given ablation mode."""
    if mode == 'baseline':
        return base_feats
    elif mode == 'no_rolling7d':
        return [c for c in base_feats if not is_rolling7d(c)]
    elif mode == 'no_rolling3d':
        return [c for c in base_feats if not is_rolling3d(c)]
    elif mode == 'no_all_rolling':
        return [c for c in base_feats if not is_any_rolling(c)]
    elif mode == 'no_lags':
        return [c for c in base_feats if not is_lag(c)]
    elif mode == 'no_lags_rolling':
        return [c for c in base_feats if not is_lag(c) and not is_any_rolling(c)]
    elif mode == 'no_nwp':
        return [c for c in base_feats if not is_nwp(c)]
    elif mode == 'no_clearsky':
        return [c for c in base_feats if not is_clearsky(c)]
    elif mode == 'no_time':
        return [c for c in base_feats if not is_timecyclic(c)]
    elif mode == 'minimal':
        return [c for c in base_feats if any(k in c for k in MINIMAL_KEEP)]
    else:
        raise ValueError(f'Unknown mode: {mode}')

ABLATION_MODES = [
    'baseline',
    'no_rolling7d',
    'no_rolling3d',
    'no_all_rolling',
    'no_lags',
    'no_lags_rolling',
    'no_nwp',
    'no_clearsky',
    'no_time',
    'minimal',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_saved_hparams(tgt_label, horizon):
    path = XGB_MODEL_DIR / f'xgb_sota_{tgt_label}_D{horizon}.json'
    if not path.exists():
        return {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.7}
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    cfg = model.get_xgb_params()
    safe_keys = {'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                 'colsample_bytree', 'min_child_weight', 'reg_alpha',
                 'reg_lambda', 'gamma'}
    out = {k: v for k, v in cfg.items() if k in safe_keys and v is not None}
    out['n_estimators'] = int(model.get_booster().num_boosted_rounds())
    return out

MONO_INCREASING = ['irradiance', 'ghi', 'fcst_ghi', 'fcst_dni', 'fcst_dhi',
                   'poa_global', 'ghi_clearsky', 'sun_elevation',
                   'poa_global_forecast', 'poa_global_clearsky']

def build_monotone(features):
    return tuple(1 if any(k in f for k in MONO_INCREASING) else 0 for f in features)

def nrmse_pct(y_true, y_pred, tgt_label, norm):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return rmse, rmse / max(norm, 1e-9) * 100

# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------
results = []
print(f'\nRunning ablations: {len(ABLATION_MODES)} modes × {len(TARGETS)} targets × {len(HORIZONS)} horizons\n')

for tgt_label, tgt_col in TARGETS.items():
    for horizon in HORIZONS:
        hparams = load_saved_hparams(tgt_label, horizon)
        print(f'  [{tgt_label}_D{horizon}]')

        # Base features for this horizon (drop lag_2d for D+2, lag_3d for D+3)
        base_feats = list(ALL_SAFE)
        if horizon >= 2:
            base_feats = [c for c in base_feats if 'lag_1d' not in c]
        if horizon >= 3:
            base_feats = [c for c in base_feats if 'lag_2d' not in c]

        # Data slices
        train_mask = df.index <= TRAIN_END
        test_mask  = (df.index >= TEST_START) & (df.index <= TEST_END)
        dl_train = daylight.loc[df.index[train_mask]].values
        dl_test  = daylight.loc[df.index[test_mask]].values

        y_train = df.loc[train_mask, tgt_col]
        y_test  = df.loc[test_mask, tgt_col]

        # Normalization from training set (daytime)
        if tgt_label == 'indirect':
            yk = y_train.values[dl_train]
            norm = float(np.nanmean(yk[yk > 0.01])) if yk[yk > 0.01].size > 0 else 0.628
        else:
            norm = AC_CAPACITY_KW

        # Filter daytime for test evaluation
        yte_day = y_test.values[dl_test]
        valid_test = ~np.isnan(yte_day)
        yte_day = yte_day[valid_test]

        for mode in ABLATION_MODES:
            t0 = time.time()
            feats = get_ablation_feats(mode, base_feats)

            # Intersect with available columns
            feats = [c for c in feats if c in df.columns]
            if len(feats) < 2:
                print(f'    [{mode}] SKIP — only {len(feats)} feature(s)')
                continue

            X_train = df.loc[train_mask, feats].fillna(0)
            X_test  = df.loc[test_mask, feats].fillna(0)

            mono = build_monotone(feats)
            params = dict(hparams)
            params.update({'tree_method': 'hist', 'device': 'cpu',
                           'verbosity': 0, 'random_state': args.seed,
                           'monotone_constraints': mono})

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            y_pred = np.clip(model.predict(X_test), 0, None)

            # Daytime-filtered test
            ypred_day = y_pred[dl_test][valid_test]
            rmse, nrmse = nrmse_pct(yte_day, ypred_day, tgt_label, norm)
            r2 = r2_score(yte_day, ypred_day)
            mae = float(np.mean(np.abs(yte_day - ypred_day)))

            results.append({
                'target': tgt_label, 'horizon': horizon, 'mode': mode,
                'n_features': len(feats),
                'nRMSE_pct': nrmse, 'RMSE': rmse, 'R2': r2, 'MAE': mae,
                'norm_used': norm,
                'n_test_day': int(valid_test.sum()),
                'elapsed_s': round(time.time() - t0, 1),
            })
            print(f'    [{mode}] feats={len(feats)}, nRMSE={nrmse:.2f}%, R2={r2:.3f} ({time.time()-t0:.0f}s)')

        # Print delta vs baseline for this target/horizon
        base_row = next((r for r in results if r['target'] == tgt_label and
                         r['horizon'] == horizon and r['mode'] == 'baseline'), None)
        if base_row:
            print(f'    --- dNRMSE vs baseline ({base_row["nRMSE_pct"]:.2f}%) ---')
            for r in results:
                if r['target'] == tgt_label and r['horizon'] == horizon and r['mode'] != 'baseline':
                    delta = r['nRMSE_pct'] - base_row['nRMSE_pct']
                    sign = '+' if delta >= 0 else ''
                    print(f'       {r["mode"]:20s} {sign}{delta:.2f}pp')
        print()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df_res = pd.DataFrame(results)
df_res['seed'] = args.seed
suffix = f'_seed{args.seed}' if args.seed != 42 else ''
csv_path = OUT_DIR / f'ablation_results{suffix}.csv'
df_res.to_csv(csv_path, index=False)
print(f'Saved: {csv_path}')

# ---------------------------------------------------------------------------
# Figure 14: ablation impact bar chart
# ---------------------------------------------------------------------------
# Compute ΔnRMSE relative to baseline, for each (target, horizon, mode)
rows_fig = []
for (tgt, h), grp in df_res.groupby(['target', 'horizon']):
    base = grp[grp['mode'] == 'baseline']['nRMSE_pct'].values
    if len(base) == 0:
        continue
    base_val = base[0]
    for _, row in grp.iterrows():
        if row['mode'] == 'baseline':
            continue
        rows_fig.append({
            'target': tgt, 'horizon': h, 'mode': row['mode'],
            'delta_nRMSE': row['nRMSE_pct'] - base_val,
            'nRMSE': row['nRMSE_pct'],
            'n_features': row['n_features'],
        })
df_fig = pd.DataFrame(rows_fig)

if df_fig.empty:
    print('No ablation deltas to plot. Skipping fig14.')
else:
    # Readable mode labels
    MODE_LABELS = {
        'no_rolling7d':   'Remove rolling-7d',
        'no_rolling3d':   'Remove rolling-3d',
        'no_all_rolling': 'Remove all rolling',
        'no_lags':        'Remove lag features',
        'no_lags_rolling':'Remove lags+rolling',
        'no_nwp':         'Remove NWP (fcst_*)',
        'no_clearsky':    'Remove clearsky',
        'no_time':        'Remove time cyclics',
        'minimal':        'Minimal 2-feature set',
    }

    # Average ΔnRMSE across horizons, group by (target, mode)
    agg = df_fig.groupby(['target', 'mode'])['delta_nRMSE'].mean().reset_index()
    agg['label'] = agg['mode'].map(MODE_LABELS)

    targets = ['indirect', 'direct']
    colors = {'indirect': '#0072B2', 'direct': '#D55E00'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)

    for ax, tgt in zip(axes, targets):
        sub = agg[agg['target'] == tgt].sort_values('delta_nRMSE', ascending=False)
        if sub.empty:
            ax.set_title(f'{tgt.capitalize()} (no data)')
            continue

        bars = ax.barh(sub['label'], sub['delta_nRMSE'],
                       color=colors[tgt], edgecolor='black', alpha=0.80)

        # Color positive bars differently (performance degradation)
        for bar, val in zip(bars, sub['delta_nRMSE']):
            bar.set_facecolor('#D55E00' if val > 0 else '#009E73')

        ax.axvline(0, color='black', linewidth=1.2)
        ax.set_xlabel('dNRMSE (pp vs. full model)', fontsize=11)
        ax.set_title(f'{tgt.capitalize()} target\n(mean across D+1–D+3)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=-0.3)

        # Value labels
        for bar, val in zip(bars, sub['delta_nRMSE']):
            sign = '+' if val >= 0 else ''
            ax.text(val + (0.05 if val >= 0 else -0.05),
                    bar.get_y() + bar.get_height() / 2,
                    f'{sign}{val:.2f}pp', va='center',
                    ha='left' if val >= 0 else 'right', fontsize=8)

    fig.suptitle(
        'Fig. 14 - Ablation impact: dNRMSE from removing feature groups\n'
        'Orange bars = performance degradation; Green bars = unexpected improvement',
        fontsize=12
    )
    fig.subplots_adjust(left=0.40)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = FIG_DIR / 'fig14_ablation_impact.png'
    pdf_path = FIG_DIR / 'fig14_ablation_impact.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')

print('\nDone.')

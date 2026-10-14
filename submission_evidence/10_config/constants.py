"""
constants.py - Single Source of Truth for Solar DS Capstone (Production)
========================================================================
Import this in every script. Do NOT duplicate these values.

Usage:
    import sys
    sys.path.insert(0, 'path/to/production/00_config')
    from constants import TRAIN_END, AC_CAPACITY_KW, HORIZONS
"""

# ── Temporal Splits ────────────────────────────────────────────────────────
TRAIN_END   = "2024-01-31 23:59"
VAL_START   = "2024-02-01 00:00"
VAL_END     = "2024-09-30 23:59"
TEST_START  = "2024-10-01 00:00"
TEST_END    = "2025-05-31 23:59"

# ── Site Parameters ────────────────────────────────────────────────────────
AC_CAPACITY_KW  = 3449.0   # Nameplate AC capacity [kW]
SITE_LAT        = 31.5     # Approximate latitude [deg N]
SITE_LON        = 34.9     # Approximate longitude [deg E]

# ── Daytime Filter ─────────────────────────────────────────────────────────
SUN_ELEV_MIN = 5.0         # Minimum sun elevation for "daytime" [degrees]
# All metrics (nRMSE, MAE, R2) are computed on daytime rows only.

# ── Normalisation Denominators ─────────────────────────────────────────────
# Direct target:   nRMSE = RMSE / AC_CAPACITY_KW * 100   (% of nameplate)
# Indirect target: nRMSE = RMSE / MEAN_DAYTIME_K * 100   (CvRMSE)
MEAN_DAYTIME_K = 0.6279    # mean(target_k | sun_elev > 5deg) on training set

# ── Forecast Horizons ──────────────────────────────────────────────────────
HORIZONS = [1, 2, 3]       # D+1, D+2, D+3

# ── Target Columns ─────────────────────────────────────────────────────────
# target_k is a clear-sky PERFORMANCE INDEX (not the atmospheric clearness index Kt).
# Definition: target_k = AC_kW_observed / (GHI_clearsky * AC_CAPACITY_KW / 1000)
# Range [0, 1.2]; night hours (sun_elev <= 5°) set to 0.
# Inverse: AC_kW = target_k * GHI_clearsky * AC_CAPACITY_KW / 1000  (exact)
TARGET_INDIRECT = "target_k"           # Clear-sky performance index, dimensionless [0, 1.2]
TARGET_DIRECT   = "site_total_AC_kW"   # Absolute AC power [kW]

# ── Seed Protocol (for reproducibility and seed-variability reporting) ──────
RANDOM_SEEDS = [42, 123, 777]          # Three seeds for mean ± std reporting

# ── Leakage Rules (conservative) ──────────────────────────────────────────
# lag_1d = yesterday's data -> always available at 00:00 issuance.
# The following conservative filtering is applied (stricter than necessary):
#   D+1: all lags available
#   D+2: lag_1d excluded
#   D+3: lag_1d + lag_2d excluded
LAG_DROP = {
    1: [],
    2: ["lag_1d"],
    3: ["lag_1d", "lag_2d"],
}

# ── Model Stopping Criterion ───────────────────────────────────────────────
# If a deep model's nRMSE > min(XGB, Persistence) + threshold, it is marked
# as "failed stopping criterion" and noted as proof-of-concept only.
DEEP_MODEL_DELTA_THRESHOLD = 0.5      # percentage points

# ── Bootstrap CI Parameters ───────────────────────────────────────────────
BOOTSTRAP_BLOCK_SIZE = 24             # hours (preserves daily autocorrelation)
BOOTSTRAP_REPS       = 1000

# ── Diebold-Mariano Test ───────────────────────────────────────────────────
DM_LOSS = "squared"                   # MSE-based loss function
# Correction: Harvey-Leybourne-Newbold (HLN), Bonferroni multiple testing


def norm_factor(target_label: str) -> float:
    """Return the nRMSE denominator for a given target formulation."""
    if target_label == "indirect":
        return MEAN_DAYTIME_K
    elif target_label == "direct":
        return AC_CAPACITY_KW
    raise ValueError(f"Unknown target: {target_label!r}. Use 'direct' or 'indirect'.")


def get_features(all_features: list, horizon: int) -> list:
    """Remove lagged features that would leak future information for a given horizon."""
    drop_patterns = LAG_DROP.get(horizon, [])
    return [f for f in all_features if not any(p in f for p in drop_patterns)]


if __name__ == "__main__":
    print("constants.py self-test")
    print(f"  TRAIN_END        : {TRAIN_END}")
    print(f"  TEST_START       : {TEST_START}")
    print(f"  AC_CAPACITY_KW   : {AC_CAPACITY_KW}")
    print(f"  MEAN_DAYTIME_K   : {MEAN_DAYTIME_K}")
    print(f"  norm_factor(direct)   = {norm_factor('direct'):.4f}")
    print(f"  norm_factor(indirect) = {norm_factor('indirect'):.4f}")
    feats = ["fcst_ghi", "site_total_AC_kW_lag_1d", "site_total_AC_kW_lag_2d"]
    for h in HORIZONS:
        print(f"  D+{h} features: {get_features(feats, h)}")

"""Single source of truth for project constants (PROJECT_MASTER.md §6).

Import from here; never duplicate these values in scripts or notebooks.
Site coordinates are not here: they live in configs/site.private.yaml (see site.py, decision D-05).
"""

import re
from collections.abc import Iterable

import yaml

from solarpv.config.paths import CONFIGS_DIR

# ── Time (F-01, D-16) ──────────────────────────────────────────────────────
TIMEZONE_LOCAL = "Asia/Jerusalem"  # raw t_stamp is local clock time with DST
ISSUANCE_LOCAL_TIME = "00:00"  # T0: forecasts are issued at local midnight (A-01)
DATA_START_UTC = "2021-08-31 21:00"  # first raw minute 2021-09-01 00:00 IDT
DATA_END_UTC = "2025-05-31 20:59"  # last raw minute 2025-05-31 23:59 IDT
N_MINUTES_EXPECTED = 1_971_360  # 1,369 days x 1,440
N_HOURS_EXPECTED = 32_856  # 1,369 days x 24
HOURLY_MIN_COVERAGE = 45  # minutes of valid data needed for an hourly mean (A-09)

# ── Splits, UTC, inclusive (§6.8) ──────────────────────────────────────────
TRAIN_END = "2024-01-31 23:59"
VAL_START = "2024-02-01 00:00"
VAL_END = "2024-09-30 23:59"
TEST_START = "2024-10-01 00:00"
TEST_END = "2025-05-31 23:59"

# ── Plant ──────────────────────────────────────────────────────────────────
AC_CAPACITY_KW = 3449.0  # nameplate AC; observed max is ~3,000 kW (C-04, A-08)

# ── Targets (§6.5, fix F-06) ───────────────────────────────────────────────
TARGET_INDIRECT = "target_k"
TARGET_DIRECT = "site_total_AC_kW"
THEO_MIN_KW = 50.0  # below this clear-sky power the index k is undefined (k_valid=False)
K_SUSPECT_MAX = 1.5  # k above this is flagged as a probable data error; k is never clipped
MEAN_DAYTIME_K_LEGACY = 0.6279  # old (buggy) pipeline; only for the evaluate.py fixture (T-15)
DERIVED_CONSTANTS_FILE = CONFIGS_DIR / "derived_constants.yaml"  # written by stage 6

# ── Horizons and leakage rules (§6.6) ──────────────────────────────────────
HORIZONS = (1, 2, 3)
LAG_DROP = {1: (), 2: ("lag_1d",), 3: ("lag_1d", "lag_2d")}

# ── NWP from the GFS archive (§6.12, D-12) ──────────────────────────────────
GFS_RUN_HOUR_UTC = 12  # 12Z run of the previous UTC day
NWP_MIN_AVAILABILITY_HOURS = 6  # a run may be used only if init + 6h <= T0 (A-13)

# ── Evaluation (§6.9) ──────────────────────────────────────────────────────
SUN_ELEV_MIN = 5.0  # daytime = sun elevation above 5 deg ...
EVAL_SUN_OFFSET_MIN = 30  # ... evaluated at the middle of the hour (F-02)
BOOTSTRAP_BLOCK = 24
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42
DM_NW_LAGS = 24
SIGNIFICANCE_LEVEL = 0.05

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEEDS = (42, 123, 777)
DEV_SEED = 42
DEV_OPTUNA_TRIALS = 20  # D-17
FINAL_OPTUNA_TRIALS = 60


def mean_daytime_k() -> float:
    """Indirect nRMSE denominator, computed on the corrected training set in stage 6."""
    if not DERIVED_CONSTANTS_FILE.exists():
        raise FileNotFoundError(
            f"{DERIVED_CONSTANTS_FILE.name} not found: MEAN_DAYTIME_K is produced by stage 6. "
            "Use MEAN_DAYTIME_K_LEGACY only for the old-predictions fixture."
        )
    with open(DERIVED_CONSTANTS_FILE, encoding="utf-8") as f:
        return float(yaml.safe_load(f)["mean_daytime_k"])


def norm_factor(target_label: str, mean_k: float | None = None) -> float:
    """nRMSE denominator: AC capacity for 'direct', mean daytime k for 'indirect'."""
    if target_label == "direct":
        return AC_CAPACITY_KW
    if target_label == "indirect":
        return mean_daytime_k() if mean_k is None else mean_k
    raise ValueError(f"Unknown target: {target_label!r}. Use 'direct' or 'indirect'.")


def get_features(features: Iterable[str], horizon: int) -> list[str]:
    """Drop lag features that are not known at issuance for the given horizon.

    Matches the lag token exactly at the end of the name, so 'lag_1d' never removes 'lag_14d'.
    """
    if horizon not in LAG_DROP:
        raise ValueError(f"Unknown horizon {horizon}; expected one of {HORIZONS}")
    drop = [re.compile(rf"_{tag}$") for tag in LAG_DROP[horizon]]
    return [f for f in features if not any(p.search(f) for p in drop)]


if __name__ == "__main__":
    print("constants self-test")
    print(f"  data UTC        : {DATA_START_UTC} .. {DATA_END_UTC} ({N_HOURS_EXPECTED} h)")
    print(f"  train end       : {TRAIN_END}")
    print(f"  test            : {TEST_START} .. {TEST_END}")
    print(f"  capacity        : {AC_CAPACITY_KW} kW")
    print(f"  norm(direct)    : {norm_factor('direct')}")
    feats = ["site_total_AC_kW_lag_1d", "site_total_AC_kW_lag_2d", "site_total_AC_kW_lag_14d"]
    for h in HORIZONS:
        print(f"  D+{h} features  : {get_features(feats, h)}")

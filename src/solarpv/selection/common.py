"""Data, folds, score and candidate groups of the stage-7 selection (docs/stage7_plan.md §3).

Every selection statistic is computed on train rows only (P1): `train_table` refuses any other
split. One score drives every decision (P6): nRMSE in kW, % of the AC capacity, on the held-out
daytime hours of expanding date folds inside train (P11); for the indirect target the predicted
k is multiplied by theo_kw first, exactly as in the D-12 value check.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from solarpv.config.constants import AC_CAPACITY_KW
from solarpv.features import build
from solarpv.features.contract import FEATURE_GROUPS, GFS_PREFIX, features_for_horizon
from solarpv.features.leakage import assert_no_leakage
from solarpv.nwp import check as nwp_check

TARGETS = ("indirect", "direct")
CORE = ("solar_zenith", "solar_azimuth", "aoi", "fcst_gfs_ghi", "fcst_gfs_poa", "fcst_gfs_t2m",
        "fcst_gfs_tcc", "hour_sin", "hour_cos", "doy_sin", "doy_cos")
# the core of the January base notebooks, mapped: fcst_ghi -> fcst_gfs_ghi; fcst_dni and
# fcst_dhi -> fcst_gfs_poa (GFS gives neither); fcst_temp_2m -> fcst_gfs_t2m;
# fcst_cloud_cover -> fcst_gfs_tcc
SELECTION_XGB = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8,
                 "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0.1,
                 "reg_alpha": 0.1, "reg_lambda": 1.0, "tree_method": "hist", "n_jobs": -1,
                 "verbosity": 0}  # the base notebooks' RFE model
SEED = 42
N_FOLDS = 5
GAP_DAYS = 3
THEO_MIN_KW = 50.0  # L-04
K_PREFIX = "target_k_"
PREVDAY = "irradiance_clear_index_prevday"
LOCAL_TZ = "Asia/Jerusalem"


# ── tables ────────────────────────────────────────────────────────────────


def train_table(horizon: int) -> pd.DataFrame:
    """The stage-6 table of the horizon joined with its GFS table, train rows only."""
    table = nwp_check.load(horizon)
    table = table[table["split"] == "train"]
    assert (table["split"] == "train").all()
    return table


def check_train_only(table: pd.DataFrame) -> None:
    if not (table["split"] == "train").all():
        raise ValueError("stage-7 selection reads train rows only (P1)")


def gfs_candidates(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith(GFS_PREFIX)] + [PREVDAY]


def pool(table: pd.DataFrame, horizon: int, extra: tuple[str, ...] = ()) -> list[str]:
    """Allow-list (P2): the contract of the horizon, the admitted candidates, optional extras
    (the k history). Anything else, or anything the leakage rules reject, raises."""
    feats = features_for_horizon(horizon) + gfs_candidates(table) + list(extra)
    allowed = set(features_for_horizon(horizon)) | set(gfs_candidates(table)) | {
        c for c in extra if c.startswith(K_PREFIX)}
    bad = [f for f in feats if f not in allowed or f not in table.columns]
    if bad:
        raise ValueError(f"not in the allow-list or not in the table: {bad}")
    assert_no_leakage(feats)
    return feats


def group(feature: str) -> str:
    for name, members in FEATURE_GROUPS.items():
        if feature in members:
            return name
    if feature.startswith(GFS_PREFIX):
        return "nwp_gfs"
    if feature.startswith(K_PREFIX):
        return "k_history"
    if feature == PREVDAY:
        return "history_other"
    raise KeyError(feature)


AVAILABILITY = {"clearsky": "FORECAST_SAFE", "time_encodings": "FORECAST_SAFE",
                "other": "FORECAST_SAFE", "rolling_stats": "FORECAST_SAFE",
                "lag_power": "HORIZON_DEPENDENT", "k_history": "HORIZON_DEPENDENT",
                "history_other": "HORIZON_DEPENDENT", "nwp_gfs": "PIPELINE_DEPENDENT"}


def availability(feature: str) -> str:
    """The operational classes of the report (§3.6.6)."""
    return AVAILABILITY[group(feature)]


# ── k history (S0) ────────────────────────────────────────────────────────


def k_history(table: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """target_k_lag_<d>d and target_k_rolling_mean_<N>d_same_tod for D+horizon, built with the
    stage-6 rules (B6): k only on valid, not op-limited hours with theo > 50 kW; the DST guard of
    `past_value`; the lags kept at the horizon; past-only imputation. The table's hourly index must
    be complete (it is a contiguous split)."""
    expected = pd.date_range(table.index[0], table.index[-1], freq="h")
    if not table.index.equals(expected):
        raise ValueError("k history needs a complete hourly index")
    ok = table["k_valid"] & (table["theo_kw"] > THEO_MIN_KW) & ~table["op_limited_flag"]
    k = table["target_k"].where(ok)
    t0 = build.issuance_utc(table.index, horizon)
    cols = {f"{K_PREFIX}lag_{d}d": build.past_value(k, d, t0) for d in build.kept_lags(horizon)}
    for n in build.ROLLING_WINDOWS_DAYS:
        stack = np.vstack([build.past_value(k, d, t0) for d in range(horizon, horizon + n)])
        count = np.isfinite(stack).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"{K_PREFIX}rolling_mean_{n}d_same_tod"] = np.where(
                count > 0, np.nansum(stack, axis=0) / count, np.nan)
    filled, _ = build.impute_same_hour(pd.DataFrame(cols, index=table.index))
    return filled


# ── the problem: rows, target, truth ──────────────────────────────────────


@dataclass
class Problem:
    target: str
    X: pd.DataFrame
    y: pd.Series        # training target: k (indirect) or kW (direct)
    fit: pd.Series      # rows a model may train on
    score: pd.Series    # rows a model is scored on (daytime, AC present)
    ac: pd.Series       # truth in kW
    theo: pd.Series     # theo_kw (indirect only)
    day: pd.Series      # local calendar day of each row


def problem(table: pd.DataFrame, target: str) -> Problem:
    """Training rows as in stage 6 and the D-12 check (D-11, D-22, L-04); scoring rows are the
    daytime hours with AC, as in the evaluation."""
    base = ~table["warmup"] & ~table["op_limited_flag"]
    if target == "indirect":
        fit = base & table["k_valid"] & (table["theo_kw"] > THEO_MIN_KW)
        y = table["target_k"]
    elif target == "direct":
        fit = base & table["site_total_AC_kW"].notna()
        y = table["site_total_AC_kW"]
    else:
        raise ValueError(target)
    score = ~table["warmup"] & table["daytime"] & table["site_total_AC_kW"].notna()
    day = pd.Series(table.index.tz_convert(LOCAL_TZ).normalize().tz_localize(None),
                    index=table.index)
    return Problem(target, table, y, fit, score, table["site_total_AC_kW"], table["theo_kw"],
                   day)


def to_kw(problem: Problem, pred: np.ndarray, rows: pd.Series) -> np.ndarray:
    if problem.target == "indirect":
        return np.clip(np.clip(pred, 0, None) * problem.theo[rows].to_numpy(), 0, AC_CAPACITY_KW)
    return pred


def date_folds(days: pd.Series, n_folds: int = N_FOLDS, gap_days: int = GAP_DAYS
               ) -> list[tuple[pd.Series, pd.Series]]:
    """Expanding folds by date (P11): the days are cut into n_folds + 1 contiguous blocks; fold i
    trains on blocks 0..i without its last `gap_days` days and is scored on block i + 1."""
    unique = np.unique(days.to_numpy())
    blocks = np.array_split(unique, n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_days = np.concatenate(blocks[: i + 1])
        train_days = train_days[train_days < blocks[i + 1][0] - np.timedelta64(gap_days, "D")]
        folds.append((days.isin(train_days), days.isin(blocks[i + 1])))
    return folds


def model(params: dict, seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**{**params, "random_state": seed})


def cv_score(problem: Problem, feats: list[str], params: dict = SELECTION_XGB, seed: int = SEED,
             with_shap: bool = True, folds=None) -> tuple[float, pd.Series | None]:
    """Pooled nRMSE in kW (% of capacity) over the held-out daytime hours of the date folds, and
    the mean |SHAP| of every feature on those held-out hours (P6, P7)."""
    folds = folds if folds is not None else date_folds(problem.day)
    errors, shap_sum, n = [], np.zeros(len(feats)), 0
    for train_rows, test_rows in folds:
        fit_rows = train_rows & problem.fit
        m = model(params, seed)
        m.fit(problem.X.loc[fit_rows, feats], problem.y[fit_rows])
        rows = test_rows & problem.score
        X = problem.X.loc[rows, feats]
        errors.append(to_kw(problem, m.predict(X), rows) - problem.ac[rows].to_numpy())
        if with_shap:
            contrib = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
            shap_sum += np.abs(contrib).sum(axis=0)
            n += len(X)
    err = np.concatenate(errors)
    nrmse = float(np.sqrt(np.mean(err ** 2)) / AC_CAPACITY_KW * 100)
    return nrmse, (pd.Series(shap_sum / n, index=feats) if with_shap else None)

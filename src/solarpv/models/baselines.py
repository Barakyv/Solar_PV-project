"""Reference forecasts (PROJECT_MASTER.md §6.9, §6.11; docs/stage8b_plan.md §2).

Three references, computed from the truth table by one code path for any split:
- persistence(h): the target at the same UTC hour h days earlier, if that hour ended by the
  issuance time T0 (the DST guard of stage 6, D-16). Otherwise the median of the same hour
  over the previous 30 days, from earlier days only (the OP-38 rule). It is naive:
  operational limits are not masked. It is the reference of the skill score;
- climatology: the train mean of the target per (UTC month, UTC hour), over hours with a known
  target and normal operation, i.e. the rows a model learns from (D-22);
- clear sky x MEAN_K:
  - direct: clip(theo_kw x MEAN_DAYTIME_K, 0, capacity);
  - indirect: MEAN_DAYTIME_K where theo_kw > 50 kW, else 0.

Climatology and clear sky do not depend on the horizon. The clear-sky indirect forecast
converted to kW equals its direct forecast exactly, which checks the route code (stage 8B, T-39).
"""

import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, HORIZONS, THEO_MIN_KW
from solarpv.features.build import IMPUTE_WINDOW_DAYS, impute_same_hour, issuance_utc, past_value

TARGETS = ("indirect", "direct")
REFERENCES = ("persistence", "climatology", "clearsky_k")


def _check_grid(series: pd.Series) -> None:
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError("series must have a tz-aware UTC index")
    if len(series) > 1 and not (series.index[1:] - series.index[:-1] == pd.Timedelta("1h")).all():
        raise ValueError("series must be on a complete hourly grid")


def persistence(truth: pd.Series, horizon: int,
                window_days: int = IMPUTE_WINDOW_DAYS) -> pd.Series:
    """Horizon-matched persistence on the full hourly grid of `truth` (NaN where nothing earlier
    exists, i.e. the first days of the series)."""
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon}")
    _check_grid(truth)
    t0 = issuance_utc(truth.index, horizon)
    raw = pd.DataFrame({"p": past_value(truth, horizon, t0)}, index=truth.index)
    filled, _ = impute_same_hour(raw, window_days)
    return filled["p"]


def climatology(truth: pd.Series, train_rows: pd.Series) -> pd.Series:
    """Train mean per (UTC month, UTC hour), mapped onto every hour of `truth`."""
    _check_grid(truth)
    use = train_rows.reindex(truth.index, fill_value=False).to_numpy() & truth.notna().to_numpy()
    if not use.any():
        raise ValueError("no training rows for the climatology")
    key = pd.MultiIndex.from_arrays([truth.index.month, truth.index.hour])
    table = truth[use].groupby([truth.index.month[use], truth.index.hour[use]]).mean()
    out = pd.Series(table.reindex(key).to_numpy(), index=truth.index)
    if out.isna().any():
        raise ValueError("climatology has (month, hour) cells without training data")
    return out


def clearsky_k(theo_kw: pd.Series, mean_k: float, target: str) -> pd.Series:
    """The clear-sky reference: MEAN_DAYTIME_K times the theoretical power (or k itself)."""
    if target == "direct":
        return (theo_kw * mean_k).clip(0.0, AC_CAPACITY_KW)
    if target == "indirect":
        return pd.Series(np.where(theo_kw > THEO_MIN_KW, mean_k, 0.0), index=theo_kw.index)
    raise ValueError(f"unknown target {target!r}")


def reference_predictions(truth: pd.DataFrame, split: str, mean_k: float,
                          horizons=HORIZONS) -> pd.DataFrame:
    """The three references for every hour of `split`, in the prediction-file contract (§6.10).

    `truth` is the table of solarpv.evaluation.truth (columns indirect, direct, theo_kw, split,
    op_limited_flag, ...). Rows without truth are kept; the evaluation drops them for all models.
    """
    rows = truth["split"] == split
    if not rows.any():
        raise ValueError(f"no rows in split {split!r}")
    train_rows = (truth["split"] == "train") & ~truth["op_limited_flag"].astype(bool)
    frames = []
    for target in TARGETS:
        series = truth[target]
        fixed = {"climatology": climatology(series, train_rows),
                 "clearsky_k": clearsky_k(truth["theo_kw"], mean_k, target)}
        for h in horizons:
            forecasts = {"persistence": persistence(series, h), **fixed}
            for model, pred in forecasts.items():
                frames.append(pd.DataFrame({
                    "timestamp": truth.index[rows], "y_true": series[rows].to_numpy(),
                    "y_pred": pred[rows].to_numpy(), "target": target, "horizon": h,
                    "model": model, "seed": -1, "split": split}))
    return pd.concat(frames, ignore_index=True)


def to_prediction_frame(
    pred: pd.Series, truth: pd.Series, target: str, horizon: int, model: str, seed=None
) -> pd.DataFrame:
    """Predictions in the file contract of §6.10 (only rows where both exist). Used by the
    fixture mode, where the truth is the prediction file itself."""
    common = pred.dropna().index.intersection(truth.index)
    return pd.DataFrame(
        {
            "timestamp": common,
            "y_true": truth.loc[common].to_numpy(),
            "y_pred": pred.loc[common].to_numpy(),
            "target": target,
            "horizon": horizon,
            "model": model,
            "seed": seed,
            "split": "test",
        }
    )

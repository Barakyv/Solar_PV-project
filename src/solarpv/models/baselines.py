"""Reference forecasts (PROJECT_MASTER.md §6.11).

Implemented now: persistence, needed by the evaluation for skill scores.
Stage 9 adds climatology (train mean per month x hour) and clear-sky x mean daytime k.
"""

import pandas as pd

from solarpv.config.constants import HORIZONS


def persistence(truth: pd.Series, horizon: int) -> pd.Series:
    """Forecast for hour t = observed value at t - 24*h hours (same hour, last day known at T0).

    `truth` is an hourly series on a tz-aware UTC index. Hours whose source value is missing
    are dropped.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon}")
    if truth.index.tz is None or str(truth.index.tz) != "UTC":
        raise ValueError("truth must have a tz-aware UTC index")
    lag = pd.Timedelta(hours=24 * horizon)
    source = truth.copy()
    source.index = source.index + lag
    return source.reindex(truth.index).dropna()


def to_prediction_frame(
    pred: pd.Series, truth: pd.Series, target: str, horizon: int, model: str, seed=None
) -> pd.DataFrame:
    """Predictions in the file contract of §6.10 (only rows where both exist)."""
    common = pred.index.intersection(truth.index)
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

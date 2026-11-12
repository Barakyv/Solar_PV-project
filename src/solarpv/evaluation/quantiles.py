"""Ordered quantiles for every use of a forecast distribution (D-33, X53).

The quantile heads of the TFT are separate outputs of one layer, and nothing orders them: on the
validation hours of the development run they are out of order on most direct-route hours of two
TFTs (X53). Where the quantiles are used as a distribution (intervals, coverage, the quantile
loss, the potential skill of L-14), they are sorted per row first: the monotone rearrangement.

For any truth value y, sorting never raises the summed quantile loss. With
rho_t(u) = t * u + max(-u, 0), the sum over the quantiles is
sum_k t_k * (y - q_k) + sum_k max(q_k - y, 0); the second sum does not depend on the order, and
the first is smallest when the q_k rise with the t_k (the rearrangement inequality).

The point forecast is not touched: it stays the model's 0.5 head, as in the submitted study, and
the quantile files keep the raw model output. Sorting happens where the quantiles are read.
"""

import numpy as np
import pandas as pd

PREFIX = "q"


def quantile_columns(frame: pd.DataFrame) -> list[str]:
    """The quantile columns ("q0.1", "q0.5", ...) ordered by their level."""
    cols = [c for c in frame.columns if c.startswith(PREFIX) and _is_level(c[len(PREFIX):])]
    if not cols:
        raise ValueError("no quantile columns (q<level>) in the frame")
    return sorted(cols, key=level)


def _is_level(text: str) -> bool:
    try:
        return 0 < float(text) < 1
    except ValueError:
        return False


def level(column: str) -> float:
    return float(column[len(PREFIX):])


def crossed(frame: pd.DataFrame) -> pd.Series:
    """True on the rows whose quantiles are out of order."""
    v = frame[quantile_columns(frame)].to_numpy(dtype=float)
    return pd.Series((np.diff(v, axis=1) < 0).any(axis=1), index=frame.index)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy with the quantile columns sorted per row; every other column is unchanged."""
    cols = quantile_columns(frame)
    out = frame.copy()
    out[cols] = np.sort(frame[cols].to_numpy(dtype=float), axis=1)
    return out


def pinball(y, frame: pd.DataFrame) -> float:
    """Mean quantile loss over the rows and the quantile levels."""
    cols = quantile_columns(frame)
    taus = np.array([level(c) for c in cols])
    d = np.asarray(y, dtype=float)[:, None] - frame[cols].to_numpy(dtype=float)
    return float(np.mean(np.maximum(taus * d, (taus - 1) * d)))


def coverage(y, frame: pd.DataFrame, low: float = 0.1, high: float = 0.9) -> float:
    """Share of the rows whose truth lies between the `low` and `high` quantiles."""
    lo, hi = frame[f"{PREFIX}{low}"].to_numpy(float), frame[f"{PREFIX}{high}"].to_numpy(float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))

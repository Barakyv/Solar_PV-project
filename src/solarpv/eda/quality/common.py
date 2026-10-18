"""Result type and plotting helpers shared by the stage-3b analyses."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from solarpv.config.constants import TEST_START, TRAIN_END, VAL_START

SPLIT_COLOURS = {"train": "#2e7d32", "val": "#ef6c00", "test": "#1a237e"}
RULE_COLOURS = {"R1_range": "#c62828", "R2_offset": "#8d6e63", "R3_frozen": "#1565c0",
                "R4_night_zero": "#9e9e9e", "R5_night_stale": "#6a1b9a", "R6_dc_dead": "#f9a825",
                "R7_scale_fault": "#00838f"}


@dataclass
class Result:
    """One analysis: a table, a figure and the facts its notebook note relies on."""

    name: str
    title: str
    table: pd.DataFrame | None = None
    figure: Figure | None = None
    facts: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name.split("_")[0]

    def show(self, rows: int = 40) -> None:
        """Display in a notebook: the table (first `rows` rows) and the figure."""
        from IPython.display import display
        if self.table is not None:
            with pd.option_context("display.max_columns", 40, "display.width", 200):
                display(self.table if len(self.table) <= rows else self.table.head(rows))
        if self.figure is not None:
            display(self.figure)
            plt.close(self.figure)  # otherwise the inline backend shows it a second time


def py(value):
    """JSON-safe scalar (numpy -> python, NaN -> None, floats rounded)."""
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("UTC").tz_localize(None)


def month_index(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return naive(index).to_period("M")


def shade_splits(ax, alpha: float = 0.07) -> None:
    import matplotlib.dates as mdates
    lo, hi = ax.get_xlim()
    bounds = {"train": (None, TRAIN_END), "val": (VAL_START, TEST_START),
              "test": (TEST_START, None)}
    for s, (a, b) in bounds.items():
        x0 = mdates.date2num(pd.Timestamp(a)) if a else lo
        x1 = mdates.date2num(pd.Timestamp(b)) if b else hi
        ax.axvspan(x0, x1, color=SPLIT_COLOURS[s], alpha=alpha, lw=0)
    ax.set_xlim(lo, hi)


def heatmap(ax, frame: pd.DataFrame, label: str, vmax: float | None = None, cmap: str = "viridis",
            fmt_months: bool = True):
    """Rows = frame.index, columns = months (PeriodIndex)."""
    data = frame.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0,
                   vmax=vmax if vmax is not None else np.nanmax(data))
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index, fontsize=7)
    cols = list(frame.columns)
    if fmt_months:
        ticks = [i for i, c in enumerate(cols) if str(c).endswith(("-01", "-07"))]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(cols[i]) for i in ticks], fontsize=7, rotation=45, ha="right")
    plt.colorbar(im, ax=ax, label=label, fraction=0.025, pad=0.01)
    return im


def dilate(mask: np.ndarray, minutes: int) -> np.ndarray:
    """mask widened by `minutes` on both sides."""
    if minutes <= 0:
        return mask
    return (pd.Series(mask.astype(np.int8)).rolling(2 * minutes + 1, center=True, min_periods=1)
            .max().to_numpy().astype(bool))


def segments(mask: np.ndarray) -> np.ndarray:
    """(start, stop) index pairs of the True stretches of mask (stop exclusive)."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return edges.reshape(-1, 2)


def sample(mask: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Indices of at most n True positions of mask (reproducible)."""
    idx = np.flatnonzero(mask)
    if len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))

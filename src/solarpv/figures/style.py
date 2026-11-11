"""One style for every figure of the figure book (docs/figures_plan.md §3, rule 5).

Okabe-Ito colours, colour always paired with a line style, marker or hatch, one value axis per
panel, white background, 200 DPI PNG.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from solarpv.config.constants import TEST_START, TRAIN_END, VAL_START  # noqa: E402

C = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7",
    "grey": "#8C8C8C", "light": "#E8E8E8",
}
SPLIT_COLOURS = {"train": C["blue"], "validation": C["orange"], "test": C["grey"]}
DPI = 200

RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False, "axes.titleweight": "normal",
}


def subplots(nrows=1, ncols=1, width=7.2, height=3.2, **kw):
    return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kw)


def shade_splits(ax, alpha=0.07, labels=True):
    """Shade train, validation and test on a date axis (UTC-naive datetimes)."""
    edges = [pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)]
    lo, hi = mdates.num2date(ax.get_xlim()[0]).replace(tzinfo=None), \
        mdates.num2date(ax.get_xlim()[1]).replace(tzinfo=None)
    spans = [("train", lo, edges[0]), ("validation", edges[0], edges[1]), ("test", edges[1], hi)]
    for name, a, b in spans:
        if b <= a:
            continue
        ax.axvspan(a, b, color=SPLIT_COLOURS[name], alpha=alpha, lw=0, zorder=0)
        if labels:
            ax.text(a + (b - a) / 2, 1.0, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=SPLIT_COLOURS[name])
    ax.set_xlim(lo, hi)


def box(ax, x, y, w, h, text, face=C["light"], edge=C["grey"], size=8, weight="normal",
        color="black"):
    """A labelled rounded box in axes-fraction or data coordinates (whatever ax uses)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=face, ec=edge, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=C["grey"]):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, shrinkA=0, shrinkB=0))


def blank(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def month_axis(ax, interval=6):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


TRAIN_END_TS = pd.Timestamp(TRAIN_END)

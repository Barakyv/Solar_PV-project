# ruff: noqa: E501  (registered texts; see s11_part_b)
"""Stage 11, part D: the quantile outputs of the final run (docs/stage11_plan.md section 7, OP-69).

S11-21 reads `outputs/11_analyses/quantile_crossing.csv` (made by `solarpv.figures.analyses11` from the
frozen quantile files) and draws every model in one layout and one style (rule 9): the BiLSTM-Opt and
the three TFTs write quantiles, XGB-Opt and PatchTST forecast a point only and are named as such.
Validation stands beside test in the same view (S10-03). Nothing here trains or scores anything.
"""

import numpy as np

from solarpv.figures import _data as D
from solarpv.figures.analyses11 import OUT
from solarpv.figures.registry import register
from solarpv.figures.s09_models import MODEL_STYLE, MODELS, NAMES
from solarpv.figures.style import subplots

TABLE = f"{OUT}/quantile_crossing.csv"
TFTS = ("tft_96", "tft_168", "tft")
SPLIT_NAME = {"val": "validation", "test": "test"}
PANELS = (("crossed_pct", "hours with quantiles out of order (%)"),
          ("point_not_median_pct", "hours with the point not the sorted median (%)"),
          ("gap_mean_kw", "distance point to sorted median (kW)"))


def _pooled_gap(g) -> float:
    """Mean distance between the point and the sorted median over the hours where they differ."""
    off = g["point_not_median_pct"] / 100 * g["n"]
    return float((g["gap_mean_kw"] * off).sum() / off.sum()) if off.sum() else 0.0


@register(
    "S11-21", "11", "quantile_crossing_final", "The quantile outputs of the final run",
    "How often are the quantiles of the final run out of order, on validation and on test, and what does "
    "sorting them cost the point forecast?",
    "Rows: validation and test. Left: the share of the scored hours whose quantiles are out of order, per "
    "model and route (indirect solid, direct hatched), the mean over the horizons and the seeds. Middle: the "
    "share of the hours on which the 0.5 quantile, the point forecast, is not the median of the sorted "
    "quantiles. Right: the mean distance between the two on those hours, in kW. XGB-Opt and PatchTST forecast "
    "a point only.",
    "The quantiles of the TFTs are out of order on {v_lo:.1f} to {v_hi:.1f} % of the hours of a model and "
    "route on validation and {t_lo:.1f} to {t_hi:.1f} % on test, most often for TFT-96 on the direct route; "
    "those of the BiLSTM-Opt never are. Sorting moves the median away from the point forecast on up to "
    "{p_v:.1f} % of the hours on validation and {p_t:.1f} % on test, by {g_v:.1f} and {g_t:.1f} kW on "
    "average on those hours, and it lowers the quantile loss in every cell of the TFTs, by at most {drop:.1f} %.",
    "Section 6.5 of report v7: the point forecast stays the 0.5 head and is the only output scored, and the "
    "quantiles are sorted wherever they are read as a distribution (D-33).",
    (TABLE, f"{OUT}/manifest.json"))
def s11_quantile_crossing():
    t = D.csv(TABLE)
    assert (t[t["model"] == "lstm_sota"]["crossed_pct"] == 0).all()
    assert (t["pinball_sorted_kw"] <= t["pinball_raw_kw"] + 1e-9).all()
    m = t.groupby(["model", "target", "split"])[[p for p, _ in PANELS[:2]]].mean()
    names = [n for _, n, _ in MODELS]
    y = np.arange(len(names))
    fig, axes = subplots(2, 3, width=9.6, height=5.0, sharey=True)
    facts = {}
    for i, split in enumerate(("val", "test")):
        s = t[t["split"] == split]
        for j, (col, label) in enumerate(PANELS):
            ax = axes[i, j]
            for k, (target, hatch) in enumerate((("indirect", ""), ("direct", "//"))):
                vals = []
                for model in names:
                    g = s[(s["model"] == model) & (s["target"] == target)]
                    if g.empty:
                        vals.append(np.nan)
                    elif col == "gap_mean_kw":
                        vals.append(_pooled_gap(g))
                    else:
                        vals.append(float(g[col].mean()))
                ax.barh(y + (k - 0.5) * 0.38, vals, 0.36, hatch=hatch, edgecolor="white",
                        color=[MODEL_STYLE[n][0] for n in names])
            for r, model in enumerate(names):
                if model not in set(t["model"]):
                    ax.text(0.02, r, "point forecast only", transform=ax.get_yaxis_transform(),
                            va="center", fontsize=6.5, color="dimgray")
                elif model == "lstm_sota":
                    ax.text(0.02, r, "never out of order", transform=ax.get_yaxis_transform(),
                            va="center", fontsize=6.5, color="dimgray")
            ax.set_yticks(y, [NAMES[n] for n in names], fontsize=7)
            if i == 1:
                ax.set_xlabel(label, fontsize=7.5)
            ax.set_title(SPLIT_NAME[split], fontsize=8, loc="left")
            if col != "gap_mean_kw":
                ax.set_xlim(0, 100)
        c = m.xs(split, level="split").loc[list(TFTS)]
        assert c["crossed_pct"].idxmax() == ("tft_96", "direct"), split
        key = "v" if split == "val" else "t"
        facts[f"{key}_lo"] = float(c["crossed_pct"].min())
        facts[f"{key}_hi"] = float(c["crossed_pct"].max())
        facts[f"p_{key}"] = float(c["point_not_median_pct"].max())
        facts[f"g_{key}"] = _pooled_gap(s[s["model"].isin(TFTS)])
    axes[0, 0].set_ylim(len(names) - 0.4, -0.6)  # top to bottom, with room for the last label
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="0.6", label="indirect route"),
                        Patch(facecolor="0.6", hatch="//", edgecolor="white", label="direct route")],
               fontsize=7, ncol=2, loc="outside lower center")
    drop = 100 * (1 - t["pinball_sorted_kw"] / t["pinball_raw_kw"])
    assert (drop[t["model"].isin(TFTS)] > 0).all(), "sorting lowers the loss in every cell of the TFTs"
    assert (drop[t["model"] == "lstm_sota"].abs() < 1e-9).all()  # never crossed, so unchanged
    facts["drop"] = float(drop.max())
    return fig, facts

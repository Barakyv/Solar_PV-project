# ruff: noqa: E501  (registered texts; see s11_part_b)
"""Stage 11, part C: the figures of report v7 (docs/stage11_plan.md section 6, OP-68).

S11-20 is a process figure: the figures of the report by chapter, and the figures of the submitted report v6 with
their counterpart in v7. Every number in the takeaway is computed from `docs/report_v7_figures.csv`, which
`scripts/make_report_v7.py` writes with the report; each claim is asserted here.
"""

import numpy as np
from matplotlib.patches import Rectangle

from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, subplots

FIG_TABLE = "docs/report_v7_figures.csv"
N_V6 = 25  # the submitted report has figures 1 to 25 (submission_evidence/13_report)


@register(
    "S11-20", "11", "report_figures", "The figures of report v7 against those of the submitted report",
    "Which figures does the final report hold, where, and which figure of the submitted report does each one "
    "replace?",
    "Left: the figures of the report by chapter, counterparts of a v6 figure (blue) and figures added to complete "
    "the account of the method (orange). Right: the 25 figures of v6, each with the number of its counterpart in "
    "v7, or hatched when it is not redrawn.",
    "The report has {n_v7} figures in {n_chapters} chapters: {n_counter} are the counterparts of v6 figures and "
    "{n_added} are added where they complete the account of the method or answer a sentence of v6 that had no "
    "figure ({added_sections}). Of the {n_v6} figures of v6, {n_redrawn} are redrawn from the final run and {n_not} are not: {n_selection} "
    "drew the feature-selection protocol of v6, which was replaced (F-26, D-23), and {n_radar} repeated the "
    "seasonal numbers as a radar (C-84).",
    "docs/report_v7.md (the figures chosen in part C) and docs/report_v7_README.md.",
    (FIG_TABLE,))
def s11_report_figures():
    t = D.csv(FIG_TABLE, dtype=str, keep_default_na=False)
    v7 = t[t["figure_id"] != ""].copy()
    v7["no"] = v7["number"].astype(int)
    assert list(v7["no"]) == list(range(1, len(v7) + 1))
    v7["chapter"] = v7["section"].str.split(".").str[0].astype(int)
    counter = v7[v7["v6_figure"] != ""]
    not_redrawn = t[t["figure_id"] == ""]
    v6_all = sorted(int(x) for x in t["v6_figure"] if x)
    assert v6_all == list(range(1, N_V6 + 1)), "every v6 figure is accounted for exactly once"
    n_sel = int(not_redrawn["relation"].str.contains("selection protocol").sum())
    n_radar = int(not_redrawn["relation"].str.contains("radar").sum())
    assert n_sel + n_radar == len(not_redrawn)

    fig, (ax, ax2) = subplots(1, 2, width=9.0, height=3.6, gridspec_kw={"width_ratios": [1.25, 1]})
    chapters = sorted(v7["chapter"].unique())
    x = np.arange(len(chapters))
    nc = [int(((v7["chapter"] == ch) & (v7["v6_figure"] != "")).sum()) for ch in chapters]
    na = [int(((v7["chapter"] == ch) & (v7["v6_figure"] == "")).sum()) for ch in chapters]
    ax.bar(x, nc, color=C["blue"], label="counterpart of a v6 figure")
    ax.bar(x, na, bottom=nc, color=C["orange"], hatch="//", edgecolor="white", label="added for the method")
    ax.set_xticks(x, [str(ch) for ch in chapters])
    ax.set_xlabel("chapter of report v7")
    ax.set_ylabel("figures")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title("Figures of v7 by chapter", fontsize=9)

    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    by_v6 = {int(r["v6_figure"]): r for _, r in t.iterrows() if r["v6_figure"]}
    for k in range(1, N_V6 + 1):
        r = by_v6[k]
        col, row = (k - 1) % 5, 4 - (k - 1) // 5
        redrawn = r["figure_id"] != ""
        ax2.add_patch(Rectangle(
            (col + 0.05, row + 0.05), 0.9, 0.9, facecolor=C["sky"] if redrawn else "white",
            edgecolor=C["blue"] if redrawn else C["grey"], hatch=None if redrawn else "////", lw=0.8))
        box = None if redrawn else dict(fc="white", ec="none", pad=0.6)
        ax2.text(col + 0.5, row + 0.66, f"v6 {k}", ha="center", va="center", fontsize=7, bbox=box)
        label = f"v7 {r['number']}" if redrawn else ("radar" if "radar" in r["relation"] else "selection")
        ax2.text(col + 0.5, row + 0.32, label, ha="center", va="center", fontsize=6.5,
                 color="black" if redrawn else "dimgray", bbox=box)
    ax2.set_title("The 25 figures of v6 and their counterparts", fontsize=9)
    return fig, {"n_v7": len(v7), "n_chapters": len(chapters), "n_counter": len(counter),
                 "n_added": len(v7) - len(counter), "n_v6": N_V6,
                 "added_sections": "sections " + ", ".join(v7[v7["v6_figure"] == ""]["section"].drop_duplicates()), "n_redrawn": len(counter),
                 "n_not": len(not_redrawn), "n_selection": n_sel, "n_radar": n_radar}

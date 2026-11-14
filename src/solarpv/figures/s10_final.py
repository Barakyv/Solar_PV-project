"""Figures of stage 10: the final run `20261114T083930Z_final` (docs/runbook_final.md, OP-65).

Every figure reads the files of that run (its report tables and the step `test`), never the
development runs. Every model-level figure covers the six models and the three references in one
layout and one style (`MODEL_STYLE`, `REF_STYLE`; rule 9). Validation numbers are tuning-split
numbers (D-10); the test numbers were read after the acceptance was written, and nothing was
changed after them (D-37).
"""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.s07_s09 import REF_STYLE
from solarpv.figures.s09_models import MODEL_STYLE, MODELS, NAMES
from solarpv.figures.style import C, subplots
from solarpv.pipeline.report import SEED_SPREAD_WARN_PP

FINAL = "outputs/runs/20261114T083930Z_final"
REPORT = f"{FINAL}/report"
STEPS = f"{REPORT}/report_steps.csv"
BENCH_TEST = f"{REPORT}/report_benchmark_test.csv"
BENCH_VAL = f"{REPORT}/report_benchmark_val.csv"
DM_TEST = f"{REPORT}/report_dm_kw_test.csv"
SPREAD = f"{REPORT}/report_seed_spread.csv"
CV_TEST = f"{REPORT}/report_cv_vs_test.csv"
VIEWS_TEST = f"{FINAL}/test/test/benchmark_views.csv"
VIEWS_VAL = f"{FINAL}/evaluate/val/benchmark_views.csv"
ACCEPTANCE = f"{FINAL}/stage10_acceptance.md"

CELLS = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]
REFS = ("climatology", "clearsky_k", "persistence")
ALL = [m for _, m, _ in MODELS] + list(REFS)
LABEL = {**NAMES, "climatology": "climatology", "clearsky_k": "clear sky x k",
         "persistence": "persistence"}
STYLE = {**MODEL_STYLE, **{m: REF_STYLE[m] for m in REFS}}
GPU_STEPS = ("bilstm", "tft96", "tft168", "tftbase", "patchtst")
STEP_MODEL = {"xgb": "xgb_sota", "bilstm": "lstm_sota", "tft96": "tft_96", "tft168": "tft_168",
              "tftbase": "tft", "patchtst": "patchtst"}


def _cell_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """The mean over seeds of `col` per model and cell (route, horizon)."""
    return df.groupby(["model", "target", "horizon"])[col].mean()


def _bench(rel: str) -> pd.DataFrame:
    df = D.csv(rel)
    assert set(ALL) <= set(df["model"]), "a model is missing from the benchmark"
    return df


# ── S10-01: the run ─────────────────────────────────────────────────────────

@register(
    "S10-01", "10", "run_steps", "The final run: every step, its time and its device",
    "How long did each step of the final run take, and did any step need a restart?",
    "One bar per step of the run, in seconds on a logarithmic axis, the model steps in the "
    "colour of their model and the others in grey; the text at the end of a bar is the "
    "duration recorded in the step's record. The GPU steps are hatched.",
    "All {n_steps} steps finished at one commit with no restart from a checkpoint. The "
    "model and analysis steps took {train_h:.1f} hours, {gpu_h:.1f} of them on the GPU; the "
    "longest step "
    "is {longest} ({longest_h:.1f} hours). The step test, run in the second phase, took "
    "{test_s:.0f} seconds.",
    "The acceptance of phase 1 (stage10_acceptance.md) and the step test of phase 2.",
    (STEPS, ACCEPTANCE))
def s10_run_steps():
    st = D.csv(STEPS)
    assert (st["status"] == "ok").all() and (st["n_resumed"] == 0).all()
    assert st["commit"].nunique() == 1
    st = st.sort_values("duration_s")
    names = {"select": "select", "baselines": "baselines", "xgb": "XGB", "cv": "cv",
             "ablation": "ablation", "bilstm": "BiLSTM", "tft96": "TFT-96", "tft168": "TFT-168",
             "tftbase": "TFT baseline", "patchtst": "PatchTST", "evaluate": "evaluate",
             "explain": "explain", "test": "test (phase 2)"}
    fig, ax = subplots(height=3.6, width=7.4)
    for i, r in enumerate(st.itertuples()):
        colour = MODEL_STYLE[STEP_MODEL[r.step]][0] if r.step in STEP_MODEL else C["grey"]
        hatch = "//" if r.step in GPU_STEPS else None
        ax.barh(i, r.duration_s, color=colour, hatch=hatch, edgecolor="white", alpha=0.9)
        ax.text(r.duration_s * 1.15, i, f"{r.duration_s:,.0f} s", va="center", fontsize=7)
    ax.set_yticks(range(len(st)), [names[s] for s in st["step"]], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_xlabel("seconds (logarithmic axis); hatched: GPU")
    gpu = st[st["step"].isin(GPU_STEPS)]["duration_s"].sum()
    train = st[st["step"].isin(["xgb", "cv", "ablation", *GPU_STEPS])]["duration_s"].sum()
    longest = st.iloc[-1]
    facts = {"n_steps": len(st), "train_h": train / 3600, "gpu_h": gpu / 3600,
             "longest": names[longest["step"]], "longest_h": longest["duration_s"] / 3600,
             "test_s": float(st.loc[st["step"] == "test", "duration_s"].iloc[0])}
    return fig, facts


# ── S10-02: the test score of every model ───────────────────────────────────

@register(
    "S10-02", "10", "test_scores", "The six models and the references on the test split",
    "How does each model score on the test split, in kW, against the references, on the same "
    "hours?",
    "Two panels, the indirect and the direct route: nRMSE in kW as a percentage of the plant "
    "capacity at D+1, D+2 and D+3, one line per model (mean over its seeds, the bar is the "
    "range over seeds) and one per reference. The y axis does not start at zero.",
    "The six models score between {lo:.2f} and {hi:.2f} per cent, the references between "
    "{ref_lo:.2f} and {ref_hi:.2f}. PatchTST is worse than climatology in {patch_worse} of 6 "
    "cells and no other model is. The lowest seed-mean among the six models is: {winners}. "
    "Whether such differences are significant is S10-04.",
    "S10-04 (which of these differences are significant), S10-03 (where the level comes from).",
    (BENCH_TEST,))
def s10_test_scores():
    b = _bench(BENCH_TEST)
    assert set(b["split"]) == {"test"} and b["n"].nunique() == 1
    mean = _cell_mean(b, "nRMSE_kW_pct")
    lo_ = b.groupby(["model", "target", "horizon"])["nRMSE_kW_pct"].min()
    hi_ = b.groupby(["model", "target", "horizon"])["nRMSE_kW_pct"].max()
    fig, axes = subplots(1, 2, width=7.6, height=3.5, sharey=True)
    for ax, t in zip(axes, ("indirect", "direct"), strict=False):
        for m in ALL:
            col, fmt = STYLE[m]
            xs = [1, 2, 3]
            ys = [mean[(m, t, h)] for h in xs]
            if m in REFS:
                ax.plot(xs, ys, fmt, color=col, lw=1.2, ms=4, label=LABEL[m], zorder=1)
            else:
                err = [[mean[(m, t, h)] - lo_[(m, t, h)] for h in xs],
                       [hi_[(m, t, h)] - mean[(m, t, h)] for h in xs]]
                ax.errorbar(xs, ys, yerr=err, fmt=fmt, color=col, ms=4, capsize=2,
                            label=LABEL[m], zorder=3)
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
        ax.set_title(f"{t} route", fontsize=9)
    axes[0].set_ylabel("nRMSE in kW, % (axis not from 0)")
    axes[1].legend(fontsize=6.5, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    models = [m for _, m, _ in MODELS]
    vals = [mean[(m, t, h)] for m in models for t, h in CELLS]
    refs = [mean[(m, t, h)] for m in REFS for t, h in CELLS]
    patch_worse = sum(mean[("patchtst", t, h)] > mean[("climatology", t, h)] for t, h in CELLS)
    others_worse = sum(mean[(m, t, h)] > mean[("climatology", t, h)]
                       for m in models if m != "patchtst" for t, h in CELLS)
    assert others_worse == 0, "a model other than PatchTST is worse than climatology"
    best = [min(models, key=lambda m: mean[(m, t, h)]) for t, h in CELLS]
    counts = pd.Series(best).value_counts()
    winners = ", ".join(f"{NAMES[m]} in {n} of 6" for m, n in counts.items())
    facts = {"lo": min(vals), "hi": max(vals), "ref_lo": min(refs), "ref_hi": max(refs),
             "patch_worse": int(patch_worse), "winners": winners}
    return fig, facts


# ── S10-03: validation, test and the operation-limited hours ────────────────

@register(
    "S10-03", "10", "val_test_views", "From validation to test: where the extra error is",
    "Why is every model's test error higher than its validation error, and is it the models?",
    "Grouped bars per model and reference, the mean over seeds and over the six route and "
    "horizon cells, in kW as a percentage of the plant capacity: validation and test, each on "
    "all daytime hours and on the hours of normal operation (no curtailment, stop or derate; "
    "D-22). The bar value is printed.",
    "Every model and reference rises from validation to test: XGB {x_val:.2f} to {x_test:.2f} "
    "on all hours, climatology {c_val:.2f} to {c_test:.2f}. On the hours of normal operation "
    "the rise is smaller (XGB {x_val_n:.2f} to {x_norm:.2f}, climatology {c_val_n:.2f} to "
    "{c_norm:.2f}): the {n_lim} operation-limited hours of the test split account for "
    "{sh_lo:.0%} to {sh_hi:.0%} of the rise, and the rest is a shift of the period that every "
    "model and reference shares.",
    "The documents state both views (the all-hours view is the primary one, D-22), and compare "
    "a test number only with the validation number of the same view. Whether the "
    "operation-limited test hours are an event or a trend is a question "
    "for stage 11.",
    (BENCH_VAL, BENCH_TEST, VIEWS_VAL, VIEWS_TEST))
def s10_val_test_views():
    bv, bt = _bench(BENCH_VAL), _bench(BENCH_TEST)
    vv, vt = D.csv(VIEWS_VAL), D.csv(VIEWS_TEST)

    def avg(df, col, view=None):
        d = df if view is None else df[df["view"] == view]
        cell = d.groupby(["model", "target", "horizon"])[col].mean()
        return {m: float(cell.loc[m].mean()) for m in ALL}

    series = {
        "validation, all hours": avg(bv, "nRMSE_kW_pct"),
        "validation, normal operation": avg(vv, "nRMSE_kW_pct", "normal_operation"),
        "test, all hours": avg(bt, "nRMSE_kW_pct"),
        "test, normal operation": avg(vt, "nRMSE_kW_pct", "normal_operation"),
    }
    cols = [C["light"], C["sky"], C["grey"], C["blue"]]
    fig, ax = subplots(height=3.6, width=7.6)
    w = 0.2
    for k, ((name, vals), col) in enumerate(zip(series.items(), cols, strict=False)):
        xs = np.arange(len(ALL)) + (k - 1.5) * w
        ax.bar(xs, [vals[m] for m in ALL], w, color=col, edgecolor="black", lw=0.4, label=name)
    ax.set_xticks(range(len(ALL)), [LABEL[m] for m in ALL], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("nRMSE in kW, % of capacity (axis starts at 0)")
    ax.legend(fontsize=7, ncol=2)
    n_all = int(vt[(vt["view"] == "all") & (vt["model"] == "xgb_sota")]["n"].iloc[0])
    n_norm = int(vt[(vt["view"] == "normal_operation") & (vt["model"] == "xgb_sota")]["n"].iloc[0])
    sv, st_, sn = series["validation, all hours"], series["test, all hours"], \
        series["test, normal operation"]
    svn = series["validation, normal operation"]
    assert all(st_[m] > sv[m] for m in ALL), "a model or reference did not rise on test"
    # the share of the rise from validation to test that the operation-limited hours account for
    share = {m: ((st_[m] - sv[m]) - (sn[m] - svn[m])) / (st_[m] - sv[m]) for m in ALL}
    assert all(0 < s < 1 for s in share.values())
    facts = {"c_val": sv["climatology"], "c_test": st_["climatology"], "x_val": sv["xgb_sota"],
             "x_test": st_["xgb_sota"], "x_norm": sn["xgb_sota"], "c_norm": sn["climatology"],
             "x_val_n": svn["xgb_sota"], "c_val_n": svn["climatology"],
             "n_lim": n_all - n_norm, "sh_lo": min(share.values()), "sh_hi": max(share.values())}
    return fig, facts


# ── S10-04: significance against XGB on the test split ──────────────────────

@register(
    "S10-04", "10", "test_dm_vs_xgb",
    "Every model and reference against XGB: Diebold-Mariano in kW on test",
    "Which differences to XGB survive the Diebold-Mariano test in kW on the test split?",
    "One row per other model or reference, one column per route and horizon. Green: XGB is "
    "significantly better (HLN, Bonferroni level of the family); red: significantly worse; grey: "
    "not significant. The text is the effect as a share of the minimum detectable effect.",
    "XGB is significantly better than persistence in {n_pers} of 6 cells, than clear sky x k in "
    "{n_cs}, than PatchTST in {n_pt} and than climatology in {n_clim}; the climatology effect "
    "is {clim_lo:.0f} to {clim_hi:.0f} per cent of the minimum detectable effect, so the test "
    "has too little power to separate them. BiLSTM is significantly worse than XGB in {n_lstm} "
    "cells; the three TFTs in {n_tft}. XGB is significantly worse than no model ({n_worse}).",
    "The statement of the result (S10-02) and the limitation of D-37; no model is declared the "
    "winner.",
    (DM_TEST,))
def s10_test_dm():
    dm = D.csv(DM_TEST)
    assert dm["split"].eq("test").all() and (dm["model_b"] == "xgb_sota").all()
    order = ["persistence", "clearsky_k", "climatology", "patchtst", "lstm_sota", "tft",
             "tft_168", "tft_96"]
    fig, ax = subplots(height=3.3, width=7.2)
    n_better, n_worse = {}, 0
    for i, m in enumerate(order):
        n_better[m] = 0
        for j, (t, h) in enumerate(CELLS):
            r = dm[(dm["model_a"] == m) & (dm["target"] == t) & (dm["horizon"] == h)]
            assert len(r) == 1, (m, t, h)
            r = r.iloc[0]
            sig, xgb_better = bool(r["significant"]), r["mean_diff"] > 0
            face = (C["green"] if xgb_better else C["red"]) + "66" if sig else C["light"]
            n_better[m] += int(sig and xgb_better)
            n_worse += int(sig and not xgb_better)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=face, ec="white", lw=1))
            ax.text(j, i, f"{r['pct_of_mde']:.0f} %", ha="center", va="center", fontsize=6.5)
    ax.set_xlim(-0.5, len(CELLS) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(range(len(CELLS)), [f"{t}\nD+{h}" for t, h in CELLS], fontsize=7)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order], fontsize=7)
    ax.grid(False)
    ax.set_title("share of the minimum detectable effect; green: XGB significantly better",
                 fontsize=8)
    clim = dm[dm["model_a"] == "climatology"]["pct_of_mde"]
    # the sign convention: model_a minus model_b of the loss, so a positive value favours XGB
    n_worse_than_xgb = {m: sum(
        bool(r.significant) and r.mean_diff < 0
        for r in dm[dm["model_a"] == m].itertuples()) for m in order}
    # "BiLSTM significantly worse than XGB" is the mirror of the grid: a significant positive value
    # of its mean_diff, which is counted above as XGB better
    facts = {"n_pers": n_better["persistence"], "n_cs": n_better["clearsky_k"],
             "n_pt": n_better["patchtst"], "n_clim": n_better["climatology"],
             "clim_lo": float(clim.min()), "clim_hi": float(clim.max()),
             "n_lstm": n_better["lstm_sota"],
             "n_tft": n_better["tft"] + n_better["tft_168"] + n_better["tft_96"],
             "n_worse": int(n_worse)}
    assert n_worse == sum(n_worse_than_xgb.values())
    return fig, facts


# ── S10-05: the seed spread (the acceptance rule of stage 10) ───────────────

@register(
    "S10-05", "10", "seed_spread", "Seed spread in kW against the rule of stage 10",
    "Do the seeds of a model agree to within the rule of stage 10 (0.15 pp in kW)?",
    "Standard deviation over seeds of the nRMSE in kW, per route and horizon, for the models "
    "that were trained on three seeds, with the rule as a dashed line. The models with one "
    "seed are named in the panel: a spread cannot be computed for them.",
    "The largest spread is {x_max:.3f} pp for XGB and {l_max:.3f} pp for BiLSTM, against the "
    "rule of {limit:.2f} pp; {n_above} of {n_cells} cells are above it. {single} have one "
    "seed each, so their spread is unknown; the differences among models of a similar level "
    "(S10-02) are read with that caveat.",
    "The acceptance of stage 10 (passed) and the limitation on the single-seed neural models.",
    (SPREAD,))
def s10_seed_spread():
    sp = D.csv(SPREAD)
    multi = sp[sp["n_seeds"] >= 3]
    single = sorted(set(sp[sp["n_seeds"] < 3]["model"]))
    assert set(multi["model"]) == {"xgb_sota", "lstm_sota"}
    fig, ax = subplots(height=3.2, width=7.2)
    for m in ("xgb_sota", "lstm_sota"):
        col, fmt = MODEL_STYLE[m]
        d = multi[multi["model"] == m].set_index(["target", "horizon"])
        ax.plot(range(len(CELLS)), [d.loc[c, "std_kw_pp"] for c in CELLS], fmt, color=col,
                label=f"{NAMES[m]} (three seeds)")
    ax.axhline(SEED_SPREAD_WARN_PP, color=C["grey"], ls="--", lw=1.2, label="rule of stage 10")
    ax.set_xticks(range(len(CELLS)), [f"{t}\nD+{h}" for t, h in CELLS], fontsize=7)
    ax.set_ylim(0, SEED_SPREAD_WARN_PP * 1.25)
    ax.set_ylabel("std over seeds, pp (axis starts at 0)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(0.99, 0.62, "one seed, no spread: " + ", ".join(NAMES[m] for m in single),
            transform=ax.transAxes, ha="right", fontsize=7, color=C["grey"])
    facts = {"x_max": float(multi[multi["model"] == "xgb_sota"]["std_kw_pp"].max()),
             "l_max": float(multi[multi["model"] == "lstm_sota"]["std_kw_pp"].max()),
             "limit": SEED_SPREAD_WARN_PP, "n_above": int(multi["above_rule"].sum()),
             "n_cells": len(multi), "single": ", ".join(NAMES[m] for m in single)}
    return fig, facts


# ── S10-06: cross-validation, validation and test of the XGB ────────────────

@register(
    "S10-06", "10", "cv_val_test", "XGB: the five-fold CV on train, validation and test",
    "How far is the test error of XGB from the error that the train CV and validation "
    "predicted?",
    "For each route and horizon the XGB nRMSE in kW as three points on one line: the mean of "
    "the five expanding folds inside train (the bar is the standard deviation over folds), the "
    "validation, and the test; the mean over the three seeds.",
    "The test error is {gap_lo:.1f} to {gap_hi:.1f} pp above the train CV in every one of the "
    "{n_fits} fits, and validation lies {val_lo:.1f} to {val_hi:.1f} pp below it. The mean "
    "spread between folds is {fold_std:.1f} pp, so the folds themselves differ by a large "
    "part of the gap; the references rise from validation to test as well (S10-03).",
    "The statement that the CV, the validation and the test are three different periods, each "
    "harder or easier for every model; no decision used the test.",
    (CV_TEST,))
def s10_cv_val_test():
    cv = D.csv(CV_TEST)
    g = cv.groupby(["target", "horizon"]).mean(numeric_only=True)
    fig, ax = subplots(height=3.3, width=7.2)
    xs = np.arange(len(CELLS))
    ax.errorbar(xs - 0.15, [g.loc[c, "cv_mean"] for c in CELLS],
                yerr=[g.loc[c, "cv_std"] for c in CELLS], fmt="o", color=C["blue"], capsize=3,
                label="train CV (mean and std over folds)")
    ax.plot(xs, [g.loc[c, "validation_uncorrected"] for c in CELLS], "s", color=C["orange"],
            label="validation")
    ax.plot(xs + 0.15, [g.loc[c, "test_uncorrected"] for c in CELLS], "D", color="black",
            label="test")
    ax.set_xticks(xs, [f"{t}\nD+{h}" for t, h in CELLS], fontsize=7)
    ax.set_ylim(0, 14)
    ax.set_ylabel("XGB nRMSE in kW, % (axis starts at 0)")
    ax.legend(fontsize=7, loc="upper center", ncol=3)
    gap = cv["test_minus_cv_pp"]
    val_gap = cv["validation_minus_cv_pp"]
    assert (gap > 0).all() and (val_gap < 0).all()
    facts = {"gap_lo": float(gap.min()), "gap_hi": float(gap.max()),
             "val_lo": float(-val_gap.max()), "val_hi": float(-val_gap.min()),
             "n_fits": len(cv), "fold_std": float(cv["cv_std"].mean())}
    return fig, facts

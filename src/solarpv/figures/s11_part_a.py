"""Figures of stage 11, part A: the frozen final run (docs/stage11_plan.md).

Every figure reads the tables of `outputs/11_analyses/` (made by `solarpv.figures.analyses11`) and
the tables of the run `20261114T083930Z_final`; none trains or scores anything. Every model-level
figure covers the six models and the three references in one layout and one style (rule 9). Where a
figure shows the test split, validation stands beside it in the same view (S10-03).
"""

import re

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from solarpv.figures import _data as D
from solarpv.figures.analyses11 import MIN_HOURS, OUT
from solarpv.figures.registry import register
from solarpv.figures.s09_models import MODELS, NAMES
from solarpv.figures.s10_final import ALL, CELLS, FINAL, LABEL, STYLE
from solarpv.figures.style import C, subplots

MODEL6 = [m for _, m, _ in MODELS]
SEASONS = ("Winter", "Spring", "Summer", "Fall")
ABOUT = f"{OUT}/manifest.json"


def T(name: str) -> pd.DataFrame:
    return D.csv(f"{OUT}/{name}.csv")


def _cell(t: str, h: int) -> str:
    return f"{t}\nD+{h}"


def _marker(model: str) -> str:
    return STYLE[model][1][0]


# ── S11-01: the benchmark with intervals in kW (S-23) ───────────────────────

@register(
    "S11-01", "11", "benchmark_intervals", "The benchmark in kW with 95 % intervals",
    "How large are the differences between the models compared with the uncertainty of one "
    "score, on validation and on test?",
    "One panel per split. For every route and horizon, one marker per model and reference: the "
    "mean over seeds of the nRMSE in kW (percent of the plant capacity); the bar is the 95 % "
    "circular block bootstrap interval (block of 24 hours), averaged over the model's seeds.",
    "On test the interval of a single score is about {hw_test:.2f} pp on each side, and the six "
    "models differ by at most {sp_test:.2f} pp in a cell; on validation the figures are "
    "{hw_val:.2f} and {sp_val:.2f} pp. The intervals of all six models overlap in {ov_test} of 6 "
    "cells on test and in {ov_val} of 6 on validation. These intervals are not paired, so "
    "overlap is not a test of a difference: the paired test of S10-04 separates PatchTST from XGB "
    "in every cell although the intervals overlap.",
    "S10-04 and S11-09 (the paired tests that use the same errors) and the statement of the "
    "results: the intervals are not paired, so they are wider than the paired test.",
    (f"{OUT}/benchmark_ci_kw.csv", ABOUT))
def s11_benchmark_intervals():
    b = T("benchmark_ci_kw")
    g = b.groupby(["model", "target", "horizon", "split"])[["nRMSE_kW_pct", "ci_lo", "ci_hi"]
                                                        ].mean()
    fig, axes = subplots(2, 1, width=8.2, height=5.8, sharex=True)
    facts = {}
    for ax, split in zip(axes, ("val", "test"), strict=False):
        for k, m in enumerate(ALL):
            xs = np.arange(len(CELLS)) + (k - (len(ALL) - 1) / 2) * 0.085
            ys = np.array([g.loc[(m, t, h, split), "nRMSE_kW_pct"] for t, h in CELLS])
            lo = np.array([g.loc[(m, t, h, split), "ci_lo"] for t, h in CELLS])
            hi = np.array([g.loc[(m, t, h, split), "ci_hi"] for t, h in CELLS])
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=_marker(m), color=STYLE[m][0],
                        ms=4, lw=0.9, capsize=1.5, label=LABEL[m])
        ax.set_title("validation" if split == "val" else "test", fontsize=9)
        ax.set_ylabel("nRMSE in kW, % (axis not from 0)")
        hw, sp, ov = [], [], 0
        for t, h in CELLS:
            rows = g.loc[[(m, t, h, split) for m in MODEL6]]
            hw += list((rows["ci_hi"] - rows["ci_lo"]) / 2)
            sp.append(rows["nRMSE_kW_pct"].max() - rows["nRMSE_kW_pct"].min())
            ov += int(rows["ci_lo"].max() <= rows["ci_hi"].min())
        facts.update({f"hw_{split}": float(np.mean(hw)), f"sp_{split}": float(max(sp)),
                      f"ov_{split}": ov})
    axes[1].set_xticks(range(len(CELLS)), [_cell(t, h) for t, h in CELLS], fontsize=7)
    axes[0].legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    return fig, facts


# ── S11-02: the horizon (S-26) ──────────────────────────────────────────────

@register(
    "S11-02", "11", "horizon_robustness", "Error against the horizon, validation and test",
    "Does the error grow from D+1 to D+3, and is the answer the same on validation and on test?",
    "Four panels: validation and test, indirect and direct route. One line per model and "
    "reference: the mean over seeds of the nRMSE in kW at D+1, D+2 and D+3 (all daytime hours).",
    "On validation the error at D+3 is lower than at D+1 for {val_down} of the 12 pairs of a model "
    "and a route (change {val_lo:+.2f} to {val_hi:+.2f} pp); on test it is higher for {test_up} "
    "of 12 (change {test_lo:+.2f} to {test_hi:+.2f} pp). The sign of the horizon effect reverses "
    "between the two periods, and its size is at most {big:.2f} pp, so the order of the horizons "
    "is a property of the period and not of the horizon.",
    "The reading of the horizon in the report: the order of the horizons is a property of the "
    "period (R6b-07, R7-05), not of the models.",
    (f"{OUT}/horizon_robustness.csv", ABOUT))
def s11_horizon():
    h = T("horizon_robustness")
    fig, axes = subplots(2, 2, width=7.8, height=5.2, sharex=True)
    for i, split in enumerate(("val", "test")):
        for j, t in enumerate(("indirect", "direct")):
            ax = axes[i, j]
            d = h[(h["split"] == split) & (h["target"] == t)].set_index("model")
            for m in ALL:
                col, fmt = STYLE[m]
                ax.plot([1, 2, 3], [d.loc[m, "D1"], d.loc[m, "D2"], d.loc[m, "D3"]], fmt,
                        color=col, ms=4, lw=1.1, label=LABEL[m])
            ax.set_title(f"{'validation' if split == 'val' else 'test'}, {t} route", fontsize=8)
            ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
    axes[0, 0].set_ylabel("nRMSE in kW, % (axis not from 0)")
    axes[1, 0].set_ylabel("nRMSE in kW, % (axis not from 0)")
    axes[0, 1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    six = h[h["model"].isin(MODEL6)]
    out = {}
    for split in ("val", "test"):
        s = six[six["split"] == split]["D3_minus_D1_pp"]
        out[f"{split}_down"] = int((s < 0).sum())
        out[f"{split}_up"] = int((s > 0).sum())
        out[f"{split}_lo"], out[f"{split}_hi"] = float(s.min()), float(s.max())
    assert len(six) == 24
    assert out["val_down"] > 6 and out["test_up"] > 6, "the sign no longer reverses"
    out["big"] = float(six["D3_minus_D1_pp"].abs().max())
    return fig, out


# ── S11-03: predicted against observed (S-24) ───────────────────────────────

@register(
    "S11-03", "11", "scatter_test", "Forecast against observed power on the test split",
    "How close does each model come to the observed power, hour by hour, on the test split?",
    "One panel per model: the number of daytime hours (log scale) in a grid of observed power "
    "(horizontal) against the forecast (vertical), both in kW, D+1, indirect route (the forecast "
    "k times the clear-sky power), lowest seed. The diagonal is the perfect forecast.",
    "The correlation between forecast and observation on test is between {r_lo:.3f} and "
    "{r_hi:.3f} for the six models. In every panel a group of hours lies above the "
    "diagonal at observed values near zero: forecasts of high power while "
    "the plant delivered little. Some of these hours are operation-limited (D-22; S11-12).",
    "S11-11 and S11-12 (the bias, and where the hours with the large errors are).",
    (f"{OUT}/scatter_hist.csv", f"{OUT}/decomposition.csv", ABOUT))
def s11_scatter():
    s = T("scatter_hist")
    s = s[(s["target"] == "indirect") & (s["split"] == "test")]
    dec = T("decomposition")
    dec = dec[(dec["view"] == "all") & (dec["split"] == "test") & (dec["target"] == "indirect")
              & (dec["horizon"] == 1)]
    fig, axes = subplots(2, 3, width=8.0, height=5.4, sharex=True, sharey=True)
    cap = 3449.0
    from solarpv.figures.analyses11 import SCATTER_BINS
    r = {}
    for ax, m in zip(axes.flat, MODEL6, strict=False):
        g = s[s["model"] == m]
        grid = np.zeros((SCATTER_BINS, SCATTER_BINS))
        grid[g["obs_bin"], g["fc_bin"]] = g["count"]
        ax.imshow(np.log10(grid.T + 1), origin="lower", extent=[0, cap, 0, cap], cmap="Greys",
                  aspect="equal")
        ax.plot([0, cap], [0, cap], color=C["red"], lw=0.9)
        r[m] = float(dec[dec["model"] == m]["corr_forecast_observed"].mean())
        ax.set_title(f"{NAMES[m]}, r = {r[m]:.3f}", fontsize=8, color=STYLE[m][0])
        # hours with observed < 15 % of capacity and a forecast above 30 %
        low = g[g["obs_bin"] < SCATTER_BINS * 0.15]
        over = int(low[low["fc_bin"] >= SCATTER_BINS * 0.3]["count"].sum())
        assert over > 0, f"{m}: no hour of high forecast against low observation"
        ax.set_xlim(0, cap)
        ax.set_ylim(0, cap)
    for ax in axes[1]:
        ax.set_xlabel("observed power, kW")
    for ax in axes[:, 0]:
        ax.set_ylabel("forecast power, kW")
    return fig, {"r_lo": min(r.values()), "r_hi": max(r.values())}


# ── S11-04: the seasons (S-29, L-20) ────────────────────────────────────────

@register(
    "S11-04", "11", "seasonal_heatmap", "Error by season, validation and test",
    "How does the error of each model depend on the season, and are the two periods comparable "
    "season by season?",
    "Two panels, validation and test: the nRMSE in kW (percent of the capacity) of every model and "
    "reference by season (local months), D+1, indirect route, mean over seeds. Grey: no hour of "
    "that season in the split. Each season is named with the months the split holds.",
    "The test split has no summer, and its autumn is {fall_test} against {fall_val} on "
    "validation, so a season is not the same weather in the two periods. XGB's error is highest "
    "in {x_val_worst} on validation ({x_val_v:.2f} per cent) and in {x_test_worst} on test "
    "({x_test_v:.2f} per cent). From validation to test the error of XGB changes by {dx_w:+.1f} "
    "(Winter), {dx_s:+.1f} (Spring) and {dx_f:+.1f} (Fall) points, and that of climatology by "
    "{dc_w:+.1f}, {dc_s:+.1f} and {dc_f:+.1f}: the rise of the test period is concentrated in the "
    "autumn, where {ol_f_test:.0%} of the test hours are operation-limited against {ol_f_val:.0%} "
    "of the validation hours; Spring is almost unchanged.",
    "L-20: the by-season half; no comparison of a season across the two periods without naming "
    "the months (X104).",
    (f"{OUT}/seasonal_kw.csv", ABOUT))
def s11_seasonal_heatmap():
    s = T("seasonal_kw")
    s = s[(s["target"] == "indirect") & (s["horizon"] == 1)]
    vmax = s["nRMSE_kW_pct"].max()
    fig, axes = subplots(1, 2, width=8.2, height=3.9, sharey=True)
    months = {}
    for ax, split in zip(axes, ("val", "test"), strict=False):
        d = s[s["split"] == split]
        mat = np.full((len(ALL), len(SEASONS)), np.nan)
        for i, m in enumerate(ALL):
            for j, se in enumerate(SEASONS):
                r = d[(d["model"] == m) & (d["season"] == se)]
                if len(r):
                    mat[i, j] = r["nRMSE_kW_pct"].iloc[0]
                    months[(split, se)] = r["months"].iloc[0]
        ax.imshow(np.ma.masked_invalid(mat), cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(ALL)):
            for j in range(len(SEASONS)):
                if np.isnan(mat[i, j]):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=C["light"], ec="white"))
                else:
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(SEASONS)),
                      [f"{se}\n{months.get((split, se), 'no hours')}" for se in SEASONS],
                      fontsize=6)
        ax.set_title("validation" if split == "val" else "test", fontsize=9)
        ax.grid(False)
    axes[0].set_yticks(range(len(ALL)), [LABEL[m] for m in ALL], fontsize=7)
    x = s[s["model"] == "xgb_sota"]
    worst = {sp: x[x["split"] == sp].sort_values("nRMSE_kW_pct").iloc[-1] for sp in ("val", "test")}
    shared = [se for se in SEASONS if ((s["split"] == "val") & (s["season"] == se)).any()
              and ((s["split"] == "test") & (s["season"] == se)).any()]
    assert "Summer" not in shared and len(shared) == 3
    def rise(m, se):
        v = s[(s["split"] == "val") & (s["model"] == m) & (s["season"] == se)]["nRMSE_kW_pct"]
        t = s[(s["split"] == "test") & (s["model"] == m) & (s["season"] == se)]["nRMSE_kW_pct"]
        return float(t.iloc[0] - v.iloc[0])

    rises = {(m, se): rise(m, se) for m in ("xgb_sota", "climatology") for se in shared}
    for m in ("xgb_sota", "climatology"):
        assert rises[(m, "Fall")] == max(rises[(m, se)] for se in shared), "Fall is not the largest"
        assert abs(rises[(m, "Spring")]) < 1.0, "Spring changed by a point or more"

    def ol(split):
        r = s[(s["split"] == split) & (s["model"] == "xgb_sota") & (s["season"] == "Fall")]
        return float(r["op_limited_share"].iloc[0])

    return fig, {"fall_test": months[("test", "Fall")], "fall_val": months[("val", "Fall")],
                 "x_val_worst": worst["val"]["season"], "x_val_v": worst["val"]["nRMSE_kW_pct"],
                 "x_test_worst": worst["test"]["season"],
                 "x_test_v": worst["test"]["nRMSE_kW_pct"],
                 "dx_w": rises[("xgb_sota", "Winter")], "dx_s": rises[("xgb_sota", "Spring")],
                 "dx_f": rises[("xgb_sota", "Fall")], "dc_w": rises[("climatology", "Winter")],
                 "dc_s": rises[("climatology", "Spring")], "dc_f": rises[("climatology", "Fall")],
                 "ol_f_test": ol("test"), "ol_f_val": ol("val")}


# ── S11-05: local hour by season (S-30) ─────────────────────────────────────

def _hour_season_figure(view: str):
    s = T("hour_season")
    all_hours = s[(s["view"] == "all") & (s["split"] == "test") & (s["target"] == "indirect")
                  & (s["horizon"] == 1)]
    s = s[(s["view"] == view) & (s["split"] == "test") & (s["target"] == "indirect")
          & (s["horizon"] == 1)]
    seasons = [se for se in ("Fall", "Winter", "Spring") if (s["season"] == se).any()]
    hours = sorted(s["hour"].unique())
    vmax = all_hours[all_hours["n"] >= MIN_HOURS]["nRMSE_kW_pct"].max()  # one scale for both views
    fig, axes = subplots(2, 3, width=8.4, height=4.6, sharex=True, sharey=True)
    peaks = {}
    for ax, m in zip(axes.flat, MODEL6, strict=False):
        mat = np.full((len(seasons), len(hours)), np.nan)
        for i, se in enumerate(seasons):
            for j, hr in enumerate(hours):
                r = s[(s["model"] == m) & (s["season"] == se) & (s["hour"] == hr)]
                if len(r) and r["n"].iloc[0] >= MIN_HOURS:
                    mat[i, j] = r["nRMSE_kW_pct"].iloc[0]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="YlOrRd", vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_facecolor(C["light"])
        ax.set_title(NAMES[m], fontsize=8, color=STYLE[m][0])
        ax.set_xticks(range(0, len(hours), 2), [str(h) for h in hours[::2]], fontsize=7)
        ax.set_yticks(range(len(seasons)), seasons, fontsize=7)
        ax.grid(False)
        i, j = np.unravel_index(np.nanargmax(mat), mat.shape)
        peaks[m] = (seasons[i], hours[j], mat[i, j])
    fig.colorbar(im, ax=axes, shrink=0.8, label="nRMSE in kW, % of capacity")
    for ax in axes[1]:
        ax.set_xlabel("local hour")
    big = max(peaks.items(), key=lambda kv: kv[1][2])
    hrs = sorted({int(v[1]) for v in peaks.values()})
    wide = s[s["n"] >= MIN_HOURS].groupby(["model", "hour"])["nRMSE_kW_pct"].mean().unstack(
        "model")[MODEL6]
    hour_range = float(wide["xgb_sota"].max() - wide["xgb_sota"].min())
    model_range = float((wide.max(axis=1) - wide.min(axis=1)).max())
    assert model_range < hour_range, "the models differ more than the hours do"
    # the same cell on all hours and in normal operation, and the share of limited hours in it
    cell = all_hours[(all_hours["model"] == "xgb_sota") & (all_hours["season"] == "Fall")
                     & (all_hours["hour"] == 11)]
    norm = T("hour_season")
    norm = norm[(norm["view"] == "normal_operation") & (norm["split"] == "test")
                & (norm["target"] == "indirect") & (norm["horizon"] == 1)
                & (norm["model"] == "xgb_sota") & (norm["season"] == "Fall")
                & (norm["hour"] == 11)]
    assert float(norm["nRMSE_kW_pct"].iloc[0]) < 0.7 * float(cell["nRMSE_kW_pct"].iloc[0])
    return fig, {"pk_hours": ", ".join(str(h) for h in hrs), "big_season": big[1][0],
                 "big_hour": int(big[1][1]), "big_v": big[1][2], "big_m": NAMES[big[0]],
                 "hour_range": hour_range, "model_range": model_range,
                 "x_all": float(cell["nRMSE_kW_pct"].iloc[0]),
                 "x_norm": float(norm["nRMSE_kW_pct"].iloc[0]),
                 "lim_share": float(1 - norm["n"].iloc[0] / cell["n"].iloc[0])}


@register(
    "S11-05", "11", "hour_season", "Error by local hour and season on the test split",
    "At which local hours of the day, in which season, does each model err most?",
    "One panel per model: the nRMSE in kW (percent of the capacity) by local hour (horizontal) and "
    "season (vertical), D+1, indirect route, test split, all daytime hours, mean over seeds, one "
    "colour scale for all panels. A cell with too few hours (the minimum is recorded in the "
    "manifest of the analyses) is left grey. The six models are shown together, not the best "
    "four (M47).",
    "The largest cell of each model is in the local hour(s) {pk_hours}, and the largest cell of "
    "the figure is {big_season} at {big_hour}:00 ({big_v:.1f} per cent of the capacity, "
    "{big_m}). Averaged over the seasons, the XGB error changes by {hour_range:.1f} points "
    "between the hours of the day, and the six models differ by at most {model_range:.1f} points "
    "within an hour: the shape over the day is shared by the models. For XGB in the autumn at "
    "11:00 the error is {x_all:.1f} per cent on all hours and {x_norm:.1f} in normal operation; "
    "{lim_share:.0%} of those hours are operation-limited (S11-17).",
    "S11-17 (the same figure in normal operation) and the report's section on the diurnal "
    "profile. Nothing was selected from this figure.",
    (f"{OUT}/hour_season.csv", ABOUT))
def s11_hour_season():
    return _hour_season_figure("all")


@register(
    "S11-17", "11", "hour_season_normal", "The same, in the hours of normal operation",
    "Does the pattern of S11-05 remain when the hours of curtailment, stops and derate are taken "
    "out?",
    "As S11-05, with only the hours of normal operation (no operation-limited flag, D-22), on the "
    "same colour scale.",
    "In normal operation the largest cell of each model is in the local hour(s) {pk_hours}, and "
    "the largest cell is {big_season} at {big_hour}:00 ({big_v:.1f} per cent, {big_m}). For XGB "
    "in the autumn at 11:00 the error falls from {x_all:.1f} per cent on all hours to "
    "{x_norm:.1f} in normal operation: the autumn peak of S11-05 is mostly the operation-limited "
    "hours.",
    "The report states both views (D-22); the diurnal profile is read in normal operation, and "
    "the limited hours are a separate topic.",
    (f"{OUT}/hour_season.csv", ABOUT))
def s11_hour_season_normal():
    return _hour_season_figure("normal_operation")


# ── S11-06: the two routes (S-31) ───────────────────────────────────────────

@register(
    "S11-06", "11", "route_comparison", "Indirect route against direct route, validation and test",
    "Is it better to forecast the clearness index and multiply by the clear-sky power, or to "
    "forecast the power directly?",
    "The nRMSE in kW of the direct route minus that of the indirect route (percentage points), for "
    "each model and horizon, mean over seeds, scored on the same hours against the actual AC "
    "(C-16). Positive: the indirect route is better. One panel per split.",
    "The direct route is better in {neg_val} of the 18 pairs of a model and a horizon on "
    "validation and in {neg_test} of 18 on test. The differences are between {lo:+.2f} and "
    "{hi:+.2f} pp, and the sign differs between the two periods for {flip} of the 18 pairs: the "
    "direct route is nominally better on validation, and the evidence on test is mixed.",
    "The statement on the routes: a difference of this size is within the seed and period "
    "uncertainty (S11-01, S10-05).",
    (f"{FINAL}/evaluate/val/route_comparison.csv", f"{FINAL}/test/test/route_comparison.csv"))
def s11_route():
    parts = []
    for split, rel in (("val", f"{FINAL}/evaluate/val/route_comparison.csv"),
                       ("test", f"{FINAL}/test/test/route_comparison.csv")):
        r = D.csv(rel)
        r = r[(r["route"] == "direct") & r["model"].isin(MODEL6)]
        parts.append(r.groupby(["model", "horizon"])["advantage_pp"].mean().reset_index().assign(
            split=split))
    adv = pd.concat(parts)
    fig, axes = subplots(1, 2, width=8.0, height=3.4, sharey=True)
    for ax, split in zip(axes, ("val", "test"), strict=False):
        d = adv[adv["split"] == split]
        for k, m in enumerate(MODEL6):
            ys = [d[(d["model"] == m) & (d["horizon"] == h)]["advantage_pp"].iloc[0]
                  for h in (1, 2, 3)]
            ax.plot(np.array([1, 2, 3]) + (k - 2.5) * 0.06, ys, _marker(m), color=STYLE[m][0],
                    ms=5, label=NAMES[m])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
        ax.set_title("validation" if split == "val" else "test", fontsize=9)
    axes[0].set_ylabel("direct minus indirect nRMSE, pp\n(positive: indirect better)")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    pv = adv[adv["split"] == "val"].set_index(["model", "horizon"])["advantage_pp"]
    pt = adv[adv["split"] == "test"].set_index(["model", "horizon"])["advantage_pp"]
    assert len(pv) == len(pt) == 18
    assert not (pv == 0).any() and not (pt == 0).any()
    assert (pv < 0).sum() > 9 and (pt < 0).sum() < (pv < 0).sum() and (pt > 0).any()
    return fig, {"neg_val": int((pv < 0).sum()), "neg_test": int((pt < 0).sum()),
                 "lo": float(min(pv.min(), pt.min())), "hi": float(max(pv.max(), pt.max())),
                 "flip": int(((pv > 0) != (pt.reindex(pv.index) > 0)).sum())}


# ── S11-07: the SHAP groups (S-32) ──────────────────────────────────────────

SHAP_VAL = f"{FINAL}/explain/xgb/xgb_shap_groups.csv"
SHAP_TEST = f"{FINAL}/test/shap_test/xgb_shap_groups_test.csv"
RANK_VAL = f"{FINAL}/explain/xgb/xgb_shap_rank_stability.csv"
RANK_TEST = f"{FINAL}/test/shap_test/xgb_shap_rank_stability_test.csv"


@register(
    "S11-07", "11", "shap_groups", "XGB: what the SHAP values attribute the forecast to",
    "Which groups of features carry the XGB forecast, and is it the same on validation and on "
    "test?",
    "Left: the share of the mean absolute SHAP value (in kW) of each feature group, for each route "
    "and horizon, mean over the three seeds; validation (upper bar) and test (lower bar) of each "
    "cell. Right: the Spearman rank correlation of the feature importances between two horizons "
    "of the same route, and between two seeds.",
    "The group with the largest share is the same on validation and on test in {same_top} of the 6 "
    "cells ({top}). The share of the weather-forecast group (GFS) is between {nwp_lo:.0f} and "
    "{nwp_hi:.0f} per cent. The rank correlation between two horizons, or between two seeds, is "
    "at least {rho_val:.2f} on validation and {rho_test:.2f} on test.",
    "The report's section on explanations: the model relies on the forecast fields, not on "
    "same-hour power.",
    (SHAP_VAL, SHAP_TEST, RANK_VAL, RANK_TEST))
def s11_shap_groups():
    sv, st = D.csv(SHAP_VAL), D.csv(SHAP_TEST)
    s = pd.concat([sv, st])
    m = s.groupby(["split", "target", "horizon", "group"])["share_pct"].mean().reset_index()
    groups = list(m.groupby("group")["share_pct"].mean().sort_values(ascending=False).index)
    palette = [C["blue"], C["orange"], C["grey"], C["green"], C["purple"], C["red"], C["sky"],
               C["yellow"]]
    assert len(groups) <= len(palette)
    colour = dict(zip(groups, palette, strict=False))
    fig, (ax, ax2) = subplots(1, 2, width=8.4, height=4.0, gridspec_kw={"width_ratios": [2, 1]})
    y, ticks, labels = 0.0, [], []
    top = {}
    for t, h in CELLS:
        for split in ("val", "test"):
            d = m[(m["split"] == split) & (m["target"] == t) & (m["horizon"] == h)]
            left = 0.0
            for g in groups:
                v = float(d[d["group"] == g]["share_pct"].sum())
                ax.barh(y, v, left=left, color=colour[g], edgecolor="white", lw=0.4,
                        label=g if (t, h, split) == CELLS[0] + ("val",) else None)
                left += v
            top[(t, h, split)] = d.sort_values("share_pct").iloc[-1]["group"]
            ticks.append(y)
            labels.append(f"{t} D+{h} {'val' if split == 'val' else 'test'}")
            y += 1.0
        y += 0.4
    ax.set_yticks(ticks, labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("share of mean |SHAP| in kW, %")
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    rk = pd.concat([D.csv(RANK_VAL).assign(split="val"), D.csv(RANK_TEST).assign(split="test")])
    rk = rk.groupby(["split", "target", "compare"])["spearman"].mean().reset_index()
    for k, split in enumerate(("val", "test")):
        d = rk[rk["split"] == split]
        ax2.plot(d["spearman"], np.arange(len(d)) + (k - 0.5) * 0.25, "os"[k],
                 color=(C["orange"], "black")[k], label="validation" if split == "val" else "test")
        if k == 0:
            names = [f"{a} {b}" for a, b in zip(d["target"], d["compare"], strict=False)]
            ax2.set_yticks(range(len(d)), names, fontsize=6.5)
    ax2.set_xlim(0.5, 1.0)
    ax2.invert_yaxis()
    ax2.set_xlabel("Spearman correlation of importances")
    ax2.legend(fontsize=7, loc="lower left")
    same = sum(top[(t, h, "val")] == top[(t, h, "test")] for t, h in CELLS)
    nwp = m[m["group"] == "nwp_gfs"]["share_pct"]
    return fig, {"same_top": same, "top": ", ".join(sorted(set(top.values()))),
                 "nwp_lo": float(nwp.min()), "nwp_hi": float(nwp.max()),
                 "rho_val": float(rk[rk["split"] == "val"]["spearman"].min()),
                 "rho_test": float(rk[rk["split"] == "test"]["spearman"].min())}


# ── S11-08: the TFT attention (S-33) ────────────────────────────────────────

ATT_LAG = f"{FINAL}/explain/tft/tft_attention_lag.csv"
ATT_SUM = f"{FINAL}/explain/tft/tft_attention_summary.csv"
TFT_STEP = {"tft96": "tft_96", "tft168": "tft_168", "tftbase": "tft"}


@register(
    "S11-08", "11", "tft_attention", "TFT: how the attention is spread over the encoder window",
    "Over which past hours does each TFT look, and is there a pattern at 24-hour lags?",
    "The mean attention weight at each hour before the forecast origin (encoder position, "
    "negative hours), for the three TFT models and the two routes, validation hours, all blocks "
    "together. The dotted verticals are multiples of 24 hours.",
    "The mean attention over the encoder positions is almost entirely a daily cycle: the "
    "24-hour pattern accounts for {cyc_lo:.0%} to {cyc_hi:.0%} of its variance in the six "
    "cases ({cyc_96i:.0%} for TFT-96 indirect), and the clock hour of the strongest attention "
    "differs between the cases (from {pk_lo} to {pk_hi}). The attention on the clock hour of the "
    "forecast hour is {lo_same:.2f} to {hi_same:.2f} times what a uniform attention would give, "
    "and the weight on the oldest encoder position relative to uniform is between "
    "{lo_first:.2f} and {hi_first:.2f}; only the TFT-168 indirect model is above one. The "
    "attention of a model is descriptive: it does not show that the model uses those hours.",
    "The report's section on the TFT attention (the submitted text described sharper peaks for the "
    "shorter window); no change of any model.",
    (ATT_LAG, ATT_SUM))
def s11_tft_attention():
    lag, summ = D.csv(ATT_LAG), D.csv(ATT_SUM)
    lag = lag[lag["block"].astype(str) == "all"]
    summ = summ[summ["block"].astype(str) == "all"]
    fig, axes = subplots(2, 1, width=7.6, height=5.2, sharex=True)
    for ax, t in zip(axes, ("indirect", "direct"), strict=False):
        for step, m in TFT_STEP.items():
            d = lag[(lag["step"] == step) & (lag["target"] == t)].sort_values("lag_h")
            assert len(d) > 0, (step, t)
            ax.plot(d["lag_h"], d["attention"], STYLE[m][1], color=STYLE[m][0], ms=0, lw=1.1,
                    label=NAMES[m])
        for k in range(1, 8):
            ax.axvline(-24 * k, color=C["grey"], lw=0.6, ls=":")
        ax.set_title(f"{t} route", fontsize=8)
        ax.set_ylabel("mean attention weight")
    axes[1].set_xlabel("hours before the forecast origin")
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    s = summ[summ["step"].isin(TFT_STEP)]
    assert len(s) == 6
    above = s[s["first_position_ratio"] > 1]
    assert list(above["step"]) == ["tft168"] and list(above["target"]) == ["indirect"]
    cyc, pk = {}, {}
    for (step, t), d in lag[lag["step"].isin(TFT_STEP)].groupby(["step", "target"]):
        a = d.sort_values("lag_h")["attention"].reset_index(drop=True)
        hr = (d.sort_values("lag_h")["lag_h"] % 24).reset_index(drop=True)
        folded = a.groupby(hr).transform("mean")
        cyc[(step, t)] = float(((folded - a.mean()) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        pk[(step, t)] = int(a.groupby(hr).mean().idxmax())
    assert len(cyc) == 6 and min(cyc.values()) > 0.7, "the daily cycle is no longer dominant"
    return fig, {"cyc_lo": min(cyc.values()), "cyc_hi": max(cyc.values()),
                 "cyc_96i": cyc[("tft96", "indirect")], "pk_lo": min(pk.values()),
                 "pk_hi": max(pk.values()), "lo_same": float(s["same_hour_ratio"].min()),
                 "hi_same": float(s["same_hour_ratio"].max()),
                 "lo_first": float(s["first_position_ratio"].min()),
                 "hi_first": float(s["first_position_ratio"].max())}


# ── S11-09: the encoder length (S-34) ───────────────────────────────────────

@register(
    "S11-09", "11", "encoder_length", "TFT encoder of 168 hours against 96 hours",
    "Does the longer encoder help, and is the answer the same on validation and on test?",
    "For each route and horizon, the nRMSE of the TFT with a 168-hour encoder minus that of the "
    "TFT "
    "with a 96-hour encoder, in kW (percentage points). Negative: the longer encoder is better. "
    "A filled marker is a significant paired Diebold-Mariano test (HLN, Bonferroni over the six "
    "cells of the split). Each model was tuned separately, one seed each.",
    "On validation the difference is between {v_lo:+.2f} and {v_hi:+.2f} pp, and no cell is "
    "significant. On test it is between {t_lo:+.2f} and {t_hi:+.2f} pp, with the sign negative for "
    "the indirect route in {t_ind_neg} of 3 horizons and positive for the direct route in "
    "{t_dir_pos} of 3; the number of significant cells is {n_sig} of 12. A consistent advantage "
    "of the shorter window is not found: the sign depends on the route.",
    "The report's section on the encoder length; the submitted statement of a consistent "
    "advantage of the 96-hour window (M48) is not supported.",
    (f"{OUT}/encoder_length.csv", ABOUT))
def s11_encoder_length():
    e = T("encoder_length")
    fig, ax = subplots(height=3.3, width=7.4)
    for k, split in enumerate(("val", "test")):
        d = e[e["split"] == split].reset_index(drop=True)
        for i, r in d.iterrows():
            ax.plot(i + (k - 0.5) * 0.2, r["diff_a_minus_b_pp"], "o" if split == "val" else "D",
                    color=C["orange"] if split == "val" else "black",
                    mfc=(C["orange"] if split == "val" else "black") if r["significant"]
                    else "white", ms=6, label=("validation" if split == "val" else "test")
                    if i == 0 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(CELLS)), [_cell(t, h) for t, h in CELLS], fontsize=7)
    ax.set_ylabel("TFT-168 minus TFT-96, pp\n(negative: longer encoder better)")
    ax.legend(fontsize=7, loc="upper right")
    v, t = e[e["split"] == "val"], e[e["split"] == "test"]
    assert not v["significant"].any()
    tt = t.set_index(["target", "horizon"])["diff_a_minus_b_pp"]
    return fig, {"v_lo": float(v["diff_a_minus_b_pp"].min()),
                 "v_hi": float(v["diff_a_minus_b_pp"].max()),
                 "t_lo": float(t["diff_a_minus_b_pp"].min()),
                 "t_hi": float(t["diff_a_minus_b_pp"].max()),
                 "t_ind_neg": int((tt.loc["indirect"] < 0).sum()),
                 "t_dir_pos": int((tt.loc["direct"] > 0).sum()),
                 "n_sig": int(e["significant"].sum())}


# ── S11-10: the TFT-96 search and its hyper-parameters (S-36) ───────────────

TRIALS96 = f"{FINAL}/tft96/trials.csv"
HPARAM = f"{FINAL}/explain/hparam_importance.csv"


@register(
    "S11-10", "11", "tft96_search", "TFT-96: how the search converged and what mattered",
    "Did the 20 trials of TFT-96 converge, and which hyper-parameter explained the differences?",
    "Top: the validation nRMSE in kW of every trial (points) and the running best (line), for each "
    "route; the star is the winning trial. Bottom: the fANOVA importance of each hyper-parameter "
    "(mean and standard deviation over ten seeds of the importance model).",
    "The best trial is number {bi_ind} of {n_tr} for the indirect route and number {bi_dir} for "
    "the "
    "direct route; a winner that late means that the direct search had not visibly converged. "
    "The most important hyper-parameter is {top_ind} for the indirect route ({imp_ind:.0f} per "
    "cent) and {top_dir} for the direct route ({imp_dir:.0f} per cent). With {n_tr} trials the "
    "importance model has few points; the values are descriptive.",
    "The report's section on the hyper-parameters of the TFT; the budget stays that of the "
    "submitted study (D-37).",
    (TRIALS96, HPARAM))
def s11_tft96_search():
    tr, hp = D.csv(TRIALS96), D.csv(HPARAM)
    hp = hp[hp["step"] == "tft96"]
    fig, axes = subplots(2, 2, width=8.0, height=5.4)
    best, topn = {}, {}
    for j, t in enumerate(("indirect", "direct")):
        d = tr[(tr["target"] == t) & (tr["state"] == "COMPLETE")].sort_values("trial")
        ax = axes[0, j]
        ax.plot(d["trial"] + 1, d["value_kw"], "o", color=C["grey"], ms=4)
        run = d["value_kw"].cummin()
        ax.plot(d["trial"] + 1, run, "-", color=C["blue"])
        b = d.loc[d["value_kw"].idxmin()]
        ax.plot(b["trial"] + 1, b["value_kw"], "*", color=C["red"], ms=12)
        ax.set_title(f"{t} route", fontsize=8)
        ax.set_xlabel("trial (counted from 1)")
        ax.set_ylabel("validation nRMSE in kW, %")
        best[t] = (int(b["trial"]) + 1, int((run.diff().fillna(-1) < 0).to_numpy().nonzero()[0][-1])
                   + 1)
        h = hp[hp["target"] == t].sort_values("importance_mean")
        axes[1, j].barh(h["param"], h["importance_mean"] * 100, xerr=h["importance_std"] * 100,
                        color=STYLE["tft_96"][0], capsize=2)
        axes[1, j].set_xlabel("fANOVA importance, %")
        topn[t] = (h.iloc[-1]["param"], float(h.iloc[-1]["importance_mean"]) * 100)
    n_tr = int(tr[tr["state"] == "COMPLETE"].groupby("target")["trial"].count().min())
    assert best["direct"][0] > 0.75 * n_tr, "the direct search converged early"
    return fig, {"bi_ind": best["indirect"][0], "bi_dir": best["direct"][0], "n_tr": n_tr,
                 "top_ind": topn["indirect"][0], "imp_ind": topn["indirect"][1],
                 "top_dir": topn["direct"][0], "imp_dir": topn["direct"][1]}


# ── S11-11: bias and variance (S-39, L-15) ──────────────────────────────────

@register(
    "S11-11", "11", "bias_variance",
    "Bias² and variance of the error, and the bias in normal operation",
    "How much of the squared error is a systematic offset, and where does the offset come from?",
    "Left: the share of the mean squared error that is the squared mean error (bias²; the rest is "
    "the variance of the error), mean over the six route and horizon cells, validation and test, "
    "all daytime hours. Middle: the mean error in kW on all hours and on the hours of normal "
    "operation (no curtailment, stop or derate; D-22). Right: the share of the squared error of a "
    "seed that is the variance between the three seeds (XGB and BiLSTM).",
    "Bias² is below {b2_max:.1%} of the squared error for the six models on validation and below "
    "{b2_test_max:.1%} on test, except PatchTST ({pt_b2:.1%} on test). Every model has a positive "
    "mean error on all hours, {me_lo:+.0f} to {me_hi:+.0f} kW on test; on the hours of normal "
    "operation it is {me_n_lo:+.0f} to {me_n_hi:+.0f} kW for XGB, BiLSTM and the three TFTs, so "
    "nearly all of their positive mean error is made on the {n_lim} operation-limited hours; "
    "PatchTST keeps {pt_n:+.0f} kW in normal operation. The variance between seeds is at most "
    "{sv_max:.1%} of the squared error.",
    "The statement of the bias of the neural forecasts (D-37) as a property of the "
    "operation-limited hours, and the report's section on the error decomposition.",
    (f"{OUT}/decomposition.csv", f"{OUT}/decomposition_seeds.csv", ABOUT))
def s11_bias_variance():
    d, e = T("decomposition"), T("decomposition_seeds")
    fig, axes = subplots(1, 3, width=8.8, height=3.6)
    a = d[d["view"] == "all"]
    shares = a.groupby(["model", "split"])["bias2_share"].mean().unstack()
    x = np.arange(len(ALL))
    axes[0].bar(x - 0.2, [shares.loc[m, "val"] * 100 for m in ALL], 0.4, color=C["orange"],
                label="validation")
    axes[0].bar(x + 0.2, [shares.loc[m, "test"] * 100 for m in ALL], 0.4, color="black",
                label="test")
    axes[0].set_xticks(x, [LABEL[m] for m in ALL], rotation=60, ha="right", fontsize=6)
    axes[0].set_ylabel("bias² share of MSE, %")
    axes[0].legend(fontsize=7)
    me = d.groupby(["model", "split", "view"])["me_kw"].mean()
    for k, (split, view, col, hatch) in enumerate((
            ("val", "all", C["orange"], None), ("val", "normal_operation", C["orange"], "//"),
            ("test", "all", "black", None), ("test", "normal_operation", "black", "//"))):
        axes[1].bar(np.arange(len(MODEL6)) + (k - 1.5) * 0.2,
                    [me.loc[(m, split, view)] for m in MODEL6], 0.2, color=col, hatch=hatch,
                    edgecolor="white", alpha=0.85,
                    label=f"{'validation' if split == 'val' else 'test'}, "
                          f"{'all hours' if view == 'all' else 'normal operation'}")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(range(len(MODEL6)), [NAMES[m] for m in MODEL6], rotation=60, ha="right",
                       fontsize=6)
    axes[1].set_ylabel("mean error, kW (forecast minus observed)")
    axes[1].legend(fontsize=5.5, loc="upper left")
    ev = e[e["view"] == "all"].groupby(["model", "split"])["seed_variance_share"].mean().unstack()
    for k, m in enumerate(("xgb_sota", "lstm_sota")):
        axes[2].bar(np.arange(2) + (k - 0.5) * 0.35, [ev.loc[m, "val"] * 100,
                                                      ev.loc[m, "test"] * 100], 0.35,
                    color=STYLE[m][0], label=NAMES[m])
    axes[2].set_xticks([0, 1], ["validation", "test"])
    axes[2].set_ylabel("variance between seeds, % of MSE")
    axes[2].legend(fontsize=7)
    six = a[a["model"].isin(MODEL6)]
    s6 = six.groupby(["model", "split"])["bias2_share"].mean().unstack()
    no_pt = [m for m in MODEL6 if m != "patchtst"]
    me_all = me.xs("all", level="view").unstack()
    me_n = me.xs("normal_operation", level="view").unstack()
    neural_ok = ["xgb_sota", "lstm_sota", "tft_96", "tft_168", "tft"]
    assert (me_all.loc[MODEL6, "test"] > 0).all()
    assert me_n.loc[neural_ok, "test"].abs().max() < me_all.loc[neural_ok, "test"].min()
    nlim = int(round(float(T("metrics_kw").query(
        "view=='all' and model=='xgb_sota' and split=='test' and seed==42 and target=='direct' "
        "and horizon==1")["n"].iloc[0]
        - T("metrics_kw").query(
            "view=='normal_operation' and model=='xgb_sota' and split=='test' and seed==42 and "
            "target=='direct' and horizon==1")["n"].iloc[0])))
    return fig, {"b2_max": float(s6.loc[no_pt, "val"].max()),
                 "b2_test_max": float(s6.loc[no_pt, "test"].max()),
                 "pt_b2": float(s6.loc["patchtst", "test"]),
                 "me_lo": float(me_all.loc[MODEL6, "test"].min()),
                 "me_hi": float(me_all.loc[MODEL6, "test"].max()),
                 "me_n_lo": float(me_n.loc[neural_ok, "test"].min()),
                 "me_n_hi": float(me_n.loc[neural_ok, "test"].max()),
                 "sv_max": float(ev.loc[["xgb_sota", "lstm_sota"]].max().max()), "n_lim": nlim,
                 "pt_n": float(me_n.loc["patchtst", "test"])}


# ── S11-12: the production quintiles (S-40, M45) ────────────────────────────

@register(
    "S11-12", "11", "production_quintiles", "XGB: the mean error by production quintile, two ways",
    "Is the XGB biased at low and high power, or does the pattern come from how the hours are "
    "binned?",
    "The mean error of the XGB (kW, forecast minus observed), seed 42, D+1, in five bins of equal "
    "validation count. Upper row: bins of the observed power (as in the submitted report). Lower "
    "row: bins of the forecast itself, which is known at issuance. Left: direct route, right: "
    "indirect route (in kW). Validation and test. The numbers above the bars are the share of "
    "operation-limited hours in the test bin. Third row, a control: the same XGB forecast made "
    "conditionally unbiased on validation by an isotonic regression, binned by the observed power.",
    "Binned by the observed power the lowest bin has a mean error of {obs_lo:+.0f} kW and the "
    "highest {obs_hi:+.0f} kW on test (direct route): a positive then negative pattern. Binned by "
    "the forecast it is {fc_lo:+.0f} and {fc_hi:+.0f} kW: the signs are the opposite at both "
    "ends. The control forecast, unbiased by construction on validation, gives {ctl_lo:+.0f} and "
    "{ctl_hi:+.0f} kW in the same observed bins on test, so binning by the outcome produces the "
    "first pattern without any conditional bias in the model. In the lowest observed bin "
    "{ol_test:.0%} of the test hours are operation-limited against {ol_val:.0%} on validation.",
    "The report's section on the production quintiles (M45): the reading of the submitted "
    "profile as a physical effect is not supported; the same table exists for every model.",
    (f"{OUT}/quintiles.csv", f"{OUT}/quintile_control.csv", ABOUT))
def s11_quintiles():
    q = T("quintiles")
    q = q[(q["model"] == "xgb_sota") & (q["seed"] == 42) & (q["horizon"] == 1)]
    ctl = T("quintile_control")
    ctl = ctl[ctl["horizon"] == 1]
    fig, axes = subplots(3, 2, width=8.0, height=7.4, sharey=True)
    for j, t in enumerate(("direct", "indirect")):
        ax = axes[2, j]
        for k, split in enumerate(("val", "test")):
            d = ctl[(ctl["target"] == t) & (ctl["split"] == split)].sort_values("bin")
            ax.bar(d["bin"] + (k - 0.5) * 0.38, d["me_calibrated_kw"], 0.38,
                   color=C["orange"] if split == "val" else "black")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"control (unbiased forecast), binned by the observed power, {t} route",
                     fontsize=7)
        ax.set_xticks(range(1, 6), [f"Q{b}" for b in range(1, 6)])
    axes[2, 0].set_ylabel("mean error, kW")
    for i, by in enumerate(("observed", "forecast")):
        for j, t in enumerate(("direct", "indirect")):
            ax = axes[i, j]
            for k, split in enumerate(("val", "test")):
                d = q[(q["binned_by"] == by) & (q["target"] == t) & (q["split"] == split)
                      ].sort_values("bin")
                ax.bar(d["bin"] + (k - 0.5) * 0.38, d["me_kw"], 0.38,
                       color=C["orange"] if split == "val" else "black",
                       label="validation" if split == "val" else "test")
                if split == "test":
                    for b, v, o in zip(d["bin"], d["me_kw"], d["op_limited_share"], strict=False):
                        ax.text(b + 0.19, v + (8 if v >= 0 else -8), f"{o:.0%}", fontsize=6,
                                ha="center", va="bottom" if v >= 0 else "top")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"binned by the {by} power, {t} route", fontsize=8)
            ax.set_xticks(range(1, 6), [f"Q{b}" for b in range(1, 6)])
    axes[0, 0].set_ylabel("mean error, kW")
    axes[1, 0].set_ylabel("mean error, kW")
    axes[0, 0].legend(fontsize=7)

    def me(by, split, b):
        r = q[(q["binned_by"] == by) & (q["target"] == "direct") & (q["split"] == split)
              & (q["bin"] == b)]
        return float(r["me_kw"].iloc[0])

    obs_lo, obs_hi = me("observed", "test", 1), me("observed", "test", 5)
    fc_lo, fc_hi = me("forecast", "test", 1), me("forecast", "test", 5)
    assert obs_lo > 0 > obs_hi and fc_hi > 0 and fc_hi > fc_lo, "the pattern changed"
    ct = ctl[(ctl["target"] == "direct") & (ctl["split"] == "test")].set_index("bin")
    ctl_lo, ctl_hi = float(ct.loc[1, "me_calibrated_kw"]), float(ct.loc[5, "me_calibrated_kw"])
    assert ctl_lo > 0 > ctl_hi, "the control does not reproduce the pattern"

    def ol(split):
        r = q[(q["binned_by"] == "observed") & (q["target"] == "direct") & (q["split"] == split)
              & (q["bin"] == 1)]
        return float(r["op_limited_share"].iloc[0])

    return fig, {"obs_lo": obs_lo, "obs_hi": obs_hi, "fc_lo": fc_lo, "fc_hi": fc_hi,
                 "ctl_lo": ctl_lo, "ctl_hi": ctl_hi, "ol_test": ol("test"), "ol_val": ol("val")}


# ── S11-13: the ablation of the final run (S-41) ────────────────────────────

ABL_VAL = f"{FINAL}/report/report_ablation.csv"
ABL_TEST = f"{FINAL}/report/report_ablation_test.csv"
ABL_MODES = f"{FINAL}/ablation/ablation_modes.csv"


@register(
    "S11-13", "11", "ablation_final", "XGB: feature-group ablation, validation and test",
    "Which groups of features does the XGB need, and do the test results agree with validation?",
    "The change of the nRMSE in kW (percentage points) when a group of features is removed and the "
    "model is refitted, against the full model; mean over three seeds, validation and test. The "
    "letter is the verdict (H helpful, N neutral, U harmful) by the rule of the margin of two "
    "pooled seed standard deviations, at least 0.05 pp. n/a: the mode does not apply to that cell "
    "(the group is absent from the input of that cell, which the ablation record lists).",
    "Removing all but the minimal set raises the error by {min_lo:.2f} to {min_hi:.2f} pp on "
    "validation, and the GFS group by {gfs_lo:.2f} to {gfs_hi:.2f} pp; the other groups change it "
    "by at most {oth:.2f} pp. The verdicts agree between validation and test in {agree} of "
    "{n_all} cells.",
    "The report's section on the ablation: the weather-forecast group is the one that matters; "
    "the history groups change the error by about the size of the seed margin (M41, C-74).",
    (ABL_VAL, ABL_TEST, ABL_MODES))
def s11_ablation():
    v, t = D.csv(ABL_VAL), D.csv(ABL_TEST)
    mrec = D.csv(ABL_MODES)
    groups_of = {(r.target, r.horizon): set(str(r.groups).split(","))
                 for r in mrec[mrec["mode"] == "full"].itertuples()}

    def absent(tg, h, md):
        """The mode does not exist for the cell because its input has no such group."""
        gone = "time_encodings" if md == "minimal" else md.removeprefix("no_")
        return gone not in groups_of[(tg, h)]

    modes = sorted(v["mode"].unique())
    fig, axes = subplots(1, 2, width=8.8, height=4.0, sharey=True)
    letters = {"helpful": "H", "neutral": "N", "harmful": "U"}
    vmax = max(v["mean_delta_pp"].abs().max(), t["mean_delta_pp"].abs().max())
    for ax, (split, d, col) in zip(axes, (("validation", v, "verdict"),
                                          ("test", t, "test_label")), strict=False):
        mat = np.full((len(modes), len(CELLS)), np.nan)
        lab = [[""] * len(CELLS) for _ in modes]
        for i, md in enumerate(modes):
            for j, (tg, h) in enumerate(CELLS):
                r = d[(d["mode"] == md) & (d["target"] == tg) & (d["horizon"] == h)]
                if len(r):
                    mat[i, j] = r["mean_delta_pp"].iloc[0]
                    lab[i][j] = f"{mat[i, j]:+.2f} {letters.get(r[col].iloc[0], '?')}"
                else:
                    assert absent(tg, h, md), f"{md} {tg} D+{h}: neither scored nor absent"
                    lab[i][j] = "n/a"
        ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(modes)):
            for j in range(len(CELLS)):
                ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(CELLS)), [_cell(tg, h) for tg, h in CELLS], fontsize=6)
        ax.set_title(split, fontsize=9)
        ax.grid(False)
    axes[0].set_yticks(range(len(modes)), modes, fontsize=7)
    a = v[v["mode"] == "minimal"]["mean_delta_pp"]
    g = v[v["mode"] == "no_nwp_gfs"]["mean_delta_pp"]
    other = v[~v["mode"].isin(["minimal", "no_nwp_gfs"])]["mean_delta_pp"].abs()
    merged = v.merge(t, on=["target", "horizon", "mode"], suffixes=("_v", "_t"))
    agree = int((merged["verdict"] == merged["test_label"]).sum())
    return fig, {"min_lo": float(a.min()), "min_hi": float(a.max()), "gfs_lo": float(g.min()),
                 "gfs_hi": float(g.max()), "oth": float(other.max()), "agree": agree,
                 "n_all": len(merged)}


# ── S11-14: PatchTST against XGB and climatology (S-35) ────────────────────

@register(
    "S11-14", "11", "patchtst_gap", "PatchTST against XGB and against climatology",
    "How far is the univariate PatchTST from the XGB with all inputs, and from a climatology?",
    "For each route and horizon, the nRMSE of PatchTST minus that of XGB (seed 42) and minus that "
    "of climatology, in kW (percentage points), validation and test. Positive: PatchTST is worse. "
    "A filled marker is a significant paired test (HLN, Bonferroni over the six cells of the "
    "split and reference).",
    "PatchTST is worse than XGB by {x_lo:.2f} to {x_hi:.2f} pp, significantly in {x_sig} of 12 "
    "cells. Against climatology the gap is {c_lo:+.2f} to {c_hi:+.2f} pp, significant in {c_sig} "
    "of 12 cells, all of them in the worse direction. A gap of this size does not support the "
    "reading of PatchTST as a lower bound within about one point of XGB.",
    "The report's section on PatchTST (M49): the submitted bound on the value of feature "
    "engineering is not stated.",
    (f"{OUT}/patchtst_gap.csv", ABOUT))
def s11_patchtst_gap():
    g = T("patchtst_gap")
    fig, ax = subplots(height=3.4, width=7.6)
    for ref, off in (("xgb_sota", -0.18), ("climatology", 0.18)):
        for k, split in enumerate(("val", "test")):
            d = g[(g["model_b"] == ref) & (g["split"] == split)].reset_index(drop=True)
            for i, r in d.iterrows():
                ax.plot(i + off + (k - 0.5) * 0.12, r["diff_a_minus_b_pp"], "os"[k],
                        color=STYLE[ref][0],
                        mfc=STYLE[ref][0] if r["significant"] else "white", ms=5.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(CELLS)), [_cell(t, h) for t, h in CELLS], fontsize=7)
    ax.set_ylabel("PatchTST minus reference, pp")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=STYLE["xgb_sota"][0], label="against XGB"),
        Line2D([], [], marker="o", ls="", color=STYLE["climatology"][0],
               label="against climatology"),
        Line2D([], [], marker="o", ls="", color="grey", label="validation (circle)"),
        Line2D([], [], marker="s", ls="", color="grey", label="test (square)")],
        fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    x, c = g[g["model_b"] == "xgb_sota"], g[g["model_b"] == "climatology"]
    assert (x["diff_a_minus_b_pp"] > 0).all()
    assert (c[c["significant"]]["diff_a_minus_b_pp"] > 0).all()
    return fig, {"x_lo": float(x["diff_a_minus_b_pp"].min()),
                 "x_hi": float(x["diff_a_minus_b_pp"].max()), "x_sig": int(x["significant"].sum()),
                 "c_lo": float(c["diff_a_minus_b_pp"].min()),
                 "c_hi": float(c["diff_a_minus_b_pp"].max()), "c_sig": int(c["significant"].sum())}


# ── S11-15: the comparison with the literature, after aligning the normalisation (L-20) ──

@register(
    "S11-15", "11", "literature_alignment",
    "Our error and the literature, on the same normalisation",
    "How do our errors compare with the day-ahead errors of the papers, once both are divided by "
    "the mean daytime power?",
    "The nRMSE at D+1 normalised by the mean observed daytime power (L-16), mean over seeds and "
    "over the two routes, for every model and reference on validation and test; the papers' "
    "numbers (the review's C1 and C2) as vertical lines. The differences of climate, NWP and "
    "resolution are listed in the panel.",
    "On this normalisation our six models are between {lo_val:.0f} and {hi_val:.0f} per cent on "
    "validation and between {lo_test:.0f} and {hi_test:.0f} per cent on test, against "
    "{lit_lo:.1f} to {lit_hi:.1f} per cent in the papers. The numbers are not comparable one to "
    "one: a hyper-arid site where most hours are clear, a coarser NWP, hourly instead of 15-minute "
    "data, one plant instead of a mean over seven, and an eight-month test period. No ranking "
    "against the papers is made.",
    "The report's literature paragraph (L-20, L-16): the papers' methods transfer, their numbers "
    "are not targets.",
    (f"{OUT}/literature_alignment.csv", "docs/literature_review.md"))
def s11_literature():
    a = T("literature_alignment")
    ours = a[a["model"] != "literature"]
    lit = a[a["model"] == "literature"]
    g = ours.groupby(["model", "split"])["nRMSE_mean_pct"].mean().unstack()
    fig, ax = subplots(height=3.9, width=7.8)
    for i, m in enumerate(ALL):
        ax.plot(g.loc[m, "val"], i, "o", color=STYLE[m][0], ms=6)
        ax.plot(g.loc[m, "test"], i, "D", color=STYLE[m][0], ms=6)
        ax.plot([g.loc[m, "val"], g.loc[m, "test"]], [i, i], "-", color=STYLE[m][0], lw=0.8)
    for k, (_, r) in enumerate(lit.iterrows()):
        ax.axvline(r["nRMSE_mean_pct"], color=C["grey"], ls="--", lw=0.9)
        ax.text(r["nRMSE_mean_pct"] + 0.4, 0.2 + 0.9 * k,
                f"{r['source'].split(' ')[0]} {r['nRMSE_mean_pct']:.1f}", fontsize=6.5,
                ha="left", va="bottom", color="black",
                bbox={"fc": "white", "ec": "none", "pad": 0.5})
    ax.set_yticks(range(len(ALL)), [LABEL[m] for m in ALL], fontsize=7)
    ax.set_xlabel("nRMSE at D+1 normalised by the mean daytime power, % (circle: validation, "
                  "diamond: test)")
    ax.set_xlim(0, 90)
    six = g.loc[MODEL6]
    lv = lit["nRMSE_mean_pct"]
    return fig, {"lo_val": float(six["val"].min()), "hi_val": float(six["val"].max()),
                 "lo_test": float(six["test"].min()), "hi_test": float(six["test"].max()),
                 "lit_lo": float(lv[lv < 50].min()), "lit_hi": float(lv[lv < 50].max())}


# ── S11-16: the six models of the final run (S-22) ──────────────────────────

@register(
    "S11-16", "11", "model_cards_final",
    "The six models of the final run: inputs, budgets and time",
    "What does each model see, how was it tuned in the final run, and what did it cost?",
    "One row per model, read from the configurations and the step records of the final run: the "
    "number of trials per study, the seeds, the epochs or steps, the history window, the output "
    "and the time of the step (hours).",
    "The final run used the budgets of the submitted study: {xgb_trials} trials and "
    "{n_seeds_xgb} seeds for XGB and BiLSTM, {tft_trials} trials per route for the two tuned "
    "TFTs, fixed settings for the TFT baseline and PatchTST. The five neural models took "
    "{gpu_h:.1f} hours of GPU time in total and XGB {xgb_h:.1f} hours of CPU time.",
    "The report's model table; the development cards (S09-25) keep the development budgets.",
    (f"{FINAL}/config.yaml", f"{FINAL}/xgb/step.json", f"{FINAL}/bilstm/step.json",
     f"{FINAL}/tft96/step.json", f"{FINAL}/tft168/step.json", f"{FINAL}/tftbase/step.json",
     f"{FINAL}/patchtst/step.json", f"{FINAL}/xgb/xgb_config.json",
     f"{FINAL}/bilstm/lstm_sota_config.json", f"{FINAL}/tft96/tft_96_config.json",
     f"{FINAL}/tft168/tft_168_config.json", f"{FINAL}/tftbase/tft_config.json",
     f"{FINAL}/patchtst/patchtst_config.json"))
def s11_model_cards():
    import yaml
    cfg = yaml.safe_load((D.REPO_ROOT / FINAL / "config.yaml").read_text(encoding="utf-8"))
    b = cfg["budgets"]
    secs = {s: D.jload(f"{FINAL}/{s}/step.json")["duration_s"]
            for s in ("xgb", "bilstm", "tft96", "tft168", "tftbase", "patchtst")}
    xcfg = D.jload(f"{FINAL}/xgb/xgb_config.json")
    stop = re.search(r"(\d+) rounds", xcfg["early_stopping"]).group(1)
    cfg_of = {s: D.jload(f"{FINAL}/{s}/{m}_config.json") for s, m in (
        ("bilstm", "lstm_sota"), ("tft96", "tft_96"), ("tft168", "tft_168"), ("tftbase", "tft"),
        ("patchtst", "patchtst"))}

    def out(step):
        q = cfg_of[step]["quantiles"]
        return "point" if q == [0.5] else f"{len(q)} quantiles"

    def hist(step):
        return "own past series" if step == "patchtst" else f"{cfg_of[step]['encoder_hours']} h"

    rows = [
        ["XGB", "3 (one per horizon)", f"{b['xgb']['trials']} trials per study",
         f"{len(b['xgb']['seeds'])} seeds", f"early stop, {stop} rounds", "lags in the features",
         "point", f"{secs['xgb'] / 3600:.1f} (CPU)"],
        ["BiLSTM", "1 (three days)", f"{b['bilstm']['trials']} trials of "
         f"{b['bilstm']['epochs_trial']} epochs", f"{len(b['bilstm']['seeds'])} seeds",
         f"final fit {b['bilstm']['epochs_final']} epochs", hist("bilstm"), out("bilstm"),
         f"{secs['bilstm'] / 3600:.1f} (GPU)"],
        ["TFT-96", "1 (three days)", f"{b['tft96']['trials']} trials per route", "1 seed",
         f"{b['tft96']['epochs']} epochs", hist("tft96"), out("tft96"),
         f"{secs['tft96'] / 3600:.1f} (GPU)"],
        ["TFT-168", "1 (three days)", f"{b['tft168']['trials']} trials per route", "1 seed",
         f"{b['tft168']['epochs']} epochs", hist("tft168"), out("tft168"),
         f"{secs['tft168'] / 3600:.1f} (GPU)"],
        ["TFT baseline", "1 (three days)", "fixed settings", "1 seed",
         f"{b['tftbase']['epochs']} epochs", hist("tftbase"), out("tftbase"),
         f"{secs['tftbase'] / 3600:.1f} (GPU)"],
        ["PatchTST", "1 (three days)", "fixed settings", "1 seed",
         f"{b['patchtst']['max_steps']} steps", hist("patchtst"), out("patchtst"),
         f"{secs['patchtst'] / 3600:.1f} (GPU)"],
    ]
    assert all(len(b[k]["seeds"]) == 1 for k in ("tft96", "tft168", "tftbase", "patchtst"))
    fig, ax = subplots(height=2.7, width=8.8)
    ax.set_axis_off()
    cols = ["model", "models per route", "search", "seeds", "training", "history", "output",
            "step time, h"]
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(6.8)
    t.scale(1, 1.6)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_facecolor(C["light"])
    gpu = sum(secs[s] for s in ("bilstm", "tft96", "tft168", "tftbase", "patchtst"))
    assert b["xgb"]["trials"] == b["bilstm"]["trials"] == 60 and b["tft96"]["trials"] == 20
    return fig, {"xgb_trials": b["xgb"]["trials"], "n_seeds_xgb": len(b["xgb"]["seeds"]),
                 "tft_trials": b["tft96"]["trials"], "gpu_h": gpu / 3600,
                 "xgb_h": secs["xgb"] / 3600}

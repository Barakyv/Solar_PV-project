"""Correction figures: differences from the submitted project, for docs/defense_notes.md.

These are kept apart from the process figures (docs/figures_plan.md §3, rule 1). The
submission's numbers come from submission_evidence/ and the original backup audit; the
rebuild's numbers from train or validation only.
"""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from solarpv.config.paths import REPO_ROOT
from solarpv.evaluation.daytime import daytime_mask
from solarpv.features.contract import ROLLING_WINDOWS_DAYS
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, blank, subplots

TIME_BUG = "outputs/03_eda/tables/table_time_bug.csv"
SUBMITTED_FORECASTS = "submission_evidence/08_lstm_alignment/predictions_lstm_sota_aligned.csv"
SHAP_DIRECT = "00_data_audit/original_backup_2026-11-01/selection_shap_direct_top.csv"
LAG_SCAN = "00_data_audit/original_backup_2026-11-01/weather_lag_scan.csv"
PROFILE_MONTHS = (("winter", "2021-12", 2), ("summer", "2022-07", 3))


def _submitted(target: str) -> pd.DataFrame:
    p = pd.read_csv(REPO_ROOT / SUBMITTED_FORECASTS)
    p = p[(p["target"] == target) & (p["horizon"] == 1)].copy()
    p["t"] = pd.to_datetime(p["timestamp"], utc=True)
    p = p.drop_duplicates("t").set_index("t").sort_index()
    return p[daytime_mask(p.index)]          # the submission's own rule, labels read as UTC


@register(
    "C-01", "correction", "time_shift_profile", "Daily profile read as UTC and as local time",
    "What does reading local time stamps as UTC do to the data?",
    "Mean hourly AC power of six clear train days per season against the clear-sky power, "
    "by true UTC hour (solid) and as the submitted pipeline placed it (dotted, local wall-clock "
    "hour read as UTC).",
    "Read as UTC, production appears {w_off} hours (winter) and {s_off} hours (summer) after the "
    "sun: the clear-day production centre sits {w_old:.0f} and {s_old:.0f} minutes after solar "
    "noon, against {w_new:.1f} and {s_new:.1f} minutes after the conversion.",
    "Every v6 result was computed on the shifted data (F-01); the rebuild converts the export "
    "to UTC at ingestion and checks the production centre against the sun every month (T-08).",
    (D.FEATURES_D1, TIME_BUG), kind="correction")
def c01_profile():
    f = D.table(D.FEATURES_D1, ["site_total_AC_kW", "theo_kw", "op_limited_flag", "split"],
                split="train")
    tb = D.csv(TIME_BUG)
    fig, axes = subplots(1, 2, height=2.8, sharey=True)
    facts = {}
    for ax, (name, month, off) in zip(axes, PROFILE_MONTHS, strict=False):
        m = f.loc[month].copy()
        m["day"] = D.local(m.index).date
        g = m.groupby("day").agg(ac=("site_total_AC_kW", "sum"), th=("theo_kw", "sum"),
                                 lim=("op_limited_flag", "sum"))
        days = (g[g["lim"] == 0]["ac"] / g[g["lim"] == 0]["th"]).nlargest(6).index
        sel = m[m["day"].isin(days)]
        hr = sel.groupby(sel.index.hour)[["site_total_AC_kW", "theo_kw"]].mean().reindex(
            range(24), fill_value=0)
        ax.plot(hr.index, hr["theo_kw"], ls="--", color=C["grey"], label="clear-sky power")
        ax.plot(hr.index, hr["site_total_AC_kW"], color=C["blue"], label="true UTC")
        ax.plot(hr.index, np.roll(hr["site_total_AC_kW"].to_numpy(), off), ls=":", lw=2.2,
                color=C["red"], label="as the submission read it")
        ax.set_title(f"{name}, {month}, six clear days")
        ax.set_xlabel("UTC hour")
        facts[f"{name[0]}_off"] = off
        rows = tb[tb["month"].str.startswith(name)]
        facts[f"{name[0]}_old"] = float(rows[rows["reading"].str.startswith("original")]
                                        ["median_centroid_minus_solar_noon_min"].iloc[0])
        facts[f"{name[0]}_new"] = float(rows[rows["reading"].str.startswith("rebuild")]
                                        ["median_centroid_minus_solar_noon_min"].iloc[0])
    axes[0].set_ylabel("kW")
    axes[0].legend(loc="upper right", fontsize=7)
    return fig, facts


@register(
    "C-02", "correction", "false_downtime", "Zero-output daytime hours by label hour",
    "Where do the 'inverter downtime' hours of the submitted report come from?",
    "Daytime hours of the submitted forecast file (direct target, D+1) per label hour, split "
    "into hours with production and hours with zero output.",
    "{n_zero} of {n_day:,} evaluated daytime hours ({share:.1%}) have zero output, and all of them "
    "carry the labels {first:02d}:00-{last:02d}:00: local hours before sunrise that the shifted "
    "time base counted as daytime. None is a whole-day outage.",
    "The disclosure about inverter downtime is rewritten from the corrected data (C-13); real "
    "stops are now flagged from the inverter data (F1, F3).",
    (SUBMITTED_FORECASTS,), kind="correction")
def c02_downtime():
    q = _submitted("direct")
    zero = q[q["y_true"] == 0]
    by = zero.groupby(zero.index.hour).size().reindex(range(24), fill_value=0)
    allh = q.groupby(q.index.hour).size().reindex(range(24), fill_value=0)
    fig, ax = subplots(height=2.6)
    hrs = range(2, 18)
    ax.bar(hrs, [by[h] for h in hrs], color=C["red"], label="zero output", width=0.75)
    ax.bar(hrs, [allh[h] - by[h] for h in hrs], bottom=[by[h] for h in hrs], color=C["grey"],
           hatch="///", edgecolor="white", label="with production", width=0.75)
    ax.set_xticks(list(hrs), [f"{h:02d}" for h in hrs])
    ax.set_xlabel("label hour in the submitted file")
    ax.set_ylabel("evaluated daytime hours")
    ax.legend(loc="upper right")
    hours = [h for h in range(24) if by[h] > 0]
    assert hours == list(range(hours[0], hours[-1] + 1)) and hours[-1] < 6, "zeros before 06"
    facts ={"n_zero": int(len(zero)), "n_day": int(len(q)), "share": len(zero) / len(q),
             "first": hours[0], "last": hours[-1] + 1}
    return fig, facts


@register(
    "C-03", "correction", "k_clipping", "Clear-sky index of the submission and of the rebuild",
    "What did the clipping of the target at 1.2 do?",
    "Share of daytime hours per k bin of 0.05: the submitted file's indirect truth (its "
    "evaluated test hours) and the rebuild's k (valid daytime train hours).",
    "In the submission {old_cap:.1%} of daytime hours sit exactly at the 1.2 cap and "
    "{old_zero:.1%} near zero; in the rebuild {new_gt:.2%} exceed 1.2 and none is capped. The "
    "pile at the cap came mostly from the time shift: afternoon power divided by morning "
    "clear-sky power (F-06).",
    "k is now an hourly ratio of means against the panel-plane clear-sky power, never clipped, "
    "and converts back to kW exactly (F-06, T-11).",
    (SUBMITTED_FORECASTS, D.FEATURES_D1), kind="correction")
def c03_clip():
    old = _submitted("indirect")["y_true"].dropna()
    f = D.table(D.FEATURES_D1, ["target_k", "k_valid", "daytime", "split"], split="train")
    new = f.loc[f["daytime"] & f["k_valid"], "target_k"].dropna()
    bins = np.round(np.arange(0, 1.45, 0.05), 2)
    ho, _ = np.histogram(old.clip(upper=1.399), bins=bins)
    hn, _ = np.histogram(new.clip(upper=1.399), bins=bins)
    fig, ax = subplots(height=2.7)
    x = bins[:-1]
    ax.bar(x + 0.0125, 100 * ho / len(old), 0.022, color=C["red"], hatch="///",
           edgecolor="white", label=f"submission (n = {len(old):,})")
    ax.bar(x + 0.0375, 100 * hn / len(new), 0.022, color=C["blue"],
           label=f"rebuild, train (n = {len(new):,})")
    ax.set_xlabel("clear-sky index k (last bin includes values above 1.4)")
    ax.set_ylabel("% of daytime hours")
    ax.legend(loc="upper left")
    facts = {"old_cap": float(np.isclose(old, 1.2).mean()), "old_zero": float((old < 0.05).mean()),
             "new_gt": float((new > 1.2).mean())}
    return fig, facts


@register(
    "C-04", "correction", "rolling_windows", "Same-hour windows per horizon, submitted and fixed",
    "Did the multi-day averages use days that were not yet known at issuance?",
    "For each horizon and window length, the days averaged by the submitted code and by the "
    "fixed code; red cells are days after the issuance midnight.",
    "The submitted windows always started one day before the target day, so at D+2 they "
    "included {leak2} unknown day and at D+3 {leak3} unknown days; the fixed windows end on "
    "the last day before issuance at every horizon.",
    "Horizon-aware windows (F-07), with a test that plants the submitted window and must fail "
    "(T-09, T-18).",
    (), kind="correction")
def c04_rolling():
    fig, ax = subplots(height=3.4)
    blank(ax)
    days = list(range(-6, 4))
    x0, cw = 0.2, 0.78 / len(days)
    for i, d in enumerate(days):
        ax.text(x0 + (i + 0.5) * cw, 0.97, str(d), ha="center", fontsize=7.5, color=C["grey"])
    t0x = x0 + (days.index(0) + 1) * cw
    ax.plot([t0x, t0x], [0.02, 0.94], ls="--", color=C["red"], lw=1)
    ax.text(t0x, -0.03, "issuance", ha="center", fontsize=7.5, color=C["red"])
    n = ROLLING_WINDOWS_DAYS[0]
    facts = {}
    row = 0
    for h in (1, 2, 3):
        for name, used in (("submitted", [h - i for i in range(1, n + 1)]),
                           ("fixed", [h - i for i in range(h, h + n)])):
            y = 0.86 - row * 0.15
            ax.text(0.01, y + 0.03, f"D+{h} {name}", fontsize=8, va="center")
            leak = 0
            for i, d in enumerate(days):
                face = "white"
                if d in used:
                    face = C["red"] if d > 0 else C["blue"]
                    leak += d > 0
                ax.add_patch(Rectangle((x0 + i * cw + 0.002, y), cw - 0.004, 0.06, fc=face,
                                       ec=C["yellow"] if d == h else C["grey"],
                                       lw=2 if d == h else 0.5))
            if name == "submitted":
                facts[f"leak{h}"] = leak
            row += 1
    ax.text(0.01, -0.08, f"{n}-day window shown; blue: known day, red: day after issuance, "
            "yellow outline: target day", fontsize=7.5)
    ax.set_ylim(-0.12, 1.02)
    return fig, facts


@register(
    "C-05", "correction", "selection_leaks",
    "Feature selection of the submission: periods and dominance",
    "What did the submitted feature selection see?",
    "Left: the periods each selection read (submitted: all three; rebuild: train only, "
    "validation once at the end). Right: the five largest mean |SHAP| values of the submitted "
    "direct-route selection.",
    "The submitted selection ran on the whole period, test included. In its direct route the "
    "same-hour DC power of the two inverters took {dc_share:.0%} of the listed importance, the "
    "first {ratio:.0f} times the third feature; they were removed only after the ranking.",
    "The rebuild selects on train only and admits features from an allow-list (F-26, F-27).",
    (SHAP_DIRECT,), kind="correction")
def c05_selection():
    s = D.csv(SHAP_DIRECT)
    fig, (a1, a2) = subplots(1, 2, height=2.6, gridspec_kw={"width_ratios": [1, 1.2]})
    spans = [("train", 29, C["blue"]), ("validation", 8, C["orange"]), ("test", 8, C["grey"])]
    reads = {"submitted": {"train": "all", "validation": "all", "test": "all"},
             "rebuild": {"train": "all", "validation": "once", "test": "none"}}
    for r, who in enumerate(("submitted", "rebuild")):
        left = 0
        for name, w, col in spans:
            mode = reads[who][name]
            a1.barh(r, w, left=left, color=col if mode == "all" else "white", edgecolor=col,
                    hatch={"all": None, "once": "///", "none": "..."}[mode], height=0.5)
            left += w
    a1.text(0, 1.55, "hatched: read once at the end; dotted: not read", fontsize=7)
    a1.set_yticks([0, 1], ["submitted", "rebuild"])
    a1.invert_yaxis()
    a1.set_xticks([14.5, 33, 41], ["train", "val", "test"])
    a1.set_title("periods read by the selection")
    a1.grid(False)
    is_dc = s["feature"].str.contains("_DC_kW")
    a2.barh(s["feature"][::-1], s["mean_abs_shap"][::-1],
            color=np.where(is_dc[::-1], C["red"], C["grey"]))
    a2.set_xlabel("mean |SHAP|, submitted direct selection")
    a2.tick_params(axis="y", labelsize=7)
    facts = {"dc_share": float(s.loc[is_dc, "mean_abs_shap"].sum() / s["mean_abs_shap"].sum()),
             "ratio": float(s["mean_abs_shap"].iloc[0] / s["mean_abs_shap"].iloc[2])}
    return fig, facts


@register(
    "C-06", "correction", "weather_alignment",
    "Submitted weather series: correlation by time shift",
    "Was the submitted weather series a forecast, and was it aligned with the plant data?",
    "Hourly correlation of the submitted weather series with the measured clear-sky index for "
    "shifts of -3 to +3 hours.",
    "The series is the archive value of each target hour, not a forecast issued before midnight. "
    "It correlates {best:.2f} with the measured k after a 1-hour shift (end-of-hour labels), "
    "but only {r2:.2f} and {r3:.2f} at the 2- and 3-hour shifts that the time bug produced.",
    "The rebuild uses real forecast runs published before issuance and aligned in true UTC "
    "(F-28, D-12).",
    (LAG_SCAN,), kind="correction")
def c06_weather():
    s = D.csv(LAG_SCAN)
    fig, ax = subplots(height=2.4)
    col = [C["blue"] if v == -1 else (C["red"] if v in (-2, -3) else C["grey"])
           for v in s["shift_h"]]
    bars = ax.bar(s["shift_h"].astype(str) + " h", s["hourly_r"], color=col, width=0.6)
    for b, v in zip(bars, s["shift_h"], strict=False):
        if v in (-2, -3):
            b.set_hatch("///")
            b.set_edgecolor("white")
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("hourly correlation with measured k")
    ax.set_xlabel("shift of the weather series")
    r = s.set_index("shift_h")["hourly_r"]
    facts = {"best": float(r.max()), "r2": float(r[-2]), "r3": float(r[-3])}
    assert int(r.idxmax()) == -1
    return fig, facts

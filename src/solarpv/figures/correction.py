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
# W4, the neural models (docs/stage9_plan.md §18.2)
SEED_TABLE = "submission_evidence/05_seed_variability/seed_variability.csv"
LSTM_TRAIN_METRICS = "submission_evidence/08_lstm_alignment/metrics_lstm_sota.csv"
SUBMITTED_BENCH = "submission_evidence/01_benchmark/benchmark_summary.csv"
NDEV_BENCH = f"{D.NEURAL_DEV_RUN}/evaluate/val/benchmark_summary.csv"
XDEV_BENCH = f"{D.DEV_RUN}/evaluate/val/benchmark_summary.csv"
ATTENTION = {(s, t): f"{D.NEURAL_DEV_RUN}/{s}/attention_{t}_seed42.csv"
             for s in ("tft96", "tft168") for t in ("indirect", "direct")}
CLAIMED_LAG_H = 168  # v6 §11.2: a -168 h attention peak "visible particularly in TFT-96 indirect"
SUBMITTED_MODELS = (("xgb_sota", "XGB"), ("lstm_sota", "BiLSTM"), ("tft_96", "TFT-96"),
                    ("tft_168", "TFT-168"), ("tft", "TFT baseline"), ("patchtst", "PatchTST"))


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


@register(
    "C-07", "correction", "bilstm_seed_table", "Seed table of the submitted BiLSTM",
    "Which numbers does the seed spread of the submitted BiLSTM rest on?",
    "Submitted nRMSE of the BiLSTM per route and horizon: the three rows of the seed table "
    "(seeds 42, 123 and 777; markers) and the one forecast that the evaluation script scored "
    "(seed 42; bars), both as the submitted files report them.",
    "All three seed-table rows, seed 42 included, are the training script's own metrics (the "
    "seed-42 row equals the training output file): at D+1 they average {ind_tab:.1f} % "
    "(indirect) and {dir_tab:.1f} % (direct), while the scored seed-42 forecast has "
    "{ind_ev:.1f} % and {dir_ev:.1f} %. The reported relative spread of {rel:.1%} is therefore "
    "not a spread of day-ahead errors; only one seed was scored.",
    "Every seed of every neural step writes its own forecast file, scored by the run's evaluate "
    "step; no number from a training loop enters a result table (C-10, M19, T-73).",
    (SEED_TABLE, LSTM_TRAIN_METRICS, SUBMITTED_BENCH), kind="correction")
def c07_seed_table():
    tab = D.csv(SEED_TABLE)
    tab = tab[tab["model"] == "lstm_sota"]
    train42 = D.csv(LSTM_TRAIN_METRICS).set_index(["target", "horizon"])["nRMSE_pct"]
    ev = D.csv(SUBMITTED_BENCH)
    ev = ev[ev["model"] == "lstm_sota"].set_index(["target", "horizon"])["nRMSE_pct"]
    row42 = tab[tab["seed"] == 42].set_index(["target", "horizon"])["nRMSE_pct"]
    # X37: the seed-42 row of the seed table is the training output, not the scored forecast
    assert np.allclose(row42.sort_index(), train42.reindex(row42.index).sort_index(), atol=1e-3)
    assert (ev.reindex(row42.index) - row42).min() > 1, "a seed-table row equals the evaluation"
    fig, axes = subplots(1, 2, height=2.7)
    marks = {42: ("o", C["blue"], -0.12), 123: ("s", C["orange"], 0.0),
             777: ("D", C["green"], 0.12)}
    for ax, t in zip(axes, ("indirect", "direct"), strict=True):
        hs = np.array([1, 2, 3])
        ax.bar(hs, [ev[(t, h)] for h in hs], 0.5, color=C["grey"], hatch="///",
               edgecolor="white", label="scored by the evaluation (seed 42)")
        for seed, (mk, col, dx) in marks.items():
            s = tab[(tab["target"] == t) & (tab["seed"] == seed)].sort_values("horizon")
            ax.plot(s["horizon"] + dx, s["nRMSE_pct"], mk, color=col, ms=5,
                    ls="none", label=f"seed table, seed {seed}")
        ax.set_xticks(hs, ["D+1", "D+2", "D+3"])
        ax.set_title(f"{t} route")
        ax.set_ylabel("submitted nRMSE (%)")
    axes[0].legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.95)
    d1 = tab[(tab["target"] == "indirect") & (tab["horizon"] == 1)]["nRMSE_pct"]
    facts = {"ind_tab": float(d1.mean()),
             "dir_tab": float(tab[(tab["target"] == "direct")
                                  & (tab["horizon"] == 1)]["nRMSE_pct"].mean()),
             "ind_ev": float(ev[("indirect", 1)]), "dir_ev": float(ev[("direct", 1)]),
             "rel": float(d1.std(ddof=1) / d1.mean())}
    return fig, facts


@register(
    "C-08", "correction", "scored_hours", "Scored hours per model, submitted and rebuilt",
    "Were the models of the submitted study scored on the same hours?",
    "Hours that each model was not scored on, counted against the model with the most scored "
    "hours at the same horizon and route: in the submitted benchmark (left, as reported) and "
    "in the rebuild's development runs (right, validation).",
    "In the submission the six models were scored on {sub_min:,} to {sub_max:,} hours at D+1 "
    "(BiLSTM {lstm:,}, the three TFTs {tft:,}, XGB and PatchTST {xgb:,}): each model's input "
    "window decided which hours it forecast, so a difference between two models also mixes "
    "different hours. In the rebuild all {n_models} models and the {n_refs} references are "
    "scored on the same {n_rb:,} validation hours at every horizon.",
    "One coverage rule for every model (C-35, T-39) and one forecast per hour and horizon from "
    "the sequence datasets (D-25, T-67).",
    (SUBMITTED_BENCH, NDEV_BENCH, XDEV_BENCH), kind="correction")
def c08_hours():
    sub = D.csv(SUBMITTED_BENCH)
    rb = pd.concat([D.csv(NDEV_BENCH), D.csv(XDEV_BENCH).query("model == 'xgb_sota'")])
    fig, (a1, a2) = subplots(1, 2, height=2.8, sharey=True)
    xs = np.arange(len(SUBMITTED_MODELS))
    styles = {1: (C["blue"], None), 2: (C["orange"], "///"), 3: (C["green"], "...")}
    for h, (col, hatch) in styles.items():
        for ax, tab in ((a1, sub), (a2, rb)):
            g = tab[(tab["horizon"] == h) & (tab["target"] == "indirect")].set_index("model")["n"]
            gap = [int(g.max() - g[m]) for m, _ in SUBMITTED_MODELS]
            ax.bar(xs + (h - 2) * 0.27, gap, 0.25, color=col, hatch=hatch, edgecolor="white",
                   label=f"D+{h}")
    for ax, title in ((a1, "submitted benchmark"), (a2, "rebuild, development runs")):
        ax.set_xticks(xs, [n for _, n in SUBMITTED_MODELS], rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
    a1.set_ylabel("hours not scored")
    a1.legend(loc="upper right", fontsize=7)
    d1 = sub[(sub["horizon"] == 1) & (sub["target"] == "indirect")].set_index("model")["n"]
    assert d1["tft"] == d1["tft_96"] == d1["tft_168"] and d1["xgb_sota"] == d1["patchtst"]
    # the submission scored both routes on the same hours per model, the rebuild every model
    assert (sub.pivot_table("n", ["model", "horizon"], "target").nunique(axis=1) == 1).all()
    assert rb["n"].nunique() == 1, "the rebuild scored two models on different hours"
    a2.text(0.5, 0.5, f"every model and reference:\n{int(rb['n'].iloc[0]):,} hours at each "
            "horizon", transform=a2.transAxes, ha="center", fontsize=8)
    models = {m for m, _ in SUBMITTED_MODELS}
    assert models <= set(rb["model"]), "a model is missing from the rebuild's runs"
    facts = {"sub_min": int(d1.min()), "sub_max": int(d1.max()), "lstm": int(d1["lstm_sota"]),
             "tft": int(d1["tft"]), "xgb": int(d1["xgb_sota"]), "n_models": len(models),
             "n_refs": len(set(rb["model"]) - models), "n_rb": int(rb["n"].iloc[0])}
    return fig, facts


@register(
    "C-09", "correction", "attention_window", "Which lags a TFT encoder can hold",
    "Can TFT-96 show an attention peak 168 hours before issuance, as the submitted report says?",
    "The encoder of TFT-96 (top) and TFT-168 (bottom) on the axis of hours before issuance, "
    "with the rebuild's mean validation attention in each (development run, relative to the "
    "mean of its own profile; indirect route solid, direct dashed). The red line is the lag "
    "that the submitted report names for TFT-96.",
    "TFT-96 reads only the last {len96} hours, so it has no position {claim} hours before "
    "issuance. In all {n_cases} model-route pairs of the rebuild the first position of the "
    "encoder lies {rise_min:.0%} to {rise_max:.0%} above the same hour of the later days, which "
    "agree to within {flat:.2%}: a rise at the edge of the window, whatever its length. A "
    "168-hour encoder puts that edge exactly at -{claim} h.",
    "The TFT steps store the attention over each model's own encoder, with its length "
    "asserted, and a peak is claimed only at a lag inside the encoder (C-55, M22, T-75).",
    tuple(ATTENTION.values()), kind="correction")
def c09_attention():
    fig, axes = subplots(2, 1, height=3.6, sharex=True)
    rises, flats = [], []
    for ax, (step, length) in zip(axes, (("tft96", 96), ("tft168", 168)), strict=True):
        ax.axvspan(-length - 0.5, -0.5, color=C["sky"], alpha=0.15, lw=0)
        for t, ls in (("indirect", "-"), ("direct", "--")):
            a = D.csv(ATTENTION[(step, t)]).sort_values("lag_h")
            assert a["lag_h"].min() == -length and a["lag_h"].max() == -1  # M22
            v = a["attention"].to_numpy()
            ax.plot(a["lag_h"], v / v.mean(), ls, color=C["blue"], lw=1.1, label=f"{t} route")
            later = v[24::24]
            rises.append(float(v[0] / later.mean() - 1))
            flats.append(float((later.max() - later.min()) / later.mean()))
        ax.set_title(f"{step.upper().replace('TFT', 'TFT-')}: encoder of {length} hours "
                     "(shaded)", fontsize=8.5)
        ax.set_ylabel("attention / mean")
    axes[0].axvline(-CLAIMED_LAG_H, color=C["red"], lw=1.6)
    axes[0].text(-CLAIMED_LAG_H + 2, 0.06, f"submitted report: peak at -{CLAIMED_LAG_H} h "
                 "in TFT-96", color=C["red"], fontsize=7.5, va="bottom",
                 transform=axes[0].get_xaxis_transform())
    axes[1].axvline(-CLAIMED_LAG_H, color=C["red"], lw=0.8, ls=":")
    axes[1].set_xlabel("hours before issuance")
    axes[1].set_xlim(-180, 0)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.08, 1.0), fontsize=7)
    assert min(rises) > 0, "the first encoder position is not raised in every pair"
    facts = {"len96": 96, "claim": CLAIMED_LAG_H, "n_cases": len(rises),
             "rise_min": min(rises), "rise_max": max(rises), "flat": max(flats)}
    return fig, facts

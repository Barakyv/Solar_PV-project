"""Figures of stages 0, 1 and 2: the raw export, the pipeline and its splits, ingestion."""

import numpy as np
import pandas as pd

from solarpv.config.constants import (
    DATA_START_UTC,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from solarpv.evaluation.daytime import daytime_mask
from solarpv.features.contract import FEATURE_GROUPS
from solarpv.features.leakage import EXCL_EXACT, EXCL_PATTERNS
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import SPLIT_COLOURS, C, arrow, blank, box, month_axis, subplots

INVENTORY = "00_data_audit/raw_data_inventory.csv"
INGEST_MANIFEST = "data/processed/01_ingest/ingest_manifest.json"
INGEST_QA = "data/processed/01_ingest/ingest_qa.csv"
CENTROIDS = "data/processed/01_ingest/centroid_offsets.csv"
FEATURES_MANIFEST = "data/processed/06_features/features_manifest.json"
GFS_MANIFEST = "outputs/06b_gfs/gfs_manifest.json"
MODEL_INPUTS = "outputs/07_feature_selection/model_inputs.json"


@register(
    "S00-01", "0", "raw_inventory", "Raw export: minutes per monthly file against the calendar",
    "Is the raw SCADA export complete, and where does it deviate from one row per minute?",
    "Bars: rows in each of the monthly files minus the minutes of that calendar month. "
    "October files have extra rows and March files have missing rows; every other month is exact.",
    "{n_files} monthly files with {rows:,} rows. Only the daylight-saving months deviate: "
    "{n_dup_months} October files with +{dup_each} rows and {n_missing_months} March files with "
    "-{missing_each} rows; no other file has a missing or duplicate minute.",
    "The export is written in local wall-clock time, so stage 2 converts it to UTC with the "
    "daylight-saving rule instead of dropping or averaging the repeated hour.",
    (INVENTORY,))
def s00_inventory():
    inv = D.csv(INVENTORY)
    dev = inv["rows"] - inv["expected"]
    fig, ax = subplots(height=2.8)
    x = np.arange(len(inv))
    colours = np.where(dev > 0, C["red"], np.where(dev < 0, C["blue"], C["grey"]))
    bars = ax.bar(x, dev, color=colours, width=0.8)
    for b, d in zip(bars, dev, strict=False):
        if d > 0:
            b.set_hatch("///")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x[::3], [f.replace(".CSV", "") for f in inv["file"]][::3], rotation=0)
    ax.set_xlabel("monthly file (YYMM)")
    ax.set_ylabel("rows minus calendar minutes")
    ax.set_title("Raw export: rows per monthly file against the calendar")
    ax.text(0.01, 0.95, "hatched: October, repeated hour   solid blue: March, skipped hour",
            transform=ax.transAxes, fontsize=7.5, va="top")
    facts = {"n_files": int(len(inv)), "rows": int(inv["rows"].sum()),
             "n_dup_months": int((dev > 0).sum()), "dup_each": int(dev[dev > 0].max()),
             "n_missing_months": int((dev < 0).sum()), "missing_each": int(-dev[dev < 0].min()),
             "n_exact_months": int((dev == 0).sum())}
    return fig, facts


PIPELINE = [
    ("Raw export", "{n_files} monthly files", "done"),
    ("Ingestion, UTC", "{minutes:,} minutes", "done"),
    ("Clocks, hourly", "{hours:,} hours", "done"),
    ("Cleaning", "rules and flags", "done"),
    ("Target, features", "{f1} / {f2} / {f3} features", "done"),
    ("GFS forecasts", "{gfs_runs:,} runs", "done"),
    ("Feature selection", "{sel_ind} / {sel_dir} inputs at D+1", "done"),
    ("Evaluation layer", "3 references", "done"),
    ("Light models", "XGB-dev, runner", "current"),
    ("Gate G-1", "end-to-end audit", "pending"),
    ("Final training", "six models", "pending"),
    ("Report", "figures, errata", "pending"),
]


@register(
    "S01-01", "1", "pipeline", "The pipeline, stage by stage",
    "What are the stages from the raw export to the report, and what does each one produce?",
    "Twelve boxes in reading order, each with the stage's main output; finished stages solid, "
    "the current stage outlined in orange, later stages hatched.",
    "From {n_files} raw files to {minutes:,} UTC minutes, {hours:,} hourly rows, "
    "{f1} / {f2} / {f3} candidate features per horizon plus {gfs_runs:,} weather-forecast runs, "
    "and {sel_ind} (indirect) and {sel_dir} (direct) model inputs at D+1. The final training "
    "of the six models comes only after the gate.",
    "Each stage is built and checked before the next one starts; the final training runs once "
    "on frozen code after gate G-1 (D-08).",
    (INVENTORY, INGEST_MANIFEST, FEATURES_MANIFEST, GFS_MANIFEST, MODEL_INPUTS))
def s01_pipeline():
    ing, feat, gfs, mi = (D.jload(INGEST_MANIFEST), D.jload(FEATURES_MANIFEST),
                          D.jload(GFS_MANIFEST), D.jload(MODEL_INPUTS))
    fph = feat["features_per_horizon"]
    facts = {"n_files": int(len(D.csv(INVENTORY))), "minutes": int(ing["rows"]),
             "hours": int(feat["rows_per_horizon"]["D1"]),
             "f1": len(fph["D1"]), "f2": len(fph["D2"]), "f3": len(fph["D3"]),
             "gfs_runs": int(gfs["runs"]), "sel_ind": int(mi["indirect_D1"]["n_features"]),
             "sel_dir": int(mi["direct_D1"]["n_features"])}
    fig, ax = subplots(width=7.4, height=3.6)
    blank(ax)
    w, h, gx = 0.21, 0.2, 0.043
    style = {"done": (C["sky"] + "55", C["blue"], None), "current": ("white", C["orange"], None),
             "pending": ("white", C["grey"], "////")}
    pos = []
    for i, (name, sub, status) in enumerate(PIPELINE):
        r, c = divmod(i, 4)
        x, y = 0.01 + c * (w + gx), 0.74 - r * 0.34
        pos.append((x, y))
        face, edge, hatch = style[status]
        box(ax, x, y, w, h, f"{name}\n{sub.format(**facts)}", face=face, edge=edge, size=7.5)
        if hatch:
            ax.patches[-1].set_hatch(hatch)
    for i in range(len(PIPELINE) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            arrow(ax, x0 + w, y0 + h / 2, x1, y1 + h / 2)
        else:
            arrow(ax, x0 + w / 2, y0, x0 + w / 2, y0 - 0.06)
            ax.plot([x0 + w / 2, x1 + w / 2], [y0 - 0.06, y0 - 0.06], color=C["grey"], lw=0.9)
            arrow(ax, x1 + w / 2, y0 - 0.06, x1 + w / 2, y1 + h)
    ax.set_title("The pipeline, stage by stage (solid: done, orange: current, hatched: to come)")
    return fig, facts


SPLITS = [("train", DATA_START_UTC, TRAIN_END), ("validation", VAL_START, VAL_END),
          ("test", TEST_START, TEST_END)]


@register(
    "S01-02", "1", "splits", "Train, validation and test periods",
    "Which periods train, tune and finally test the models, and how many hours does each hold?",
    "Three consecutive bars on a calendar axis with the hours and the daytime hours of each "
    "period. The test period is hatched: it is read only in the final run.",
    "Chronological split without shuffling: train {train_h:,} hours ({train_d:,} daytime), "
    "validation {validation_h:,} ({validation_d:,} daytime), test {test_h:,} "
    "({test_d:,} daytime). Every selection and tuning step uses train and validation only.",
    "The split fixed the evaluation rule of every later stage: decisions on validation, the test "
    "period opened once in the final run after the code is frozen (D-08).",
    ())
def s01_splits():
    facts = {}
    fig, ax = subplots(height=2.0)
    for _i, (name, a, b) in enumerate(SPLITS):
        start = pd.Timestamp(a, tz="UTC").floor("h")
        idx = pd.date_range(start, pd.Timestamp(b, tz="UTC").floor("h"), freq="h")
        facts[f"{name}_h"] = int(len(idx))
        facts[f"{name}_d"] = int(daytime_mask(idx).sum())
        ax.barh(0, (idx[-1] - idx[0]).days + 1 / 24, left=D.naive_utc(idx[:1])[0],
                color=SPLIT_COLOURS[name], alpha=0.85, height=0.5,
                hatch="////" if name == "test" else None, edgecolor="white")
        mid = D.naive_utc(idx[:1])[0] + (idx[-1] - idx[0]) / 2
        ax.text(mid, 0.3, name, ha="center", va="bottom", fontsize=9)
        ax.text(mid, 0, f"{facts[f'{name}_h']:,} h\n{facts[f'{name}_d']:,} daytime",
                ha="center", va="center", fontsize=8, color="white" if name != "test" else "black",
                bbox=dict(fc=C["light"], ec="none", pad=1.5) if name == "test" else None)
    ax.set_ylim(-0.4, 0.7)
    ax.set_yticks([])
    ax.grid(False, axis="y")
    month_axis(ax, 6)
    ax.set_title("Chronological split (daytime: sun above 5 degrees at mid-hour)")
    return fig, facts


@register(
    "S01-03", "1", "feature_contract", "The feature contract and the leakage list",
    "Which candidate features may a model see, and how many remain at each horizon?",
    "Left: allowed features per group. Right: features available at each horizon, after the "
    "power lags that are not known at issuance are removed.",
    "{total} allowed features in {n_groups} groups; {f1} / {f2} / {f3} remain at D+1 / D+2 / "
    "D+3. The leakage list blocks {n_excl_exact} exact names and {n_excl_patterns} name patterns "
    "(same-hour power, flags, quality columns, set-points).",
    "An allow-list, not a block-list, decides what reaches a model; the leakage tests run on "
    "every table and model input from stage 6 on.",
    (FEATURES_MANIFEST,))
def s01_contract():
    groups = {g: len(v) for g, v in FEATURE_GROUPS.items()}
    fph = D.jload(FEATURES_MANIFEST)["features_per_horizon"]
    fig, (a1, a2) = subplots(1, 2, height=2.6, gridspec_kw={"width_ratios": [1.4, 1]})
    names = sorted(groups, key=groups.get)
    a1.barh([n.replace("_", " ") for n in names], [groups[n] for n in names], color=C["blue"])
    for i, n in enumerate(names):
        a1.text(groups[n] + 0.5, i, str(groups[n]), va="center", fontsize=8)
    a1.set_xlabel("allowed features")
    a1.set_title("Allowed features per group")
    hs = ["D+1", "D+2", "D+3"]
    vals = [len(fph[f"D{h}"]) for h in (1, 2, 3)]
    a2.bar(hs, vals, color=[C["blue"], C["sky"], C["green"]], width=0.6)
    for i, v in enumerate(vals):
        a2.text(i, v + 1, str(v), ha="center", fontsize=8)
    a2.set_ylabel("features available")
    a2.set_title("Per horizon")
    facts = {"total": sum(groups.values()), "n_groups": len(groups), "f1": vals[0],
             "f2": vals[1], "f3": vals[2], "n_excl_exact": len(EXCL_EXACT),
             "n_excl_patterns": len(EXCL_PATTERNS)}
    return fig, facts


@register(
    "S02-01", "2", "dst_utc_grid", "Daylight-saving days on a continuous UTC grid",
    "How are the repeated autumn hour and the skipped spring hour handled?",
    "Minutes per local wall-clock hour on the first autumn and the first spring transition day "
    "(bars), against the 60 minutes that every UTC hour holds (dashed line).",
    "{minutes:,} UTC minutes from {start} to {end}, every UTC hour with exactly 60 minutes. On "
    "the {n_autumn} autumn days the local hour {autumn_hour:02d}:00 holds 120 minutes (two "
    "distinct UTC hours); on the {n_spring} spring days {spring_hour:02d}:00 local does not exist.",
    "All later stages work on this one UTC grid, so daily and hourly features never mix two "
    "clocks.",
    (INGEST_MANIFEST,))
def s02_dst():
    df = D.table(D.INGEST_1MIN, ["iec_mode"])
    utc_counts = df.groupby(df.index.floor("h")).size()
    loc = D.local(df.index)
    per_day = pd.Series(1, index=loc).groupby(loc.date).size()
    autumn = [d for d, n in per_day.items() if n == 1500]
    spring = [d for d, n in per_day.items() if n == 1380]
    fig, axes = subplots(1, 2, height=2.6, sharey=True)
    facts = {"minutes": int(len(df)), "start": str(df.index[0])[:16], "end": str(df.index[-1])[:16],
             "utc_hours_all_60": bool((utc_counts == 60).all()), "n_autumn": len(autumn),
             "n_spring": len(spring)}
    for ax, day, label in ((axes[0], autumn[0], "autumn"), (axes[1], spring[0], "spring")):
        sel = loc[loc.date == day]
        counts = pd.Series(1, index=sel).groupby(sel.hour).size().reindex(range(24), fill_value=0)
        colours = [C["red"] if v > 60 else (C["blue"] if v == 0 else C["grey"]) for v in counts]
        ax.bar(counts.index, counts.values, color=colours, width=0.8)
        ax.axhline(60, color="black", ls="--", lw=1, label="minutes in every UTC hour")
        ax.set_title(f"{label} transition, {day}")
        ax.set_xlabel("local wall-clock hour")
        odd = counts.index[counts != 60][0]
        facts[f"{label}_hour"] = int(odd)
        if counts[odd] > 60:
            ax.text(odd, counts[odd] + 2, f"{counts[odd]} min", ha="left", fontsize=8)
        else:
            ax.text(odd, 3, "hour skipped", rotation=90, ha="center", va="bottom", fontsize=8,
                    color=C["blue"])
    axes[0].set_ylabel("minutes in the hour")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, 135)
    return fig, facts


ANOMALY_ROWS = [
    ("ambient_temp", "n_negative", "ambient temperature: fault code"),
    ("mod_temp1", "n_negative", "module temperature 1: fault code"),
    ("mod_temp2", "n_negative", "module temperature 2: fault code"),
    ("irradiance", "n_negative", "irradiance below zero (night offset)"),
    ("s01_inv1_AC_kW", "n_above_inverter_flag", "inverter 1 AC above 110 % of rating"),
    ("s01_inv2_AC_kW", "n_above_inverter_flag", "inverter 2 AC above 110 % of rating"),
]


@register(
    "S02-02", "2", "raw_anomalies", "Value anomalies found at ingestion",
    "What kinds of impossible or suspicious values does the raw export contain?",
    "Minutes affected per anomaly type, on a logarithmic axis because the counts range from a "
    "few minutes to tens of thousands.",
    "No missing or non-numeric cell. Temperature sensors carry a fault code (minimum "
    "{temp_min:.0f}) in {temp_fault_min:,}-{temp_fault_max:,} minutes each; irradiance is "
    "slightly negative at night in {irr_neg:,} minutes; {ac_spikes} inverter minutes exceed "
    "110 % of the inverter rating (up to {ac_max:,.0f} kW).",
    "Each anomaly type became an explicit cleaning rule with evidence for its threshold "
    "(stage 3b) and a logged effect (stage 5).",
    (INGEST_QA, INGEST_MANIFEST))
def s02_anomalies():
    qa = D.csv(INGEST_QA).set_index("column")
    vals = [int(qa.loc[c, k]) for c, k, _ in ANOMALY_ROWS]
    labels = [lab for _, _, lab in ANOMALY_ROWS]
    fig, ax = subplots(height=2.6)
    ax.barh(labels[::-1], vals[::-1], color=C["blue"])
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, i, f"{v:,}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1, max(vals) * 8)
    ax.set_xlabel("minutes affected (log scale)")
    ax.set_title("Value anomalies found at ingestion")
    temps = [vals[0], vals[1], vals[2]]
    facts = {"temp_min": float(qa.loc[["ambient_temp", "mod_temp1", "mod_temp2"], "min"].min()),
             "temp_fault_min": min(temps), "temp_fault_max": max(temps), "irr_neg": vals[3],
             "ac_spikes": vals[4] + vals[5],
             "ac_max": float(qa.loc[["s01_inv1_AC_kW", "s01_inv2_AC_kW"], "max"].max()),
             "missing_cells": int(qa["n_missing"].sum() + qa["n_non_numeric"].sum())}
    return fig, facts


@register(
    "S02-03", "2", "centroid_after_utc",
    "Production centre against solar noon after the UTC conversion",
    "Once the time base is UTC, does production line up with the sun on clear days?",
    "Each point is one clear day: the centre of the DC power curve (circles) and of the "
    "irradiance curve (triangles) minus solar noon, in minutes.",
    "On {n_days} clear days the monthly medians stay within {month_max:.1f} min of solar noon "
    "(the stage-2 check allows 30), so the UTC conversion is right; but the medians drift from "
    "month to month (irradiance sensor {irr_month_min:.1f} to {irr_month_max:.1f} min) and "
    "the DC and irradiance medians differ by up to {gap_max:.1f} min in the same month.",
    "The residual drift pointed to the logger clocks themselves: the per-day clock study (F-15) "
    "and the per-source correction of stage 4 (D-19).",
    (CENTROIDS,))
def s02_centroids():
    c = D.csv(CENTROIDS, parse_dates=["day"])
    fig, ax = subplots(height=2.8)
    ax.plot(c["day"], c["dc"], "o", ms=3, color=C["blue"], label="DC power")
    ax.plot(c["day"], c["irradiance"], "^", ms=3, color=C["orange"], label="irradiance sensor")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("centre minus solar noon (min)")
    month_axis(ax, 6)
    ax.legend(loc="lower left", ncols=2)
    ax.set_title("Clear-day production centre after the UTC conversion, before the clock fix")
    med = c.groupby("month")[["dc", "irradiance"]].median()
    ax.plot(pd.to_datetime(med.index) + pd.Timedelta(days=14), med["irradiance"], "-",
            color=C["orange"], lw=1, alpha=0.8)
    ax.plot(pd.to_datetime(med.index) + pd.Timedelta(days=14), med["dc"], "--",
            color=C["blue"], lw=1, alpha=0.8)
    month_max = float(med.abs().max().max())
    assert month_max <= 30, "monthly median outside the stage-2 bound"
    facts = {"n_days": int(len(c)), "month_max": month_max,
             "irr_month_min": float(med["irradiance"].min()),
             "irr_month_max": float(med["irradiance"].max()),
             "gap_max": float((med["dc"] - med["irradiance"]).abs().max())}
    return fig, facts

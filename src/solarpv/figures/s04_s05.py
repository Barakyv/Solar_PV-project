"""Figures of stages 4 and 5: logger clocks, the second data source, cleaning and flags."""

import numpy as np
import pandas as pd

from solarpv.config.constants import TEST_START, VAL_START
from solarpv.evaluation.daytime import daytime_mask
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, month_axis, shade_splits, subplots

OFFSETS = "configs/clock_offsets.csv"
MONTHLY = "00_data_audit/clock_drift_2026-10-14/monthly_minute_before_after.csv"
CLEANING_LOG = "outputs/05_cleaning/cleaning_log.csv"
CLOCK_STEPS = ("2023-05-09", "2024-01-21")


@register(
    "S04-01", "4", "clock_offsets", "Logger clock offsets applied per day",
    "How far were the logger clocks from true time, and how does the correction follow them?",
    "Daily offset (time stamp minus true time) of the weather-station clock (solid), the "
    "inverter clock (dashed) and the clock of the second data source (dotted), with the two "
    "re-synchronisations marked.",
    "Over {n_days:,} days the weather-station clock drifts down to {meteo_min:.1f} min and the "
    "inverter clock to {inv_min:.1f} min; the clocks were re-synchronised on {step1} (weather "
    "station only) and {step2}. The second source, from {polled_start}, keeps its own clock "
    "({polled_min:.1f} to {polled_max:.1f} min).",
    "Every minute is shifted by its own clock's offset of that day before the hourly means, so "
    "power, irradiance and the sun share one time base (D-19).",
    (OFFSETS,))
def s04_offsets():
    o = D.csv(OFFSETS, parse_dates=["day"])
    fig, ax = subplots(height=2.9)
    ax.plot(o["day"], o["minute_meteo"], color=C["blue"], label="weather-station clock")
    ax.plot(o["day"], o["minute_inverter"], color=C["orange"], ls="--", label="inverter clock")
    ax.plot(o["day"], o["polled_meteo"], color=C["green"], ls=":", lw=2.2,
            label="second source (15-min snapshots)")
    for s in CLOCK_STEPS:
        ax.axvline(pd.Timestamp(s), color=C["grey"], lw=0.8)
        ax.text(pd.Timestamp(s), 3.6, f" re-sync\n {s}", fontsize=7, va="top", color=C["grey"])
    shade_splits(ax)
    ax.set_ylabel("clock offset (min)")
    month_axis(ax, 6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3)
    ax.set_ylim(-18, 4)
    polled = o.dropna(subset=["polled_meteo"])
    facts = {"n_days": int(len(o)), "meteo_min": float(o["minute_meteo"].min()),
             "inv_min": float(o["minute_inverter"].min()), "step1": CLOCK_STEPS[0],
             "step2": CLOCK_STEPS[1], "polled_start": str(polled["day"].iloc[0].date()),
             "polled_min": float(polled["polled_meteo"].min()),
             "polled_max": float(polled["polled_meteo"].max())}
    return fig, facts


@register(
    "S04-02", "4", "clock_check", "Clock check after the correction",
    "Does the corrected data meet the acceptance bounds of the clock correction?",
    "Monthly median of the clear-day offset of the irradiance sensor against the clear-sky model "
    "(left) and of the inverter clock against the weather-station clock (right), before "
    "(triangles, dashed) and after (circles) the correction, with the acceptance bands.",
    "After the correction every one of the {n_months} months is inside the bands: the "
    "irradiance offset is at most {tau_after:.2f} min (before: up to {tau_before:.1f}) and the "
    "inverter-to-station gap at most {delta_after:.2f} min (before: up to {delta_before:.1f}).",
    "With the clocks aligned, the panel orientation was re-measured on the corrected data "
    "(21 degrees tilt, 185 degrees azimuth) and the hourly tables were built.",
    (MONTHLY,))
def s04_check():
    m = D.csv(MONTHLY)
    m["month"] = pd.to_datetime(m["day"])
    fig, (a1, a2) = subplots(1, 2, height=2.8, sharex=True)
    for ax, before, after, band, label in (
            (a1, "tau_met", "tau_met_after", 2, "irradiance sensor vs clear sky (min)"),
            (a2, "delta", "delta_after", 1, "inverter clock vs station clock (min)")):
        ax.axhspan(-band, band, color=C["green"], alpha=0.12, lw=0, label=f"band +/-{band} min")
        ax.plot(m["month"], m[before], "^--", ms=3.5, color=C["orange"], lw=1, label="before")
        ax.plot(m["month"], m[after], "o-", ms=3.5, color=C["blue"], lw=1.2, label="after")
        ax.set_ylabel(label)
        month_axis(ax, 9)
    a1.legend(loc="lower left")
    a1.set_title("Weather-station clock")
    a2.set_title("Inverter clock")
    assert m["tau_met_after"].abs().max() <= 2 and m["delta_after"].abs().max() <= 1
    facts = {"n_months": int(len(m)), "tau_after": float(m["tau_met_after"].abs().max()),
             "tau_before": float(m["tau_met"].abs().max()),
             "delta_after": float(m["delta_after"].abs().max()),
             "delta_before": float(m["delta"].abs().max())}
    return fig, facts


@register(
    "S04-03", "4", "second_source", "The second data source: values held for 15 minutes",
    "Did the way the data was recorded change during the period?",
    "Share of daytime minutes per day in which inverter-1 AC power repeats the previous minute "
    "exactly, over the whole period with the splits shaded.",
    "Before {switch}, the first day of the second source, the daily median of repeated minutes "
    "is {before:.1%}; from that day on it is {after:.1%} ({n_after} days, the two sources mixed "
    "hour by hour at first). The file still has a row per minute, but the value is a snapshot "
    "refreshed every 15 minutes.",
    "The second source was detected per tag and hour, rebuilt at its sampling instants with its "
    "own clock (D-20), and evaluated separately (D-21).",
    (D.INGEST_1MIN, OFFSETS))
def s04_second_source():
    df = D.table(D.INGEST_1MIN, ["s01_inv1_AC_kW"])
    ac = df["s01_inv1_AC_kW"]
    held = ac.diff().eq(0) & ac.gt(20) & ac.shift().gt(20)
    day = D.local(df.index).date
    share = held.groupby(day).sum() / ac.gt(20).groupby(day).sum().replace(0, np.nan)
    share.index = pd.to_datetime(share.index)
    roll = share.rolling(7, min_periods=3).median()
    offsets = D.csv(OFFSETS, parse_dates=["day"]).dropna(subset=["polled_meteo"])
    switch = offsets["day"].iloc[0]
    fig, ax = subplots(height=2.6)
    ax.plot(share.index, share, ".", ms=2, color=C["grey"], label="daily share")
    ax.plot(roll.index, roll, color=C["blue"], label="7-day median")
    ax.axvline(switch, color=C["red"], ls="--", lw=1)
    ax.text(switch, 0.55, f"{switch.date()} ", ha="right", fontsize=8, color=C["red"])
    shade_splits(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("held daytime minutes (share)")
    month_axis(ax, 6)
    ax.legend(loc="upper left")
    facts = {"switch": str(switch.date()), "before": float(share[share.index < switch].median()),
             "after": float(share[share.index >= switch].median()),
             "n_after": int((share.index >= switch).sum())}
    return fig, facts


RULE_LABELS = {
    "R1_range": "R1 outside the physical range", "R2_offset": "R2 night offset of irradiance",
    "R3_frozen": "R3 frozen values (logger freeze)", "R4_night_zero": "R4 power at night",
    "R6_dc_dead": "R6 dead DC sensor", "R7_scale_fault": "R7 pyranometer scale fault",
}


@register(
    "S05-01", "5", "cleaning_rules", "What each cleaning rule changed",
    "How much data does each cleaning rule touch, by day and by night?",
    "Minutes set to missing or corrected by each rule (all columns), split into daytime and "
    "night minutes, on a logarithmic axis.",
    "{n_rules} value rules touched {total:,} minutes; the largest is the pyranometer scale fault "
    "({r7:,} minutes, a QA sensor that no model uses). On the power columns the rules changed "
    "{kwh:,.0f} kWh, most of it in logger freezes that had repeated a value for hours.",
    "The cleaned minutes were re-aggregated to hours, and hours with downtime, curtailment or "
    "a partial inverter outage were flagged rather than deleted (D-22).",
    (CLEANING_LOG,))
def s05_rules():
    log = D.csv(CLEANING_LOG)
    g = log.groupby("rule")[["minutes", "daytime_minutes"]].sum()
    g = g.loc[[r for r in RULE_LABELS if r in g.index]]
    g["night"] = g["minutes"] - g["daytime_minutes"]
    fig, ax = subplots(height=2.8)
    y = np.arange(len(g))
    ax.barh(y + 0.2, g["daytime_minutes"].clip(lower=0.8), height=0.38, color=C["orange"],
            label="daytime minutes")
    ax.barh(y - 0.2, g["night"].clip(lower=0.8), height=0.38, color=C["blue"], hatch="///",
            edgecolor="white", label="night minutes")
    ax.set_yticks(y, [RULE_LABELS[r] for r in g.index])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("minutes changed (log scale; zero shown at 1)")
    ax.legend(loc="lower right")
    ax.set_title("What each cleaning rule changed")
    frozen_kwh = log.loc[log["rule"] == "R3_frozen", "kwh_changed"].sum()
    assert frozen_kwh > 0.5 * log["kwh_changed"].sum(), "takeaway says most kWh are freezes"
    assert g["minutes"].idxmax() == "R7_scale_fault"
    facts = {"n_rules": int(len(g)), "total": int(g["minutes"].sum()),
             "r7": int(g.loc["R7_scale_fault", "minutes"]),
             "kwh": float(log["kwh_changed"].sum())}
    return fig, facts


@register(
    "S05-02", "5", "operational_flags", "Operationally limited daytime hours per month",
    "When was production limited by people or equipment rather than by the weather?",
    "Daytime hours per month carrying the operational flag, by cause: grid curtailment, "
    "inverter stop, partial inverter outage; the splits are shaded.",
    "{train:,} flagged daytime hours in train, {validation:,} in validation and {test:,} in "
    "test: {curtail:,} curtailment, {downtime:,} stops and {derate:,} partial outages in total.",
    "Flagged hours stay in the data and in the main evaluation, but are left out of the training "
    "loss and set to missing in the past-day features (D-22).",
    (D.CLEAN_1H,))
def s05_flags():
    cols = ["frac_downtime_inv1_flag", "frac_downtime_inv2_flag", "frac_curtail_flag",
            "frac_derate_inv1_flag", "frac_derate_inv2_flag", "op_limited_flag"]
    h = D.table(D.CLEAN_1H, cols)
    h = h[daytime_mask(h.index) & h["op_limited_flag"].astype(bool)]
    cause = np.where(h["frac_curtail_flag"] >= 0.05, "curtailment",
                     np.where(h[["frac_downtime_inv1_flag", "frac_downtime_inv2_flag"]].max(axis=1)
                              >= 0.05, "inverter stop", "partial outage"))
    h = h.assign(cause=cause)
    month = D.naive_utc(h.index).to_period("M").to_timestamp()
    tab = pd.crosstab(month, h["cause"]).reindex(
        pd.date_range(month.min(), month.max(), freq="MS"), fill_value=0)
    fig, ax = subplots(height=2.8)
    bottom = np.zeros(len(tab))
    styles = {"curtailment": (C["orange"], None), "inverter stop": (C["red"], "///"),
              "partial outage": (C["blue"], "...")}
    for name, (col, hatch) in styles.items():
        if name in tab:
            ax.bar(tab.index, tab[name], width=25, bottom=bottom, color=col, hatch=hatch,
                   edgecolor="white", lw=0.3, label=name)
            bottom += tab[name].to_numpy()
    shade_splits(ax)
    ax.set_ylabel("flagged daytime hours")
    month_axis(ax, 6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3)
    t = D.naive_utc(h.index)
    v0, t0 = pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)
    facts = {"train": int((t < v0).sum()), "validation": int(((t >= v0) & (t < t0)).sum()),
             "test": int((t >= t0).sum()),
             "curtail": int((h["cause"] == "curtailment").sum()),
             "downtime": int((h["cause"] == "inverter stop").sum()),
             "derate": int((h["cause"] == "partial outage").sum())}
    return fig, facts

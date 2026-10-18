"""Part A: the export as delivered (Q01-Q04)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarpv.data.resolution import (
    CONTINUOUS_COLUMNS,
    MIN_DETECT_MINUTES,
    MINUTE_RATE_MIN,
    POLL_RUN,
    POLLED_SHARE_MIN,
    local_day,
)
from solarpv.eda.quality.common import Result, heatmap, month_index
from solarpv.eda.quality.data import MAIN, STAGE4_DIR, Data

# group, unit, logger, role in the project (with the reason for tags that are not used)
TAGS = {
    "site_total_AC_kW": ("power", "kW", "derived",
        "target: inverter 1 + inverter 2, minute by minute"),
    "s01_inv1_AC_kW": ("power", "kW", "inverter", "target component; model input as past-day lags"),
    "s01_inv2_AC_kW": ("power", "kW", "inverter", "target component; model input as past-day lags"),
    "s01_inv1_DC_kW": ("power", "kW", "inverter",
        "QA only: rule R6, clock check (measured at the target time, EXCL)"),
    "s01_inv2_DC_kW": ("power", "kW", "inverter",
        "QA only: rule R6, clock check (measured at the target time, EXCL)"),
    "s01_inv1_AC_kVar": ("power", "kVar", "inverter",
        "not used: reactive power, set by the grid operator (EXCL)"),
    "s01_inv2_AC_kVar": ("power", "kVar", "inverter",
        "not used: reactive power, set by the grid operator (EXCL)"),
    "s01_inv1_PF": ("power", "power factor", "inverter",
        "not used: follows the reactive-power setpoint (EXCL)"),
    "s01_inv2_PF": ("power", "power factor", "inverter",
        "not used: follows the reactive-power setpoint (EXCL)"),
    "s01_inv1_state": ("status", "code", "inverter",
        "flag F3: codes 5/6 mark a derated inverter (Q16); not a model input (EXCL)"),
    "s01_inv2_state": ("status", "code", "inverter",
        "flag F3: codes 5/6 mark a derated inverter (Q16); not a model input (EXCL)"),
    "irradiance": ("irradiance", "W/m2, panel plane", "meteo",
        "model input as past-day lags; reference of R7 and F1 (D-09)"),
    "pyranometer": ("irradiance", "W/m2, panel plane", "SiteData",
        "QA only: redundant sensor, x0.75 fault (F-19)"),
    "mod_temp1": ("temperature", "degC", "meteo", "model input as past-day lags"),
    "mod_temp2": ("temperature", "degC", "meteo", "model input as past-day lags"),
    "ambient_temp": ("temperature", "degC", "meteo",
        "model input as past-day lags (D-09: sensor choice, Q22)"),
    "site_ambient_temp": ("temperature", "degC", "SiteData",
        "QA only: second ambient sensor (Q22)"),
    "wind_speed": ("meteo", "m/s", "meteo",
        "not used: measured at the target time, no lag in the contract (Q28)"),
    "wind_direction": ("meteo", "deg", "meteo",
        "not used: measured at the target time, no lag in the contract (Q28)"),
    "precipitation": ("meteo", "undocumented", "meteo", "not used: see Q28"),
    "soil": ("soiling", "undocumented", "meteo", "not used: see Q28"),
    "soiling01": ("soiling", "undocumented", "meteo", "not used: see Q28"),
    "soiling02": ("soiling", "undocumented", "meteo", "not used: see Q28"),
    "iec_sp_w": ("grid (IEC)", "% of park power", "inverter",
        "flag F2: curtailment when below 100 (Q14)"),
    "iec_mode": ("grid (IEC)", "code 0/1/2", "inverter",
        "not used: output under modes 0/1 equals mode 2 (F-17, Q14)"),
    "iec_sp_q": ("grid (IEC)", "kVar setpoint", "inverter", "not used: reactive-power setpoint"),
    "iec_sp_v": ("grid (IEC)", "kV setpoint", "inverter", "not used: voltage setpoint"),
    "iec_sp_pf": ("grid (IEC)", "power-factor setpoint", "inverter",
        "not used: power-factor setpoint"),
}
CODED = ("s01_inv1_state", "s01_inv2_state", "iec_mode", "iec_sp_w")
# continuous measured tags whose held values are shown in Q02 (setpoints and codes are constant
# by nature; power factor is held at 1.0 by control)
HELD_COLUMNS = (*MAIN, "s01_inv1_AC_kVar", "s01_inv2_AC_kVar", "wind_speed", "wind_direction",
                "soil", "soiling01", "soiling02", "precipitation")


def role_class(role: str) -> str:
    for key in ("target", "model input", "QA only", "flag", "not used"):
        if role.startswith(key):
            return key
    return "other"


def q01_inventory(d: Data) -> Result:
    raw = d.raw
    rows = []
    for col, (group, unit, logger, role) in TAGS.items():
        r = raw[col] if col in raw else None
        s4, s5 = d.s4[col], d.s5[col]
        src = r if r is not None else s4
        q = src.quantile([0.01, 0.5, 0.99])
        rows.append({
            "column": col, "group": group, "unit": unit, "logger": logger,
            "role": role, "role_class": role_class(role),
            "min": round(float(src.min()), 2), "p1": round(float(q[0.01]), 2),
            "p50": round(float(q[0.5]), 2), "p99": round(float(q[0.99]), 2),
            "max": round(float(src.max()), 2), "distinct_values": int(src.nunique()),
            "non_integer_share": (round(float((src.round() != src).mean()), 5)
                                  if col in CODED else np.nan),
            "nan_raw": int(r.isna().sum()) if r is not None else np.nan,
            "nan_stage4": int(s4.isna().sum()), "nan_stage5": int(s5.isna().sum()),
        })
    table = pd.DataFrame(rows)
    tags = table[table["column"] != "site_total_AC_kW"]
    facts = {
        "n_tags": len(tags), "minutes": len(raw), "hours": len(d.h5),
        "raw_nan_total": int(tags["nan_raw"].sum()),
        "stage4_nan_max": int(tags["nan_stage4"].max()),
        "roles": table["role_class"].value_counts().to_dict(),
        "index_unique": bool(raw.index.is_unique), "index_regular": bool(
            (np.diff(raw.index.asi8) == 60_000_000).all()),
        "state_non_integer_share": float(table.set_index("column").loc["s01_inv1_state",
                                                                        "non_integer_share"]),
    }
    return Result("q01_inventory", "Inventory of the 27 tags", table=table, facts=facts)


def q02_held_values(d: Data) -> Result:
    day = d.daytime
    month = month_index(d.s4.index)[day]
    shares = {}
    for col in HELD_COLUMNS:
        shares[col] = pd.Series(d.frozen(col)[day]).groupby(month).mean() * 100
    frame = pd.DataFrame(shares).T
    fig, ax = plt.subplots(figsize=(14, 6))
    heatmap(ax, frame, "% of daytime minutes held >= 30 min", vmax=100, cmap="magma_r")
    ax.set_title("Q02 - Held values (a run of one value >= 30 min; power and kVar: non-zero), "
                 "share of daytime minutes per month, stage 4 (before cleaning)", fontsize=10)
    overall = {c: round(float(d.frozen(c)[day].mean() * 100), 3) for c in HELD_COLUMNS}
    train = {c: round(float(d.frozen(c)[day & d.train].mean() * 100), 3) for c in HELD_COLUMNS}
    dead = {c: [str(m) for m in frame.columns[frame.loc[c] > 90]] for c in HELD_COLUMNS}
    dead = {c: (v[0], v[-1], len(v)) for c, v in dead.items() if v}
    table = pd.DataFrame({"daytime_held_pct_all": overall, "daytime_held_pct_train": train})
    table["months_over_90pct"] = [dead.get(c, ("", "", 0))[2] for c in table.index]
    table["first_month_over_90pct"] = [dead.get(c, ("", "", 0))[0] for c in table.index]
    table = table.reset_index(names="column")
    return Result("q02_held_values", "Held values over time", table=table, figure=fig,
                  facts={"held_pct_all": overall, "held_pct_train": train,
                         "dead_months": {c: list(v) for c, v in dead.items()}})


def detection_stats(raw: pd.DataFrame, daytime: np.ndarray) -> pd.DataFrame:
    """Per tag-hour statistics of stage 4's source detection (resolution.hourly_source):
    share of daytime minutes in constant runs of POLL_RUN length, and changes per minute."""
    hour = raw.index[daytime].floor("h")
    frames = []
    for col in CONTINUOUS_COLUMNS:
        v = raw[col].to_numpy(dtype=float)
        changed = np.r_[True, v[1:] != v[:-1]]
        rid = np.cumsum(changed) - 1
        length = np.bincount(rid)[rid]
        poll = (length >= POLL_RUN[0]) & (length <= POLL_RUN[1])
        g = pd.DataFrame({"rate": changed[daytime], "poll": poll[daytime]}).groupby(hour)
        stats = g.agg(rate=("rate", "mean"), poll_share=("poll", "mean"), n=("rate", "size"))
        frames.append(stats[stats["n"] >= MIN_DETECT_MINUTES].assign(column=col))
    out = pd.concat(frames)
    out["class"] = np.where(out["poll_share"] >= POLLED_SHARE_MIN, "polled",
                            np.where(out["rate"] >= MINUTE_RATE_MIN, "minute", "other"))
    return out


def q03_sampling_regime(d: Data) -> Result:
    manifest = json.loads((STAGE4_DIR / "preprocess_manifest.json").read_text(encoding="utf-8"))
    first_day = pd.Timestamp(manifest["polled_source"]["first_day"])
    stats = detection_stats(d.raw, d.raw_elev > 15.0)
    after = np.asarray(local_day(stats.index) >= first_day)
    stats["period"] = np.where(after, f"from {first_day.date()}", f"before {first_day.date()}")
    table = stats.groupby(["period", "class"]).size().unstack(fill_value=0)
    table["tag_hours"] = table.sum(axis=1)
    table = table.reset_index()
    ambiguous = stats["poll_share"].between(0.3, 0.7, inclusive="neither")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.linspace(0, 1, 41)
    for period, colour in zip(sorted(stats["period"].unique()), ("#455a64", "#c62828"),
                              strict=True):
        s = stats[stats["period"] == period]
        a.hist(s["poll_share"], bins=bins, alpha=0.6, color=colour, label=period)
        b.hist(s.loc[s["poll_share"] < POLLED_SHARE_MIN, "rate"], bins=bins, alpha=0.6,
               color=colour, label=period)
    a.axvline(POLLED_SHARE_MIN, color="k", ls="--", lw=1, label=f"threshold {POLLED_SHARE_MIN}")
    b.axvline(MINUTE_RATE_MIN, color="k", ls="--", lw=1, label=f"threshold {MINUTE_RATE_MIN}")
    for ax in (a, b):
        ax.set_yscale("log")
        ax.set_ylabel("tag-hours (daytime, sun > 15 deg)")
        ax.legend(fontsize=7)
    a.set_xlabel(f"share of minutes in constant runs of {POLL_RUN[0]}-{POLL_RUN[1]} min")
    a.set_title("Polled signature per tag-hour", fontsize=9)
    b.set_xlabel("changes per minute (tag-hours not polled)")
    b.set_title("Change rate: minute source vs held / frozen", fontsize=9)
    fig.suptitle("Q03 - The two sampling regimes are separated per tag-hour (stage-4 detection, "
                 "raw export, 12 continuous tags)", fontsize=10)
    after_polled = stats[after]["class"].eq("polled").mean()
    before_polled = stats[~after]["class"].eq("polled").mean()
    return Result("q03_sampling_regime", "Two sampling regimes", table=table, figure=fig,
                  facts={"tag_hours": len(stats), "ambiguous_share": float(ambiguous.mean()),
                         "polled_share_before": float(before_polled),
                         "polled_share_after": float(after_polled),
                         "first_day": str(first_day.date()),
                         "minute_rate_median_before": float(
                             stats.loc[~after & (stats["class"] == "minute"), "rate"].median())})


def q04_distributions(d: Data) -> Result:
    ranges = d.policy["range"]
    cols = list(ranges)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    rows = []
    for ax, col in zip(axes.flat, cols, strict=False):
        lo, hi = ranges[col]
        v = d.s4[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        below, above = v[v < lo], v[v > hi]
        span = hi - lo
        shown = v[(v >= lo - 0.15 * span) & (v <= hi + 0.15 * span)]
        ax.hist(shown, bins=120, color="#455a64")
        ax.set_yscale("log")
        for x in (lo, hi):
            ax.axvline(x, color="#c62828", ls="--", lw=1)
        ax.set_title(col, fontsize=9)
        note = []
        if len(below):
            note.append(f"{len(below):,} < {lo:g} (min {below.min():,.0f})")
        if len(above):
            note.append(f"{len(above):,} > {hi:g} (max {above.max():,.0f})")
        ax.text(0.97, 0.97, "\n".join(note) or "all inside", transform=ax.transAxes,
                va="top", ha="right", fontsize=7, color="#c62828",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        rows.append({"column": col, "lower": lo, "upper": hi, "minutes": len(v),
                     "n_below": len(below), "n_above": len(above),
                     "min": round(float(v.min()), 2), "max": round(float(v.max()), 2)})
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle("Q04 - Value distributions before cleaning (stage 4, all periods, log count); "
                 "dashed: R1 limits of the cleaning policy", fontsize=10)
    fig.tight_layout()
    table = pd.DataFrame(rows)
    return Result("q04_distributions", "Distributions before cleaning", table=table, figure=fig,
                  facts={"outside": {r["column"]: r["n_below"] + r["n_above"] for r in rows}})

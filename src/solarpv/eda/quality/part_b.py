"""Part B: the evidence for each cleaning threshold (Q05-Q15).

Thresholds are judged on the train split of the stage-4 data (what the rules see); counts over
the whole period are given only as description.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.data.clean import (
    local_day_number,
    rule_frozen,
    source_reference,
    true_run_lengths,
    utc_hour_number,
)
from solarpv.eda.quality.common import (
    SPLIT_COLOURS,
    Result,
    dilate,
    month_index,
    naive,
    sample,
    segments,
)
from solarpv.eda.quality.data import INV_AC, INV_DC, MAIN, POWER, TEMPERATURES, Data

SOURCE_MINUTE = 0


def _source(d: Data, col: str) -> np.ndarray:
    return d.s4[source_reference(col)].to_numpy()


# ── R1 ─────────────────────────────────────────────────────────────────────


def q05_range_limits(d: Data) -> Result:
    rows, fig = [], plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    tr = d.train
    for i, (col, (lo, hi)) in enumerate(d.policy["range"].items()):
        v = d.s4[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        out = ok & ((v < lo) | (v > hi))
        m = ok & tr
        inside = m & ~out
        not_adjacent = inside & ~dilate(out, 1)
        outside = v[m & out]
        top = pd.Series(np.round(outside)).value_counts().head(4)
        row = {"column": col, "lower": lo, "upper": hi, "train_minutes": int(m.sum()),
               "n_below": int((m & (v < lo)).sum()), "n_above": int((m & (v > hi)).sum()),
               "smallest_inside": float(v[inside].min()),
               "smallest_inside_not_next_to_fault": float(v[not_adjacent].min()),
               "largest_inside": float(v[inside].max()),
               "largest_below": float(v[m & (v < lo)].max()) if (m & (v < lo)).any() else np.nan,
               "smallest_above": float(v[m & (v > hi)].min()) if (m & (v > hi)).any() else np.nan,
               "most_frequent_outside": "; ".join(f"{k:g} x{n}" for k, n in top.items())}
        rows.append(row)
        span = hi - lo
        x_in = [(row["smallest_inside_not_next_to_fault"] - lo) / span,
                (row["largest_inside"] - lo) / span]
        ax.plot(x_in, [i, i], color="#2e7d32", lw=6, solid_capstyle="butt")
        xo = (outside - lo) / span
        ax.scatter(xo, np.full(len(xo), i), s=8, color="#c62828", alpha=0.4, zorder=3)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(1, color="k", lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["column"] for r in rows], fontsize=8)
    ax.set_xlabel("position relative to the R1 limits: 0 = lower, 1 = upper (symlog)")
    ax.set_title("Q05 - Where the R1 limits sit (train, stage 4). Green: range of normal values "
                 "(not next to a fault); red: values outside the limits", fontsize=10)
    table = pd.DataFrame(rows).round(2)
    t = table.set_index("column")
    facts = {
        "power_largest_inside_over_upper": {c: float(t.loc[c, "largest_inside"] / t.loc[c, "upper"])
                                            for c in POWER},
        "power_smallest_above_over_upper": {
            c: (None if pd.isna(t.loc[c, "smallest_above"])
                else float(t.loc[c, "smallest_above"] / t.loc[c, "upper"])) for c in POWER},
        "temperature_smallest_inside_not_next": {
            c: float(t.loc[c, "smallest_inside_not_next_to_fault"]) for c in TEMPERATURES},
        "temperature_smallest_inside": {c: float(t.loc[c, "smallest_inside"])
                                        for c in TEMPERATURES},
        "temperature_largest_below": {c: (None if pd.isna(t.loc[c, "largest_below"])
                                          else float(t.loc[c, "largest_below"]))
                                      for c in TEMPERATURES},
        "irradiance_largest_inside": float(t.loc["irradiance", "largest_inside"]),
        "wind_largest_inside": float(t.loc["wind_speed", "largest_inside"]),
        "n_outside": {c: int(t.loc[c, "n_below"] + t.loc[c, "n_above"]) for c in t.index},
    }
    return Result("q05_range_limits", "R1 limits", table=table, figure=fig, facts=facts)


def _forward_ref(v: np.ndarray, sign: int) -> np.ndarray:
    """Median of the 3 minutes beyond each minute in direction `sign` (+1 later, -1 earlier)."""
    n = len(v)
    shifted = []
    for k in (1, 2, 3):
        s = np.full(n, np.nan)
        if sign > 0:
            s[:-k] = v[k:]
        else:
            s[k:] = v[:-k]
        shifted.append(s)
    return np.median(np.vstack(shifted), axis=0)


def q06_range_margin(d: Data) -> Result:
    rows, ratios = [], {}
    tr = d.train
    for col, (lo, hi) in d.policy["range"].items():
        v = d.s4[col].to_numpy(dtype=float)
        src = _source(d, col)
        ok = np.isfinite(v)
        core = ok & ((v < lo) | (v > hi))
        near = dilate(core, 4)
        fwd, bwd = _forward_ref(v, +1), _forward_ref(v, -1)
        normal_dev = np.abs(v - fwd)[tr & ok & ~near & (src == SOURCE_MINUTE) & np.isfinite(fwd)]
        if len(normal_dev) == 0:
            continue
        p999 = float(np.quantile(normal_dev, 0.999))
        devs = []
        for off, ref in ((+1, fwd), (-1, bwd)):  # neighbour after / before the fault
            j = np.flatnonzero(core & tr) + off
            j = j[(j >= 0) & (j < len(v))]
            j = j[ok[j] & ~core[j] & (src[j] == SOURCE_MINUTE)]
            r = ref[j]
            keep = np.isfinite(r) & ~dilate(core, 0)[np.clip(j + off, 0, len(v) - 1)]
            devs.append(np.abs(v[j] - r)[keep])
        dev = np.concatenate(devs)
        if len(dev) == 0:
            continue
        ratios[col] = dev / max(p999, 1e-9)
        rows.append({"column": col, "fault_minutes_train": int((core & tr).sum()),
                     "neighbours_checked": len(dev), "normal_change_p99_9": round(p999, 3),
                     "neighbour_deviation_median": round(float(np.median(dev)), 3),
                     "share_neighbours_beyond_normal": round(float((dev > p999).mean()), 3)})
    table = pd.DataFrame(rows)
    fam = {"temperatures": [c for c in table["column"] if c in TEMPERATURES],
           "power": [c for c in table["column"] if c in POWER]}
    pooled = {k: float(np.mean(np.concatenate([ratios[c] for c in cols]) > 1)) if cols else None
              for k, cols in fam.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    # example: a temperature fault on the minute source with its neighbours
    col = max(fam["temperatures"], key=lambda c: ratios[c].size) if fam["temperatures"] else None
    if col is not None:
        lo, hi = d.policy["range"][col]
        v4 = d.s4[col].to_numpy(dtype=float)
        core = np.isfinite(v4) & ((v4 < lo) | (v4 > hi)) & tr
        starts = segments(core)[:, 0]
        nxt = np.clip(starts - 1, 0, len(v4) - 1)
        cand = [s for s, j in zip(starts, nxt, strict=True)
                if np.isfinite(v4[j]) and abs(v4[j] - np.nanmedian(v4[max(j - 4, 0):j])) > 2]
        s = cand[0] if cand else starts[0]
        w = slice(s - 8, s + 12)
        t = naive(d.s4.index[w])
        a = axes[0]
        a.plot(t, v4[w], "o-", color="0.5", ms=4, label="stage 4")
        a.plot(t, d.s5[col].to_numpy()[w], "o-", color="#c62828", ms=4, label="stage 5")
        a.set_ylim(min(-25, np.nanmin(v4[w][v4[w] > -100]) - 5) if (v4[w] > -100).any() else -30,
                   np.nanmax(v4[w]) + 5)
        a.axhline(lo, color="k", ls=":", lw=1)
        a.set_title(f"{col}: fault code below the axis; the minutes next to it are blends",
                    fontsize=9)
        a.set_ylabel(f"{col} [degC]")
        a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        a.legend(fontsize=7)
    b = axes[1]
    for name, cols in fam.items():
        if cols:
            r = np.concatenate([ratios[c] for c in cols])
            b.hist(np.log10(np.maximum(r, 1e-3)), bins=40, alpha=0.6, label=name)
    b.axvline(0, color="k", ls="--", lw=1, label="normal 1-min change (p99.9)")
    b.set_xlabel("log10(neighbour deviation / normal p99.9 change)")
    b.set_ylabel("neighbours of R1 minutes (train, minute source)")
    b.legend(fontsize=7)
    fig.suptitle("Q06 - Are the neighbours of an out-of-range minute contaminated? "
                 "(R1 margin, +/-1 min on the minute source)", fontsize=10)
    return Result("q06_range_margin", "R1 margin", table=table, figure=fig,
                  facts={"share_beyond_normal": pooled,
                         "example_column": col,
                         "per_column": dict(zip(table["column"],
                                                table["share_neighbours_beyond_normal"],
                                                strict=True))})


# ── R2 ─────────────────────────────────────────────────────────────────────


def q07_night_offset(d: Data) -> Result:
    bands = [(-90, -6), (-6, 0), (0, 10), (10, 90)]
    rows = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    for col in d.policy["negative_to_zero"]:
        v = d.s4[col].to_numpy(dtype=float)
        for lo, hi in bands:
            m = d.train & np.isfinite(v) & (d.elev >= lo) & (d.elev < hi)
            x = v[m]
            rows.append({"column": col, "sun_band": f"{lo}..{hi}", "minutes": len(x),
                         "share_negative": round(float((x < 0).mean()), 5),
                         "min": round(float(x.min()), 2),
                         "p0_1": round(float(np.quantile(x, 0.001)), 2),
                         "share_below_minus_20": float((x < -20).mean())})
    v = d.s4["irradiance"].to_numpy(dtype=float)
    night = d.train & (d.elev < -6) & np.isfinite(v)
    a.hist(v[night], bins=np.arange(-9, 6, 0.25), color="#455a64")
    a.set_yscale("log")
    a.axvline(0, color="k", lw=1)
    a.set_xlabel("irradiance at night (sun < -6 deg) [W/m2]")
    a.set_title("The sensor reads a small negative offset at night", fontsize=9)
    idx = sample(d.train & np.isfinite(v) & (d.elev > -12) & (d.elev < 15) & (v < 20), 150_000)
    b.scatter(d.elev[idx], v[idx], s=1, alpha=0.2, color="#455a64")
    b.axhline(0, color="k", lw=1)
    b.set_xlabel("sun elevation [deg]")
    b.set_ylabel("irradiance [W/m2]")
    b.set_title("Negative values exist only around and below the horizon", fontsize=9)
    fig.suptitle("Q07 - R2: negative irradiance is the night offset of the sensor (train, "
                 "stage 4) -> 0", fontsize=10)
    table = pd.DataFrame(rows)
    t = table.set_index(["column", "sun_band"])
    return Result("q07_night_offset", "R2 night offset", table=table, figure=fig, facts={
        "irradiance_min_train": float(v[d.train & np.isfinite(v)].min()),
        "irradiance_negative_share_sun_above_10": float(t.loc[("irradiance", "10..90"),
                                                              "share_negative"]),
        "irradiance_negative_share_night": float(t.loc[("irradiance", "-90..-6"),
                                                       "share_negative"]),
        "pyranometer_min_train": float(np.nanmin(d.s4["pyranometer"].to_numpy()[d.train])),
        "any_below_minus_20": float(table["share_below_minus_20"].max())})


# ── R3 ─────────────────────────────────────────────────────────────────────

RUN_KS = (2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 240)


def q08_frozen_runs(d: Data) -> Result:
    rows, steps = [], {}
    fig, ax = plt.subplots(figsize=(10, 5))
    base_mask = d.train & d.daytime
    for col, minimum in d.policy["frozen_min_minutes"].items():
        v = d.s4[col].to_numpy(dtype=float)
        _, length = d.runs(col)
        base = base_mask & np.isfinite(v)
        if col in d.policy["frozen_ignore_zero"]:
            base &= v != 0
        r = length[base]
        shares = {k: float((r >= k).mean() * 100) for k in RUN_KS}
        rows.append({"column": col, "R3_minimum": minimum, **{f">={k}": round(s, 4)
                                                              for k, s in shares.items()}})
        diffs = np.abs(np.diff(v[d.train & np.isfinite(v)]))
        diffs = diffs[diffs > 0]
        steps[col] = float(pd.Series(np.round(diffs, 4)).mode().iloc[0]) if len(diffs) else None
        style = "--" if col in ("ambient_temp", "site_ambient_temp") else "-"
        ax.plot(RUN_KS, [shares[k] for k in RUN_KS], style, marker="o", ms=3, label=col)
    ax.axvline(30, color="k", ls=":", lw=1)
    ax.axvline(60, color="k", ls=":", lw=1)
    ax.text(30, ax.get_ylim()[1], " 30 min", fontsize=7, va="top")
    ax.text(60, ax.get_ylim()[1], " 60 min (ambient)", fontsize=7, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("run length k [min] (one value repeated)")
    ax.set_ylabel("% of daytime minutes in runs >= k")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Q08 - R3: how long a value stays constant (train, daytime, stage 4; power: "
                 "non-zero). The plateau after a few minutes is the freezes", fontsize=9)
    table = pd.DataFrame(rows)
    t = table.set_index("column")
    ratio_15_60 = {c: float(t.loc[c, ">=15"] / t.loc[c, ">=60"]) for c in t.index
                   if t.loc[c, ">=60"] > 0}
    ratio_30_120 = {c: float(t.loc[c, ">=30"] / t.loc[c, ">=120"]) for c in t.index
                    if t.loc[c, ">=120"] > 0}
    return Result("q08_frozen_runs", "R3 run lengths", table=table, figure=fig, facts={
        "ratio_15_60": ratio_15_60, "ratio_30_120": ratio_30_120, "value_step": steps,
        "share_ge_5": {c: float(t.loc[c, ">=5"]) for c in t.index},
        "share_ge_30": {c: float(t.loc[c, ">=30"]) for c in t.index}})


def coincidence(d: Data) -> np.ndarray:
    """Number of main columns frozen (run >= 30 min, power non-zero) at each minute."""
    return np.sum([d.frozen(c) for c in MAIN], axis=0)


def q09_freeze_events(d: Data) -> Result:
    count = coincidence(d)
    pairs = np.vstack([d.frozen(c) for c in MAIN])
    sel = pairs[:, d.train & d.daytime]
    per_pair = np.broadcast_to(count[d.train & d.daytime], sel.shape)[sel]
    share_multi = float((per_pair >= 5).mean()) if per_pair.size else np.nan
    event = count >= 5
    # join stretches less than 5 minutes apart
    gaps = segments(~event)
    for a, b in gaps:
        if 0 < a and b < len(event) and b - a < 5:
            event[a:b] = True
    rows = []
    idx = d.s4.index
    for a, b in segments(event):
        rows.append({"start_utc": idx[a], "end_utc": idx[b - 1] + pd.Timedelta(minutes=1),
                     "minutes": int(b - a), "max_columns": int(count[a:b].max()),
                     "daytime_minutes": int(d.daytime[a:b].sum()), "split": d.split[a]})
    events = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 2]})
    a.hist(per_pair, bins=np.arange(0.5, len(MAIN) + 1.5), color="#1565c0", rwidth=0.85)
    a.set_xlabel(f"main columns frozen at the same minute (of {len(MAIN)})")
    a.set_ylabel("frozen daytime minute-columns (train)")
    a.set_title("Freezes hit many sensors at once", fontsize=9)
    for s, colour in SPLIT_COLOURS.items():
        e = events[events["split"] == s]
        b.scatter(naive(pd.DatetimeIndex(e["start_utc"])), e["minutes"] / 60, s=12,
                  color=colour, label=f"{s} ({len(e)})")
    b.set_yscale("log")
    b.set_ylabel("duration [h]")
    b.set_title("Communication freezes (>= 5 main columns held >= 30 min)", fontsize=9)
    b.legend(fontsize=7)
    fig.suptitle("Q09 - R3 removes communication freezes, not sensor behaviour (stage 4)",
                 fontsize=10)
    events["start_utc"] = events["start_utc"].astype(str)
    events["end_utc"] = events["end_utc"].astype(str)
    return Result("q09_freeze_events", "Communication freezes", table=events, figure=fig,
                  facts={"share_multi_column_train": share_multi, "events_all": len(events),
                         "events_by_split": events["split"].value_counts().to_dict(),
                         "events_reaching_daytime": int((events["daytime_minutes"] > 0).sum()),
                         "longest_hours": float(events["minutes"].max() / 60)})


def q10_freeze_resume(d: Data) -> Result:
    positions, rows = [], []
    daytime = d.daytime
    for col, minimum in d.policy["frozen_min_minutes"].items():
        v = d.s4[col].to_numpy(dtype=float)
        src = _source(d, col)
        core = rule_frozen(v, minimum, daytime, col in d.policy["frozen_ignore_zero"])
        ends = np.flatnonzero(core[:-3] & ~core[1:-2])
        steps = np.abs(np.diff(v[d.train & np.isfinite(v)]))
        jump_min = float(np.quantile(steps[steps > 0], 0.99)) if (steps > 0).any() else 0.0
        for i in ends:
            x0, x1, x2 = v[i], v[i + 1], v[i + 2]
            if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(x2)):
                continue
            if src[i + 1] != SOURCE_MINUTE or abs(x2 - x0) <= jump_min:
                continue
            p = (x1 - x0) / (x2 - x0)
            positions.append(p)
            rows.append({"column": col, "resume_utc": str(d.s4.index[i + 1]),
                         "split": d.split[i + 1],
                         "stale": round(x0, 2), "first_new": round(x1, 2), "next": round(x2, 2),
                         "position": round(float(p), 3)})
    table = pd.DataFrame(rows)
    pos = np.asarray(positions)
    blend = (pos > 0.1) & (pos < 0.9)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    lo, hi = pd.Timestamp("2021-09-12 02:30", tz="UTC"), pd.Timestamp("2021-09-12 04:30", tz="UTC")
    w4 = d.s4.loc[lo:hi, "s01_inv1_AC_kW"]
    w5 = d.s5.loc[lo:hi, "s01_inv1_AC_kW"]
    a.plot(naive(w4.index), w4, "o-", ms=2, color="0.5", label="stage 4")
    a.plot(naive(w5.index), w5, "o-", ms=2, color="#c62828", label="stage 5")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a.set_ylabel("s01_inv1_AC_kW [kW]")
    a.set_title("2021-09-12 sunrise (UTC): the first minute after the freeze is a blend",
                fontsize=9)
    a.legend(fontsize=7)
    b.hist(np.clip(pos, -0.5, 1.5), bins=np.linspace(-0.5, 1.5, 41), color="#1565c0")
    b.axvspan(0.1, 0.9, color="0.9", zorder=0, label="blend: between stale and next")
    b.set_xlabel("position of the first new minute: 0 = stale value, 1 = next minute's value")
    b.set_ylabel("freeze ends (all periods, minute source)")
    b.legend(fontsize=7)
    fig.suptitle("Q10 - R3 margin: when data resume, the first 1-min average mixes the stale "
                 "and the new value", fontsize=10)
    train_pos = pos[table["split"].to_numpy() == "train"] if len(table) else pos
    return Result("q10_freeze_resume", "R3 margin", table=table, figure=fig, facts={
        "resumes": int(len(pos)), "blend_share": float(blend.mean()) if len(pos) else None,
        "resumes_train": int(len(train_pos)),
        "blend_share_train": (float(((train_pos > 0.1) & (train_pos < 0.9)).mean())
                              if len(train_pos) else None),
        "example_blend_kw": float(w4.loc["2021-09-12 03:00":"2021-09-12 04:00"]
                                  .pipe(lambda s: s[(s > 100) & (s < 400)]).iloc[0])
        if ((w4 > 100) & (w4 < 400)).any() else None})


# ── R4 / R5 ────────────────────────────────────────────────────────────────

BANDS = [(-90, -6), (-6, -3), (-3, 0), (0, 5), (5, 10)]


def q11_night(d: Data) -> Result:
    rows, near = [], {}
    for col in INV_AC:
        v = d.s4[col].to_numpy(dtype=float)
        near[col] = dilate(d.frozen(col), 15)
        for lo, hi in BANDS:
            m = d.train & np.isfinite(v) & (d.elev >= lo) & (d.elev < hi)
            x, xn = v[m], v[m & ~near[col]]
            rows.append({"column": col, "sun_band": f"{lo}..{hi}", "minutes": len(x),
                         "share_zero": round(float((x == 0).mean()), 4),
                         "share_0_to_5kW": round(float(((x > 0) & (x <= 5)).mean()), 4),
                         "share_above_5kW": round(float((x > 5).mean()), 4),
                         "p99_9_not_frozen": round(float(np.quantile(xn, 0.999)), 2),
                         "max_not_frozen": round(float(xn.max()), 2),
                         "max_any": round(float(x.max()), 1)})
    table = pd.DataFrame(rows)
    r5_in_freeze = []
    for col in INV_AC:
        v = d.s4[col].to_numpy(dtype=float)
        r5 = np.isfinite(v) & (d.elev < d.policy["night"]["stale_below_elevation"]) & (v > 5)
        r5_in_freeze.append((int(r5.sum()), int((r5 & near[col]).sum())))
    total = sum(a for a, _ in r5_in_freeze)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.5))
    col = INV_AC[0]
    v = d.s4[col].to_numpy(dtype=float)
    idx = sample(d.train & np.isfinite(v) & (d.elev > -12) & (d.elev < 8), 250_000)
    frozen = near[col][idx]
    a.scatter(d.elev[idx][~frozen], v[idx][~frozen], s=1, alpha=0.25, color="#455a64",
              label="normal")
    a.scatter(d.elev[idx][frozen], v[idx][frozen], s=2, alpha=0.6, color="#6a1b9a",
              label="in or next to a frozen run")
    a.set_yscale("symlog", linthresh=1)
    a.axhline(5, color="#c62828", ls="--", lw=1, label="5 kW (R4 / R5)")
    a.axvline(0, color="k", lw=1)
    a.axvline(-6, color="k", ls=":", lw=1)
    a.set_xlabel("sun elevation [deg]")
    a.set_ylabel(f"{col} [kW] (symlog)")
    a.set_title("Twilight output between -3 and 0 deg; frozen values far below the horizon",
                fontsize=9)
    a.legend(fontsize=7, markerscale=5)
    m = d.train & np.isfinite(v) & (d.elev < 0) & ~near[col] & (v > 0)
    b.hist(v[m], bins=np.logspace(-2, 2, 50), color="#455a64")
    b.set_xscale("log")
    b.axvline(5, color="#c62828", ls="--", lw=1)
    b.set_xlabel(f"{col} > 0 with the sun below the horizon, not frozen [kW]")
    b.set_ylabel("minutes (train)")
    b.set_title("Below the horizon: deep night is exactly 0, twilight gives a few kW",
                fontsize=9)
    fig.suptitle("Q11 - R4 (|AC| <= 5 kW below the horizon -> 0) and R5 (AC > 5 kW below -6 deg "
                 "-> NaN), stage 4", fontsize=10)
    t = table.set_index(["column", "sun_band"])
    return Result("q11_night", "R4 / R5 night", table=table, figure=fig, facts={
        "max_not_frozen_below_minus6": {c: float(t.loc[(c, "-90..-6"), "max_not_frozen"])
                                        for c in INV_AC},
        "max_not_frozen_twilight": {c: float(t.loc[(c, "-3..0"), "max_not_frozen"])
                                    for c in INV_AC},
        "p99_9_not_frozen_below_0": {c: float(np.quantile(
            d.s4[c].to_numpy()[d.train & (d.elev < 0) & ~near[c]
                               & np.isfinite(d.s4[c].to_numpy())], 0.999)) for c in INV_AC},
        "deep_night_nonzero_not_frozen": {c: int((d.train & (d.elev < -6) & ~near[c] & (
            d.s4[c].to_numpy() > 0)).sum()) for c in INV_AC},
        "r4_candidates_in_twilight_share": float(np.mean([
            (d.elev[m] >= -3).mean() for m in (
                (d.elev < 0) & ~near[c] & (d.s4[c].to_numpy() > 0) & (d.s4[c].to_numpy() <= 5)
                for c in INV_AC)])),
        "r5_minutes_all": total,
        "r5_share_in_or_next_to_freeze": (sum(b for _, b in r5_in_freeze) / total
                                          if total else None)})


# ── R6 ─────────────────────────────────────────────────────────────────────


def q12_dc_consistency(d: Data) -> Result:
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    min_ac = d.policy["dc_dead"]["min_ac_kw"]
    for k, (acc, dcc) in enumerate(zip(INV_AC, INV_DC, strict=True)):
        ac, dc = d.s4[acc].to_numpy(dtype=float), d.s4[dcc].to_numpy(dtype=float)
        ok = d.train & d.daytime & np.isfinite(ac) & np.isfinite(dc) & (ac < 1900) & (dc < 1900)
        m = ok & (dc > 300)
        eff = ac[m] / dc[m]
        dac = np.abs(np.r_[np.nan, np.diff(ac)]) / np.maximum(ac, 1)
        ddc = np.abs(np.r_[np.nan, np.diff(dc)]) / np.maximum(dc, 1)
        stable = (dac < 0.005) & (ddc < 0.005)
        eff_stable = ac[m & stable] / dc[m & stable]
        dead = np.isfinite(ac) & np.isfinite(dc) & (dc == 0) & (ac > min_ac)
        months = pd.Series(dead).groupby(month_index(d.s4.index)).sum()
        rows.append({"inverter": acc.split("_")[1], "minutes_dc_over_300": int(m.sum()),
                     "efficiency_p1": round(float(np.quantile(eff, 0.01)), 4),
                     "efficiency_median": round(float(np.median(eff)), 4),
                     "efficiency_p99": round(float(np.quantile(eff, 0.99)), 4),
                     "share_ac_above_dc": round(float((eff > 1).mean()), 4),
                     "share_ac_above_dc_stable": round(float((eff_stable > 1).mean()), 4),
                     "median_efficiency_stable": round(float(np.median(eff_stable)), 4),
                     "r6_minutes_all": int(dead.sum()),
                     "r6_top_month": str(months.idxmax()) if dead.any() else "",
                     "r6_share_top_month": round(float(months.max() / max(dead.sum(), 1)), 4)})
        axes[1].hist(eff, bins=np.linspace(0.85, 1.1, 101), histtype="step", lw=1.2,
                     label=f"{acc.split('_')[1]} all", density=True)
        axes[1].hist(eff_stable, bins=np.linspace(0.85, 1.1, 101), histtype="step", lw=1.2,
                     ls="--", label=f"{acc.split('_')[1]} steady minutes", density=True)
        axes[2].bar(months.index.to_timestamp() + pd.Timedelta(days=8 * k), months.to_numpy(),
                    width=10, label=acc.split("_")[1])
        if k == 0:
            idx = sample(ok & (dc > 0), 300_000)
            axes[0].hexbin(dc[idx], ac[idx], gridsize=80, bins="log", cmap="viridis")
            axes[0].plot([0, 1700], [0, 1700], "w--", lw=1)
            axes[0].set_xlabel(f"{dcc} [kW]")
            axes[0].set_ylabel(f"{acc} [kW]")
            axes[0].set_title("DC vs AC (train, daytime): DC = 0 column = dead DC sensor",
                              fontsize=9)
    axes[1].axvline(1, color="k", lw=1)
    axes[1].set_xlabel("AC / DC (DC > 300 kW)")
    axes[1].set_title("Efficiency: AC above DC is impossible physically", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].set_ylabel("minutes with DC = 0 and AC > 100 kW")
    axes[2].set_title("R6 minutes by month (all periods)", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle("Q12 - R6 (dead DC sensor) and the consistency of DC with AC (stage 4)",
                 fontsize=10)
    table = pd.DataFrame(rows)
    t = table.set_index("inverter")
    return Result("q12_dc_consistency", "R6 and AC/DC", table=table, figure=fig, facts={
        "efficiency_median": t["efficiency_median"].to_dict(),
        "share_ac_above_dc": t["share_ac_above_dc"].to_dict(),
        "share_ac_above_dc_stable": t["share_ac_above_dc_stable"].to_dict(),
        "r6_minutes": t["r6_minutes_all"].to_dict(), "r6_top_month": t["r6_top_month"].to_dict(),
        "r6_share_top_month": t["r6_share_top_month"].to_dict()})


# ── R7 ─────────────────────────────────────────────────────────────────────


def _period_ratio(ratio: pd.Series, key: np.ndarray) -> pd.DataFrame:
    return ratio.groupby(key).agg(["median", "size"])


def q13_scale_fault(d: Data) -> Result:
    p = d.policy["scale_fault"]
    lo, hi = p["ratio_range"]
    pyr = d.s4["pyranometer"].to_numpy(dtype=float)
    irr = d.s5["irradiance"].to_numpy(dtype=float)  # R7 compares with the cleaned reference
    day = local_day_number(d.s4.index)
    hour = utc_hour_number(d.s4.index)
    ok = np.isfinite(pyr) & np.isfinite(irr) & (d.elev > p["min_elevation"]) & \
        (irr > p["min_reference"])
    ratio = pd.Series(pyr[ok] / irr[ok])
    days = _period_ratio(ratio, day[ok])
    days = days[days["size"] >= p["min_minutes"]["day"]]
    split = pd.Series(d.split).groupby(day).first().reindex(days.index)
    dates = pd.to_datetime(days.index, unit="D")
    tr = (split == "train").to_numpy()
    r_tr = days.loc[tr, "median"]
    gap = r_tr[((r_tr >= 0.80) & (r_tr < 0.90)) | ((r_tr > 1.10) & (r_tr <= 1.20))]
    good_days = days.index[(days["median"] >= lo) & (days["median"] <= hi)]
    # hours outside the band on days the daily rule keeps: what the hourly rule adds (train)
    hours = _period_ratio(ratio, hour[ok])
    hours = hours[hours["size"] >= p["min_minutes"]["hour"]]
    hour_day = pd.Series(day, index=hour).groupby(level=0).first().reindex(hours.index)
    hour_split = pd.Series(d.split, index=hour).groupby(level=0).first().reindex(hours.index)
    extra = hours[hour_day.isin(good_days).to_numpy()
                  & ((hours["median"] < lo) | (hours["median"] > hi)).to_numpy()
                  & (hour_split == "train").to_numpy()]
    near_fault = extra["median"].between(0.72, 0.78)
    # ratio against sun elevation on good train days: why the sun must be high
    m = np.isfinite(pyr) & np.isfinite(irr) & (irr > p["min_reference"]) & \
        np.isin(day, good_days[(split.reindex(good_days) == "train").to_numpy()])
    bins = np.arange(5, 85, 5)
    by_elev = pd.Series(pyr[m] / irr[m]).groupby(pd.cut(d.elev[m], bins)).median()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    axes[0].hist(r_tr, bins=np.linspace(0.6, 1.3, 71), color="#00838f")
    for x in (lo, hi):
        axes[0].axvline(x, color="#c62828", ls="--", lw=1)
    axes[0].set_xlabel("daily median pyranometer / irradiance")
    axes[0].set_ylabel("train days")
    axes[0].set_title("Two populations: 1.00 and 0.75", fontsize=9)
    for s, colour in SPLIT_COLOURS.items():
        sel = (split == s).to_numpy()
        axes[1].scatter(dates[sel], days.loc[sel, "median"], s=3, color=colour, label=s)
    axes[1].axhspan(lo, hi, color="0.9", zorder=0)
    axes[1].set_ylabel("daily median ratio")
    axes[1].set_title("Timeline (all periods, descriptive)", fontsize=9)
    axes[1].legend(fontsize=7, markerscale=3)
    w = day == (pd.Timestamp("2021-11-05") - pd.Timestamp("1970-01-01")).days
    example = pd.Series(np.where(ok & w, pyr / np.where(irr > 0, irr, np.nan), np.nan),
                        index=d.s4.index)[w].resample("30min").median().dropna()
    axes[2].plot(naive(example.index), example.to_numpy(), "o-", color="#00838f")
    axes[2].axhspan(lo, hi, color="0.9", zorder=0)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[2].set_xlabel("UTC")
    axes[2].set_ylabel("30-min median ratio")
    axes[2].set_title("2021-11-05: the fault switches within a day", fontsize=9)
    centres = [iv.mid for iv in by_elev.index]
    axes[3].plot(centres, by_elev.to_numpy(), "o-", color="#00838f")
    axes[3].axvline(p["min_elevation"], color="#c62828", ls="--", lw=1)
    axes[3].set_xlabel("sun elevation [deg]")
    axes[3].set_ylabel("median ratio, good train days")
    axes[3].set_title("At low sun the two sensors differ", fontsize=9)
    fig.suptitle("Q13 - R7: the pyranometer's scale fault, judged per local day and per UTC hour "
                 "(sun > 30 deg, irradiance > 300 W/m2)", fontsize=10)
    table = pd.DataFrame({"local_day": dates.strftime("%Y-%m-%d"), "split": split.to_numpy(),
                          "usable_minutes": days["size"].to_numpy(),
                          "median_ratio": days["median"].round(4).to_numpy(),
                          "day_rule": ((days["median"] < lo) | (days["median"] > hi)).to_numpy()})
    return Result("q13_scale_fault", "R7 scale fault", table=table, figure=fig, facts={
        "train_days": int(tr.sum()),
        "train_gap_days": {str(pd.to_datetime(k, unit="D").date()): round(float(v), 4)
                           for k, v in gap.items()},
        "train_faulty_days": int((r_tr < lo).sum()),
        "train_faulty_median": float(r_tr[r_tr < lo].median()) if (r_tr < lo).any() else None,
        "train_good_median": float(r_tr[(r_tr >= lo) & (r_tr <= hi)].median()),
        "train_good_iqr": [float(r_tr[(r_tr >= lo) & (r_tr <= hi)].quantile(q))
                           for q in (0.25, 0.75)],
        "faulty_days_by_split": {s: int(((split == s) & (days["median"] < lo)).sum())
                                 for s in SPLIT_COLOURS},
        "train_extra_hours": int(len(extra)), "train_extra_hours_near_075": int(near_fault.sum()),
        "example_ratios": [round(float(x), 3) for x in example.to_numpy()],
        "ratio_by_elevation": {str(k): (None if pd.isna(v) else round(float(v), 4))
                               for k, v in by_elev.items()}})


# ── F1 / F2 / F3 ───────────────────────────────────────────────────────────


def q14_downtime(d: Data) -> Result:
    p = d.policy["downtime"]
    irr = d.s5["irradiance"].to_numpy(dtype=float)
    base = d.train & (d.elev > p["min_elevation"]) & np.isfinite(irr) & (irr > p["min_irradiance"])
    rows = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.5))
    for col in INV_AC:
        ac = d.s5[col].to_numpy(dtype=float)
        m = base & np.isfinite(ac)
        x = ac[m]
        low_irr = m & (irr < 200) & (ac >= p["max_ac_kw"])
        rows.append({"column": col, "minutes": len(x),
                     "share_below_20kW": round(float((x < p["max_ac_kw"]).mean()), 5),
                     "share_20_to_100kW": round(float(((x >= p["max_ac_kw"]) & (x < 100))
                                                      .mean()), 5),
                     "median_ac_at_150_200_wm2": round(float(np.median(ac[low_irr])), 1),
                     "p1_ac_at_150_200_wm2": round(float(np.quantile(ac[low_irr], 0.01)), 1)})
        a.hist(x, bins=np.arange(0, 1700, 10), histtype="step", lw=1.2, label=col)
    a.set_yscale("log")
    a.axvline(p["max_ac_kw"], color="#c62828", ls="--", lw=1, label="20 kW (F1)")
    a.set_xlabel("AC [kW] with sun > 5 deg and irradiance > 150 W/m2")
    a.set_ylabel("minutes (train)")
    a.set_title("A stop is a separate mode at 0, far below normal output", fontsize=9)
    a.legend(fontsize=7)
    ac = d.s5[INV_AC[0]].to_numpy(dtype=float)
    idx = sample(d.train & (d.elev > p["min_elevation"]) & np.isfinite(irr) & np.isfinite(ac),
                 300_000)
    b.hexbin(irr[idx], ac[idx], gridsize=80, bins="log", cmap="viridis")
    b.axvline(p["min_irradiance"], color="#c62828", ls="--", lw=1)
    b.axhline(p["max_ac_kw"], color="#c62828", ls="--", lw=1)
    b.set_xlabel("in-plane irradiance [W/m2]")
    b.set_ylabel(f"{INV_AC[0]} [kW]")
    b.set_title("Flag region: bottom right of the dashed lines", fontsize=9)
    fig.suptitle("Q14 - F1 inverter stop: sun > 5 deg, irradiance > 150 W/m2, AC < 20 kW "
                 "(train, after value rules)", fontsize=10)
    table = pd.DataFrame(rows)
    t = table.set_index("column")
    return Result("q14_downtime", "F1 downtime", table=table, figure=fig, facts={
        "share_below_20": t["share_below_20kW"].to_dict(),
        "share_20_to_100": t["share_20_to_100kW"].to_dict(),
        "median_at_150_200": t["median_ac_at_150_200_wm2"].to_dict(),
        "p1_at_150_200": t["p1_ac_at_150_200_wm2"].to_dict()})


def _relative_output(d: Data) -> np.ndarray:
    """Site AC per unit irradiance over the median of the same calendar month in normal
    operation (train, setpoint 100, no flag): 1.0 = what the plant normally makes."""
    ac = d.s4["site_total_AC_kW"].to_numpy(dtype=float)
    irr = d.s4["irradiance"].to_numpy(dtype=float)
    sp = d.s4["iec_sp_w"].to_numpy(dtype=float)
    bright = (d.elev > 10) & np.isfinite(irr) & (irr > 300) & np.isfinite(ac)
    per_irr = np.where(bright, ac / np.where(bright, irr, 1.0), np.nan)
    month = d.s4.index.month.to_numpy()
    normal = bright & d.train & (sp == 100) & ~d.s5["op_limited_flag"].to_numpy()
    reference = pd.Series(per_irr[normal]).groupby(month[normal]).median()
    return per_irr / reference.reindex(month).to_numpy()


def q15_curtailment(d: Data) -> Result:
    p = d.policy["curtailment"]
    sp = d.s4[p["setpoint_column"]].to_numpy(dtype=float)
    rel = _relative_output(d)
    limited = np.isfinite(sp) & (sp < p["below"])
    non_integer = limited & (np.abs(sp - np.round(sp)) > 1e-6)
    runs = segments(non_integer)
    lengths = runs[:, 1] - runs[:, 0]
    run_split = d.split[runs[:, 0]] if len(runs) else np.array([])
    tr_lengths = lengths[run_split == "train"]
    flag = d.s5["curtail_flag"].to_numpy()
    up = d.elev > p["min_elevation"]
    m = up & limited
    local = d.s4.index[m].tz_convert(TIMEZONE_LOCAL).strftime("%Y-%m-%d")
    days = pd.DataFrame({"day": local, "split": d.split[m], "setpoint": sp[m],
                         "integer": ~non_integer[m], "flagged": flag[m], "rel": rel[m]})
    table = days.groupby(["day", "split"]).agg(
        minutes=("setpoint", "size"), median_setpoint=("setpoint", "median"),
        integer_share=("integer", "mean"), flagged_share=("flagged", "mean"),
        relative_output=("rel", "median")).round(3).reset_index()
    tr = table[table["split"] == "train"]
    commanded = tr[tr["flagged_share"] > 0.5]
    drift = tr[tr["flagged_share"] <= 0.5]
    mode = np.round(d.s4["iec_mode"].to_numpy(dtype=float))
    normal = d.train & np.isfinite(rel) & (sp == 100) & ~d.s5["op_limited_flag"].to_numpy()
    by_mode = pd.Series(rel[normal]).groupby(mode[normal]).agg(["median", "size"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    a = axes[0]
    for name, g, colour in (("commanded step (flagged)", commanded, "#f9a825"),
                            ("non-integer drift (not flagged)", drift, "#455a64")):
        a.scatter(g["median_setpoint"], g["relative_output"], s=25, color=colour, label=name)
    a.plot([0, 100], [0, 1], "k:", lw=1, label="output = setpoint")
    a.axhline(1, color="0.6", lw=0.8)
    a.set_xlabel("daily median setpoint [%] (train days with setpoint < 100)")
    a.set_ylabel("relative output (1 = normal for the month)")
    a.set_title("Commanded steps cut output; the 2022 drift did not", fontsize=9)
    a.legend(fontsize=7)
    b = axes[1]
    b.hist(tr_lengths, bins=np.logspace(0, 4, 41), color="#455a64")
    b.set_xscale("log")
    b.axvline(p["max_noninteger_run_minutes"], color="#c62828", ls="--", lw=1,
              label=f"{p['max_noninteger_run_minutes']} min")
    b.set_xlabel("run of non-integer setpoint values [min] (train)")
    b.set_ylabel("runs")
    b.set_title("Step changes last 2-3 min; the drift lasted 3 days", fontsize=9)
    b.legend(fontsize=7)
    c = axes[2]
    c.bar([f"mode {int(k)}\n(n={int(n):,})" for k, n in by_mode["size"].items()],
          by_mode["median"], color="#455a64")
    c.set_ylabel("median relative output (train, setpoint 100)")
    c.set_ylim(0.9, 1.05)
    c.set_title("iec_mode does not change output", fontsize=9)
    fig.suptitle("Q15 - F2 curtailment: a commanded setpoint step limits output; a non-integer "
                 "drift and iec_mode do not", fontsize=10)
    step_rel = {}
    for level in (0, 25, 50, 75):
        sel = d.train & up & (sp == level) & np.isfinite(rel)
        if sel.any():
            step_rel[str(level)] = round(float(np.median(rel[sel])), 3)
    return Result("q15_curtailment", "F2 curtailment", table=table, figure=fig, facts={
        "minutes_below_100_by_split": {s: int((up & limited & (d.split == s)).sum())
                                       for s in SPLIT_COLOURS},
        "flagged_minutes_by_split": {s: int((flag & (d.split == s)).sum()) for s in SPLIT_COLOURS},
        "train_transition_run_max": int(tr_lengths[tr_lengths <= p["max_noninteger_run_minutes"]]
                                        .max()) if len(tr_lengths) else None,
        "train_drift_runs": [int(x) for x in
                             tr_lengths[tr_lengths > p["max_noninteger_run_minutes"]]],
        "relative_output_by_setpoint_train": step_rel,
        "relative_output_commanded_days_median": float(commanded["relative_output"].median()),
        "relative_output_drift_days": {r["day"]: float(r["relative_output"])
                                       for _, r in drift.iterrows()},
        "mode_relative_output": {str(int(k)): round(float(v), 4)
                                 for k, v in by_mode["median"].items()}})


def _windowed_ratio(d: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred windowed median and IQR of log(AC1 / AC2), and the minutes where both run."""
    p = d.policy["derate"]
    a1 = d.s5[INV_AC[0]].to_numpy(dtype=float)
    a2 = d.s5[INV_AC[1]].to_numpy(dtype=float)
    both = (d.elev > p["ratio_min_elevation"]) & (a1 > p["ratio_min_ac_kw"]) & \
        (a2 > p["ratio_min_ac_kw"])
    lr = np.full(len(a1), np.nan)
    lr[both] = np.log(a1[both] / a2[both])
    window = pd.Series(lr).rolling(p["ratio_window_minutes"], center=True,
                                   min_periods=p["ratio_min_valid_minutes"])
    median = window.median().to_numpy()
    iqr = (window.quantile(0.75) - window.quantile(0.25)).to_numpy()
    return median, iqr, both


def q16_derate(d: Data) -> Result:
    p = d.policy["derate"]
    median, iqr, both = _windowed_ratio(d)
    tr = d.train & both & np.isfinite(median)
    thr = np.log(p["ratio_below"])
    low = tr & (np.abs(median) > -thr)
    valley = tr & (np.abs(median) > np.log(1 / 0.88)) & (np.abs(median) < np.log(1 / 0.78))
    rows = []
    for i, (acc, other) in enumerate(((INV_AC[0], INV_AC[1]), (INV_AC[1], INV_AC[0])), start=1):
        ac = d.s5[acc].to_numpy(dtype=float)
        oth = d.s5[other].to_numpy(dtype=float)
        st = d.s5[f"s01_inv{i}_state"].to_numpy(dtype=float)
        producing = d.train & (d.elev > 15) & np.isfinite(ac) & (ac >= p["min_ac_kw"]) & \
            np.isfinite(oth) & (oth > 200)
        for code in (1, 3, 5, 6):
            sel = producing & (st == code)
            if code in p["states"]:
                long = true_run_lengths((d.elev > p["min_elevation"]) & (st == code)
                                        & (ac >= p["min_ac_kw"])) >= p["state_min_run_minutes"]
                for name, mask in (("run >= 30 min", sel & long), ("run < 30 min", sel & ~long)):
                    if mask.any():
                        rows.append({"inverter": f"inv{i}", "state": code, "episode": name,
                                     "minutes": int(mask.sum()),
                                     "median_ratio_to_other": round(float(np.median(
                                         ac[mask] / oth[mask])), 3)})
            elif sel.any():
                rows.append({"inverter": f"inv{i}", "state": code, "episode": "all",
                             "minutes": int(sel.sum()),
                             "median_ratio_to_other": round(float(np.median(ac[sel] / oth[sel])),
                                                            3)})
    table = pd.DataFrame(rows)
    flags = {f"inv{i}": d.s5[f"derate_inv{i}_flag"].to_numpy() for i in (1, 2)}
    hourly = (d.h5["frac_derate_inv1_flag"] + d.h5["frac_derate_inv2_flag"]) > 0
    lost = {}
    for i, (acc, other) in enumerate(((INV_AC[0], INV_AC[1]), (INV_AC[1], INV_AC[0])), start=1):
        ac = d.s5[acc].to_numpy(dtype=float)
        oth = d.s5[other].to_numpy(dtype=float)
        f = flags[f"inv{i}"]
        for s in SPLIT_COLOURS:
            m = f & (d.split == s) & np.isfinite(ac) & np.isfinite(oth)
            lost[s] = lost.get(s, 0.0) + float(np.sum(np.clip(oth[m] - ac[m], 0, None))) / 60
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    a = axes[0]
    a.hist(np.exp(median[tr]), bins=np.exp(np.linspace(-0.45, 0.45, 91)), color="#6a1b9a")
    a.set_yscale("log")
    for x in (p["ratio_below"], 1 / p["ratio_below"]):
        a.axvline(x, color="#c62828", ls="--", lw=1)
    a.set_xlabel("31-min median of AC1 / AC2 (train)")
    a.set_ylabel("minutes")
    a.set_title("Modes at 0.99 and 0.74 / 1.30; empty valleys between", fontsize=9)
    b = axes[1]
    b.hist(np.clip(iqr[low], 1e-4, 1), bins=np.logspace(-4, 0, 41), color="#6a1b9a")
    b.set_xscale("log")
    b.axvline(p["ratio_max_iqr"], color="#c62828", ls="--", lw=1, label="0.05")
    b.set_xlabel("IQR of log(AC1 / AC2) in low-median windows")
    b.set_ylabel("minutes (train)")
    b.set_title("A derate is a stable fraction; clouds are not", fontsize=9)
    b.legend(fontsize=7)
    c = axes[2]
    labels = [f"inv{r.inverter[-1]} code {r.state}\n{r.episode}" for r in table.itertuples()]
    c.barh(range(len(table)), table["median_ratio_to_other"], color="#455a64")
    c.set_yticks(range(len(table)))
    c.set_yticklabels(labels, fontsize=6)
    c.axvline(1, color="k", lw=0.8)
    c.axvline(p["ratio_below"], color="#c62828", ls="--", lw=1)
    c.set_xlabel("median AC ratio to the other inverter (train)")
    c.set_title("Status codes 5 / 6 for >= 30 min mean a derate", fontsize=9)
    e = axes[3]
    w = d.s5.loc["2023-06-27 03:00":"2023-06-27 16:00"]
    e.plot(naive(w.index), w[INV_AC[0]], lw=1, label="inverter 1")
    e.plot(naive(w.index), w[INV_AC[1]], lw=1, label="inverter 2")
    e.fill_between(naive(w.index), 0, 1, where=w["derate_inv2_flag"].to_numpy(),
                   transform=e.get_xaxis_transform(), color="#6a1b9a", alpha=0.15,
                   label="F3 on inverter 2")
    e.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    e.set_xlabel("2023-06-27, UTC")
    e.set_ylabel("AC [kW]")
    e.legend(fontsize=7)
    e.set_title("A derated day: inverter 2 at 77 % of inverter 1", fontsize=9)
    fig.suptitle("Q16 - F3 derated inverter (stage 3b, F-22): one inverter runs at a stable "
                 "fraction of its twin for hours or days", fontsize=10)
    hist, edges = np.histogram(median[tr], bins=np.linspace(-0.45, 0.45, 91))
    centres = np.exp((edges[:-1] + edges[1:]) / 2)
    modes = {}
    for name, sel in (("normal", (centres > p["ratio_below"]) & (centres < 1 / p["ratio_below"])),
                      ("inv1_low", centres <= p["ratio_below"]),
                      ("inv2_low", centres >= 1 / p["ratio_below"])):
        modes[name] = float(centres[sel][np.argmax(hist[sel])])
    return Result("q16_derate", "F3 derate", table=table, figure=fig, facts={
        "modes": modes,
        "train_windows": int(tr.sum()), "train_low_windows": int(low.sum()),
        "train_valley_share": float(valley.sum() / tr.sum()),
        "train_low_stable_share": float((iqr[low] < 0.01).mean()),
        "train_low_unstable_share": float((iqr[low] >= p["ratio_max_iqr"]).mean()),
        "flag_minutes_by_split": {s: int(sum((f & (d.split == s)).sum() for f in flags.values()))
                                  for s in SPLIT_COLOURS},
        "hours_by_split": {s: int((hourly.to_numpy() & d.hour_daytime & (d.hour_split == s)).sum())
                           for s in SPLIT_COLOURS},
        "lost_mwh_by_split": {s: round(v / 1000, 1) for s, v in lost.items()},
        "ratio_by_state": {f"{r.inverter} code {r.state} {r.episode}": r.median_ratio_to_other
                           for r in table.itertuples()}})


# ── hourly thresholds ──────────────────────────────────────────────────────


def q17_hourly_thresholds(d: Data) -> Result:
    day = d.hour_daytime & (d.hour_split == "train")
    n5 = d.h5["n_site_total_AC_kW"].to_numpy()
    n4 = d.h4["n_site_total_AC_kW"].to_numpy()
    frac = d.h5["frac_op_limited_flag"].to_numpy()
    cover = [30, 40, 45, 50, 55, 60]
    shares = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    rows = [{"threshold": f"coverage >= {c}/60", "daytime_train_hours_affected":
             int((day & (n5 < c)).sum())} for c in cover]
    rows += [{"threshold": f"op-limited share >= {s}", "daytime_train_hours_affected":
              int((day & (frac >= s)).sum())} for s in shares]
    table = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    a.hist([n4[day], n5[day]], bins=np.arange(-0.5, 61.5, 1), label=["stage 4", "stage 5"],
           color=["0.6", "#1565c0"])
    a.set_yscale("log")
    a.axvline(44.5, color="#c62828", ls="--", lw=1, label="45/60")
    a.set_xlabel("valid target minutes in the hour")
    a.set_ylabel("daytime hours (train)")
    a.set_title("Hours are complete or empty; the 45/60 threshold is not binding", fontsize=9)
    a.legend(fontsize=7)
    f = frac[day & (frac > 0)]
    b.hist(f, bins=np.linspace(0, 1, 61), color="#f9a825")
    b.axvline(0.05, color="#c62828", ls="--", lw=1, label="0.05 (3 of 60 minutes)")
    b.set_xlabel("share of flagged minutes (F1, F2 or F3) in the hour")
    b.set_ylabel("daytime hours with any flag (train)")
    b.legend(fontsize=7)
    b.set_title("Few flagged hours lie near the 0.05 threshold", fontsize=9)
    fig.suptitle("Q17 - Hourly thresholds: coverage 45/60 (A-09) and op-limited share 0.05 "
                 "(D-22), train", fontsize=10)
    t = table.set_index("threshold")["daytime_train_hours_affected"]
    return Result("q17_hourly_thresholds", "Hourly thresholds", table=table, figure=fig, facts={
        "daytime_train_hours": int(day.sum()),
        "hours_partial_coverage_1_44": int((day & (n5 >= 1) & (n5 < 45)).sum()),
        "hours_empty": int((day & (n5 == 0)).sum()),
        "share_partial": float((day & (n5 >= 1) & (n5 < 45)).sum() / day.sum()),
        "op_hours_at": {str(s): int(t[f"op-limited share >= {s}"]) for s in shares},
        "flagged_hours_full_share": float((f >= 0.95).mean()) if f.size else None})

"""Part C: after cleaning, stage 5 against stage 4 (Q17-Q21). Descriptive over all periods."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, TIMEZONE_LOCAL
from solarpv.data.clean import RULES, cleaned_columns
from solarpv.eda.quality.common import RULE_COLOURS, Result, heatmap, month_index, segments
from solarpv.eda.quality.data import CLEANING_LOG, INV_AC, MAIN, SPLITS, Data

TARGET = "site_total_AC_kW"


def _columns(d: Data) -> list[str]:
    return [c for c in cleaned_columns(d.policy) if f"rule_{c}" in d.s5]


def q18_changes(d: Data) -> Result:
    rows = []
    daytime = d.daytime
    for col in _columns(d):
        code = d.s5[f"rule_{col}"].to_numpy()
        before = d.s4[col].to_numpy(dtype=float)
        after = d.s5[col].to_numpy(dtype=float)
        for r, name in RULES.items():
            hit = code == r
            if not hit.any():
                continue
            for s in SPLITS:
                m = hit & (d.split == s)
                if not m.any():
                    continue
                row = {"rule": name, "column": col, "split": s, "minutes": int(m.sum()),
                       "pct_of_split_minutes": round(100 * m.sum() / (d.split == s).sum(), 4),
                       "daytime_minutes": int((m & daytime).sum()), "kwh_changed": np.nan}
                if col in INV_AC:
                    row["kwh_changed"] = round(float(np.nansum(
                        before[m] - np.nan_to_num(after[m]))) / 60, 1)
                rows.append(row)
    table = pd.DataFrame(rows)
    log = pd.read_csv(CLEANING_LOG)
    mine = table.groupby(["rule", "column"])["minutes"].sum()
    theirs = log.set_index(["rule", "column"])["minutes"]
    matches = bool(mine.sort_index().equals(theirs.sort_index().astype(mine.dtype)))
    month = month_index(d.s5.index)
    per_rule = {}
    for r, name in RULES.items():
        hits = np.zeros(len(d.s5), dtype=np.int32)
        for col in _columns(d):
            hits += d.s5[f"rule_{col}"].to_numpy() == r
        per_rule[name] = pd.Series(hits).groupby(month).sum()
    frame = pd.DataFrame(per_rule).T
    fig, ax = plt.subplots(figsize=(14, 3.8))
    heatmap(ax, np.log10(1 + frame), "log10(1 + minute-columns changed)", cmap="Reds")
    ax.set_title("Q18 - When each rule acted: minute-columns changed per month (all cleaned "
                 "columns, all periods)", fontsize=10)
    totals = table.groupby("rule")["minutes"].sum().to_dict()
    by_split = table.groupby("split")["minutes"].sum().to_dict()
    kwh = table.dropna(subset=["kwh_changed"]).groupby("split")["kwh_changed"].sum().to_dict()
    return Result("q18_changes", "What cleaning changed", table=table, figure=fig, facts={
        "matches_cleaning_log": matches, "minutes_by_rule": totals,
        "minutes_by_split": by_split, "kwh_by_split": kwh,
        "kwh_by_rule": table.dropna(subset=["kwh_changed"]).groupby("rule")["kwh_changed"]
        .sum().round(1).to_dict(),
        "busiest_month_by_rule": {k: str(frame.loc[k].idxmax()) for k in frame.index
                                  if frame.loc[k].max() > 0}})


def q19_distributions(d: Data) -> Result:
    qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    rows, worst = [], {}
    outside_after = {}
    for col in _columns(d):
        b = d.s4[col].to_numpy(dtype=float)
        a = d.s5[col].to_numpy(dtype=float)
        mb, ma = d.daytime & np.isfinite(b), d.daytime & np.isfinite(a)
        qb, qa = np.quantile(b[mb], qs), np.quantile(a[ma], qs)
        scale = max(qb[3] - qb[1], 1e-9)
        shift = np.abs(qa - qb) / scale * 100
        row = {"column": col, "daytime_minutes_before": int(mb.sum()),
               "daytime_minutes_after": int(ma.sum()),
               "min_before": round(float(b[mb].min()), 2),
               "min_after": round(float(a[ma].min()), 2),
               "max_before": round(float(b[mb].max()), 2),
               "max_after": round(float(a[ma].max()), 2)}
        for q, x, y, s in zip(qs, qb, qa, shift, strict=True):
            row[f"p{q * 100:g}_before"] = round(float(x), 2)
            row[f"p{q * 100:g}_after"] = round(float(y), 2)
            row[f"p{q * 100:g}_shift_pct"] = round(float(s), 3)
        rows.append(row)
        worst[col] = float(shift[1:4].max())
        if col in d.policy["range"]:
            lo, hi = d.policy["range"][col]
            outside_after[col] = int(((a < lo) | (a > hi)).sum())
    table = pd.DataFrame(rows)
    show = ["s01_inv1_AC_kW", "s01_inv1_DC_kW", "irradiance", "pyranometer", "mod_temp1",
            "ambient_temp"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, col in zip(axes.flat, show, strict=True):
        b, a = d.s4[col].to_numpy(dtype=float), d.s5[col].to_numpy(dtype=float)
        lo, hi = d.policy["range"][col]
        bins = np.linspace(lo, hi, 120)
        ax.hist(np.clip(b[d.daytime & np.isfinite(b)], lo, hi), bins=bins, histtype="step",
                color="0.5", lw=1.2, label="stage 4")
        ax.hist(a[d.daytime & np.isfinite(a)], bins=bins, histtype="step", color="#c62828", lw=1,
                label="stage 5")
        ax.set_yscale("log")
        ax.set_title(f"{col} (values beyond the limits drawn at the limit)", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Q19 - Daytime distributions before and after cleaning: only the tails change",
                 fontsize=10)
    fig.tight_layout()
    return Result("q19_distributions", "Distributions before and after", table=table, figure=fig,
                  facts={"max_bulk_shift_pct": worst, "outside_range_after": outside_after})


def _local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()


def q20_energy(d: Data) -> Result:
    day = _local_day(d.s4.index)
    before = d.s4[TARGET].groupby(day).sum() / 60 / 1000
    after = d.s5[TARGET].groupby(day).sum() / 60 / 1000
    change = (after - before) / before.where(before > 1) * 100
    # rule responsible for most of the change on each day (AC columns)
    kwh = {}
    for col in INV_AC:
        code = d.s5[f"rule_{col}"].to_numpy()
        diff = d.s4[col].to_numpy(dtype=float) - np.nan_to_num(d.s5[col].to_numpy(dtype=float))
        for r, name in RULES.items():
            m = code == r
            if m.any():
                kwh[name] = kwh.get(name, 0) + pd.Series(np.where(m, np.abs(diff), 0.0)).groupby(
                    day).sum().to_numpy() / 60
    rules = pd.DataFrame(kwh, index=before.index)
    dominant = rules.idxmax(axis=1).where(rules.max(axis=1) > 0)
    split = pd.Series(d.split).groupby(day).first()
    changed = change.abs() > 0.1
    table = pd.DataFrame({"local_day": before.index[changed].strftime("%Y-%m-%d"),
                          "split": split[changed].to_numpy(),
                          "mwh_before": before[changed].round(3).to_numpy(),
                          "mwh_after": after[changed].round(3).to_numpy(),
                          "change_pct": change[changed].round(2).to_numpy(),
                          "main_rule": dominant[changed].to_numpy()})
    by_split = {}
    for s in SPLITS:
        m = (split == s).to_numpy()
        by_split[s] = round(float((after[m].sum() / before[m].sum() - 1) * 100), 3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [2, 1]})
    for name in rules.columns:
        sel = (dominant == name) & changed
        a.scatter(before.index[sel], change[sel], s=10, color=RULE_COLOURS.get(name, "k"),
                  label=f"{name} ({int(sel.sum())} days)")
    a.axhline(0, color="k", lw=0.8)
    a.set_ylabel("daily energy change [%]")
    a.set_title("Days whose energy changed by more than 0.1 %", fontsize=9)
    a.legend(fontsize=7)
    b.scatter(before, after, s=3, alpha=0.4)
    m = float(before.max())
    b.plot([0, m], [0, m], "k:", lw=1)
    b.set_xlabel("daily energy, stage 4 [MWh]")
    b.set_ylabel("daily energy, stage 5 [MWh]")
    fig.suptitle("Q20 - Energy is kept except where a freeze or a spike was removed", fontsize=10)
    return Result("q20_energy", "Energy before and after", table=table, figure=fig, facts={
        "days": int(before.size), "days_changed": int(changed.sum()),
        "days_changed_by_rule": table["main_rule"].value_counts().to_dict(),
        "total_change_pct_by_split": by_split,
        "largest_increase_pct": float(change.max()), "largest_decrease_pct": float(change.min())})


def q21_missing_after(d: Data) -> Result:
    day = d.daytime
    month = month_index(d.s5.index)[day]
    cols = [c for c in (TARGET, *MAIN, "wind_speed") if c in d.s5]
    frame = pd.DataFrame({c: pd.Series(d.s5[c].isna().to_numpy()[day]).groupby(month).mean() * 100
                          for c in cols}).T
    hourly_nan = d.h5[TARGET].isna().to_numpy() & d.hour_daytime
    rows = []
    for s in SPLITS:
        m = d.hour_split == s
        nan_h = hourly_nan & m
        runs = segments(nan_h)
        longest = int((runs[:, 1] - runs[:, 0]).max()) if len(runs) else 0
        rows.append({"split": s, "daytime_hours": int((d.hour_daytime & m).sum()),
                     "target_nan_daytime_hours": int(nan_h.sum()),
                     "pct": round(100 * nan_h.sum() / max((d.hour_daytime & m).sum(), 1), 3),
                     "longest_gap_hours": longest})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    heatmap(ax, frame, "% of daytime minutes missing", vmax=max(5, float(np.nanmax(frame))),
            cmap="Blues")
    ax.set_title("Q21 - Missing values after cleaning (daytime minutes, by month)", fontsize=10)
    return Result("q21_missing_after", "Missing after cleaning", table=table, figure=fig, facts={
        "target_nan_daytime_hours": dict(zip(table["split"], table["target_nan_daytime_hours"],
                                             strict=True)),
        "missing_pct_all": {c: round(float(d.s5[c].isna().to_numpy()[day].mean() * 100), 3)
                            for c in cols}})


def _imbalance_without_f3(h: pd.DataFrame, day: np.ndarray) -> int:
    """Daytime hours with the inverter ratio outside 0.8..1.25 if F3 did not exist (only F1/F2
    hours excluded): what F3 explains."""
    share = (h[["frac_downtime_inv1_flag", "frac_downtime_inv2_flag", "frac_curtail_flag"]]
             .sum(axis=1) > 0).to_numpy()
    both = day & (h["s01_inv1_AC_kW"] > 200).to_numpy() & (h["s01_inv2_AC_kW"] > 200).to_numpy() \
        & ~share
    bal = (h["s01_inv1_AC_kW"] / h["s01_inv2_AC_kW"])[both]
    return int(((bal < 0.8) | (bal > 1.25)).sum())


def q22_residual(d: Data) -> Result:
    h = d.h5
    day = d.hour_daytime
    theo = h["poa_clearsky"] * AC_CAPACITY_KW / 1000
    csi = h[TARGET] / theo.where(theo > 50)
    items = []
    for t, v in csi[day & (csi > 1.2).to_numpy()].items():
        items.append({"check": "hourly AC / clear-sky AC > 1.2", "timestamp_utc": str(t),
                      "value": round(float(v), 3), "split": d.hour_split[h.index.get_loc(t)]})
    frozen_left = {}
    for col, minimum in d.policy["frozen_min_minutes"].items():
        frozen_left[col] = int((d.frozen(col, minimum, stage="s5") & d.daytime).sum())
    # robust outliers of AC per unit irradiance in normal operation (minute-source hours)
    ok = day & h[TARGET].notna().to_numpy() & (h["irradiance"] > 200).to_numpy() & \
        ~h["op_limited_flag"].to_numpy() & (h["hour_source"] == "minute").to_numpy()
    ratio = (h[TARGET] / h["irradiance"])[ok]
    month = month_index(ratio.index)
    med = ratio.groupby(month).transform("median")
    mad = (ratio - med).abs().groupby(month).transform("median") * 1.4826
    z = (ratio - med) / mad
    for t, v in z[z.abs() > 6].items():
        items.append({"check": "AC / irradiance robust z > 6 (normal operation)",
                      "timestamp_utc": str(t), "value": round(float(v), 2),
                      "split": d.hour_split[h.index.get_loc(t)]})
    both = day & (h["s01_inv1_AC_kW"] > 200).to_numpy() & (h["s01_inv2_AC_kW"] > 200).to_numpy() \
        & ~h["op_limited_flag"].to_numpy()
    bal = (h["s01_inv1_AC_kW"] / h["s01_inv2_AC_kW"])[both]
    for t, v in bal[(bal < 0.8) | (bal > 1.25)].items():
        items.append({"check": "inverter 1 / inverter 2 outside 0.8..1.25 (no flag)",
                      "timestamp_utc": str(t), "value": round(float(v), 3),
                      "split": d.hour_split[h.index.get_loc(t)]})
    table = pd.DataFrame(items, columns=["check", "timestamp_utc", "value", "split"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    a.hist(csi[day].dropna(), bins=np.linspace(0, 1.4, 71), color="#455a64")
    a.axvline(1.2, color="#c62828", ls="--", lw=1)
    a.set_yscale("log")
    a.set_xlabel("hourly AC / clear-sky AC (POA_cs x 3,449 kW)")
    a.set_ylabel("daytime hours")
    a.set_title("Above 1.2 only at sunrise and sunset (clear-sky model at low sun)",
                fontsize=9)
    b.hist(z.clip(-12, 12), bins=np.linspace(-12, 12, 97), color="#455a64")
    b.set_yscale("log")
    for x in (-6, 6):
        b.axvline(x, color="#c62828", ls="--", lw=1)
    b.set_xlabel("robust z of hourly AC / irradiance (per month, normal operation)")
    b.set_title("Remaining outliers in the power-irradiance relation", fontsize=9)
    fig.suptitle("Q22 - Is anything left after cleaning? (all periods, descriptive)", fontsize=10)
    counts = table["check"].value_counts().to_dict()
    return Result("q22_residual", "Residual checks", table=table, figure=fig, facts={
        "csi_above_1_2_low_sun_share": float((h.loc[csi[day & (csi > 1.2).to_numpy()].index,
                                                    "elev"] < 15).mean()),
        "csi_above_1_2_by_utc_hour": table[table["check"].str.startswith("hourly")]
        ["timestamp_utc"].str[11:13].value_counts().sort_index().to_dict(),
        "robust_outliers_by_utc_hour": table[table["check"].str.startswith("AC / irradiance")]
        ["timestamp_utc"].str[11:13].value_counts().sort_index().to_dict(),
        "robust_outliers_by_month": table[table["check"].str.startswith("AC / irradiance")]
        ["timestamp_utc"].str[5:7].value_counts().sort_index().to_dict(),
        "csi_max": float(csi[day].max()), "csi_above_1_2": int(counts.get(
            "hourly AC / clear-sky AC > 1.2", 0)),
        "frozen_daytime_minutes_left": frozen_left,
        "robust_outliers": int(counts.get("AC / irradiance robust z > 6 (normal operation)", 0)),
        "robust_outliers_by_split": table[table["check"].str.startswith("AC / irradiance")]
        ["split"].value_counts().to_dict(),
        "imbalance_hours": int(counts.get(
            "inverter 1 / inverter 2 outside 0.8..1.25 (no flag)", 0)),
        "imbalance_hours_without_f3": int(_imbalance_without_f3(h, day)),
        "imbalance_by_split": table[table["check"].str.startswith("inverter")]["split"]
        .value_counts().to_dict(),
        "ac_max_after": {c: float(np.nanmax(d.s5[c].to_numpy())) for c in INV_AC}})

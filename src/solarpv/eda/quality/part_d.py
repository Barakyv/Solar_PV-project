"""Part D: the cleaned data for modelling (Q22-Q30). Train only where a modelling choice could
follow; Q29 and Q30 describe the whole period and are used for no decision."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from solarpv.config.constants import AC_CAPACITY_KW, TIMEZONE_LOCAL
from solarpv.config.paths import OUTPUTS_DIR
from solarpv.eda.quality.common import (
    SPLIT_COLOURS,
    Result,
    month_index,
    naive,
    sample,
    shade_splits,
)
from solarpv.eda.quality.data import INV_AC, INV_DC, SPLITS, Data

TARGET = "site_total_AC_kW"
EDA_MANIFEST = OUTPUTS_DIR / "03_eda" / "eda_manifest.json"
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA",
           8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def _train_day_hours(d: Data, normal: bool = True) -> pd.DataFrame:
    m = d.hour_daytime & (d.hour_split == "train")
    if normal:
        m &= ~d.h5["op_limited_flag"].to_numpy()
    return d.h5[m]


def _clear_sky_ac(h: pd.DataFrame) -> pd.Series:
    return h["poa_clearsky"] * AC_CAPACITY_KW / 1000


def q23_inverter_ceiling(d: Data) -> Result:
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    local = d.s5.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    for k, (acc, dcc) in enumerate(zip(INV_AC, INV_DC, strict=True)):
        ac, dc = d.s5[acc].to_numpy(dtype=float), d.s5[dcc].to_numpy(dtype=float)
        ok = d.train & d.daytime & np.isfinite(ac) & np.isfinite(dc)
        top = float(np.quantile(ac[ok], 0.9999))
        bins = np.arange(1000, 1750, 50)
        med = pd.Series(ac[ok]).groupby(pd.cut(dc[ok], bins)).median()
        cnt = pd.Series(ac[ok]).groupby(pd.cut(dc[ok], bins)).size()
        med = med[cnt >= 50]
        daily_max = pd.Series(np.where(ok, ac, np.nan)).groupby(local).max().dropna()
        rows.append({"inverter": acc.split("_")[1], "ac_p99_99": round(top, 1),
                     "ac_max": round(float(ac[ok].max()), 1),
                     "dc_max": round(float(dc[ok].max()), 1),
                     "minutes_ac_above_1500": int((ok & (ac > 1500)).sum()),
                     "days_max_above_1500": int((daily_max > 1500).sum()),
                     "median_ac_at_dc_1600_1650": (round(float(med[med.index.map(
                         lambda iv: iv.left == 1600)].iloc[0]), 1)
                         if any(iv.left == 1600 for iv in med.index) else np.nan),
                     "rating_assumed_kw": AC_CAPACITY_KW / 2})
        axes[0].plot([iv.mid for iv in med.index], med.to_numpy(), "o-",
                     label=acc.split("_")[1])
        axes[1].hist(daily_max, bins=np.arange(0, 1650, 25), histtype="step", lw=1.2,
                     label=acc.split("_")[1])
        if k == 0:
            idx = sample(ok & (dc > 900), 200_000)
            axes[2].hexbin(dc[idx], ac[idx], gridsize=60, bins="log", cmap="viridis")
            axes[2].set_xlabel(f"{dcc} [kW]")
            axes[2].set_ylabel(f"{acc} [kW]")
            axes[2].set_title("Top of the AC-DC relation (train)", fontsize=9)
    axes[0].plot([1000, 1700], [1000 * 0.973, 1700 * 0.973], "k:", lw=1, label="AC = 0.973 DC")
    axes[0].set_xlabel("DC [kW] (bins of 50)")
    axes[0].set_ylabel("median AC [kW]")
    axes[0].set_title("Does AC flatten while DC still rises?", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].axvline(AC_CAPACITY_KW / 2, color="#c62828", ls="--", lw=1,
                    label="3,449 / 2 kW (A-08)")
    axes[1].set_xlabel("daily maximum AC per inverter [kW] (train)")
    axes[1].set_ylabel("days")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Daily peaks never reach the assumed rating", fontsize=9)
    fig.suptitle("Q23 - The real ceiling of each inverter (A-08, capacity used in nRMSE)",
                 fontsize=10)
    table = pd.DataFrame(rows)
    h = d.h5[d.hour_split == "train"]
    return Result("q23_inverter_ceiling", "Inverter ceiling", table=table, figure=fig, facts={
        "ac_p99_99": dict(zip(table["inverter"], table["ac_p99_99"], strict=True)),
        "ac_max": dict(zip(table["inverter"], table["ac_max"], strict=True)),
        "site_hourly_max_train": float(h[TARGET].max()),
        "site_hourly_p99_9_train": float(h[TARGET].quantile(0.999)),
        "share_of_capacity": float(h[TARGET].max() / AC_CAPACITY_KW),
        "efficiency_at_dc_1600": float(table["median_ac_at_dc_1600_1650"].mean() / 1625)})


def q24_temperature_sensors(d: Data) -> Result:
    s5 = d.s5
    tr = d.train
    rows = []
    pairs = (("ambient_temp", "site_ambient_temp"), ("mod_temp1", "mod_temp2"))
    local_hour = d.s5.index.tz_convert(TIMEZONE_LOCAL).hour
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for a_col, b_col in pairs:
        a, b = s5[a_col].to_numpy(dtype=float), s5[b_col].to_numpy(dtype=float)
        m = tr & np.isfinite(a) & np.isfinite(b)
        diff = a[m] - b[m]
        by_hour = pd.Series(diff).groupby(local_hour[m]).median()
        rows.append({"pair": f"{a_col} - {b_col}", "minutes": int(m.sum()),
                     "median_diff": round(float(np.median(diff)), 3),
                     "p1_diff": round(float(np.quantile(diff, 0.01)), 3),
                     "p99_diff": round(float(np.quantile(diff, 0.99)), 3),
                     "pearson_r": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4),
                     "diurnal_range_of_median_diff": round(float(by_hour.max()
                                                                 - by_hour.min()), 3)})
        axes[0].plot(by_hour.index, by_hour.to_numpy(), "o-", ms=3, label=f"{a_col} - {b_col}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("local hour")
    axes[0].set_ylabel("median difference [degC] (train)")
    axes[0].set_title("Sensor pairs through the day", fontsize=9)
    axes[0].legend(fontsize=7)
    faults = {}
    for col in ("ambient_temp", "site_ambient_temp", "mod_temp1", "mod_temp2"):
        code = s5[f"rule_{col}"].to_numpy()
        faults[col] = {"R1": int((code == 1).sum()), "R3": int((code == 3).sum())}
    axes[1].bar(np.arange(4) - 0.2, [faults[c]["R1"] for c in faults], width=0.4, label="R1 range")
    axes[1].bar(np.arange(4) + 0.2, [faults[c]["R3"] for c in faults], width=0.4, label="R3 frozen")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(list(faults), fontsize=7, rotation=15)
    axes[1].set_ylabel("minutes removed (all periods)")
    axes[1].set_title("Faults per sensor", fontsize=9)
    axes[1].legend(fontsize=7)
    # physics: module heating above ambient grows with irradiance (train, sun > 20 deg)
    irr = s5["irradiance"].to_numpy(dtype=float)
    mod = s5["mod_temp1"].to_numpy(dtype=float)
    phys = {}
    for col, colour in (("ambient_temp", "#1565c0"), ("site_ambient_temp", "#c62828")):
        amb = s5[col].to_numpy(dtype=float)
        m = tr & (d.elev > 20) & np.isfinite(irr) & np.isfinite(mod) & np.isfinite(amb) & \
            (irr > 100)
        slope, intercept, r, _, _ = stats.linregress(irr[m], mod[m] - amb[m])
        phys[col] = {"slope_K_per_kWm2": round(slope * 1000, 3), "r2": round(r * r, 4)}
        g = pd.Series(mod[m] - amb[m]).groupby((irr[m] // 50) * 50 + 25).median()
        axes[2].plot(g.index, g.to_numpy(), "o-", ms=3, color=colour,
                     label=f"mod_temp1 - {col} (R2 {r * r:.3f})")
    axes[2].set_xlabel("in-plane irradiance [W/m2]")
    axes[2].set_ylabel("median module - ambient [K]")
    axes[2].set_title("Module heating with each ambient sensor (train)", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle("Q24 - Temperature sensors: which agree, which fail (D-09, temperature part)",
                 fontsize=10)
    table = pd.DataFrame(rows)
    return Result("q24_temperature_sensors", "Temperature sensors", table=table, figure=fig,
                  facts={"pairs": {r["pair"]: {k: v for k, v in r.items() if k != "pair"}
                                   for r in rows},
                         "faults": faults, "physics": phys})


def q25_correlations(d: Data) -> Result:
    h = _train_day_hours(d)
    cols = [TARGET, "s01_inv1_AC_kW", "s01_inv2_AC_kW", "irradiance", "pyranometer",
            "poa_clearsky", "elev", "mod_temp1", "mod_temp2", "ambient_temp",
            "site_ambient_temp", "wind_speed", "s01_inv1_DC_kW"]
    corr = h[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Q25 - Spearman correlation, hourly, train, daytime, normal operation\n"
                 "(same-hour values: only past-day lags may enter the models)", fontsize=9)
    table = corr.round(3).reset_index(names="variable")
    t = corr[TARGET]
    return Result("q25_correlations", "Correlations", table=table, figure=fig, facts={
        "target_vs": {c: round(float(t[c]), 3) for c in cols if c != TARGET},
        "hours": len(h)})


def q26_lags(d: Data) -> Result:
    h = d.h5.copy()
    normal = d.hour_daytime & ~h["op_limited_flag"].to_numpy()
    theo = _clear_sky_ac(h)
    series = {"site AC": h[TARGET], "clear-sky index (AC / clear-sky AC)":
              h[TARGET] / theo.where(theo > 50), "irradiance": h["irradiance"],
              "mod_temp1": h["mod_temp1"]}
    train = d.hour_split == "train"
    rows = []
    for name, s in series.items():
        s = s.where(normal)
        for k in range(1, 15):
            lagged = s.shift(24 * k)
            m = train & s.notna().to_numpy() & lagged.notna().to_numpy()
            rows.append({"series": name, "lag_days": k,
                         "pearson_r": round(float(np.corrcoef(s[m], lagged[m])[0, 1]), 4)})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, g in table.groupby("series", sort=False):
        ax.plot(g["lag_days"], g["pearson_r"], "o-", ms=4, label=name)
    for k in (1, 2, 3, 7, 14):
        ax.axvline(k, color="0.85", lw=6, zorder=0)
    ax.set_xlabel("lag [days], same hour of the day")
    ax.set_ylabel("correlation (train, daytime, normal operation)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Q26 - What the same hour on past days tells (grey: lags in the feature "
                 "contract)", fontsize=9)
    t = table.set_index(["series", "lag_days"])["pearson_r"]
    ci = "clear-sky index (AC / clear-sky AC)"
    return Result("q26_lags", "Lag correlations", table=table, figure=fig, facts={
        "target": {k: float(t[("site AC", k)]) for k in (1, 2, 3, 7, 14)},
        "clear_sky_index": {k: float(t[(ci, k)]) for k in (1, 2, 3, 7, 14)},
        "mod_temp1": {k: float(t[("mod_temp1", k)]) for k in (1, 7, 14)}})


def q27_sky_conditions(d: Data) -> Result:
    h = _train_day_hours(d)
    h = h[h["poa_clearsky"] > 100]
    kt = (h["irradiance"] / h["poa_clearsky"]).dropna()
    season = pd.Series(naive(kt.index).month, index=kt.index).map(SEASONS)
    cls = pd.cut(kt, [-np.inf, 0.3, 0.8, np.inf], labels=["overcast", "partly cloudy", "clear"])
    table = pd.crosstab(season, cls, normalize="index").mul(100).round(1)
    table["hours"] = season.value_counts()
    table["median_kt"] = kt.groupby(season).median().round(3)
    table = table.loc[["DJF", "MAM", "JJA", "SON"]].reset_index(names="season")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    for s in ("DJF", "MAM", "JJA", "SON"):
        a.hist(kt[season == s], bins=np.linspace(0, 1.5, 61), histtype="step", lw=1.2,
               density=True, label=s)
    a.axvline(0.3, color="0.5", ls=":", lw=1)
    a.axvline(0.8, color="0.5", ls=":", lw=1)
    a.set_xlabel("hourly irradiance / clear-sky POA (kt)")
    a.set_ylabel("density")
    a.legend(fontsize=7)
    a.set_title("Hourly sky condition (train, daytime)", fontsize=9)
    t = table.set_index("season")
    t[["clear", "partly cloudy", "overcast"]].plot.bar(stacked=True, ax=b, rot=0,
                                                       color=["#f9a825", "#90a4ae", "#455a64"])
    b.set_ylabel("% of daytime hours")
    b.legend(fontsize=7)
    b.set_title("Clear, partly cloudy (0.3-0.8) and overcast (< 0.3) hours", fontsize=9)
    fig.suptitle("Q27 - How cloudy is it? Most of the forecasting difficulty is in winter and "
                 "spring", fontsize=10)
    return Result("q27_sky_conditions", "Sky conditions", table=table, figure=fig, facts={
        "clear_pct": t["clear"].to_dict(), "overcast_pct": t["overcast"].to_dict(),
        "median_kt": t["median_kt"].to_dict(),
        "kt_p95": float(kt.quantile(0.95))})


def _daily_index(d: Data, split: str | None = "train") -> pd.Series:
    """Daily AC energy over clear-sky AC energy, complete days in normal operation."""
    h = d.h5 if split is None else d.h5[d.hour_split == split]
    day = h.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    g = h.groupby(day)
    ok = (g[TARGET].count() >= 23) & ~g["op_limited_flag"].any()
    return (g[TARGET].sum() / (g["poa_clearsky"].sum() * AC_CAPACITY_KW / 1000))[ok]


def q28_weekday(d: Data) -> Result:
    idx = _daily_index(d)
    dow = pd.Series(idx.index.dayofweek, index=idx.index)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    groups = [idx[dow == k].to_numpy() for k in range(7)]
    h_stat, p = stats.kruskal(*groups)
    table = pd.DataFrame({"day": names, "days": [len(g) for g in groups],
                          "median_daily_index": [round(float(np.median(g)), 4) for g in groups],
                          "mean_daily_index": [round(float(np.mean(g)), 4) for g in groups]})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(groups, tick_labels=names, showfliers=False)
    ax.set_ylabel("daily AC / clear-sky AC")
    ax.set_title(f"Q28 - No weekday effect (train, Kruskal-Wallis p = {p:.2f})", fontsize=10)
    return Result("q28_weekday", "Weekday effect", table=table, figure=fig, facts={
        "kruskal_p": float(p), "kruskal_h": float(h_stat),
        "median_range": float(table["median_daily_index"].max()
                              - table["median_daily_index"].min()),
        "days": int(len(idx))})


def _slope(months: pd.PeriodIndex, values: np.ndarray) -> tuple[float, float]:
    """Linear trend in % of the mean per year, and its p-value."""
    years = np.array([(p.to_timestamp() - pd.Timestamp("2021-09-01")).days / 365.25
                      for p in months])
    fit = stats.linregress(years, values)
    return float(100 * fit.slope / np.mean(values)), float(fit.pvalue)


def q29_long_term(d: Data) -> Result:
    gamma = json.loads(EDA_MANIFEST.read_text(encoding="utf-8"))[
        "temperature_coefficient_pct_per_K"] / 100
    h = d.h5
    normal = d.hour_daytime & ~h["op_limited_flag"].to_numpy() & h[TARGET].notna().to_numpy()
    x = h[normal & h["irradiance"].between(700, 900).to_numpy() & h["mod_temp1"].notna()
          .to_numpy()]
    pi = x[TARGET] / x["irradiance"] / (1 + gamma * (x["mod_temp1"] - 25))
    month = month_index(pi.index)
    monthly = pi.groupby(month).median()
    count = pi.groupby(month).size()
    monthly = monthly[count >= 20]
    # independent of the irradiance sensor: clearest hours against the clear-sky model
    c = h[normal & (h["poa_clearsky"] > 600).to_numpy()]
    cm = month_index(c.index)
    plant = (c[TARGET] / (c["poa_clearsky"] * AC_CAPACITY_KW / 1000)).groupby(cm).quantile(0.9)
    sensor = (c["irradiance"] / c["poa_clearsky"]).groupby(cm).quantile(0.9)
    split_of_month = pd.Series(d.hour_split, index=month_index(h.index)).groupby(level=0).first()
    split = split_of_month.reindex(monthly.index)
    tr = (split == "train").to_numpy()
    slope_pi, p_pi = _slope(monthly.index[tr], monthly.to_numpy()[tr])
    trm = (split_of_month.reindex(plant.index) == "train").to_numpy()
    slope_plant, p_plant = _slope(plant.index[trm], plant.to_numpy()[trm])
    slope_sensor, p_sensor = _slope(sensor.index[trm], sensor.to_numpy()[trm])
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4.2))
    dates = monthly.index.to_timestamp() + pd.Timedelta(days=14)
    for s, colour in SPLIT_COLOURS.items():
        sel = (split == s).to_numpy()
        a.plot(dates[sel], monthly[sel], "o", color=colour, label=s)
    years = np.array([(p.to_timestamp() - pd.Timestamp("2021-09-01")).days / 365.25
                      for p in monthly.index[tr]])
    fit = stats.linregress(years, monthly.to_numpy()[tr])
    a.plot(dates[tr], fit.intercept + fit.slope * years, "k--", lw=1,
           label=f"train trend {slope_pi:+.2f} % per year")
    a.set_ylabel("AC / irradiance at 25 degC [kW per W/m2]")
    a.legend(fontsize=7)
    a.set_title(f"Performance at 700-900 W/m2, temperature-corrected ({100 * gamma:.2f} %/K)",
                fontsize=9)
    t2 = plant.index.to_timestamp() + pd.Timedelta(days=14)
    b.plot(t2, plant / plant[trm].mean(), "o-", ms=3, color="#c62828",
           label=f"site AC / clear-sky AC, p90 (train {slope_plant:+.2f} %/yr, p={p_plant:.2f})")
    b.plot(t2, sensor / sensor[trm].mean(), "o-", ms=3, color="#1565c0",
           label=f"irradiance / clear-sky POA, p90 (train {slope_sensor:+.2f} %/yr, "
                 f"p={p_sensor:.2f})")
    b.set_ylabel("relative to the train mean")
    b.legend(fontsize=7)
    b.set_title("Clearest hours vs the clear-sky model: the sensor is stable, the plant follows "
                "the seasons",
                fontsize=9)
    for ax in (a, b):
        shade_splits(ax)
    fig.suptitle("Q29 - Long-term performance (trend estimated on train; val and test shown for "
                 "description only)", fontsize=10)
    table = pd.DataFrame({"month": monthly.index.astype(str), "split": split.to_numpy(),
                          "hours": count.reindex(monthly.index).to_numpy(),
                          "performance_index": monthly.round(4).to_numpy(),
                          "plant_vs_clear_sky_p90": plant.reindex(monthly.index).round(4)
                          .to_numpy(),
                          "sensor_vs_clear_sky_p90": sensor.reindex(monthly.index).round(4)
                          .to_numpy()})
    seasonal = monthly[tr].groupby([p.month for p in monthly.index[tr]]).mean()
    return Result("q29_long_term", "Long-term performance", table=table, figure=fig, facts={
        "slope_pct_per_year_train": slope_pi, "p_value_train": p_pi,
        "plant_vs_clear_sky_slope_train": slope_plant, "plant_vs_clear_sky_p": p_plant,
        "sensor_vs_clear_sky_slope_train": slope_sensor, "sensor_vs_clear_sky_p": p_sensor,
        "gamma_pct_per_K": float(100 * gamma),
        "seasonal_range_pct": float(100 * (seasonal.max() - seasonal.min()) / seasonal.mean()),
        "same_month_plant": {str(m): {str(p): round(float(plant[p]), 3) for p in plant.index
                                      if p.month == m} for m in (10, 11, 12, 1, 2)}})


def q30_unused_tags(d: Data) -> Result:
    s4 = d.s4
    month = month_index(s4.index)
    rows = []
    prec = s4["precipitation"]
    rows.append({"tag": "precipitation", "evidence": f"max {prec.max():.2f}, minutes > 0: "
                 f"{int((prec > 0).sum())}, minutes < 0: {int((prec < 0).sum())}",
                 "conclusion": "never records rain in 45 months (the site has rain events): "
                               "not usable"})
    last = {}
    for col in ("soil", "soiling01", "soiling02"):
        med = s4[col].groupby(month).median()
        active = med[med.abs() > 1]
        last[col] = str(active.index.max()) if len(active) else "never"
        rows.append({"tag": col, "evidence": f"monthly median > 1 until {last[col]}; then "
                     f"{med[med.index > active.index.max()].abs().max():.2f} or less"
                     if len(active) else "flat",
                     "conclusion": "sensor stops reporting during the data; undocumented units: "
                                   "not usable"})
    held = {c: float((d.frozen(c) & d.daytime).sum() / d.daytime.sum() * 100)
            for c in ("wind_speed", "wind_direction")}
    r1 = int((d.s5["rule_wind_speed"] == 1).sum())
    rows.append({"tag": "wind_speed / wind_direction",
                 "evidence": f"held >= 30 min: {held['wind_speed']:.1f} % / "
                             f"{held['wind_direction']:.1f} % of daytime minutes; "
                             f"{r1} minutes outside 0..60 m/s",
                 "conclusion": "measured at the target time (EXCL), no lag in the contract; "
                               "frequent stuck periods"})
    pf = s4["s01_inv1_PF"]
    pf_one = float((pf.round(3) == 1).mean()) * 100
    rows.append({"tag": "kVar, PF", "evidence": f"PF = 1.000 in {pf_one:.1f} "
                 "% of minutes; kVar follows the grid operator's reactive-power setpoint",
                 "conclusion": "grid control, not weather: no information on active power"})
    st = s4["s01_inv1_state"].round().value_counts(normalize=True).head(4)
    rows.append({"tag": "inverter states", "evidence": "; ".join(f"code {int(k)}: {v * 100:.1f} %"
                                                                 for k, v in st.items()),
                 "conclusion": "status at the target time (EXCL); used only by flag F3 "
                               "(codes 5/6 = derated, Q16)"})
    for col in ("iec_sp_q", "iec_sp_v", "iec_sp_pf"):
        v = s4[col]
        rows.append({"tag": col, "evidence": f"p1 {v.quantile(.01):.1f}, median {v.median():.1f}, "
                     f"p99 {v.quantile(.99):.1f}",
                     "conclusion": "grid setpoint for reactive power / voltage: not related to "
                                   "the forecast"})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for col in ("soil", "soiling01", "soiling02"):
        med = s4[col].groupby(month).median()
        axes[0].plot(med.index.to_timestamp(), med.to_numpy(), "o-", ms=3, label=col)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_title("Soiling sensors, monthly median", fontsize=9)
    axes[0].legend(fontsize=7)
    hw = pd.Series((d.frozen("wind_speed") & d.daytime)[d.daytime]).groupby(
        month[d.daytime]).mean() * 100
    axes[1].bar(hw.index.to_timestamp(), hw.to_numpy(), width=25, color="#455a64")
    axes[1].set_ylabel("% of daytime minutes held >= 30 min")
    axes[1].set_title("wind_speed stuck periods", fontsize=9)
    axes[2].bar([str(int(k)) for k in st.index], st.to_numpy() * 100, color="#455a64")
    axes[2].set_xlabel("s01_inv1_state code")
    axes[2].set_ylabel("% of minutes")
    axes[2].set_title("Inverter state codes", fontsize=9)
    fig.suptitle("Q30 - Why the remaining tags are not used", fontsize=10)
    return Result("q30_unused_tags", "Unused tags", table=table, figure=fig, facts={
        "precipitation_max": float(prec.max()), "precipitation_positive_minutes":
        int((prec > 0).sum()), "soiling_last_active_month": last, "wind_held_pct": held,
        "pf_one_pct": float((pf.round(3) == 1).mean() * 100)})


def q31_splits(d: Data) -> Result:
    h = d.h5[d.hour_daytime]
    split = d.hour_split[d.hour_daytime]
    months = pd.Series(naive(h.index).month, index=h.index)
    season = months.map(SEASONS)
    theo = _clear_sky_ac(h)
    kt = h["irradiance"] / h["poa_clearsky"].where(h["poa_clearsky"] > 100)
    rows = []
    for s in SPLITS:
        m = split == s
        rows.append({"split": s, "daytime_hours": int(m.sum()),
                     "calendar_months": ",".join(str(x) for x in sorted(months[m].unique())),
                     **{f"hours_{x}": int((season[m] == x).sum()) for x in ("DJF", "MAM", "JJA",
                                                                              "SON")},
                     "mean_ac_kw": round(float(h.loc[m, TARGET].mean()), 1),
                     "median_kt": round(float(kt[m].median()), 3),
                     "clear_hour_pct": round(float((kt[m] >= 0.8).mean() * 100), 1),
                     "mean_ac_over_clear_sky": round(float((h.loc[m, TARGET] / theo[m])
                                                           .mean()), 3)})
    table = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4))
    width = 0.27
    for k, s in enumerate(SPLITS):
        counts = months[split == s].value_counts().reindex(range(1, 13), fill_value=0)
        a.bar(np.arange(1, 13) + (k - 1) * width, counts, width=width, color=SPLIT_COLOURS[s],
              label=s)
    a.set_xticks(range(1, 13))
    a.set_xlabel("calendar month")
    a.set_ylabel("daytime hours")
    a.legend(fontsize=7)
    a.set_title("Which months each split contains", fontsize=9)
    data = [h.loc[(split == s) & (season == x).to_numpy(), TARGET].dropna().to_numpy()
            for x in ("DJF", "MAM", "SON") for s in SPLITS]
    labels = [f"{x}\n{s}" for x in ("DJF", "MAM", "SON") for s in SPLITS]
    b.boxplot(data, tick_labels=labels, showfliers=False)
    b.set_ylabel("daytime hourly site AC [kW]")
    b.set_title("Same season, different splits (JJA has no test hours)", fontsize=9)
    b.tick_params(axis="x", labelsize=7)
    fig.suptitle("Q31 - How the splits differ (descriptive only, used for no decision)",
                 fontsize=10)
    t = table.set_index("split")
    return Result("q31_splits", "Split comparison", table=table, figure=fig, facts={
        "test_calendar_months": t.loc["test", "calendar_months"],
        "test_hours_JJA": int(t.loc["test", "hours_JJA"]),
        "mean_ac_kw": t["mean_ac_kw"].to_dict(), "clear_hour_pct": t["clear_hour_pct"].to_dict()})

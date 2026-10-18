"""Stage 3 figures and tables (docs/stage3_eda_plan.md).

    python -m solarpv.eda

EDA changes no data. Anything that could inform a modelling choice is computed on the train
split only and labelled so; whole-period overviews mark val and test.
Outputs: outputs/03_eda/{figures/*.png, tables/*.csv, eda_manifest.json}.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pvlib  # noqa: E402

from solarpv.config.constants import (  # noqa: E402
    AC_CAPACITY_KW,
    TEST_START,
    TIMEZONE_LOCAL,
    TRAIN_END,
    VAL_START,
)
from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from solarpv.config.provenance import git_commit, sha256, text_sha256  # noqa: E402
from solarpv.config.site import Site, load_site  # noqa: E402
from solarpv.data.clean import split_of  # noqa: E402
from solarpv.data.clock import Orientation, clear_sky_components, poa_from_components  # noqa: E402
from solarpv.data.ingest import TAG_MAP, TIME_COLUMN  # noqa: E402
from solarpv.data.preprocess import AC_COLUMNS, TARGET  # noqa: E402
from solarpv.data.qa import clear_days  # noqa: E402
from solarpv.data.resolution import dehold  # noqa: E402
from solarpv.evaluation.daytime import daytime_mask  # noqa: E402

OUT = OUTPUTS_DIR / "03_eda"
FIG, TAB = OUT / "figures", OUT / "tables"
STAGE4 = PROCESSED_DIR / "04_preprocess"
STAGE5 = PROCESSED_DIR / "05_clean"
INGEST = PROCESSED_DIR / "01_ingest" / "scada_1min_utc.parquet"
CLOCK_DIR = AUDIT_DIR / "clock_drift_2026-10-14"
POLLED_SIM = AUDIT_DIR / "polled_reconstruction_2026-10-15" / "summary.csv"
CLEANING_LOG = OUTPUTS_DIR / "05_cleaning" / "cleaning_log.csv"
TIME_BUG_MONTHS = {"winter (Dec 2021, UTC+2)": "2112.CSV", "summer (Jul 2022, UTC+3)": "2207.CSV"}
POLLED_EXAMPLE = ("2025-04-10 07:50", "2025-04-10 09:10")  # UTC; 10:50-12:10 local
FREEZE_EXAMPLE = ("2021-09-10 18:00", "2021-09-12 21:00")  # UTC; a communication freeze
CLOCK_STEPS = ("2023-05-09", "2024-01-21")
SPLIT_COLOURS = {"train": "#2e7d32", "val": "#ef6c00", "test": "#1a237e"}
DPI = 150


# ── helpers ────────────────────────────────────────────────────────────────


def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _table(df: pd.DataFrame, name: str) -> str:
    TAB.mkdir(parents=True, exist_ok=True)
    path = TAB / f"{name}.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    return path.name


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("UTC").tz_localize(None)


def _shade_splits(ax, alpha=0.08):
    starts = {"train": None, "val": VAL_START, "test": TEST_START}
    ends = {"train": TRAIN_END, "val": TEST_START, "test": None}
    lo, hi = ax.get_xlim()
    for s, colour in SPLIT_COLOURS.items():
        a = matplotlib.dates.date2num(pd.Timestamp(starts[s])) if starts[s] else lo
        b = matplotlib.dates.date2num(pd.Timestamp(ends[s])) if ends[s] else hi
        ax.axvspan(a, b, color=colour, alpha=alpha, lw=0, label=f"{s}")
    ax.set_xlim(lo, hi)


def hourly_clear_sky(minute_index: pd.DatetimeIndex, site: Site) -> pd.DataFrame:
    """Clear-sky POA (measured orientation) and sun elevation, as hourly means of minutes."""
    comp = clear_sky_components(minute_index, site)
    poa = poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth))
    frame = pd.DataFrame({"poa_clearsky": poa, "elev": comp["elev"].to_numpy()},
                         index=minute_index)
    return frame.resample("1h").mean()


def daily_centroid_offsets(power: pd.Series, days, site: Site) -> pd.Series:
    """Minutes between the power-weighted centre of each day's production and solar noon.
    Days are grouped by UTC date (production never crosses UTC midnight at this site)."""
    p = power.clip(lower=0).dropna()
    date = pd.DatetimeIndex(p.index.tz_convert("UTC").date)
    t = (p.index - p.index.normalize()).total_seconds().to_numpy() / 60 + 0.5
    frame = pd.DataFrame({"date": date, "t": t, "w": p.to_numpy()})
    frame = frame[frame["date"].isin(pd.DatetimeIndex(days))]
    g = frame.assign(tw=frame["t"] * frame["w"]).groupby("date")
    centroid = g["tw"].sum() / g["w"].sum()
    midnights = pd.DatetimeIndex(centroid.index).tz_localize("UTC")
    transit = pvlib.solarposition.sun_rise_set_transit_spa(midnights, site.latitude,
                                                           site.longitude)["transit"]
    noon = (transit - midnights).dt.total_seconds().to_numpy() / 60
    return pd.Series(centroid.to_numpy() - noon, index=centroid.index)


# ── T1 split table ─────────────────────────────────────────────────────────


def table_splits(hourly: pd.DataFrame, site: Site) -> pd.DataFrame:
    day = daytime_mask(hourly.index, site)
    split = split_of(hourly.index)
    valid = hourly[TARGET].notna().to_numpy()
    op = hourly["op_limited_flag"].to_numpy()
    src = hourly["hour_source"].to_numpy()
    rows = []
    for s in ("train", "val", "test", "all"):
        m = np.ones(len(hourly), bool) if s == "all" else split == s
        idx = hourly.index[m]
        rows.append({
            "split": s, "start_utc": str(idx[0]), "end_utc": str(idx[-1] + pd.Timedelta(hours=1)),
            "hours": int(m.sum()), "daytime_hours": int((m & day).sum()),
            "daytime_valid_target": int((m & day & valid).sum()),
            "daytime_op_limited": int((m & day & op).sum()),
            "daytime_valid_normal_operation": int((m & day & valid & ~op).sum()),
            "daytime_polled": int((m & day & (src == "polled")).sum()),
            "daytime_mixed": int((m & day & (src == "mixed")).sum()),
        })
    return pd.DataFrame(rows)


# ── T2 / F2 time-zone bug (F-01) ───────────────────────────────────────────


def _raw_power_as_utc(path) -> pd.Series:
    """The original reading: raw local stamps labelled UTC, no conversion (F-01)."""
    raw_of = {v: k for k, v in TAG_MAP.items()}
    cols = [raw_of[c] for c in AC_COLUMNS]
    df = pd.read_csv(path, usecols=[TIME_COLUMN, *cols], dtype=str)
    stamps = pd.to_datetime(df[TIME_COLUMN], format="%Y-%m-%d %H:%M:%S.%f")
    power = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=2)
    s = pd.Series(power.to_numpy(), index=pd.DatetimeIndex(stamps).tz_localize("UTC"))
    return s[~s.index.duplicated()]


def time_bug(stage4: pd.DataFrame, site: Site) -> tuple[pd.DataFrame, str]:
    rows, fig, axes = [], *plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (label, fname) in zip(axes, TIME_BUG_MONTHS.items(), strict=True):
        original = _raw_power_as_utc(RAW_DIR / fname)
        month = original.index[len(original) // 2].strftime("%Y-%m")
        rebuild = stage4.loc[month, TARGET]
        comp = clear_sky_components(rebuild.index, site)
        elev = pd.Series(comp["elev"].to_numpy(), index=rebuild.index)
        ten = rebuild.resample("10min").mean()
        days = [pd.Timestamp(d) for d in clear_days(ten, elev.resample("10min").mean(),
                                                    per_month=6)]
        poa = pd.Series(poa_from_components(comp, Orientation(site.surface_tilt,
                                                              site.surface_azimuth)),
                        index=rebuild.index)
        for reading, series in (("original (local stamps read as UTC)", original),
                                ("rebuild (true UTC, clock-corrected)", rebuild)):
            off = daily_centroid_offsets(series, days, site)
            rows.append({"month": label, "reading": reading, "clear_days": len(off),
                         "median_centroid_minus_solar_noon_min": round(float(off.median()), 1)})
            sel = series[pd.DatetimeIndex(series.index.date).isin(pd.DatetimeIndex(days))]
            prof = sel.groupby(sel.index.hour + sel.index.minute / 60).mean()
            ax.plot(prof.index, prof / prof.max(), label=reading.split(" (")[0])
        sel = poa[pd.DatetimeIndex(poa.index.date).isin(pd.DatetimeIndex(days))]
        prof = sel.groupby(sel.index.hour + sel.index.minute / 60).mean()
        ax.plot(prof.index, prof / prof.max(), "k--", lw=1, label="clear-sky POA (true UTC)")
        ax.set_title(label)
        ax.set_xlabel("hour of day, UTC")
        ax.set_xlim(0, 24)
    axes[0].set_ylabel("normalised mean on clear days")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("F-01: the original read Israel local time as UTC — production shifted "
                 "2 h (winter) / 3 h (summer) against solar geometry")
    return pd.DataFrame(rows), _save(fig, "fig02_time_bug")


# ── F1 overview ────────────────────────────────────────────────────────────


def overview(hourly: pd.DataFrame, polled_first_day: str) -> str:
    local_day = hourly.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    energy = hourly[TARGET].groupby(local_day).sum(min_count=18) / 1000
    peak = hourly[TARGET].groupby(local_day).max().rolling(30, min_periods=10).max()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(energy.index, energy, lw=0.6, color="0.3", label="daily energy [MWh]")
    ax2 = ax.twinx()
    ax2.plot(peak.index, peak, color="#c62828", lw=1, label="30-day max hourly power [kW]")
    ax2.axhline(AC_CAPACITY_KW, color="#c62828", ls=":", lw=1)
    ax2.set_ylim(0, AC_CAPACITY_KW * 1.15)
    ax2.set_ylabel("kW")
    ax.set_xlim(energy.index[0], energy.index[-1])
    _shade_splits(ax)
    ax.axvspan(pd.Timestamp(polled_first_day), energy.index[-1], hatch="///", fill=False,
               ec="0.5", lw=0, label="15-min polled source (F-18)")
    for step in CLOCK_STEPS:
        ax.axvline(pd.Timestamp(step), color="0.4", ls="--", lw=0.8)
        ax.text(pd.Timestamp(step), ax.get_ylim()[1] * 0.98, f" clock step {step}", fontsize=7,
                va="top", rotation=90)
    ax.set_ylabel("MWh per day")
    ax.set_title("Two inverters of the Ketura Solar park (3,449 kW AC), Sep 2021 – May 2025, "
                 "after stages 2–5")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7, ncol=3)
    return _save(fig, "fig01_overview")


# ── F3 clocks ──────────────────────────────────────────────────────────────


def clocks() -> str:
    off = pd.read_csv(CONFIGS_DIR / "clock_offsets.csv", parse_dates=["day"]).set_index("day")
    mon = pd.read_csv(CLOCK_DIR / "monthly_minute_before_after.csv")
    mon["t"] = pd.to_datetime(mon["day"]) + pd.Timedelta(days=14)
    pol = pd.read_csv(CLOCK_DIR / "monthly_polled_before_after.csv")
    pol["t"] = pd.to_datetime(pol["day"]) + pd.Timedelta(days=14)
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    a.plot(off.index, off["minute_meteo"], color="#1565c0", label="meteo logger (applied)")
    a.plot(off.index, off["minute_inverter"], color="#c62828", label="inverter logger (applied)")
    a.plot(off.index, off["polled_meteo"], color="#6a1b9a", label="polled source (applied)")
    a.scatter(mon["t"], mon["tau_met"], s=10, color="#1565c0", label="monthly estimate, meteo")
    a.scatter(mon["t"], mon["tau_met"] + mon["delta"], s=10, color="#c62828",
              label="monthly estimate, inverter")
    a.scatter(pol["t"], pol["tau_met"], s=10, color="#6a1b9a")
    for step in CLOCK_STEPS:
        a.axvline(pd.Timestamp(step), color="0.4", ls="--", lw=0.8)
    a.set_ylabel("stamped − true time [min]")
    a.legend(fontsize=7, ncol=3, loc="lower left")
    a.set_title("F-15: logger clock offsets (D-19); dashed: resync steps")
    b.axhspan(-2, 2, color="0.9")
    b.plot(mon["t"], mon["tau_met_after"], "o-", ms=3, color="#1565c0", label="meteo")
    b.plot(mon["t"], mon["delta_after"], "o-", ms=3, color="#c62828", label="inverter − meteo")
    b.plot(pol["t"], pol["tau_met_after"], "o-", ms=3, color="#6a1b9a", label="polled")
    b.set_ylabel("residual after\ncorrection [min]")
    b.set_ylim(-5, 5)
    b.legend(fontsize=7, ncol=3)
    return _save(fig, "fig03_clock")


# ── F4 polled source ───────────────────────────────────────────────────────


def polled(ingest: pd.DataFrame) -> str:
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in POLLED_EXAMPLE)
    day = ingest.loc[lo.normalize():lo.normalize() + pd.Timedelta(days=1)]
    sim = pd.read_csv(POLLED_SIM)
    fig = plt.figure(figsize=(13, 5))
    grid = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    tags = [("irradiance", "W/m²"), ("s01_inv1_AC_kW", "kW"), ("s01_inv1_DC_kW", "kW")]
    for i, (tag, unit) in enumerate(tags):
        ax = fig.add_subplot(grid[i, 0])
        held = day[tag].loc[lo:hi]
        rebuilt = dehold(day[tag]).loc[lo:hi]
        ax.step(_naive(held.index), held, where="post", color="0.5", lw=1, label="as exported")
        ax.plot(_naive(rebuilt.index), rebuilt, color="#c62828", lw=1.2,
                label="rebuilt at poll instants")
        ax.set_ylabel(f"{tag}\n[{unit}]", fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if i == 0:
            ax.legend(fontsize=7)
            ax.set_title("2025-04-10 (UTC): one value per tag every 15 min, each tag on its own "
                         "phase", fontsize=9)
    ax = fig.add_subplot(grid[:, 1])
    pivot = sim.pivot(index="method", columns="days", values="rmse_pct_capacity")
    pivot[["clear or steady", "variable", "all"]].plot.bar(ax=ax, rot=20)
    ax.set_ylabel("hourly-mean RMSE [% of capacity]")
    ax.set_title("Simulation on train (Mar–May 2022/23)", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("F-18: the 2025 export holds 15-min snapshots; rebuilding them linearly at the "
                 "poll instants (M1) cuts the hourly error from 2.6 % to 1.6 % of capacity")
    return _save(fig, "fig04_polled")


# ── F5 cleaning ────────────────────────────────────────────────────────────


def cleaning(stage4_min: pd.DataFrame, stage5_min: pd.DataFrame, stage4_h: pd.DataFrame,
             hourly: pd.DataFrame, site: Site) -> str:
    log = pd.read_csv(CLEANING_LOG)
    per_rule = log.groupby("rule")["minutes"].sum()
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in FREEZE_EXAMPLE)
    col = "s01_inv1_AC_kW"
    before = stage4_min.loc[lo:hi, col]
    after = stage5_min.loc[lo:hi, col]
    elev = clear_sky_components(before.index, site)["elev"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2, 1]})
    per_rule.plot.barh(ax=axes[0], color="#455a64")
    for i, v in enumerate(per_rule):
        axes[0].text(v, i, f" {v:,}", va="center", fontsize=7)
    axes[0].set_xlim(0, per_rule.max() * 1.35)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}k"))
    axes[0].set_xlabel("minutes changed (all columns)")
    axes[0].set_title("Value rules (stage 5)", fontsize=9)
    t = _naive(before.index)
    axes[1].fill_between(t, 0, 1, where=elev < 0, transform=axes[1].get_xaxis_transform(),
                         color="0.85", lw=0, label="sun below horizon")
    axes[1].plot(t, before, color="0.5", lw=1, label="stage 4")
    axes[1].plot(t, after, color="#c62828", lw=1.2, label="stage 5")
    axes[1].set_ylabel(f"{col} [kW]")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    axes[1].set_xlabel("UTC")
    axes[1].set_title("Communication freeze 2021-09-11/12: last value repeated for hours "
                      "(R3 with margin → NaN)", fontsize=9)
    axes[1].legend(fontsize=7)
    day4 = stage4_h[TARGET].groupby(_naive(stage4_h.index).normalize()).sum() / 1000
    day5 = hourly[TARGET].groupby(_naive(hourly.index).normalize()).sum() / 1000
    axes[2].scatter(day4, day5.reindex(day4.index), s=4, alpha=0.5)
    m = float(max(day4.max(), day5.max()))
    axes[2].plot([0, m], [0, m], "k:", lw=1)
    axes[2].set_xlabel("daily energy, stage 4 [MWh]")
    axes[2].set_ylabel("daily energy, stage 5 [MWh]")
    axes[2].set_title("Days off the line: spikes or freezes removed", fontsize=9)
    return _save(fig, "fig05_cleaning")


# ── train-only figures ─────────────────────────────────────────────────────


def _train(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly[split_of(hourly.index) == "train"]


def diurnal(hourly: pd.DataFrame) -> str:
    tr = _train(hourly)
    local = tr.index.tz_convert(TIMEZONE_LOCAL)
    pivot = tr[TARGET].groupby([local.hour, local.month]).mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="magma",
                   extent=[0.5, 12.5, -0.5, 23.5])
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month")
    ax.set_ylabel("local hour")
    fig.colorbar(im, ax=ax, label="mean site AC [kW]")
    ax.set_title("Mean production by local hour and month (train, after cleaning)")
    return _save(fig, "fig06_diurnal")


def power_irradiance(hourly: pd.DataFrame, site: Site) -> tuple[pd.DataFrame, str]:
    tr = _train(hourly)
    day = daytime_mask(tr.index, site)
    ok = day & tr[TARGET].notna() & tr["irradiance"].gt(50) & ~tr["op_limited_flag"] \
        & tr["mod_temp1"].notna()
    d = tr[ok]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    sc = a.scatter(d["irradiance"], d[TARGET], c=d["mod_temp1"], s=2, alpha=0.4, cmap="coolwarm")
    fig.colorbar(sc, ax=a, label="module temperature [°C]")
    a.set_xlabel("in-plane irradiance [W/m²]")
    a.set_ylabel("site AC [kW]")
    a.set_title("Hourly AC vs in-plane irradiance (train, daytime, normal operation)", fontsize=9)
    band = d[d["irradiance"].between(700, 900)]
    ratio = band[TARGET] / band["irradiance"]
    bins = (band["mod_temp1"] // 2) * 2 + 1
    med = ratio.groupby(bins).median()
    cnt = ratio.groupby(bins).size()
    med = med[cnt >= 30]
    slope, intercept = np.polyfit(med.index.to_numpy(), med.to_numpy(), 1)
    at25 = slope * 25 + intercept
    b.plot(med.index, med, "o", ms=4)
    xs = np.linspace(med.index.min(), med.index.max(), 10)
    b.plot(xs, slope * xs + intercept, "k--", lw=1,
           label=f"{100 * slope / at25:+.2f} % per K (relative to 25 °C)")
    b.set_xlabel("module temperature [°C]")
    b.set_ylabel("AC / irradiance [kW per W/m²]")
    b.set_title("Temperature effect, irradiance 700–900 W/m² (train)", fontsize=9)
    b.legend(fontsize=8)
    table = pd.DataFrame([{"quantity": "relative temperature coefficient of AC [%/K]",
                           "value": round(100 * slope / at25, 3), "basis": "train, 700-900 W/m2"},
                          {"quantity": "AC per irradiance at 25 degC [kW per W/m2]",
                           "value": round(at25, 4), "basis": "train, 700-900 W/m2"}])
    return table, _save(fig, "fig07_power_irradiance")


def variability(hourly: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    tr = _train(hourly)
    local_day = tr.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    g = tr.groupby(local_day)
    complete = g[TARGET].count() >= 23
    normal = ~g["op_limited_flag"].any()
    energy = g[TARGET].sum()
    theo = g["poa_clearsky"].sum() * AC_CAPACITY_KW / 1000
    idx = (energy / theo)[complete & normal]
    by_month = idx.groupby(idx.index.month)
    daily = idx.reindex(pd.date_range(idx.index.min(), idx.index.max(), freq="D"))
    acf = [daily.autocorr(lag) for lag in range(1, 15)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    a.boxplot([by_month.get_group(m).to_numpy() for m in range(1, 13)],
              tick_labels=[str(m) for m in range(1, 13)],
              showfliers=False)
    a.set_xlabel("month")
    a.set_ylabel("daily AC energy / clear-sky POA energy")
    a.set_title("Daily clear-sky index (train, complete normal-operation days)", fontsize=9)
    b.bar(range(1, 15), acf, color="#455a64")
    b.set_xlabel("lag [days]")
    b.set_ylabel("autocorrelation")
    b.set_title("Persistence of the daily index (train)", fontsize=9)
    table = pd.DataFrame({"month": range(1, 13),
                          "days": [int(by_month.get_group(m).size) for m in range(1, 13)],
                          "median_daily_index": [round(float(by_month.get_group(m).median()), 3)
                                                 for m in range(1, 13)],
                          "p10_daily_index": [round(float(by_month.get_group(m).quantile(.1)), 3)
                                              for m in range(1, 13)]})
    table = pd.concat([table, pd.DataFrame({"month": [f"acf_lag_{k}" for k in range(1, 15)],
                                            "median_daily_index": np.round(acf, 3)})])
    return table, _save(fig, "fig08_variability")


# ── F9 operations, F10 sensors ─────────────────────────────────────────────


def operations(hourly: pd.DataFrame, site: Site) -> str:
    day = daytime_mask(hourly.index, site)
    h = hourly[day]
    month = _naive(h.index).to_period("M")
    down = ((h["frac_downtime_inv1_flag"] + h["frac_downtime_inv2_flag"]) > 0).groupby(month).sum()
    curt = (h["frac_curtail_flag"] > 0).groupby(month).sum()
    derate = ((h["frac_derate_inv1_flag"] + h["frac_derate_inv2_flag"]) > 0).groupby(month).sum()
    miss = h[TARGET].isna().groupby(month).sum()
    t = down.index.to_timestamp() + pd.Timedelta(days=14)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.bar(t, down, width=25, color="#c62828", label="hours with an inverter stop (F1)")
    ax.bar(t, curt, width=25, bottom=down, color="#f9a825", label="hours with curtailment (F2)")
    ax.bar(t, derate, width=25, bottom=down + curt, color="#6a1b9a",
           label="hours with a derated inverter (F3)")
    ax.bar(t, miss, width=25, bottom=down + curt + derate, color="0.3", label="target missing")
    _shade_splits(ax)
    ax.set_ylabel("daytime hours per month")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Operational events and missing target (flags never change values, D-22)")
    return _save(fig, "fig09_operations")


def sensors(hourly: pd.DataFrame, stage4_h: pd.DataFrame, site: Site
            ) -> tuple[pd.DataFrame, str]:
    """F-19: which in-plane sensor is right? Each against clear-sky POA (90th percentile of the
    month, i.e. the clearest hours) and against each other, before cleaning (stage 4); the
    pyranometer's faulty days are removed by R7 in stage 5."""
    day = daytime_mask(stage4_h.index, site)
    h = stage4_h[day & stage4_h["irradiance"].gt(300) & stage4_h["pyranometer"].gt(300)
                 & hourly["poa_clearsky"].gt(300)]
    month = _naive(h.index).to_period("M")
    cs = hourly.loc[h.index, "poa_clearsky"]
    table = pd.DataFrame({
        "irradiance_over_clearsky_p90": (h["irradiance"] / cs).groupby(month).quantile(0.9),
        "pyranometer_over_clearsky_p90": (h["pyranometer"] / cs).groupby(month).quantile(0.9),
        "pyranometer_over_irradiance_median": (h["pyranometer"] / h["irradiance"])
        .groupby(month).median(),
    }).round(3)
    after = hourly[day & hourly["irradiance"].gt(300) & hourly["pyranometer"].gt(300)]
    after_ratio = (after["pyranometer"] / after["irradiance"]).groupby(
        _naive(after.index).to_period("M")).median()
    table["pyranometer_over_irradiance_after_cleaning"] = after_ratio.reindex(table.index).round(3)
    t = table.index.to_timestamp() + pd.Timedelta(days=14)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4))
    a.plot(t, table["irradiance_over_clearsky_p90"], "o-", ms=3, label="irradiance (meteo st01)")
    a.plot(t, table["pyranometer_over_clearsky_p90"], "o-", ms=3, label="pyranometer (SiteData)")
    a.set_ylabel("measured / clear-sky POA\n(90th pct of the month)")
    a.set_title("Against clear-sky POA: the pyranometer drops by 25 %, the irradiance does not",
                fontsize=9)
    a.legend(fontsize=7)
    b.plot(t, table["pyranometer_over_irradiance_median"], "o-", ms=3, color="0.5",
           label="before cleaning (stage 4)")
    b.plot(t, table["pyranometer_over_irradiance_after_cleaning"], "o-", ms=3, color="#c62828",
           label="after R7 (stage 5)")
    b.axhline(1, color="k", ls=":", lw=1)
    b.set_ylabel("pyranometer / irradiance (monthly median)")
    b.set_title("x0.75 calibration fault: 2023-05-28 .. 2023-12-31 and days in 2024-12 .. 2025-04",
                fontsize=9)
    b.legend(fontsize=7)
    for ax in (a, b):
        ax.set_xlim(t[0] - pd.Timedelta(days=20), t[-1] + pd.Timedelta(days=20))
        _shade_splits(ax)
    fig.suptitle("F-19: the in-plane irradiance sensor used by the models is consistent; the "
                 "redundant pyranometer is not (D-09 confirmed)")
    table = table.reset_index(names="month")
    table["month"] = table["month"].astype(str)
    return table, _save(fig, "fig10_sensors")


# ── run ────────────────────────────────────────────────────────────────────


def run() -> dict:
    site = load_site()
    hourly = pd.read_parquet(STAGE5 / "scada_1H_clean.parquet")
    stage4_h = pd.read_parquet(STAGE4 / "scada_1H_utc.parquet",
                               columns=[TARGET, "irradiance", "pyranometer"])
    stage4_min = pd.read_parquet(STAGE4 / "scada_1min_true_utc.parquet",
                                 columns=[TARGET, "s01_inv1_AC_kW"])
    stage5_min = pd.read_parquet(STAGE5 / "scada_1min_clean.parquet", columns=["s01_inv1_AC_kW"])
    cs = hourly_clear_sky(stage5_min.index, site)
    hourly = hourly.join(cs)
    ingest = pd.read_parquet(INGEST, columns=["irradiance", "s01_inv1_AC_kW", "s01_inv1_DC_kW"])
    stage4_manifest = json.loads((STAGE4 / "preprocess_manifest.json").read_text(encoding="utf-8"))

    splits = table_splits(hourly, site)
    bug_table, fig_bug = time_bug(stage4_min, site)
    physics, fig_phys = power_irradiance(hourly, site)
    var_table, fig_var = variability(hourly)
    sensor_table, fig_sensors = sensors(hourly, stage4_h, site)
    figures = [
        overview(hourly, stage4_manifest["polled_source"]["first_day"]),
        fig_bug, clocks(), polled(ingest),
        cleaning(stage4_min, stage5_min, stage4_h, hourly, site),
        diurnal(hourly), fig_phys, fig_var, operations(hourly, site), fig_sensors,
    ]
    tables = [_table(splits, "table_splits"), _table(bug_table, "table_time_bug"),
              _table(physics, "table_physics"), _table(var_table, "table_variability"),
              _table(sensor_table, "table_sensors")]
    inputs = {p.relative_to(OUTPUTS_DIR.parent).as_posix(): sha256(p) for p in (
        STAGE5 / "scada_1H_clean.parquet", STAGE4 / "scada_1H_utc.parquet",
        *(RAW_DIR / f for f in TIME_BUG_MONTHS.values()))}
    inputs.update({p.relative_to(OUTPUTS_DIR.parent).as_posix(): text_sha256(p)
                   for p in (CONFIGS_DIR / "clock_offsets.csv", POLLED_SIM, CLEANING_LOG)})
    manifest = {
        "stage": "03_eda",
        "git_commit": git_commit(),
        "inputs_sha256": inputs,
        "figures": sorted(figures),
        "tables": sorted(tables),
        "split_hours": splits.set_index("split")["hours"].to_dict(),
        "time_bug_median_offset_min": {f"{r['month']} | {r['reading']}":
                                       r["median_centroid_minus_solar_noon_min"]
                                       for r in bug_table.to_dict("records")},
        "temperature_coefficient_pct_per_K": float(physics.iloc[0]["value"]),
    }
    (OUT / "eda_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8",
                                           newline="\n")
    return manifest


def main() -> int:
    manifest = run()
    print(json.dumps({k: manifest[k] for k in ("figures", "split_hours",
                                               "time_bug_median_offset_min",
                                               "temperature_coefficient_pct_per_K")}, indent=2))
    return 0

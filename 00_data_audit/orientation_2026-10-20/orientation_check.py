"""A-04 re-measured: orientation of the panel plane on the corrected data (stage 6, S1).

F-13 (2026-10-14) measured tilt ~20 deg / azimuth ~180 deg on clocks that were not yet
corrected. After the clock correction (D-19) the DC production centre lies 1-2.7 min after
solar noon, so the panels may face a little west of south. This script repeats the fit on the
stage-5 minutes (true UTC, clock-corrected, cleaned), on the train split only.

Data: 10-minute means of the summed DC power of both inverters (temperature-corrected with the
coefficient measured in stage 3) and of the in-plane irradiance sensor; sun elevation > 15 deg
(eastern mountain horizon, low-sun effects); clear days chosen from the DC curve alone
(smooth and high-energy), with no op-limited minute and no missing DC.

Methods (each gives tilt and/or azimuth):
  M1  DC vs clear-sky POA (Ineichen + Hay-Davies), one scale for all days: uses the seasonal
      level, so it constrains the tilt.
  M2  DC vs clear-sky POA, one scale per day: daily shape only (azimuth well, tilt weakly).
  M3  DC production centre minus the centre of the clear-sky POA curve, as a function of the
      azimuth (tilt fixed at the M1 value): the azimuth at which the median offset is zero.
  M0  the in-plane irradiance sensor with M2: the sensor plane, which is the frame of the
      clock correction (stage 4 used 20/180).

Decision rule (docs/stage6_plan.md, S1): if M1-M3 agree on a panel plane that differs from
20/180 by >= 2 deg in azimuth or in tilt, the panel plane is used for the theoretical power and
the POA features of stage 6; otherwise 20/180 stays.

Run: python 00_data_audit/orientation_2026-10-20/orientation_check.py
Writes orientation_results.csv, orientation_summary.json, orientation_fit.png next to this file
and a run log under outputs/06_features/runs/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pvlib  # noqa: E402

from solarpv.config.constants import TIMEZONE_LOCAL, TRAIN_END  # noqa: E402
from solarpv.config.paths import OUTPUTS_DIR, PROCESSED_DIR  # noqa: E402
from solarpv.config.runlog import RunLog  # noqa: E402
from solarpv.config.site import load_site  # noqa: E402

OUT = Path(__file__).parent
MINUTES = PROCESSED_DIR / "05_clean" / "scada_1min_clean.parquet"
EDA_PHYSICS = OUTPUTS_DIR / "03_eda" / "tables" / "table_physics.csv"
COLUMNS = ["s01_inv1_DC_kW", "s01_inv2_DC_kW", "mod_temp1", "irradiance", "op_limited_flag"]
TILTS = np.arange(10.0, 30.5, 0.5)
AZIMUTHS = np.arange(170.0, 195.5, 0.5)
MIN_ELEV = 15.0
DAYS_PER_MONTH = 6


def ten_minute(site) -> pd.DataFrame:
    """10-min means of the train minutes, with sun position and clear-sky irradiance at the
    middle of each interval. A 10-min block with any op-limited minute is marked."""
    df = pd.read_parquet(MINUTES, columns=COLUMNS)
    df = df[df.index <= pd.Timestamp(TRAIN_END, tz="UTC")]
    df["dc"] = df["s01_inv1_DC_kW"] + df["s01_inv2_DC_kW"]  # NaN if either is missing
    g = df.resample("10min")
    out = g[["dc", "mod_temp1", "irradiance"]].mean()
    out["n_dc"] = g["dc"].count()
    out["limited"] = g["op_limited_flag"].max().astype(bool)
    loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC",
                                  altitude=site.altitude_m)
    mid = out.index + pd.Timedelta(minutes=5)
    sp = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen", solar_position=sp)
    out["zenith"] = sp["apparent_zenith"].to_numpy()
    out["azimuth"] = sp["azimuth"].to_numpy()
    out["elev"] = sp["apparent_elevation"].to_numpy()
    for c in ("ghi", "dni", "dhi"):
        out[c] = cs[c].to_numpy()
    out["dni_extra"] = np.asarray(pvlib.irradiance.get_extra_radiation(mid))
    return out


def clear_days(df: pd.DataFrame) -> list:
    """Smooth, high-energy days per month, from the DC curve only."""
    day = df.index.tz_convert(TIMEZONE_LOCAL).date
    rows = []
    for d, g in df.assign(day=day).groupby("day"):
        g = g[g["elev"] > MIN_ELEV]
        p = g["dc"].to_numpy()
        if (len(p) < 30 or np.isnan(p).any() or p.max() <= 0 or (g["n_dc"] < 10).any()
                or g["limited"].any()):  # incomplete, stopped or op-limited day
            continue
        rows.append((d, p.sum(), np.abs(np.diff(p, 2)).mean() / p.max()))
    s = pd.DataFrame(rows, columns=["day", "energy", "rough"])
    s["month"] = [pd.Timestamp(d).strftime("%Y-%m") for d in s["day"]]
    keep = []
    for _, g in s.groupby("month"):
        g = g[g["energy"] >= g["energy"].quantile(0.6)]
        keep += list(g.nsmallest(DAYS_PER_MONTH, "rough")["day"])
    return keep


def poa(df: pd.DataFrame, tilt: float, azimuth: float) -> np.ndarray:
    return np.asarray(pvlib.irradiance.get_total_irradiance(
        tilt, azimuth, df["zenith"], df["azimuth"], df["dni"], df["ghi"], df["dhi"],
        dni_extra=df["dni_extra"], model="haydavies")["poa_global"])


def fit_errors(y: np.ndarray, day: np.ndarray, sel: pd.DataFrame, per_day: bool) -> np.ndarray:
    """nRMSE of y against scale x POA over the (tilt, azimuth) grid."""
    err = np.full((len(TILTS), len(AZIMUTHS)), np.nan)
    days = pd.factorize(day)[0]
    for i, tilt in enumerate(TILTS):
        for j, az in enumerate(AZIMUTHS):
            x = poa(sel, tilt, az)
            if per_day:  # least-squares scale per day: shape only
                num = np.bincount(days, weights=x * y)
                den = np.bincount(days, weights=x * x)
                pred = (num / den)[days] * x
            else:
                pred = (x @ y) / (x @ x) * x
            err[i, j] = np.sqrt(np.mean((y - pred) ** 2)) / y.mean()
    return err


def centroid_offset(t_min: np.ndarray, w: np.ndarray, day: np.ndarray) -> pd.Series:
    """Power-weighted centre of each day [min of day]."""
    frame = pd.DataFrame({"day": day, "tw": t_min * w, "w": w})
    g = frame.groupby("day")
    return g["tw"].sum() / g["w"].sum()


def main() -> None:
    site = load_site()
    with RunLog("06_features", "python 00_data_audit/orientation_2026-10-20/orientation_check.py",
                out_dir=OUTPUTS_DIR / "06_features") as log:
        log.inputs([MINUTES, EDA_PHYSICS])
        gamma = float(pd.read_csv(EDA_PHYSICS).iloc[0]["value"]) / 100
        log.value("temperature_coefficient_per_K", gamma)
        with log.step("10-minute data and clear days (train)"):
            df = ten_minute(site)
            days = clear_days(df)
            log.value("clear_days", len(days))
            log.note(f"clear days per month: {DAYS_PER_MONTH} (smooth, top-40 % energy, no "
                     "op-limited minute, DC complete); months covered: "
                     f"{len({pd.Timestamp(d).strftime('%Y-%m') for d in days})}")
        local_day = df.index.tz_convert(TIMEZONE_LOCAL).date
        sel = df[np.isin(local_day, days) & (df["elev"] > MIN_ELEV)].copy()
        sel["day"] = sel.index.tz_convert(TIMEZONE_LOCAL).date
        y_dc = (sel["dc"] / (1 + gamma * (sel["mod_temp1"] - 25))).to_numpy()
        y_irr = sel["irradiance"].to_numpy()
        log.value("ten_minute_samples", len(sel))

        results = []
        with log.step("grid fits (M1 one scale, M2 scale per day, M0 sensor)"):
            grids = {"M1 DC one scale": fit_errors(y_dc, sel["day"].to_numpy(), sel, False),
                     "M2 DC scale per day": fit_errors(y_dc, sel["day"].to_numpy(), sel, True),
                     "M0 sensor scale per day": fit_errors(y_irr, sel["day"].to_numpy(), sel,
                                                           True)}
            for name, err in grids.items():
                i, j = np.unravel_index(np.nanargmin(err), err.shape)
                near = err <= err[i, j] * 1.01  # within 1 % of the best nRMSE
                ti, aj = np.where(near)
                results.append({"method": name, "tilt": TILTS[i], "azimuth": AZIMUTHS[j],
                                "nrmse": round(float(err[i, j]), 5),
                                "tilt_range_1pct": f"{TILTS[ti].min()}-{TILTS[ti].max()}",
                                "azimuth_range_1pct": f"{AZIMUTHS[aj].min()}-{AZIMUTHS[aj].max()}"})
                log.note(f"{name}: tilt {TILTS[i]}, azimuth {AZIMUTHS[j]} (within 1 %: tilt "
                         f"{TILTS[ti].min()}-{TILTS[ti].max()}, azimuth "
                         f"{AZIMUTHS[aj].min()}-{AZIMUTHS[aj].max()})")

        with log.step("M3 production centre vs clear-sky centre"):
            tilt_m1 = results[0]["tilt"]
            t_min = ((sel.index - sel.index.normalize()).total_seconds().to_numpy() / 60 + 5)
            day = sel["day"].to_numpy()
            measured = centroid_offset(t_min, y_dc, day)
            offsets = []
            for az in AZIMUTHS:
                model = centroid_offset(t_min, poa(sel, tilt_m1, az), day)
                offsets.append(float((measured - model).median()))
            offsets = np.array(offsets)
            az_m3 = float(np.interp(0.0, offsets[::-1], AZIMUTHS[::-1]))
            results.append({"method": "M3 DC centre vs clear-sky centre", "tilt": tilt_m1,
                            "azimuth": round(az_m3, 1), "nrmse": np.nan,
                            "tilt_range_1pct": "", "azimuth_range_1pct": ""})
            log.note(f"M3: median offset of the DC centre against the clear-sky POA centre is "
                     f"{offsets[np.argmin(np.abs(AZIMUTHS - 180))]:+.2f} min at azimuth 180; "
                     f"zero at azimuth {az_m3:.1f}")

        table = pd.DataFrame(results)
        table.to_csv(OUT / "orientation_results.csv", index=False, lineterminator="\n")
        panel = table[table["method"].str.startswith(("M1", "M2", "M3"))]
        az_panel = float(panel["azimuth"].median())
        tilt_panel = float(table.loc[0, "tilt"])
        differs = abs(az_panel - 180) >= 2 or abs(tilt_panel - 20) >= 2
        summary = {"clear_days": len(days), "temperature_coefficient_per_K": gamma,
                   "panel_tilt_m1": tilt_panel, "panel_azimuth_median_m1_m3": az_panel,
                   "sensor_plane_m0": {"tilt": float(table.loc[2, "tilt"]),
                                       "azimuth": float(table.loc[2, "azimuth"])},
                   "differs_from_20_180": bool(differs),
                   "decision": (f"use the panel plane {tilt_panel:.1f}/{az_panel:.1f}"
                                if differs else "keep 20/180")}
        (OUT / "orientation_summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8", newline="\n")
        log.value("decision", summary["decision"])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for ax, (name, err) in zip(axes[:2], list(grids.items())[:2], strict=True):
            im = ax.contourf(AZIMUTHS, TILTS, err * 100, levels=20, cmap="viridis")
            ax.set_xlabel("azimuth [deg]")
            ax.set_ylabel("tilt [deg]")
            ax.set_title(f"{name}: nRMSE [%]", fontsize=9)
            plt.colorbar(im, ax=ax)
        axes[2].plot(AZIMUTHS, offsets, "o-", ms=3)
        axes[2].axhline(0, color="k", lw=0.8)
        axes[2].axvline(180, color="0.6", ls=":")
        axes[2].set_xlabel("azimuth of the clear-sky model [deg]")
        axes[2].set_ylabel("median centre offset, DC - model [min]")
        axes[2].set_title("M3: azimuth where the DC curve is centred", fontsize=9)
        fig.suptitle("A-04 re-measured on the corrected data (train clear days, sun > 15 deg)",
                     fontsize=10)
        fig.savefig(OUT / "orientation_fit.png", dpi=130, bbox_inches="tight")
        log.outputs([OUT / "orientation_results.csv", OUT / "orientation_summary.json",
                     OUT / "orientation_fit.png"])
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

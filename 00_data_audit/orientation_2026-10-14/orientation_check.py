"""Exploratory check of panel tilt and azimuth (assumption A-04) from the raw data.

Method 1: fit modelled clear-sky POA (Ineichen + Hay-Davies) to each radiation sensor on clear
          days. Result: both sensors are in the module plane (no horizontal GHI on site).
Method 2: fit the same POA to temperature-corrected DC power, with one scale over all seasons
          (uses the winter/summer ratio, constrains tilt) and with one scale per day (shape only).
Method 3: clear-day DC centroid minus solar noon (azimuth symmetry).
Clear days are chosen from DC smoothness and energy only, without any orientation assumption.
Hours with sun elevation < 15 deg are excluded (eastern mountain horizon, low-sun effects).
Results (2026-10-14): tilt ~20-22 deg, azimuth ~178-180 deg. See PROJECT_MASTER.md F-13.

Run: python 00_data_audit/orientation_2026-10-14/orientation_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.paths import RAW_DIR
from solarpv.config.site import load_site

OUT_DIR = Path(__file__).parent
MONTHS = ["2112", "2201", "2204", "2206", "2207", "2209"]  # no DST switch inside these months
COLS = {0: "t_stamp", 7: "irrad", 8: "mod_temp1", 18: "inv1_dc", 22: "inv2_dc",
        16: "inv1_ac", 20: "inv2_ac", 24: "pyranometer"}
TILTS = np.arange(0, 46, 1)
AZIMUTHS = np.arange(130, 232, 2)
GAMMA = -0.004  # power temperature coefficient [1/K], typical c-Si


def load_month(m: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{m}.CSV", usecols=sorted(COLS))
    df.columns = [COLS[i] for i in sorted(COLS)]
    t = pd.to_datetime(df.pop("t_stamp"), format="%Y-%m-%d %H:%M:%S.%f")
    df.index = pd.DatetimeIndex(t).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    return df.apply(pd.to_numeric, errors="coerce").resample("10min").mean()


def clear_days(df: pd.DataFrame, elev: pd.Series) -> list:
    """Smooth, high-energy days, chosen from the DC curve only (no orientation assumption)."""
    day = df.index.tz_convert(TIMEZONE_LOCAL).date
    stats = []
    for d, g in df.assign(day=day, elev=elev.values).groupby("day"):
        g = g[g["elev"] > 15]
        p = g["dc"].to_numpy()
        if len(p) < 30 or np.isnan(p).any() or p.max() <= 0:
            continue
        stats.append((d, p.sum(), np.abs(np.diff(p, 2)).mean() / p.max()))
    s = pd.DataFrame(stats, columns=["day", "energy", "rough"])
    s["month"] = [d.month for d in s["day"]]
    keep = []
    for _, g in s.groupby("month"):
        g = g[g["energy"] >= g["energy"].quantile(0.6)]
        keep += list(g.nsmallest(6, "rough")["day"])
    return keep


def fit_orientation(y, zen, azi, dni, ghi, dhi, dni_extra, per_day=None):
    """Grid search of (tilt, azimuth) minimising normalised RMSE of y ~ scale * POA."""
    best = (np.inf, None, None)
    surface = np.full((len(TILTS), len(AZIMUTHS)), np.nan)
    for i, tilt in enumerate(TILTS):
        for j, az in enumerate(AZIMUTHS):
            poa = np.asarray(pvlib.irradiance.get_total_irradiance(
                tilt, az, zen, azi, dni, ghi, dhi, dni_extra=dni_extra, model="haydavies"
            )["poa_global"])
            if per_day is None:
                scale = (y @ poa) / (poa @ poa)
                resid = y - scale * poa
            else:
                resid = np.empty_like(y)
                for d in np.unique(per_day):
                    k = per_day == d
                    resid[k] = y[k] - (y[k] @ poa[k]) / (poa[k] @ poa[k]) * poa[k]
            nrmse = np.sqrt(np.mean(resid**2)) / y.mean()
            surface[i, j] = nrmse
            if nrmse < best[0]:
                best = (nrmse, tilt, az)
    return best, surface


def main():
    site = load_site()
    loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC", altitude=site.altitude_m)
    print(f"site: {site.latitude}, {site.longitude}, {site.altitude_m} m  (private={site.is_private})")

    df = pd.concat(load_month(m) for m in MONTHS)
    df["dc"] = df["inv1_dc"] + df["inv2_dc"]
    df["ac"] = df["inv1_ac"] + df["inv2_ac"]
    tc = df.index + pd.Timedelta(minutes=5)  # centre of each 10-min bin
    sp = loc.get_solarposition(tc)
    cs = loc.get_clearsky(tc, model="ineichen", solar_position=sp)
    dni_extra = pvlib.irradiance.get_extra_radiation(tc)
    elev = pd.Series(sp["apparent_elevation"].to_numpy(), index=df.index)

    days = clear_days(df, elev)
    local_day = df.index.tz_convert(TIMEZONE_LOCAL).date
    m = np.isin(local_day, days) & (elev.to_numpy() > 15)
    m &= df[["dc", "irrad", "pyranometer", "mod_temp1"]].notna().all(axis=1).to_numpy()
    print(f"clear days: {len(days)}; 10-min points used: {int(m.sum())}")

    # plain arrays from here on: the solar-position index is shifted by 5 min from the data index
    zen = sp["apparent_zenith"].to_numpy()[m]
    azi = sp["azimuth"].to_numpy()[m]
    cs_dni, cs_ghi, cs_dhi = (cs[c].to_numpy()[m] for c in ("dni", "ghi", "dhi"))
    dni_extra = np.asarray(dni_extra)[m]
    sub = df[m]

    # Method 1: sensors vs clear-sky POA
    rows = []
    for sensor in ["pyranometer", "irrad"]:
        (err, t, a), _ = fit_orientation(
            sub[sensor].to_numpy(), zen, azi, cs_dni, cs_ghi, cs_dhi, dni_extra
        )
        rows.append(("sensor vs clear-sky POA", sensor, t, a, err))

    # Both sensors turned out to be tilted (in the module plane), so there is no measured GHI
    # to decompose. Method 2 therefore fits temperature-corrected DC to clear-sky POA directly.
    dc_tc = (sub["dc"] / (1 + GAMMA * (sub["mod_temp1"] - 25))).to_numpy()
    k = dc_tc < 0.95 * np.nanmax(dc_tc)  # drop a possible clipping plateau
    days_m = np.asarray(local_day)[m]
    for label, per_day in [("DC vs clear-sky POA, one scale", None),
                           ("DC vs clear-sky POA, scale per day", days_m)]:
        (err, t, a), surf = fit_orientation(
            dc_tc[k], zen[k], azi[k], cs_dni[k], cs_ghi[k], cs_dhi[k], dni_extra[k],
            None if per_day is None else per_day[k],
        )
        rows.append((label, "dc", t, a, err))
        near = np.argwhere(surf <= err * 1.02)  # orientations within 2 % of the best fit
        print(f"{label}: tilt {TILTS[near[:, 0]].min()}-{TILTS[near[:, 0]].max()}, "
              f"azimuth {AZIMUTHS[near[:, 1]].min()}-{AZIMUTHS[near[:, 1]].max()} "
              "(within 2 % of best nRMSE)")
        if per_day is None:
            pd.DataFrame(surf, index=TILTS, columns=AZIMUTHS).to_csv(OUT_DIR / "dc_fit_surface.csv")

    ratio = (sub["pyranometer"] / sub["irrad"]).median()
    corr_dc = np.corrcoef(sub["dc"], sub["irrad"])[0, 1]
    print(f"pyranometer / irrad median ratio: {ratio:.3f}; corr(DC, irrad) = {corr_dc:.4f}")

    res = pd.DataFrame(rows, columns=["method", "series", "tilt", "azimuth", "nrmse"])
    res.to_csv(OUT_DIR / "orientation_results.csv", index=False)
    print(res.to_string(index=False))

    # Production centroid vs solar noon on the clear days (azimuth symmetry check)
    noon = loc.get_sun_rise_set_transit(pd.DatetimeIndex(
        [pd.Timestamp(d, tz=TIMEZONE_LOCAL).tz_convert("UTC") + pd.Timedelta(hours=12)
         for d in days]), method="spa")["transit"]
    offsets = []
    for d, tn in zip(days, noon, strict=True):
        g = df[(local_day == d) & (elev.to_numpy() > 15)]
        w = g["dc"].clip(lower=0)
        centroid = g.index[0] + pd.Timedelta(
            seconds=float(((g.index - g.index[0]).total_seconds() * w).sum() / w.sum()) + 300)
        offsets.append((centroid - tn).total_seconds() / 60)
    print(f"DC centroid - solar noon on clear days: median {np.median(offsets):+.1f} min "
          f"(IQR {np.percentile(offsets, 25):+.1f} .. {np.percentile(offsets, 75):+.1f})")


if __name__ == "__main__":
    main()

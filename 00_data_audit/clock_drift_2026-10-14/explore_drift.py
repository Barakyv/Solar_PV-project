"""Daily clock-offset estimates for the two logger clocks (finding F-15, decision D-19).

tau_met : shift (min) that best aligns the in-plane irradiance sensor with modelled clear-sky
          POA on clear days (positive = logger timestamps later than true time).
delta   : inverter-clock minus meteo-clock offset, from the daily cross-correlation of the
          first differences of DC power and irradiance (all days with enough signal).
Output: daily_offsets.csv next to this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.paths import PROCESSED_DIR
from solarpv.config.site import load_site

OUT = Path(__file__).parent / "daily_offsets.csv"
MAX_LAG = 30  # minutes
MIN_ELEV = 15.0


def parabolic_peak(values: np.ndarray, lags: np.ndarray, maximise: bool) -> float:
    i = int(np.argmax(values) if maximise else np.argmin(values))
    if 0 < i < len(values) - 1:
        y0, y1, y2 = values[i - 1], values[i], values[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            return float(lags[i] + 0.5 * (y0 - y2) / denom)
    return float(lags[i])


def main():
    site = load_site()
    cols = ["s01_inv1_DC_kW", "s01_inv2_DC_kW", "irradiance"]
    df = pd.read_parquet(PROCESSED_DIR / "01_ingest" / "scada_1min_utc.parquet", columns=cols)
    dc = (df["s01_inv1_DC_kW"] + df["s01_inv2_DC_kW"]).clip(lower=0, upper=4000)
    irr = df["irradiance"].clip(lower=0, upper=1500)

    loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC",
                                  altitude=site.altitude_m)
    t_mid = df.index + pd.Timedelta(seconds=30)  # 1-min averages labelled at the start
    sp = loc.get_solarposition(t_mid)
    cs = loc.get_clearsky(t_mid, model="ineichen", solar_position=sp)
    poa = pvlib.irradiance.get_total_irradiance(
        site.surface_tilt, site.surface_azimuth, sp["apparent_zenith"], sp["azimuth"],
        cs["dni"], cs["ghi"], cs["dhi"],
        dni_extra=pvlib.irradiance.get_extra_radiation(t_mid), model="haydavies",
    )["poa_global"].to_numpy()
    elev = sp["apparent_elevation"].to_numpy()

    day = df.index.tz_convert(TIMEZONE_LOCAL).date
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    rows = []
    for d, pos in pd.Series(np.arange(len(df)), index=day).groupby(level=0):
        i = pos.to_numpy()
        sel = i[elev[i] > MIN_ELEV]
        if len(sel) < 240:
            continue
        a, b = sel[0], sel[-1]
        x = irr.to_numpy()[a - MAX_LAG:b + MAX_LAG + 1]
        m = poa[a - MAX_LAG:b + MAX_LAG + 1]
        p = dc.to_numpy()[a - MAX_LAG:b + MAX_LAG + 1]
        core = slice(MAX_LAG, MAX_LAG + (b - a + 1))
        xi = x[core]
        # clear-sky likeness of the irradiance trace
        ratio = xi.sum() / max(m[core].sum(), 1)
        rough = np.abs(np.diff(xi, 2)).mean() / max(xi.max(), 1)
        # tau_met: measured(t) ~ s * model(t - tau); lag L means model shifted by L minutes
        sse = []
        for lag in lags:
            mm = m[MAX_LAG - lag:MAX_LAG - lag + len(xi)]
            s = (xi @ mm) / max(mm @ mm, 1e-9)
            sse.append(((xi - s * mm) ** 2).sum())
        tau_met = parabolic_peak(np.array(sse), lags, maximise=False)
        # delta: corr(d irr(t), d DC(t + lag)) peaks at lag = tau_inv - tau_met
        dx = np.diff(xi)
        corr = []
        for lag in lags:
            dp = np.diff(p[MAX_LAG + lag:MAX_LAG + lag + len(xi)])
            corr.append(np.corrcoef(dx, dp)[0, 1] if dx.std() > 0 and dp.std() > 0 else np.nan)
        corr = np.array(corr)
        ok = np.isfinite(corr).all()
        delta = parabolic_peak(corr, lags, maximise=True) if ok else np.nan
        rows.append({"day": d, "clear_ratio": ratio, "roughness": rough, "tau_met": tau_met,
                     "delta": delta, "delta_peak_corr": np.nanmax(corr) if ok else np.nan,
                     "variability": dx.std()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    out["month"] = pd.to_datetime(out["day"]).dt.to_period("M")
    clear = out[(out["roughness"] < out["roughness"].quantile(0.35))
                & out["clear_ratio"].between(0.85, 1.15)]
    good = out[out["delta_peak_corr"] > 0.5]
    summary = pd.DataFrame({
        "n_clear": clear.groupby("month").size(),
        "tau_met_med": clear.groupby("month")["tau_met"].median(),
        "n_delta": good.groupby("month").size(),
        "delta_med": good.groupby("month")["delta"].median(),
    }).round(2)
    summary["tau_inv_est"] = (summary["tau_met_med"] + summary["delta_med"]).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()

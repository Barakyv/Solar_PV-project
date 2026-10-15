"""How well can hourly means be recovered from 15-min polled data? (F-18, decision D-20)

Truth: minute-source data from the train period (Mar-May 2022 and 2023, clock-corrected, stage 4).
Each of the six tags that see the cloud field (irradiance, pyranometer, AC and DC of both
inverters) is degraded the way the 2025 export is: one sample every 15 min at a random phase per
tag and day, held until the next one. The hourly mean of site_total_AC_kW is then rebuilt with:
  M0 hold      the raw 2025 export as it is
  M1 linear    linear interpolation between poll instants (current stage-4 method)
  M2 kc-linear linear interpolation of the clear-sky index AC/POA_cs, times POA_cs
  M3 fused     clear-sky indices of all six tags, each scaled to site AC units per day, merged
               in time and interpolated; the AC samples themselves are among them
Errors are on daytime hours (sun > 5 deg at mid-hour), split by day variability.
Output: results.csv and summary.csv next to this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, TIMEZONE_LOCAL
from solarpv.config.paths import PROCESSED_DIR
from solarpv.config.site import load_site
from solarpv.data.clock import Orientation, clear_sky_components, poa_from_components
from solarpv.evaluation.daytime import daytime_mask

OUT = Path(__file__).parent
PERIODS = [("2022-03-01", "2022-05-31"), ("2023-03-01", "2023-05-31")]
TAGS = ["irradiance", "pyranometer", "s01_inv1_AC_kW", "s01_inv2_AC_kW", "s01_inv1_DC_kW",
        "s01_inv2_DC_kW"]
AC = ["s01_inv1_AC_kW", "s01_inv2_AC_kW"]
POLL = 15
CS_MIN = 30.0  # W/m2: below this the clear-sky index is not used
SEED = 42


def poll(values: np.ndarray, phase: int):
    """Sample positions and values every POLL minutes from `phase`; held series with blend."""
    pos = np.arange(phase, len(values), POLL)
    samples = values[pos]
    held = np.empty_like(values)
    k = np.searchsorted(pos, np.arange(len(values)), side="right") - 1
    held[:] = np.where(k >= 0, samples[np.clip(k, 0, None)], samples[0])
    blend = pos[pos > 0]
    held[blend] = 0.5 * held[blend - 1] + 0.5 * samples[np.searchsorted(pos, blend)]
    return pos + 0.5, samples, held  # poll instant at the centre of the poll minute


def interp(t, x, n):
    return np.interp(np.arange(n) + 0.5, t, x)


def kc_interp(t, x, cs, n):
    """Linear interpolation of x / cs, times cs; plain linear where cs is too small."""
    pos = (t - 0.5).astype(int)
    ok = cs[pos] > CS_MIN
    if ok.sum() < 2:
        return interp(t, x, n)
    k = np.interp(np.arange(n) + 0.5, t[ok], x[ok] / cs[pos][ok])
    return np.where(cs > CS_MIN, k * cs, interp(t, x, n))


def one_day(day: pd.DataFrame, cs: np.ndarray, rng) -> dict[str, np.ndarray]:
    n = len(day)
    polled = {c: poll(day[c].to_numpy(), int(rng.integers(POLL))) for c in TAGS}
    out = {"truth": day[AC].sum(axis=1).to_numpy()}
    out["M0 hold"] = sum(polled[c][2] for c in AC)
    out["M1 linear"] = sum(interp(polled[c][0], polled[c][1], n) for c in AC)
    out["M2 kc-linear"] = sum(kc_interp(polled[c][0], polled[c][1], cs, n) for c in AC)
    # M3: every tag's clear-sky index, scaled per day to site-AC units against M2
    ref = out["M2 kc-linear"]
    day_ok = cs > CS_MIN
    times, values = [], []
    for c in TAGS:
        t, x, _ = polled[c]
        own = kc_interp(t, x, cs, n)
        gain = ref[day_ok].sum() / max(own[day_ok].sum(), 1e-9)  # tag -> site-AC units
        pos = (t - 0.5).astype(int)
        keep = cs[pos] > CS_MIN
        times.append(t[keep])
        values.append(x[keep] * gain / cs[pos][keep])
    t_all = np.concatenate(times)
    k_all = np.concatenate(values)
    order = np.argsort(t_all)
    k = np.interp(np.arange(n) + 0.5, t_all[order], k_all[order])
    out["M3 fused"] = np.where(day_ok, k * cs, ref)
    return out


def main():
    site = load_site()
    cols = TAGS
    data = pd.read_parquet(PROCESSED_DIR / "04_preprocess" / "scada_1min_true_utc.parquet",
                           columns=cols)
    data = pd.concat([data[a:b] for a, b in PERIODS]).interpolate(limit=5)
    comp = clear_sky_components(data.index, site)
    cs_all = poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth))
    rng = np.random.default_rng(SEED)
    local_day = pd.Series(data.index.tz_convert(TIMEZONE_LOCAL).date, index=data.index)
    frames = []
    for idx in data.groupby(local_day).indices.values():
        day = data.iloc[idx]
        if day.isna().any().any() or len(day) < 1380:
            continue
        rec = one_day(day, cs_all[idx], rng)
        frames.append(pd.DataFrame(rec, index=day.index))
    minute = pd.concat(frames)
    hourly = minute.resample("1h").mean().dropna()
    hourly = hourly[daytime_mask(hourly.index, site)]
    # day variability from the truth: std of minute-to-minute changes / mean, per local day
    var = (minute["truth"].diff().abs().groupby(minute.index.tz_convert(TIMEZONE_LOCAL).date)
           .mean() / minute["truth"].groupby(minute.index.tz_convert(TIMEZONE_LOCAL).date)
           .mean())
    hd = pd.Series(hourly.index.tz_convert(TIMEZONE_LOCAL).date, index=hourly.index)
    hourly["day_type"] = np.where(var.reindex(hd).to_numpy() > var.quantile(0.75), "variable",
                                  "clear or steady")
    rows = []
    for dtype, g in [("all", hourly), *hourly.groupby("day_type")]:
        for m in ["M0 hold", "M1 linear", "M2 kc-linear", "M3 fused"]:
            err = g[m] - g["truth"]
            rows.append({"days": dtype, "method": m, "n_hours": len(g),
                         "rmse_kW": round(float(np.sqrt((err ** 2).mean())), 1),
                         "rmse_pct_capacity": round(float(np.sqrt((err ** 2).mean())
                                                          / AC_CAPACITY_KW * 100), 2),
                         "bias_kW": round(float(err.mean()), 1),
                         "p95_abs_kW": round(float(err.abs().quantile(0.95)), 1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    hourly.to_csv(OUT / "results.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Clock-offset estimation and correction (findings F-15, F-18; decision D-19).

Three clocks stamp the raw data. Offset convention: offset = stamped time - true time [min];
true time = stamped time - offset.
  minute source  inverter logger (s01_*, iec_* tags) and meteo logger (all other tags). Both drift
                 and were resynchronised independently (F-15). Used until 2025-02 and for some
                 hours after that.
  polled source  one clock for every tag (checked by the estimation script), from 2025-02-26
                 (F-18).
The source of every tag-minute comes from resolution.harmonise_sources().

Daily estimates, each on the minutes of one source only:
  tau_met  shift that best aligns the irradiance sensor (in the panel plane, F-13) with the
           modelled clear-sky POA at the measured orientation, on clear days.
  tau_dc   the same for the summed DC power. Temperature effects bias it by a roughly constant
           amount; it is only used to compare the two clock groups of the polled source.
  delta    inverter minus meteo offset, from the cross-correlation of the first differences of DC
           power and irradiance on days with short-term variability (clouds). Minute source only:
           polled tags are sampled at different instants, so their differences do not line up.
The absolute frame is that of the clear-sky model at orientation 20/180. An azimuth error and a
constant shift are confounded (8 deg ~ 6 min), so the azimuth is taken from the DC data (F-13)
and the frame is checked against resync anchors.

Smoothing: rolling median within segments separated by step events (resyncs). The applied shift
is constant over a local day for a given source, so daytime minutes are neither lost nor
duplicated except where a tag changes source.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.site import Site

INVERTER_PREFIXES = ("s01_", "iec_")
SOURCES = ("minute", "polled")
GROUPS = ("meteo", "inverter")
MAX_LAG = 30  # minutes searched
MIN_ELEV = 15.0  # deg, avoids the eastern horizon and low-sun model errors
MIN_DAY_MINUTES = 240
DC_COLUMNS = ("s01_inv1_DC_kW", "s01_inv2_DC_kW")


def clock_group(column: str) -> str:
    return "inverter" if column.startswith(INVERTER_PREFIXES) else "meteo"


def offset_column(source: str, group: str) -> str:
    return f"{source}_{group}"


@dataclass(frozen=True)
class Orientation:
    tilt: float
    azimuth: float


# ── modelled clear-sky POA on the 1-minute grid ────────────────────────────


def clear_sky_components(index: pd.DatetimeIndex, site: Site) -> pd.DataFrame:
    """Solar position and Ineichen clear-sky irradiance at the middle of each 1-min interval."""
    loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC",
                                  altitude=site.altitude_m)
    t = index + pd.Timedelta(seconds=30)
    sp = loc.get_solarposition(t)
    cs = loc.get_clearsky(t, model="ineichen", solar_position=sp)
    return pd.DataFrame({
        "zenith": sp["apparent_zenith"].to_numpy(),
        "azimuth": sp["azimuth"].to_numpy(),
        "elev": sp["apparent_elevation"].to_numpy(),
        "dni": cs["dni"].to_numpy(), "ghi": cs["ghi"].to_numpy(), "dhi": cs["dhi"].to_numpy(),
        "dni_extra": np.asarray(pvlib.irradiance.get_extra_radiation(t)),
    }, index=index)


def poa_from_components(comp: pd.DataFrame, orient: Orientation) -> np.ndarray:
    return np.asarray(pvlib.irradiance.get_total_irradiance(
        orient.tilt, orient.azimuth, comp["zenith"].to_numpy(), comp["azimuth"].to_numpy(),
        comp["dni"].to_numpy(), comp["ghi"].to_numpy(), comp["dhi"].to_numpy(),
        dni_extra=comp["dni_extra"].to_numpy(), model="haydavies",
    )["poa_global"])


# ── daily estimates ────────────────────────────────────────────────────────


def _parabolic(values: np.ndarray, lags: np.ndarray, maximise: bool) -> float:
    i = int(np.argmax(values) if maximise else np.argmin(values))
    if 0 < i < len(values) - 1:
        y0, y1, y2 = values[i - 1], values[i], values[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            return float(lags[i] + 0.5 * (y0 - y2) / denom)
    return float(lags[i])


def _day_windows(index: pd.DatetimeIndex, elev: np.ndarray):
    """(local day, first, last) positions of daytime minutes with a MAX_LAG margin."""
    day = index.tz_convert(TIMEZONE_LOCAL).date
    pos = pd.Series(np.arange(len(index)), index=day)
    for d, p in pos.groupby(level=0):
        i = p.to_numpy()
        sel = i[elev[i] > MIN_ELEV]
        if len(sel) >= MIN_DAY_MINUTES and sel[0] >= MAX_LAG and sel[-1] + MAX_LAG < len(index):
            yield d, sel[0], sel[-1]


def shift_fit(meas: np.ndarray, model: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """Shift [min] of `model` that best fits meas[a:b+1] with one scale factor.
    Returns (shift, relative rmse)."""
    x = meas[a:b + 1]
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    sse = np.empty(lags.size)
    for k, lag in enumerate(lags):
        m = model[a - lag:b + 1 - lag]
        s = (x @ m) / max(m @ m, 1e-9)
        sse[k] = ((x - s * m) ** 2).sum()
    tau = _parabolic(sse, lags, maximise=False)
    return tau, float(np.sqrt(sse.min() / x.size) / max(x.mean(), 1e-9))


def daily_estimates(df: pd.DataFrame, comp: pd.DataFrame, orient: Orientation,
                    usable: pd.DataFrame | None = None, with_delta: bool = True) -> pd.DataFrame:
    """One row per local day: tau_met and tau_dc with clear-sky quality, delta with its peak
    correlation. `comp` = clear_sky_components(df.index, site). `usable` (bool, columns
    'irradiance' and 'dc') limits each estimate to days whose whole window is usable."""
    irr_raw = df["irradiance"].to_numpy(dtype=float)
    dc_raw = df[list(DC_COLUMNS)].sum(axis=1, min_count=2).to_numpy(dtype=float)
    ok_irr = np.isfinite(irr_raw)
    ok_dc = np.isfinite(dc_raw)
    if usable is not None:
        ok_irr &= usable["irradiance"].to_numpy()
        ok_dc &= usable["dc"].to_numpy()
    irr = np.clip(np.nan_to_num(irr_raw), 0, 1500)
    dc = np.clip(np.nan_to_num(dc_raw), 0, 4000)
    poa, elev = poa_from_components(comp, orient), comp["elev"].to_numpy()
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    rows = []
    for d, a, b in _day_windows(df.index, elev):
        win = slice(a - MAX_LAG, b + MAX_LAG + 1)
        use_irr, use_dc = ok_irr[win].all(), ok_dc[win].all()
        if not (use_irr or use_dc):
            continue
        row = {"day": d}
        x = irr[a:b + 1]
        if use_irr:
            row["tau_met"], row["clear_rel_rmse"] = shift_fit(irr, poa, a, b)
            row["clear_ratio"] = x.sum() / max(poa[a:b + 1].sum(), 1.0)
            row["variability"] = float(np.diff(x).std() / max(x.mean(), 1e-9))
        if use_dc:
            row["tau_dc"], row["dc_rel_rmse"] = shift_fit(dc, poa, a, b)
        if with_delta and use_irr and use_dc:
            dx = np.diff(x)
            corr = np.full(lags.size, np.nan)
            if dx.std() > 0:
                for k, lag in enumerate(lags):
                    dp = np.diff(dc[a + lag:b + 1 + lag])
                    if dp.std() > 0:
                        corr[k] = np.corrcoef(dx, dp)[0, 1]
            if np.isfinite(corr).all():
                row["delta"] = _parabolic(corr, lags, maximise=True)
                row["delta_corr"] = float(corr.max())
        rows.append(row)
    cols = ["day", "tau_met", "clear_rel_rmse", "clear_ratio", "variability", "tau_dc",
            "dc_rel_rmse", "delta", "delta_corr"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    out["day"] = pd.to_datetime(out["day"])
    return out


# ── steps (resync events) ──────────────────────────────────────────────────


def coarse_steps(days: pd.Series, values: pd.Series, threshold: float,
                 neighbours: int = 7, merge_days: int = 20) -> list[pd.Timestamp]:
    """Days where the median of the next `neighbours` estimates differs from the median of the
    previous ones by more than `threshold`; candidates closer than `merge_days` are merged."""
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(days)).sort_index().dropna()
    k, cands = neighbours, []
    for i in range(k, len(s) - k + 1):
        jump = s.iloc[i:i + k].median() - s.iloc[i - k:i].median()
        if abs(jump) > threshold:
            cands.append((s.index[i], abs(jump)))
    steps, group = [], []
    for day, jump in cands:
        if group and (day - group[-1][0]).days > merge_days:
            steps.append(max(group, key=lambda g: g[1])[0])
            group = []
        group.append((day, jump))
    if group:
        steps.append(max(group, key=lambda g: g[1])[0])
    return steps


def refine_step(days: pd.Series, values: pd.Series, coarse: pd.Timestamp,
                radius: int = 40) -> pd.Timestamp:
    """Exact first day of the new regime: the split within +-radius days of `coarse` that
    minimises the summed absolute deviation from each side's median."""
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(days)).sort_index().dropna()
    lo, hi = coarse - pd.Timedelta(days=radius), coarse + pd.Timedelta(days=radius)
    s = s[(s.index >= lo) & (s.index <= hi)]
    best, best_cost = coarse, np.inf
    for split in s.index[2:-1]:
        left, right = s[s.index < split], s[s.index >= split]
        cost = (left - left.median()).abs().sum() + (right - right.median()).abs().sum()
        if cost < best_cost:
            best, best_cost = split, cost
    return best


# ── building and applying the correction ───────────────────────────────────


def smooth_within_segments(days: pd.Series, values: pd.Series, steps: list[pd.Timestamp],
                           calendar: pd.DatetimeIndex, window: int = 31,
                           min_obs: int = 4) -> pd.Series:
    """Rolling median per segment, interpolated over every calendar day of that segment."""
    edges = [calendar[0]] + sorted(steps) + [calendar[-1] + pd.Timedelta(days=1)]
    obs = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(days)).sort_index()
    out = pd.Series(np.nan, index=calendar)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        seg_cal = calendar[(calendar >= lo) & (calendar < hi)]
        seg = obs[(obs.index >= lo) & (obs.index < hi)].dropna()
        if len(seg) == 0 or len(seg_cal) == 0:
            continue
        daily = seg.groupby(level=0).median().reindex(seg_cal)
        med = daily.rolling(window, center=True, min_periods=min_obs).median()
        if med.notna().sum() == 0:
            med = pd.Series(seg.median(), index=seg_cal)
        out.loc[seg_cal] = med.interpolate(limit_direction="both").to_numpy()
    return out


def load_offsets(path) -> pd.DataFrame:
    """configs/clock_offsets.csv -> DataFrame indexed by local day (naive midnight)."""
    off = pd.read_csv(path, parse_dates=["day"]).set_index("day")
    off.index = pd.DatetimeIndex(off.index).normalize()
    return off


def apply_correction(df: pd.DataFrame, offsets: pd.DataFrame,
                     source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Re-stamp every tag-minute at true time.

    `offsets`: index = local calendar day (naive midnight), columns '<source>_<group>' [min].
    `source`: per tag-minute 'minute' or 'polled' (default: all 'minute'). Shifts are rounded to
    whole minutes. Minutes that land outside the grid or on an occupied minute are dropped; the
    result is on the original grid, with NaN where nothing landed.
    """
    day = pd.DatetimeIndex(df.index.tz_convert(TIMEZONE_LOCAL).date)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        group = clock_group(col)
        src = (source[col].to_numpy() if source is not None
               else np.full(len(df), "minute", dtype=object))
        shift = np.full(len(df), np.nan)
        for s in np.unique(src):
            m = src == s
            shift[m] = offsets[offset_column(s, group)].reindex(day[m]).to_numpy()
        if np.isnan(shift).any():
            raise ValueError(f"missing clock offset for {col!r} on some days")
        stamped = df.index - pd.to_timedelta(np.round(shift).astype(int), unit="min")
        moved = pd.Series(df[col].to_numpy(dtype=float), index=stamped)
        moved = moved[~moved.index.duplicated(keep="first")]
        out[col] = moved.reindex(df.index).to_numpy()
    return out

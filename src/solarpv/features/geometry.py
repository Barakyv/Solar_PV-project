"""Hourly sun geometry and clear-sky irradiance for stage 6 (docs/stage6_plan.md, S2).

Everything here is deterministic astronomy and a clear-sky model, so it is known at any
issuance time and can describe a future hour. Each hourly value is the mean of the 60 minute
values of the hour [t, t+1h), evaluated at the middle of each minute. Averaging minute values
(not evaluating at mid-hour) matches how the hourly AC mean is formed, which matters most at
sunrise and sunset, where geometry changes fast inside the hour.

Two planes exist in this project:
  sensor plane  20/180 (configs/site.private.yaml): frame of the clock correction (stage 4)
  panel plane   21/185 (configs/panel_orientation.yaml, A-04 re-measured, F-23): used here,
                for the theoretical power (target k) and the POA / aoi features
Clear sky: Ineichen with pvlib's Linke-turbidity climatology; transposition: Hay-Davies,
albedo 0.2 (the model used for clear-sky POA since stage 2).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import yaml

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.paths import CONFIGS_DIR
from solarpv.config.site import Site, load_site

PANEL_FILE = CONFIGS_DIR / "panel_orientation.yaml"
POA_PARTS = {"poa_global": "poa_global_clearsky", "poa_direct": "poa_direct_clearsky",
             "poa_diffuse": "poa_diffuse_clearsky", "poa_sky_diffuse": "poa_sky_diffuse_clearsky",
             "poa_ground_diffuse": "poa_ground_diffuse_clearsky"}


@dataclass(frozen=True)
class PanelPlane:
    tilt: float
    azimuth: float
    albedo: float
    model: str


def load_panel_plane(path: Path = PANEL_FILE) -> PanelPlane:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PanelPlane(float(cfg["tilt"]), float(cfg["azimuth"]), float(cfg["albedo"]),
                      str(cfg["transposition"]))


def minute_geometry(index: pd.DatetimeIndex, site: Site, plane: PanelPlane) -> pd.DataFrame:
    """Per minute (at its middle): sun position, clear-sky GHI/DNI/DHI, the clear-sky POA
    components on the panel plane, angle of incidence and relative air mass."""
    loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC",
                                  altitude=site.altitude_m)
    mid = index + pd.Timedelta(seconds=30)
    sp = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen", solar_position=sp)
    zen, azi = sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy()
    poa = pvlib.irradiance.get_total_irradiance(
        plane.tilt, plane.azimuth, zen, azi, cs["dni"].to_numpy(), cs["ghi"].to_numpy(),
        cs["dhi"].to_numpy(), dni_extra=np.asarray(pvlib.irradiance.get_extra_radiation(mid)),
        albedo=plane.albedo, model=plane.model)
    out = pd.DataFrame({
        "solar_zenith": zen, "solar_azimuth": azi,
        "sun_elevation": sp["apparent_elevation"].to_numpy(),
        "ghi_clearsky": cs["ghi"].to_numpy(), "dni_clearsky": cs["dni"].to_numpy(),
        "dhi_clearsky": cs["dhi"].to_numpy(),
        "aoi": np.asarray(pvlib.irradiance.aoi(plane.tilt, plane.azimuth, zen, azi)),
        # NaN when the sun is below the horizon: air mass is undefined there
        "airmass": np.asarray(pvlib.atmosphere.get_relative_airmass(zen)),
    }, index=index)
    for part, name in POA_PARTS.items():
        out[name] = np.nan_to_num(np.asarray(poa[part], dtype=float))  # 0 at night
    return out


def circular_mean_deg(values: pd.Series, groups) -> pd.Series:
    """Mean angle in degrees (0-360), so that 359 and 1 average to 0 and not 180."""
    rad = np.deg2rad(values.to_numpy())
    s = pd.Series(np.sin(rad), index=values.index).groupby(groups).mean()
    c = pd.Series(np.cos(rad), index=values.index).groupby(groups).mean()
    return np.rad2deg(np.arctan2(s, c)) % 360


def hourly_geometry(hours: pd.DatetimeIndex, site: Site | None = None,
                    plane: PanelPlane | None = None) -> pd.DataFrame:
    """Hourly means of the minute geometry for every hour in `hours` (left-labelled, UTC).
    Azimuth uses a circular mean; air mass is averaged over the minutes with the sun up
    (NaN for a night hour)."""
    site = site or load_site()
    plane = plane or load_panel_plane()
    minutes = pd.date_range(hours[0], hours[-1] + pd.Timedelta(minutes=59), freq="min")
    m = minute_geometry(minutes, site, plane)
    hour = m.index.floor("h")
    out = m.drop(columns=["solar_azimuth"]).groupby(hour).mean()  # pandas mean skips NaN
    out["solar_azimuth"] = circular_mean_deg(m["solar_azimuth"], hour)
    out.index.name = "timestamp_utc"
    return out.reindex(hours)


def daily_sun_times(hours: pd.DatetimeIndex, site: Site | None = None) -> pd.DataFrame:
    """Per hour: transit (solar noon), sunrise and sunset of the hour's local calendar day,
    as UTC timestamps (pvlib SPA, geometric sun centre)."""
    site = site or load_site()
    local_day = hours.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(set(local_day)))
    # pvlib expects local midnights of the days, tz-aware
    midnights = days.tz_localize(TIMEZONE_LOCAL)
    times = pvlib.solarposition.sun_rise_set_transit_spa(midnights, site.latitude,
                                                         site.longitude)
    times.index = days
    out = times.reindex(local_day)
    out.index = hours
    return out.apply(lambda col: pd.to_datetime(col).dt.tz_convert("UTC"))

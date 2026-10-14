"""Data-quality checks shared by stages 2-6.

production_centroid_offsets: on clear days, the power-weighted centre of the production curve
should coincide with solar noon if timestamps are true UTC and the panels face south (test T-08,
findings F-01 and F-13). Clear days are chosen from the smoothness and energy of the DC curve
alone, so the check makes no orientation assumption.
"""

import numpy as np
import pandas as pd
import pvlib

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.site import Site, load_site

MIN_ELEVATION = 15.0  # deg; avoids the eastern mountain horizon and low-sun effects
CLEAR_DAYS_PER_MONTH = 6


def _ten_minute(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return df[list(columns)].sum(axis=1).resample("10min").mean()


DC_COLUMNS = ("s01_inv1_DC_kW", "s01_inv2_DC_kW")


def clear_days(dc: pd.Series, elevation: pd.Series, per_month: int = CLEAR_DAYS_PER_MONTH):
    """Smooth, high-energy local days, chosen per calendar month."""
    local_day = dc.index.tz_convert(TIMEZONE_LOCAL).date
    frame = pd.DataFrame({"dc": dc.to_numpy(), "elev": elevation.to_numpy(), "day": local_day})
    stats = []
    for day, g in frame[frame["elev"] > MIN_ELEVATION].groupby("day"):
        p = g["dc"].to_numpy()
        if len(p) < 30 or np.isnan(p).any() or p.max() <= 0:
            continue
        stats.append((day, p.sum(), np.abs(np.diff(p, 2)).mean() / p.max()))
    s = pd.DataFrame(stats, columns=["day", "energy", "rough"])
    s["month"] = [(d.year, d.month) for d in s["day"]]
    keep = []
    for _, g in s.groupby("month"):
        g = g[g["energy"] >= g["energy"].quantile(0.6)]
        keep += list(g.nsmallest(per_month, "rough")["day"])
    return sorted(keep)


def production_centroid_offsets(
    df: pd.DataFrame,
    site: Site | None = None,
    columns: tuple[str, ...] = DC_COLUMNS,
    days: list | None = None,
) -> pd.DataFrame:
    """Minutes between the clear-day centroid of `columns` and solar noon, one row per day.

    Clear days are selected from DC power unless `days` is given, so different signals can be
    compared on the same days.
    """
    site = site or load_site()
    dc = _ten_minute(df, columns)
    centre = dc.index + pd.Timedelta(minutes=5)
    pos = pvlib.solarposition.get_solarposition(centre, site.latitude, site.longitude,
                                                altitude=site.altitude_m)
    elevation = pd.Series(pos["apparent_elevation"].to_numpy(), index=dc.index)
    if days is None:
        days = clear_days(_ten_minute(df, DC_COLUMNS), elevation)

    local_day = dc.index.tz_convert(TIMEZONE_LOCAL).date
    noon_utc = pd.DatetimeIndex(
        [pd.Timestamp(d, tz=TIMEZONE_LOCAL).tz_convert("UTC") for d in days])
    transit = pvlib.solarposition.sun_rise_set_transit_spa(
        noon_utc + pd.Timedelta(hours=12), site.latitude, site.longitude)["transit"]

    rows = []
    for day, noon in zip(days, transit, strict=True):
        sel = (local_day == day) & (elevation.to_numpy() > MIN_ELEVATION)
        w = dc[sel].clip(lower=0)
        t = (centre[sel] - centre[sel][0]).total_seconds().to_numpy()
        centroid = centre[sel][0] + pd.Timedelta(seconds=float((t * w.to_numpy()).sum()
                                                               / w.sum()))
        rows.append({"day": day, "offset_min": (centroid - noon).total_seconds() / 60})
    return pd.DataFrame(rows)

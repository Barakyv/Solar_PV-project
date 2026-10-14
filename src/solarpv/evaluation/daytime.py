"""Evaluation daytime mask (finding F-02, assumption A-12).

An hourly row labelled t (UTC, start of hour) is daytime when the true solar elevation at
t + 30 min exceeds 5 deg. With the Ketura coordinates this reproduces the original n exactly:
2,580 test hours, and 2,510 / 2,512 / 2,514 on the surviving LSTM timestamps. The counts are the
same at the 40 MW field centre, at the nearby 4.95 MW field and at the old kibbutz point.
"""

import numpy as np
import pandas as pd
import pvlib

from solarpv.config.constants import EVAL_SUN_OFFSET_MIN, SUN_ELEV_MIN
from solarpv.config.site import Site, load_site


def sun_elevation_mid_hour(index_utc: pd.DatetimeIndex, site: Site | None = None) -> np.ndarray:
    """True (not refraction-corrected) solar elevation at the middle of each hourly row."""
    if index_utc.tz is None or str(index_utc.tz) != "UTC":
        raise ValueError("index must be tz-aware UTC")
    site = site or load_site()
    mid = index_utc + pd.Timedelta(minutes=EVAL_SUN_OFFSET_MIN)
    pos = pvlib.solarposition.get_solarposition(
        mid, site.latitude, site.longitude, altitude=site.altitude_m
    )
    return pos["elevation"].to_numpy()


def daytime_mask(index_utc: pd.DatetimeIndex, site: Site | None = None) -> np.ndarray:
    """Boolean mask of evaluation daytime hours."""
    return sun_elevation_mid_hour(index_utc, site) > SUN_ELEV_MIN

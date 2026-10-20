"""Feature contract: the exact model inputs (PROJECT_MASTER.md §6.6, finding F-05).

Derived from the 76 features the original XGB-Opt actually used (submission_evidence/07_shap/
shap_rank_stability.csv) minus theoretical_power (decision D-03) and sun_azimuth (Q-05, stage 6:
identical to solar_azimuth), so 74. Rolling features keep their names but are built per horizon
(decision D-04). GFS features (stage 6b) are appended only if they pass decision D-12.
"""

from solarpv.config.constants import get_features

ROLLING_SOURCES = ("s01_inv1", "s01_inv2", "site_total")
LAG_SOURCES = (
    "s01_inv1_AC_kW",
    "s01_inv2_AC_kW",
    "site_total_AC_kW",
    "irradiance",
    "mod_temp1",
    "mod_temp2",
    "ambient_temp",
)
LAG_DAYS = (1, 2, 3, 7, 14)
ROLLING_WINDOWS_DAYS = (3, 7)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "clearsky": (
        "ghi_clearsky",
        "dhi_clearsky",
        "dni_clearsky",
        "sun_elevation",
        "poa_global_clearsky",
        "poa_direct_clearsky",
        "poa_diffuse_clearsky",
        "poa_sky_diffuse_clearsky",
        "poa_ground_diffuse_clearsky",
    ),
    "rolling_stats": tuple(
        f"{src}_rolling_{stat}_{w}d_same_tod"
        for src in ROLLING_SOURCES
        for stat in ("mean", "std")
        for w in ROLLING_WINDOWS_DAYS
    ),
    "lag_power": tuple(f"{src}_lag_{d}d" for src in LAG_SOURCES for d in LAG_DAYS),
    "time_encodings": (
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "month",
        "season",
        "is_weekend",
        "hour_local",
    ),
    "other": (
        "aoi",
        "solar_azimuth",
        "solar_zenith",
        "airmass",
        "doy_sin",
        "doy_cos",
        "day_length",
        "time_to_solar_noon",
        "daylight_fraction",
        "day_of_week",
    ),
}

FEATURES: tuple[str, ...] = tuple(f for group in FEATURE_GROUPS.values() for f in group)

# Present in the original model inputs but removed on purpose
REMOVED_FROM_LEGACY = {
    "theoretical_power": "listed as leakage (category 4), SHAP importance 0 (D-03)",
    "sun_azimuth": "identical to solar_azimuth (both the pvlib azimuth); in the original it came "
                   "from the lost preprocessing step, on the wrong time base (F-01). Q-05 closed "
                   "in stage 6",
}

# Prefix reserved for leakage-safe NWP features from the GFS archive (stage 6b, D-12)
GFS_PREFIX = "fcst_gfs_"


def features_for_horizon(horizon: int, extra: tuple[str, ...] = ()) -> list[str]:
    """Model inputs for D+horizon: the contract (plus approved extras) minus unavailable lags."""
    return get_features(FEATURES + tuple(extra), horizon)


def group_of(feature: str) -> str:
    """Feature group as used in the SHAP group-importance tables."""
    for group, names in FEATURE_GROUPS.items():
        if feature in names:
            return group
    if feature.startswith(GFS_PREFIX):
        return "nwp_gfs"
    raise KeyError(f"{feature!r} is not part of the feature contract")

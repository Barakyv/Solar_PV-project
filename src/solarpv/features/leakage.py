"""Leakage rules: what may never enter a model matrix (PROJECT_MASTER.md §6.7, leakage_audit.md).

Two complementary guards:
  * same-timestamp information: exact names and name patterns below (this module);
  * information after issuance time T0: horizon lag rules (constants.get_features) and the
    perturbation tests T-09 / T-18, which catch anything the name rules cannot.
Lagged versions of measurements (``*_lag_<n>d``) and rolling statistics are not blocked here;
their horizon safety is enforced by LAG_DROP and the perturbation tests.
"""

import re
from collections.abc import Iterable

# Exact names. Sources: leakage_audit.md (20), LEAKY_EXCL in run_ablation_study.py (17),
# alternative names in REPORT_EVIDENCE_MAP.md, and raw same-timestamp measurements.
EXCL_EXACT: frozenset[str] = frozenset(
    {
        # targets and target duplicates (category 1)
        "target_k",
        "site_total_AC_kW",
        "actual_power_kW",
        "sum_inv_ac_kW",
        "performance_ratio",
        "s01_inv1_AC_kW",
        "s01_inv2_AC_kW",
        "s01_inv1_DC_kW",
        "s01_inv2_DC_kW",
        "site_total_DC_kW",
        "actual_power_dc_kW",
        "site_total_AC_power_kW",
        "inv1_AC_power_kW",
        "inv2_AC_power_kW",
        "k_valid",
        # old reanalysis NWP and derivatives (category 2)
        "fcst_ghi",
        "fcst_dni",
        "fcst_dhi",
        "fcst_temp_2m",
        "fcst_cloud_cover",
        "fcst_wind_speed",
        "poa_global_forecast",
        "poa_direct_forecast",
        "poa_diffuse_forecast",
        "wind_speed_squared",
        "mod_temp_forecast",
        # same-timestamp sensors (category 3 and raw tags)
        "mod_temp1",
        "mod_temp2",
        "ambient_temp",
        "site_ambient_temp",
        "irradiance",
        "pyranometer",
        "ghi_measured",
        "irradiance_proxy",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "soil",
        "soiling01",
        "soiling02",
        # derived from same-timestamp power (category 4 and ablation list)
        "theoretical_power",
        "site_loading",
        "inv1_efficiency",
        "inv2_efficiency",
        "site_avg_efficiency",
        "site_power_per_irr",
        "_daylight",
    }
)

# Patterns for same-timestamp columns that may appear under other names.
EXCL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^fcst_(?!gfs_)",  # any NWP except the leakage-checked GFS archive features
        r"_forecast$",
        r"^iec_",  # grid-operator setpoints / mode
        r"_(AC|DC)_kW$",
        r"_kVar$",
        r"_PF$",
        r"_state$",
        r"_is_running$",
        r"_efficiency$",
        r"_flag(_auto)?$",  # cleaning / QA flags are computed from same-timestamp data
        r"^has_",
    )
)

_HISTORY_SUFFIX = re.compile(r"(_lag_\d+d|_rolling_(mean|std)_\d+d_same_tod)$")


def is_leaky(column: str) -> bool:
    """True if the column carries same-timestamp information and must never be a model input."""
    if column in EXCL_EXACT:
        return True
    if _HISTORY_SUFFIX.search(column):
        return False
    return any(p.search(column) for p in EXCL_PATTERNS)


def leaky_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if is_leaky(c)]


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise if any model input column is leaky (test T-01 uses this on every model matrix)."""
    bad = leaky_columns(columns)
    if bad:
        raise ValueError(f"Leaky columns in model input: {sorted(bad)}")

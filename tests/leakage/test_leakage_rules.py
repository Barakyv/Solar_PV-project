"""T-01 (rule level): the contract is leak-free and the blocklist covers every known leaky name."""

import pytest

from solarpv.features.contract import FEATURES, GFS_PREFIX
from solarpv.features.leakage import assert_no_leakage, is_leaky

# Every name listed as leakage anywhere in the original project
ORIGINAL_LEAKY = [
    # leakage_audit.md
    "actual_power_kW", "performance_ratio", "s01_inv1_AC_kW", "s01_inv2_AC_kW",
    "s01_inv1_DC_kW", "s01_inv2_DC_kW", "sum_inv_ac_kW", "fcst_ghi", "fcst_dni", "fcst_dhi",
    "fcst_temp_2m", "fcst_cloud_cover", "fcst_wind_speed", "poa_global_forecast",
    "poa_direct_forecast", "poa_diffuse_forecast", "wind_speed_squared", "mod_temp1",
    "ambient_temp", "theoretical_power",
    # LEAKY_EXCL in run_ablation_study.py
    "target_k", "site_total_AC_kW", "_daylight", "site_loading", "inv1_efficiency",
    "inv2_efficiency", "site_avg_efficiency", "site_power_per_irr", "site_total_DC_kW",
    # alternative names in REPORT_EVIDENCE_MAP.md
    "site_total_AC_power_kW", "inv1_AC_power_kW", "inv2_AC_power_kW", "ghi_measured",
    "irradiance_proxy", "mod_temp_forecast",
    # same-timestamp columns seen in the old 05e feature sets and notebooks
    "irradiance", "mod_temp2", "has_statistical_outlier", "s01_inv1_is_running",
    "ac_clip_flag", "ac_clip_flag_auto", "curtail_flag",
]


@pytest.mark.parametrize("name", ORIGINAL_LEAKY)
def test_known_leaky_names_are_blocked(name):
    assert is_leaky(name)


def test_contract_is_leak_free():
    assert_no_leakage(FEATURES)


@pytest.mark.parametrize(
    "name", ["irradiance_lag_1d", "s01_inv1_AC_kW_lag_2d", "site_total_rolling_mean_7d_same_tod"]
)
def test_history_features_are_not_blocked_by_name(name):
    assert not is_leaky(name)


def test_gfs_features_are_allowed_but_old_nwp_is_not():
    assert not is_leaky(f"{GFS_PREFIX}ghi")
    assert is_leaky("fcst_ghi")
    assert is_leaky("fcst_anything_new")


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError, match="mod_temp1"):
        assert_no_leakage(["hour_sin", "mod_temp1"])

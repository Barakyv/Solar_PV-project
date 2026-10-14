"""T-10: the feature contract (finding F-05, decision D-03)."""

import pandas as pd

from solarpv.features.contract import (
    FEATURE_GROUPS,
    FEATURES,
    REMOVED_FROM_LEGACY,
    features_for_horizon,
    group_of,
)

EXPECTED_GROUP_SIZES = {
    "clearsky": 9,
    "rolling_stats": 12,
    "lag_power": 35,
    "time_encodings": 8,
    "other": 11,
}


def test_contract_has_75_unique_features():
    assert len(FEATURES) == 75
    assert len(set(FEATURES)) == 75


def test_group_sizes():
    assert {g: len(v) for g, v in FEATURE_GROUPS.items()} == EXPECTED_GROUP_SIZES


def test_contract_equals_original_inputs_minus_removed(evidence_dir):
    """The original XGB inputs (SHAP table) minus the deliberately removed features."""
    shap = pd.read_csv(evidence_dir / "07_shap" / "shap_rank_stability.csv", index_col=0)
    original = set(shap.index)
    assert len(original) == 76
    assert set(FEATURES) == original - set(REMOVED_FROM_LEGACY)


def test_horizon_drops():
    d1, d2, d3 = (set(features_for_horizon(h)) for h in (1, 2, 3))
    assert len(d1) == 75
    assert len(d2) == 75 - 7  # lag_1d of 7 sources
    assert len(d3) == 75 - 14  # lag_1d and lag_2d of 7 sources
    assert not any(f.endswith("_lag_1d") for f in d2 | d3)
    assert not any(f.endswith("_lag_2d") for f in d3)
    assert all(f in d3 for f in FEATURES if f.endswith("_lag_14d"))


def test_every_feature_has_a_group():
    for f in FEATURES:
        assert group_of(f) in FEATURE_GROUPS

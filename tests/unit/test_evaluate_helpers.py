"""T-04 (one truth across models), T-12 (route truth is actual AC), loading and persistence."""

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import AC_CAPACITY_KW
from solarpv.evaluation.evaluate import (
    check_truth_consistency,
    load_predictions,
    route_comparison,
)
from solarpv.models.baselines import persistence

TS = pd.date_range("2024-10-01 06:00", periods=6, freq="h", tz="UTC")


def _frame(model, target, y_true, y_pred, horizon=1):
    return pd.DataFrame({"timestamp": TS, "y_true": y_true, "y_pred": y_pred,
                         "target": target, "horizon": horizon, "model": model, "seed": 42})


def test_truth_consistency_passes_and_fails():
    y = np.arange(6.0)
    ok = pd.concat([_frame("a", "direct", y, y), _frame("b", "direct", y, y + 1)])
    check_truth_consistency(ok)
    bad = pd.concat([_frame("a", "direct", y, y), _frame("b", "direct", y + 0.5, y)])
    with pytest.raises(ValueError, match="y_true differs"):
        check_truth_consistency(bad)


def test_fixture_truth_is_identical_across_horizons(lstm_fixture):
    check_truth_consistency(lstm_fixture)


def test_route_comparison_scores_both_routes_against_actual_ac():
    ac = np.array([100.0, 800.0, 1500.0, 2000.0, 1200.0, 300.0])
    theo = pd.Series([200.0, 1000.0, 2000.0, 2500.0, 1500.0, 400.0], index=TS)
    k_pred = ac / theo.to_numpy() * 1.1  # indirect route overshoots by 10 %
    preds = pd.concat([
        _frame("m", "indirect", ac / theo.to_numpy(), k_pred),
        _frame("m", "direct", ac, ac * 0.8),  # direct route undershoots by 20 %
    ])
    preds["daytime"] = True
    out = route_comparison(preds, theo).set_index("route")
    assert out.loc["indirect_to_power", "n"] == out.loc["direct", "n"] == 6
    expected_ind = np.sqrt(np.mean((ac - np.clip(k_pred * theo.to_numpy(), 0, AC_CAPACITY_KW))
                                   ** 2)) / AC_CAPACITY_KW * 100
    expected_dir = np.sqrt(np.mean((0.2 * ac) ** 2)) / AC_CAPACITY_KW * 100
    assert out.loc["indirect_to_power", "nRMSE_pct"] == pytest.approx(expected_ind)
    assert out.loc["direct", "nRMSE_pct"] == pytest.approx(expected_dir)
    assert out.loc["direct", "advantage_pp"] == pytest.approx(expected_dir - expected_ind)


def test_load_predictions_rejects_duplicates(tmp_path):
    df = _frame("a", "direct", np.arange(6.0), np.arange(6.0))
    pd.concat([df, df.iloc[:1]]).to_csv(tmp_path / "p.csv", index=False)
    with pytest.raises(ValueError, match="duplicated"):
        load_predictions(tmp_path / "p.csv")


def test_load_predictions_adds_missing_seed(tmp_path):
    _frame("a", "direct", np.arange(6.0), np.arange(6.0)).drop(columns="seed").to_csv(
        tmp_path / "p.csv", index=False)
    assert (load_predictions(tmp_path / "p.csv")["seed"] == -1).all()


def test_persistence_uses_value_h_days_earlier():
    idx = pd.date_range("2024-10-01", periods=24 * 5, freq="h", tz="UTC")
    truth = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    for h in (1, 2, 3):
        pred = persistence(truth, h).dropna()  # full grid; NaN where nothing earlier exists
        assert pred.index[0] == idx[24 * h]
        assert (pred.to_numpy() == truth.loc[pred.index].to_numpy() - 24 * h).all()

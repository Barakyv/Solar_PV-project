import numpy as np
import pytest

from solarpv.evaluation.metrics import (
    bootstrap_nrmse_ci,
    circular_block_indices,
    mae,
    nrmse_pct,
    point_metrics,
    r2,
    rmse,
    skill_score,
)

Y = np.array([1.0, 2.0, 3.0, 4.0])
P = np.array([1.5, 2.0, 2.0, 4.5])  # errors: -0.5, 0, 1, -0.5


def test_point_metrics_by_hand():
    assert rmse(Y, P) == pytest.approx(np.sqrt((0.25 + 0 + 1 + 0.25) / 4))
    assert mae(Y, P) == pytest.approx((0.5 + 0 + 1 + 0.5) / 4)
    assert r2(Y, P) == pytest.approx(1 - 1.5 / 5.0)
    assert nrmse_pct(Y, P, 2.0) == pytest.approx(rmse(Y, P) / 2.0 * 100)
    m = point_metrics(Y, P, 2.0)
    assert m["n"] == 4 and m["RMSE"] == pytest.approx(rmse(Y, P))


def test_skill_score():
    assert skill_score(0.8, 1.0) == pytest.approx(0.2)


def test_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        rmse(np.array([1.0, np.nan, 3.0, 4.0]), P)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        rmse(np.array([1.0, 2.0]), P)


def test_circular_block_indices():
    idx = circular_block_indices(10, 4, 3, np.random.default_rng(0))
    assert idx.shape == (3, 10)
    assert idx.min() >= 0 and idx.max() < 10
    # within a block, indices are consecutive modulo n
    assert all(((idx[r, 1:4] - idx[r, 0:3]) % 10 == 1).all() for r in range(3))


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.random(500)
    p = y + rng.normal(0, 0.1, 500)
    ci1 = bootstrap_nrmse_ci(y, p, 0.5, reps=300)
    ci2 = bootstrap_nrmse_ci(y, p, 0.5, reps=300)
    assert ci1 == ci2
    assert ci1[0] < nrmse_pct(y, p, 0.5) < ci1[1]

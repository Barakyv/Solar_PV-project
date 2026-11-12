"""T-92: ordered quantiles (D-33, X53).

Fails when: sorting changes a row that was already in order, leaves a row out of order, touches
the point forecast or any other column, raises the quantile loss for some truth value, orders the
columns by their position instead of their level, or accepts a frame without quantiles.
"""

import numpy as np
import pandas as pd
import pytest

from solarpv.evaluation.quantiles import (
    coverage,
    crossed,
    ordered,
    pinball,
    quantile_columns,
)


def frame(rows, cols=("q0.1", "q0.5", "q0.9")):
    f = pd.DataFrame(rows, columns=list(cols))
    f["y_pred"] = f["q0.5"]  # the point forecast is the 0.5 head
    f["timestamp"] = pd.date_range("2024-02-01", periods=len(f), freq="h", tz="UTC")
    return f


def test_rows_in_order_are_unchanged_and_crossed_rows_are_sorted():
    f = frame([[1.0, 2.0, 3.0], [5.0, 1.0, 3.0]])
    assert crossed(f).tolist() == [False, True]
    o = ordered(f)
    assert o.loc[0, ["q0.1", "q0.5", "q0.9"]].tolist() == [1.0, 2.0, 3.0]
    assert o.loc[1, ["q0.1", "q0.5", "q0.9"]].tolist() == [1.0, 3.0, 5.0]
    assert not crossed(o).any()


def test_the_point_forecast_and_other_columns_are_untouched():
    f = frame([[5.0, 1.0, 3.0]])
    o = ordered(f)
    assert o.loc[0, "y_pred"] == 1.0  # the raw 0.5 head, although the sorted median is 3.0
    assert o["timestamp"].equals(f["timestamp"])
    assert f.loc[0, "q0.1"] == 5.0  # the input frame is not modified


def test_sorting_never_raises_the_quantile_loss():
    rng = np.random.default_rng(7)
    cols = ("q0.02", "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q0.98")
    f = frame(rng.normal(size=(500, len(cols))), cols)
    y = rng.normal(size=500)
    for i in range(len(f)):  # row by row, for every truth value
        row = f.iloc[[i]]
        assert pinball(y[[i]], ordered(row)) <= pinball(y[[i]], row) + 1e-12
    assert pinball(y, ordered(f)) < pinball(y, f)  # strictly lower when rows cross


def test_columns_are_ordered_by_level_not_position():
    f = frame([[1.0, 2.0, 3.0]], cols=("q0.9", "q0.1", "q0.5"))
    assert quantile_columns(f) == ["q0.1", "q0.5", "q0.9"]
    o = ordered(f)
    assert (o.loc[0, "q0.1"], o.loc[0, "q0.5"], o.loc[0, "q0.9"]) == (1.0, 2.0, 3.0)


def test_coverage_counts_the_central_interval():
    f = frame([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert coverage([1.0, 2.0, -0.1, 2.5], f) == 0.5


def test_a_frame_without_quantiles_is_refused():
    with pytest.raises(ValueError):
        quantile_columns(pd.DataFrame({"y_pred": [1.0], "quality": [2.0]}))

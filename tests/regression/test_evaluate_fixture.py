"""T-15: the evaluation code reproduces the old benchmark from the old LSTM predictions.

The old predictions come from the buggy pipeline (F-01); they are used only to prove that the
metric code itself is right, so any later change in reported numbers comes from the data and
models, not from the evaluation. Tolerances: PROJECT_MASTER.md §8.3.
"""

import pandas as pd
import pytest

from solarpv.config.constants import MEAN_DAYTIME_K_LEGACY
from solarpv.config.site import load_site
from solarpv.evaluation.evaluate import add_daytime, benchmark_table, seasonal_table

pytestmark = pytest.mark.needs_private_site

TOL_NRMSE = {"indirect": 1e-3, "direct": 5e-3}  # pp
TOL_R2 = 5e-4
TOL_CI = 0.3  # pp; the old bootstrap RNG is unknown


@pytest.fixture(scope="module")
def evaluated(lstm_fixture):
    preds = lstm_fixture.copy()
    preds["seed"] = -1
    preds = add_daytime(preds, load_site())
    bench = benchmark_table(preds, MEAN_DAYTIME_K_LEGACY).set_index(["target", "horizon"])
    seasonal = seasonal_table(preds, MEAN_DAYTIME_K_LEGACY)
    return bench, seasonal


@pytest.fixture(scope="module")
def old_benchmark(evidence_dir):
    old = pd.read_csv(evidence_dir / "01_benchmark" / "benchmark_summary.csv")
    return old[old["model"] == "lstm_sota"].set_index(["target", "horizon"])


@pytest.mark.parametrize("target", ["indirect", "direct"])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_benchmark_rows_match(evaluated, old_benchmark, target, horizon):
    new = evaluated[0].loc[(target, horizon)]
    old = old_benchmark.loc[(target, horizon)]
    assert new["n"] == old["n"]
    assert new["nRMSE_pct"] == pytest.approx(old["nRMSE_pct"], abs=TOL_NRMSE[target])
    assert new["RMSE"] == pytest.approx(old["RMSE"], rel=5e-4)
    assert new["MAE"] == pytest.approx(old["MAE"], rel=1e-3)
    assert new["R2"] == pytest.approx(old["R2"], abs=TOL_R2)
    assert new["nRMSE_CI_lo"] == pytest.approx(old["nRMSE_CI_lo"], abs=TOL_CI)
    assert new["nRMSE_CI_hi"] == pytest.approx(old["nRMSE_CI_hi"], abs=TOL_CI)


def test_seasonal_rows_match(evaluated, evidence_dir):
    """Seasons by local month; one boundary hour differs from the old file (n +-1)."""
    old = pd.read_csv(evidence_dir / "04_seasonal" / "seasonal_metrics.csv")
    old = old[old["model"] == "lstm_sota"].set_index(["target", "horizon", "season"])
    new = evaluated[1].set_index(["target", "horizon", "season"])
    for key, row in old.iterrows():
        assert abs(new.loc[key, "n"] - row["n"]) <= 1
        assert new.loc[key, "nRMSE_pct"] == pytest.approx(row["nRMSE_pct"], abs=0.05)

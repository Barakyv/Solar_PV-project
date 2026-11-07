"""T-55: the stage-9 runner on the real tables (docs/stage9_plan.md §9, §10.3).

- A dev run of `baselines,evaluate` reproduces the committed stage-8B validation tables
  (`outputs/07_evaluation/val/`) exactly, although the runner never loads a test row, takes the
  references from its own prediction files and checks every input against its stage manifest.
- A smoke run covers every daytime hour of its 28-day window, which contains the spring DST day.
- Neither run writes outside its run folder: the committed evaluation outputs and the test-access
  log are unchanged.

Skipped when the stage-6 tables are not present (they are not in git).
"""

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.paths import OUTPUTS_DIR
from solarpv.config.provenance import sha256
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.truth import FEATURES_FILE, load_truth
from solarpv.pipeline import steps
from solarpv.pipeline.config import evaluation_windows
from solarpv.pipeline.runner import Runner

pytestmark = [pytest.mark.skipif(not FEATURES_FILE.exists(), reason="stage 6 output not present"),
              pytest.mark.needs_private_site, pytest.mark.slow]
COMMITTED = OUTPUTS_DIR / "07_evaluation" / "val"
TABLES = ("benchmark_summary", "benchmark_views", "dm_tests", "seasonal_metrics",
          "common_mask_summary", "common_mask_sensitivity", "route_comparison",
          "seed_variability_summary", "dm_tests_kw")


def _snapshot():
    files = sorted((OUTPUTS_DIR / "07_evaluation").rglob("*"))
    return {p: sha256(p) for p in files if p.is_file()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    before = _snapshot()
    seen = []
    real = steps.load_truth

    def spy(path, exclude_splits=()):
        out = real(path, exclude_splits=exclude_splits)
        seen.append(set(out["split"]))
        return out
    mp = pytest.MonkeyPatch()
    mp.setattr(steps, "load_truth", spy)
    root = tmp_path_factory.mktemp("runs")
    try:
        r = Runner(runs_dir=root, echo=False)
        codes = {mode: r.run(mode, ["baselines", "evaluate"], command=f"T-55 {mode}")
                 for mode in ("dev", "smoke")}
    finally:
        mp.undo()
    dirs = {d.name.rsplit("_", 1)[-1]: d for d in root.iterdir()}
    return {"codes": codes, "dirs": dirs, "seen": seen, "before": before,
            "after": _snapshot()}


def test_both_runs_pass_without_loading_a_test_row(runs):
    assert runs["codes"] == {"dev": 0, "smoke": 0}
    assert runs["seen"] and all("test" not in s for s in runs["seen"])


def test_dev_run_reproduces_the_stage_8b_validation_tables(runs):
    out = runs["dirs"]["dev"] / "evaluate" / "val"
    for name in TABLES:
        a, b = pd.read_csv(COMMITTED / f"{name}.csv"), pd.read_csv(out / f"{name}.csv")
        assert list(a.columns) == list(b.columns) and len(a) == len(b), name
        num = a.select_dtypes("number").columns
        np.testing.assert_allclose(b[num].to_numpy(float), a[num].to_numpy(float),
                                   rtol=0, atol=1e-6, equal_nan=True, err_msg=name)
        other = [c for c in a.columns if c not in num]
        pd.testing.assert_frame_equal(a[other], b[other], obj=name)


def test_smoke_window_is_fully_covered_and_contains_the_dst_day(runs):
    (window,) = evaluation_windows("smoke")
    truth = load_truth(exclude_splits=("test",))
    inside = truth[(truth.index >= window.start_ts) & (truth.index < window.end_ts)]
    assert (inside["split"] == "val").all() and len(inside) == 28 * 24 - 1
    assert pd.Timestamp("2024-03-29").date() in set(inside.index.tz_convert(TIMEZONE_LOCAL).date)
    bench = pd.read_csv(runs["dirs"]["smoke"] / "evaluate" / "val" / "benchmark_summary.csv")
    for target in ("indirect", "direct"):  # truth columns as renamed by load_truth
        n = int((inside["daytime"] & inside[target].notna()).sum())
        assert (bench.loc[bench["target"] == target, "n"] == n).all()
    clear = bench[bench["model"] == "clearsky_k"].groupby("horizon")["nRMSE_kW_pct"]
    assert (clear.max() - clear.min()).abs().max() < 1e-9  # same kW forecast in both routes


def test_nothing_is_written_outside_the_run_folder(runs):
    assert runs["before"] == runs["after"]
    assert not E.TEST_ACCESS_LOG.exists() or E.TEST_ACCESS_LOG in runs["before"]

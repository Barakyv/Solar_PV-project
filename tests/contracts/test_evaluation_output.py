"""T-41: the stage-8B evaluation on the real tables (docs/stage8b_plan.md §5).

The run evaluates the three references on validation only. It checks:
- the truth equals the feature table;
- daytime equals the table's flag (F-02);
- every reference is scored on every validation daytime hour with truth;
- the views follow the flags;
- the clear-sky route advantage is 0;
- no test row is read;
- the committed outputs are current.

Skipped when stage 6 has not been run (the parquet files are not in git).
"""

import json

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import HORIZONS, mean_daytime_k
from solarpv.config.paths import OUTPUTS_DIR
from solarpv.config.provenance import sha256
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.truth import FEATURES_FILE, load_truth

pytestmark = [pytest.mark.skipif(not FEATURES_FILE.exists(), reason="stage 6 output not present"),
              pytest.mark.needs_private_site]
COMMITTED = OUTPUTS_DIR / "07_evaluation" / "val"


@pytest.fixture(scope="module")
def truth():
    return load_truth()


@pytest.fixture(scope="module")
def tables(truth, tmp_path_factory):
    log_before = E.TEST_ACCESS_LOG.read_bytes() if E.TEST_ACCESS_LOG.exists() else None
    out = E.run([], tmp_path_factory.mktemp("eval"), mean_daytime_k(), truth=truth,
                split="val", ci=False)
    log_after = E.TEST_ACCESS_LOG.read_bytes() if E.TEST_ACCESS_LOG.exists() else None
    assert log_before == log_after  # a validation run never touches the test log
    return out


def test_truth_is_the_feature_table(truth):
    raw = pd.read_parquet(FEATURES_FILE, columns=["target_k", "site_total_AC_kW", "theo_kw",
                                                  "daytime", "split"])
    np.testing.assert_array_equal(truth["indirect"].to_numpy(), raw["target_k"].to_numpy())
    np.testing.assert_array_equal(truth["direct"].to_numpy(), raw["site_total_AC_kW"].to_numpy())
    np.testing.assert_array_equal(truth["theo_kw"].to_numpy(), raw["theo_kw"].to_numpy())
    for h in (2, 3):  # the targets are the same in every horizon table (T-33)
        other = pd.read_parquet(FEATURES_FILE.with_name(f"features_1H_D{h}.parquet"),
                                columns=["target_k", "site_total_AC_kW"])
        pd.testing.assert_frame_equal(other, raw[["target_k", "site_total_AC_kW"]])


def test_every_reference_is_scored_on_every_validation_daytime_hour(tables, truth):
    bench = tables["benchmark_summary"]
    val_day = truth[(truth["split"] == "val") & truth["daytime"]]
    assert len(bench) == 3 * 2 * len(HORIZONS)
    for target in ("indirect", "direct"):
        n = int(val_day[target].notna().sum())
        assert (bench.loc[bench["target"] == target, "n"] == n).all()
    assert (bench.loc[bench["model"] != "persistence", "n_skill"] == bench.loc[
        bench["model"] != "persistence", "n"]).all()  # A1: the reference covers every hour


def test_views_follow_the_flags(tables, truth):
    views = tables["benchmark_views"]
    day = truth[(truth["split"] == "val") & truth["daytime"] & truth["direct"].notna()]
    row = views[(views["model"] == "climatology") & (views["target"] == "direct")
                & (views["horizon"] == 1)].set_index("view")
    assert row.loc["all", "n"] == len(day)
    assert row.loc["normal_operation", "n"] == int((~day["op_limited_flag"]).sum())
    assert row.loc["source_minute", "n"] == int((day["hour_source"] == "minute").sum())


def test_clearsky_route_advantage_is_zero(tables):
    route = tables["route_comparison"]
    clear = route[(route["model"] == "clearsky_k") & (route["route"] == "direct")]
    assert len(clear) == len(HORIZONS)
    assert clear["advantage_pp"].abs().max() < 1e-9
    assert (route.groupby(["model", "horizon"])["n"].nunique() == 1).all()  # paired (E2)


def test_persistence_does_not_beat_the_cliper_reference(tables):
    bench = tables["benchmark_summary"]
    assert (bench.loc[bench["model"] == "persistence", "skill_cliper"] <= 0.02).all()


def test_committed_outputs_are_current():
    manifest = json.loads((COMMITTED / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "val"
    assert manifest["mean_daytime_k"] == mean_daytime_k()
    assert manifest["truth_sha256"] == sha256(FEATURES_FILE), \
        "run: python -m solarpv.evaluation.evaluate"

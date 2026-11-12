"""T-93 (docs/stage9_plan.md §20.5; M32-M34, M36): `scripts/reproduce_paper_results.py`.

- the plan starts from the raw data and covers every stage of docs/data_processing_record.md §19
  up to the two development runs, with all six models;
- the comparison passes on equal content with other bytes (a parquet written differently) and
  fails on a changed value; columns of times are left out; JSON differing only in volatile keys
  is equal; a differing decision file under configs/ fails;
- the script installs nothing (no pip, conda or mamba install command).
"""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from solarpv.config.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "reproduce_paper_results.py"


@pytest.fixture(scope="module")
def R():
    spec = importlib.util.spec_from_file_location("reproduce_paper_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def test_the_plan_runs_every_stage_from_the_raw_data(R):
    names = [s.name for s in R.plan({"neural": True})]
    assert names[:2] == ["checksums", "ingest"]
    for stage in ("clock", "preprocess", "clean", "orientation", "features", "decide_k",
                  "features_final", "gfs", "nwp_check", "evaluate", "fixture", "sequence_data",
                  "dev_xgb", "dev_neural"):
        assert stage in names
    assert names.index("clean") < names.index("features") < names.index("gfs") \
        < names.index("dev_xgb")
    models = R.XGB_STEPS.split(",") + R.NEURAL_STEPS.split(",")
    for m in ("xgb", "bilstm", "tft96", "tft168", "tftbase", "patchtst", "baselines", "select"):
        assert m in models
    assert "dev_neural" not in [s.name for s in R.plan({"neural": False})]
    optional = [s.name for s in R.plan({"neural": True, "with_eda": True,
                                        "rerun_selection": True})]
    assert {"eda", "quality", "selection"} <= set(optional)
    sel = next(s for s in R.plan({"rerun_selection": True}) if s.name == "selection")
    assert "--skip-validation" in sel.args  # never a new validation look


def test_the_plan_prints_and_writes_nothing(R, capsys, tmp_path):
    assert R.main(["--plan"]) == 0
    out = capsys.readouterr().out
    assert "dev_neural" in out and "reads validation" in out


def test_parquet_with_other_bytes_and_the_same_content_is_equal(R, tmp_path):
    df = pd.DataFrame({"a": np.arange(100.0), "b": list("xy" * 50)},
                      index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"))
    ref, new = tmp_path / "ref.parquet", tmp_path / "new.parquet"
    df.to_parquet(ref, compression="snappy")
    df.to_parquet(new, compression="gzip")
    assert R.sha256(ref) != R.sha256(new)
    row = R.compare_file(ref, new)
    assert row["equal"] is True and row["kind"] == "table"
    changed = df.copy()
    changed.iloc[50, 0] += 1e-9  # relative 2e-11: floating-point noise, reported, not failed
    changed.to_parquet(new)
    row = R.compare_file(ref, new)
    assert row["equal"] is False and row["float_noise"] is True and row["cells"] == 1
    assert not R.failing([{"path": "data/x.parquet", **row}])
    changed.iloc[60, 0] += 1.0  # a real difference fails
    changed.to_parquet(new)
    row = R.compare_file(ref, new)
    assert row["float_noise"] is False and row["max_abs"] == pytest.approx(1.0)
    assert R.failing([{"path": "data/x.parquet", **row}])
    missing = df.copy()
    missing.iloc[10, 0] = np.nan  # a value missing on one side only is never noise
    missing.to_parquet(new)
    row = R.compare_file(ref, new)
    assert row["float_noise"] is False and row["other_cells"] == 1


def test_csv_times_are_left_out_and_values_are_not(R, tmp_path):
    ref, new = tmp_path / "ref.csv", tmp_path / "new.csv"
    pd.DataFrame({"trial": [0, 1], "value": [6.5, 6.7], "duration": [1.0, 2.0]}).to_csv(
        ref, index=False)
    pd.DataFrame({"trial": [0, 1], "value": [6.5, 6.7], "duration": [9.0, 8.0]}).to_csv(
        new, index=False)
    assert R.compare_file(ref, new)["equal"] is True
    pd.DataFrame({"trial": [0, 1], "value": [6.5, 6.8], "duration": [1.0, 2.0]}).to_csv(
        new, index=False)
    assert R.compare_file(ref, new)["equal"] is False


def test_json_volatile_keys_and_decisions(R, tmp_path):
    ref, new = tmp_path / "ref.json", tmp_path / "new.json"
    base = {"git_commit": "abc", "result": {"k_denominator": "poa", "duration_s": 1.0},
            "inputs_sha256": {"x.parquet": "0" * 64}}
    ref.write_text(json.dumps(base), encoding="utf-8")
    other = {"git_commit": "def", "result": {"k_denominator": "poa", "duration_s": 9.0},
             "inputs_sha256": {"x.parquet": "1" * 64}}
    new.write_text(json.dumps(other), encoding="utf-8")
    assert R.compare_file(ref, new)["equal"] is True
    other["result"]["k_denominator"] = "ghi"
    new.write_text(json.dumps(other), encoding="utf-8")
    row = R.compare_file(ref, new)
    assert row["equal"] is None and "k_denominator" in row["detail"]
    assert R.failing([{"path": "configs/target.yaml", **row}])  # a decision may not differ
    assert not R.failing([{"path": "outputs/runs/x/step.json", **row}])  # listed for review


def test_arrays_are_compared_exactly(R, tmp_path):
    ref, new = tmp_path / "ref.npz", tmp_path / "new.npz"
    a = np.random.default_rng(0).normal(size=(4, 73, 3)).astype(np.float32)
    np.savez_compressed(ref, val_q=a)
    np.savez(new, val_q=a)
    assert R.compare_file(ref, new)["equal"] is True
    b = a.copy()
    b[0, 0, 0] += 1e-3
    np.savez(new, val_q=b)
    row = R.compare_file(ref, new)
    assert row["equal"] is False and row["float_noise"] is False


def test_the_script_installs_nothing_and_refuses_a_folder_inside_the_checkout(R):
    text = SCRIPT.read_text(encoding="utf-8")
    for word in ("pip install", "conda install", "mamba install", "conda create",
                 "\"install\""):
        assert word not in text, word
    with pytest.raises(SystemExit):
        R.make_clone(REPO_ROOT / "outputs" / "not_here")

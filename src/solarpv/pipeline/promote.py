"""`python -m solarpv.pipeline promote <run_id>`: the separate, explicit copy of a final run to
the paths of §6.10 (X8 of docs/stage9_plan.md, §19.7).

Refused unless the run is a complete final run of the whole v6 scope (X39): every model step,
`cv`, `ablation` (D-36), `select`, `evaluate`, `explain`, `report` and the second-phase step
`test` (D-32) completed,
every step at one commit that carries the tag `g1-frozen`, none `+dirty`, and every output as
recorded. It never overwrites an existing file unless `--replace`, which moves the old file aside.
It copies, for every model of the run:
  outputs/06_<model>/predictions_<model>[_seedN].csv      the run's prediction files, with the
                                                          y_true of the test rows (phase 1 wrote
                                                          it blank) from the step test (R-G1-16)
  outputs/06_<model>/metrics_<model>[_seedN].csv          its rows of the test benchmark
  outputs/06_<model>/models/                              boosters and checkpoints
  outputs/06_xgb_sota/shap/shap_<target>_D<h>.csv         feature, shap_importance (mean |SHAP|
                                                          on test, booster unit, mean over seeds)
  outputs/06_xgb_sota/analysis/                           the tables of the steps cv and ablation
                                                          and their test values (S-38, S-41)
and writes `outputs/06_promoted.json` with the source run and the SHA-256 of every file.
"""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from solarpv.config.constants import HORIZONS
from solarpv.config.paths import OUTPUTS_DIR
from solarpv.config.provenance import sha256
from solarpv.pipeline import checks
from solarpv.pipeline.config import FROZEN_TAG, MODEL_STEPS, TARGETS
from solarpv.pipeline.runner import completed_records, outputs_intact, read_json
from solarpv.pipeline.steps import TEST_TRUTH_FILE

REQUIRED = (*MODEL_STEPS, "cv", "ablation", "select", "evaluate", "explain", "report", "test")
ANALYSIS_FILES = {"cv": ("cv_folds.csv", "cv_summary.csv", "cv_config.json"),
                  "ablation": ("ablation_results.csv", "ablation_modes.csv",
                               "ablation_summary.csv", "ablation_config.json")}
ANALYSIS_TEST = ("analyses/cv_vs_test.csv", "analyses/ablation_test.csv",
                 "analyses/ablation_test_summary.csv")
PRED = re.compile(r"^predictions_(?P<model>.+?)(?:_seed(?P<seed>\d+))?\.csv$")
MANIFEST_NAME = "06_promoted.json"


class PromoteError(RuntimeError):
    pass


def plan(run_dir: Path, out_root: Path = OUTPUTS_DIR, check_tag: bool = True
         ) -> list[tuple[Path, Path | pd.DataFrame]]:
    """(target, source file or table) of every copy, after every check."""
    run_dir = Path(run_dir)
    state = read_json(run_dir / "run.json")
    if state["mode"] != "final":
        raise PromoteError(f"run {state['run_id']} is a {state['mode']} run: only a final run "
                           "is promoted (X8)")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if (list(config["targets"]) != list(TARGETS) or list(config["horizons"]) != list(HORIZONS)
            or config.get("seeds") is not None):
        raise PromoteError("the run is not the whole v6 scope (X39)")
    records = completed_records(run_dir, state)
    missing = [s for s in REQUIRED if s not in records]
    if missing:
        raise PromoteError(f"the run is not complete: {missing} (D-32: test after the "
                           "stage-10 acceptance)")
    done = {s: records[s].get("finished_utc") for s in ("test", "report")}
    if all(done.values()) and done["report"] < done["test"]:  # R-G1-02
        raise PromoteError("the report was written before the step test finished: rerun it "
                           "(--only test,report --force --allow-test) so that it holds the test "
                           "tables")
    commits = {r["commit"] for r in records.values()}
    if len(commits) != 1 or any(c.endswith("+dirty") for c in commits):
        raise PromoteError(f"the steps ran at {sorted(commits)}: one clean commit is required")
    commit = commits.pop()
    if check_tag:
        tags = checks._git("tag", "--points-at", commit).stdout.split()
        if FROZEN_TAG not in tags:
            raise PromoteError(f"commit {commit} does not carry the tag {FROZEN_TAG}")
    for step, rec in records.items():
        if not outputs_intact(run_dir, rec):
            raise PromoteError(f"{step}: outputs differ from its record")
    # R-G1-19: what is promoted is what the step test scored, and nothing but the report was
    # written after it (a model step rerun after the test would leave old test numbers beside
    # new forecasts)
    scored = {f"{Path(k).parent.name}/{Path(k).name}": v
              for k, v in records["test"].get("inputs", {}).items()
              if Path(k).name.startswith("predictions_")}
    promoted = {f"{records[s]['folder']}/{n}": digest for s in MODEL_STEPS
                for n, digest in records[s]["outputs"].items() if PRED.match(n)}
    if scored != promoted:
        raise PromoteError("the prediction files differ from those the step test scored: "
                           f"{sorted(set(scored) ^ set(promoted))[:3]}")
    tested = records["test"].get("finished_utc")
    later = sorted(s for s, r in records.items() if s not in ("report", "test") and tested
                   and r.get("finished_utc") and r["finished_utc"] > tested)
    if later:
        raise PromoteError(f"{later} finished after the step test: its numbers describe an "
                           "earlier state")
    copies: list[tuple[Path, Path | pd.DataFrame]] = []
    test_folder = run_dir / records["test"]["folder"]
    bench = pd.read_csv(test_folder / "test" / "benchmark_summary.csv")
    truth_file = test_folder / TEST_TRUTH_FILE  # R-G1-16: phase 1 wrote the test y_true blank
    if not truth_file.exists():
        raise PromoteError(f"the step test wrote no {TEST_TRUTH_FILE}: the prediction files of "
                           "§6.10 cannot get the truth of their test rows")
    test_truth = pd.read_csv(truth_file)
    test_truth.index = pd.DatetimeIndex(pd.to_datetime(test_truth.pop("timestamp"), utc=True))
    for step in MODEL_STEPS:
        folder = run_dir / records[step]["folder"]
        for name in sorted(records[step]["outputs"]):
            m = PRED.match(name)
            if m:
                model, seed = m["model"], m["seed"]
                dest = out_root / f"06_{model}"
                copies.append((dest / name, with_test_truth(folder / name, test_truth,
                                                            run_dir)))
                rows = bench[(bench["model"] == model)
                             & (bench["seed"] == (int(seed) if seed else -1))]
                metrics = name.replace("predictions_", "metrics_")
                copies.append((dest / metrics, rows.reset_index(drop=True)))
            elif name.endswith((".ubj", ".pt")) and "/models/" in f"/{name}":
                model = {"xgb": "xgb_sota"}.get(step) or _neural_model(step)
                copies.append((out_root / f"06_{model}" / "models" / Path(name).name,
                               folder / name))
    for step, names in ANALYSIS_FILES.items():  # X73: the tables of S-38 and S-41
        for name in names:
            copies.append((out_root / "06_xgb_sota" / "analysis" / name,
                           run_dir / records[step]["folder"] / name))
    for name in ANALYSIS_TEST:
        source = run_dir / records["test"]["folder"] / name
        if source.exists():
            copies.append((out_root / "06_xgb_sota" / "analysis" / Path(name).name, source))
    shap_file = run_dir / records["test"]["folder"] / "shap_test" / "xgb_shap_features_test.csv"
    if shap_file.exists():
        shap = pd.read_csv(shap_file)
        for (target, h), g in shap.groupby(["target", "horizon"]):
            table = (g.groupby("feature", as_index=False)
                     .agg(shap_importance=("mean_abs_native", "mean"),
                          shap_importance_kw=("mean_abs_kw", "mean"),
                          n_seeds=("seed", "nunique"))
                     .sort_values("shap_importance", ascending=False))
            copies.append((out_root / "06_xgb_sota" / "shap" / f"shap_{target}_D{h}.csv",
                           table.reset_index(drop=True)))
    return copies


def with_test_truth(source: Path, truth: pd.DataFrame, run_dir: Path) -> Path | pd.DataFrame:
    """R-G1-16: a prediction file of phase 1 has a blank `y_true` on its test rows (D-32). The
    promoted copy of §6.10 gets it from the truth the step `test` wrote (a read it logged); every
    other cell is the file's own. A file without such rows is copied unchanged."""
    frame = pd.read_csv(source)
    if not {"timestamp", "target", "split", "y_true"} <= set(frame.columns):
        return source
    rows = (frame["split"] == "test") & frame["y_true"].isna()
    if not rows.any():
        return source
    stamps = pd.DatetimeIndex(pd.to_datetime(frame.loc[rows, "timestamp"], utc=True))
    for target in frame.loc[rows, "target"].unique():
        sel = (frame.loc[rows, "target"] == target).to_numpy()
        values = truth[target].reindex(stamps[sel]).to_numpy()
        frame.loc[frame.index[rows][sel], "y_true"] = values
    frame.attrs["origin"] = (f"{source.relative_to(run_dir).as_posix()} with the y_true of its "
                             f"test rows from the step test ({TEST_TRUTH_FILE})")
    return frame


def _neural_model(step: str) -> str:
    from solarpv.models.neural_job import STEP_MODEL

    return STEP_MODEL[step]


def promote(run_dir: Path, out_root: Path = OUTPUTS_DIR, replace: bool = False,
            check_tag: bool = True) -> Path:
    run_dir = Path(run_dir)
    copies = plan(run_dir, out_root, check_tag)
    existing = [t for t, _ in copies if t.exists()]
    if existing and not replace:
        raise PromoteError(f"{len(existing)} target files exist (e.g. {existing[0]}); "
                           "--replace moves them aside")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for t in existing:
        t.rename(t.with_name(f"{t.name}.superseded-{stamp}"))
    written = {}
    for target, source in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, pd.DataFrame):
            source.to_csv(target, index=False, lineterminator="\n")
            origin = source.attrs.get("origin", "table of the run")
        else:
            shutil.copyfile(source, target)
            if sha256(target) != sha256(source):
                raise PromoteError(f"the copy of {source} differs")
            origin = source.relative_to(run_dir).as_posix()
        written[target.relative_to(out_root).as_posix()] = {"source": origin,
                                                            "sha256": sha256(target)}
    state = read_json(run_dir / "run.json")
    manifest = out_root / MANIFEST_NAME
    manifest.write_text(json.dumps({"run_id": state["run_id"], "promoted_utc": stamp,
                                    "files": written}, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    return manifest

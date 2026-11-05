"""The work steps implemented so far (W1): `baselines` and `evaluate` (docs/stage9_plan.md §3.2,
§10.3). Every step gets a `Context` and writes only into `ctx.out`, a temporary folder that the
runner renames when the step succeeds.

Rules (the self-review of §10):
- the truth of smoke and dev runs is read without the test split (X3); the test split is read
  only to evaluate it, in final mode with allow_test;
- the evaluation window: rows of the split inside the window are scored; earlier rows of the split
  are context for the persistence reference; later rows are dropped (§10.3);
- `evaluate` reads only the prediction files of the run's completed model steps, each checked
  against the hash in its step record (M1, X1), and in smoke and dev refuses any row in the test
  split (X7).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from solarpv.config.constants import DERIVED_CONSTANTS_FILE, TEST_START, mean_daytime_k
from solarpv.config.provenance import sha256
from solarpv.config.runlog import RunLog
from solarpv.config.site import load_site
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.truth import FEATURES_FILE, load_truth
from solarpv.models.baselines import reference_predictions
from solarpv.pipeline.config import MODEL_STEPS, Window


class RunnerError(RuntimeError):
    """A rule of the runner is broken (missing input, test access, modified file)."""


@dataclass
class Context:
    run_id: str
    run_dir: Path
    mode: str
    config: dict
    windows: list[Window]
    out: Path
    log: RunLog
    completed: dict[str, dict] = field(default_factory=dict)  # step -> record (step.json)
    failed: list[str] = field(default_factory=list)  # steps of the run whose last try failed
    command: str = ""


@dataclass
class Result:
    outputs: list[Path]
    values: dict = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


# ── truth and windows ───────────────────────────────────────────────────────


def truth_for(window: Window, path: Path | None = None) -> pd.DataFrame:
    """The truth table for one evaluation window: without test rows unless the window is the
    test split, restricted to the window."""
    exclude = () if window.split == "test" else ("test",)
    truth = load_truth(path or FEATURES_FILE, exclude_splits=exclude)
    if exclude and (truth["split"] == "test").any():  # load_truth checks this too
        raise RunnerError("test rows were loaded outside a test evaluation")
    return restrict_to_window(truth, window)


def restrict_to_window(truth: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Rows of `window.split` inside [start, end) keep their label; earlier rows of that split
    become '<split>_context' (they feed persistence only); rows from `end` on are dropped."""
    start, end = window.start_ts, window.end_ts
    in_split = truth["split"] == window.split
    inside = (truth.index >= start) & (truth.index < end)
    if not (in_split & inside).any():
        raise RunnerError(f"the window {window.label} has no row of the split {window.split}")
    if (inside & ~in_split.to_numpy()).any():
        raise RunnerError(f"the window {window.label} crosses the boundary of {window.split}")
    out = truth[truth.index < end].copy()
    before = (out["split"] == window.split) & (out.index < start)
    out.loc[before, "split"] = f"{window.split}_context"
    return out


def assert_no_test_rows(paths: list[Path]) -> None:
    """Smoke and dev: a prediction for a test hour means the model read test features (X7)."""
    first_test = pd.Timestamp(TEST_START, tz="UTC")
    for path in paths:
        ts = pd.to_datetime(pd.read_csv(path, usecols=["timestamp"])["timestamp"], utc=True)
        n = int((ts >= first_test).sum())
        if n:
            raise RunnerError(f"{path.name}: {n} rows in the test split in a development run")


# ── baselines ───────────────────────────────────────────────────────────────


def baselines_inputs(ctx: Context) -> list[Path]:
    return [FEATURES_FILE, DERIVED_CONSTANTS_FILE]


def run_baselines(ctx: Context) -> Result:
    """The three references (OP-40) for every row of each evaluation window, written as
    prediction files in the contract of §6.10, so they pass the checks every model passes."""
    mean_k = mean_daytime_k()
    targets, horizons = ctx.config["targets"], ctx.config["horizons"]
    frames = []
    for window in ctx.windows:
        with ctx.log.step(f"references on {window.split}, {window.label}"):
            truth = truth_for(window)
            refs = reference_predictions(truth, window.split, mean_k, horizons=horizons)
            refs = refs[refs["target"].isin(targets)]
            day = truth["daytime"] & (truth["split"] == window.split)
            ctx.log.note(f"{window.split}: {int((truth['split'] == window.split).sum())} hours "
                         f"in the window, {int(day.sum())} of them daytime; context rows "
                         f"{int((truth['split'] == window.split + '_context').sum())}")
            frames.append(refs)
    refs = pd.concat(frames, ignore_index=True)
    outputs = []
    for model, g in refs.groupby("model", sort=True):
        path = ctx.out / f"predictions_{model}.csv"
        g.sort_values(["target", "horizon", "timestamp"]).to_csv(
            path, index=False, lineterminator="\n")
        outputs.append(path)
        ctx.log.value(f"rows_{model}", len(g))
    return Result(outputs, values={"mean_daytime_k": mean_k, "rows": len(refs)},
                  summary=[f"{len(outputs)} reference files, {len(refs):,} rows"])


# ── evaluate ────────────────────────────────────────────────────────────────


def prediction_files(ctx: Context) -> tuple[list[Path], list[str]]:
    """The prediction files of the completed model steps of this run, checked against the hashes
    in their records; and the model steps of the run that are not complete."""
    files = []
    for step in MODEL_STEPS:
        record = ctx.completed.get(step)
        if record is None:
            continue
        folder = ctx.run_dir / record["folder"]
        for name, digest in sorted(record["outputs"].items()):
            if not name.startswith("predictions_"):
                continue
            path = folder / name
            if not path.exists() or sha256(path) != digest:
                raise RunnerError(f"{record['folder']}/{name} is missing or was modified after "
                                  "its step completed; rerun the step with --force")
            files.append(path)
    missing = [s for s in MODEL_STEPS if s in ctx.failed and s not in ctx.completed]
    return files, missing


def evaluate_inputs(ctx: Context) -> list[Path]:
    return [FEATURES_FILE, DERIVED_CONSTANTS_FILE, *prediction_files(ctx)[0]]


def run_evaluate(ctx: Context) -> Result:
    if "baselines" not in ctx.completed:
        raise RunnerError("evaluate needs the references of this run: run the baselines step "
                          "first (they are the skill reference and the DM reference)")
    files, missing = prediction_files(ctx)
    if ctx.mode != "final":
        assert_no_test_rows(files)
    mean_k = mean_daytime_k()
    site = load_site()
    outputs, summary, notes = [], [], {"models_missing": missing}
    if missing:
        ctx.log.warn(f"model steps of this run that did not complete: {missing}")
    for window in ctx.windows:
        out = ctx.out / window.split
        with ctx.log.step(f"evaluate {window.split}, {window.label}"):
            truth = truth_for(window)
            tables = E.run(files, out, mean_k, site=site, truth=truth, split=window.split,
                           allow_test=ctx.config["allow_test"] and window.split == "test",
                           strict=True, with_references=False, ci=True, log=ctx.log,
                           command=ctx.command,
                           record_dir=ctx.run_dir / "evaluate" / window.split)
        outputs += sorted(out.glob("*.csv")) + [out / "manifest.json"]
        summary += benchmark_lines(out / "benchmark_summary.csv", window)
        ctx.log.value(f"rows_benchmark_{window.split}", len(tables["benchmark_summary"]))
    return Result(outputs, values={"files_scored": [f.name for f in files]}, summary=summary,
                  notes=notes)


def benchmark_lines(path: Path, window: Window) -> list[str]:
    """The run summary, read from the written table (M2: no result number in the code)."""
    bench = pd.read_csv(path)
    lines = [f"{window.split}, {window.label}: nRMSE in kW (% of capacity), n daytime hours"]
    for row in bench.sort_values(["target", "horizon", "model", "seed"]).itertuples():
        lines.append(f"  {row.target:<8} D+{row.horizon} {row.model:<12} seed {row.seed:>3}: "
                     f"{row.nRMSE_kW_pct:6.2f} %  (n = {row.n})")
    return lines


# ── registry ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Implementation:
    inputs: Callable[[Context], list[Path]]
    run: Callable[[Context], Result]


IMPLEMENTED = {
    "baselines": Implementation(baselines_inputs, run_baselines),
    "evaluate": Implementation(evaluate_inputs, run_evaluate),
}

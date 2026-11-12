"""The `select` step (docs/stage9_plan.md §19.4, D-30): it verifies the stage-7 selection and
records the input that every model step of the run reads; it reruns nothing.

Stage 7 is closed (R7-04, R7-05): selection on train only, the inside-train check, and one
registered validation look. A rerun would end with a fourth look at validation (X44). The step
reads no row of any table: only the committed stage-7 outputs, the stage-7 run logs and the
column names (parquet schemas) of the current stage-6 and 6b tables. It checks:

1. the stage-7 outputs that decide the input are tracked by git and unmodified;
2. the tables named in the stage-7 run logs are the current tables (X11);
3. the manifest records selection on train only and the single validation look;
4. `model_inputs.json` equals `selection.run.decide_input` applied to the committed checks: the
   indirect sets are the recommended 05e sets, the direct ones the full pool of the current
   tables (F-29), in their recorded order;
5. the stage-7 code at the selection commit against HEAD (reported, not an error).

Outputs: `select.json` (every check, the SHA-256 of `model_inputs.json`), copies of
`model_inputs.json` and of the six 05e files. The model steps of the run refuse another
`model_inputs.json` (M25, `steps.check_selected_inputs`).
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import sha256
from solarpv.pipeline import checks, data
from solarpv.pipeline.steps import Context, Result, RunnerError

TARGETS = ("indirect", "direct")
HORIZONS = (1, 2, 3)
CODE_PATHS = ("src/solarpv/selection",)


def model_inputs_file() -> Path:
    from solarpv.models import xgb_sota as X

    return Path(X.MODEL_INPUTS_FILE)


def selection_dir() -> Path:
    """The stage-7 folder of the inputs file (tests point both at a synthetic copy)."""
    return model_inputs_file().parent


def decided_files() -> dict[str, Path]:
    base = selection_dir()
    files = {"model_inputs": model_inputs_file(),
             "manifest": base / "selection_manifest.json",
             "inner_train_check": base / "inner_train_check.csv",
             "validation_check": base / "validation_check.csv"}
    for t in TARGETS:
        for h in HORIZONS:
            files[f"05e_{t}_D{h}"] = base / f"final_features_{t}_D{h}.json"
    return files


def select_inputs(ctx: Context) -> list[Path]:
    files = list(decided_files().values())
    for h in ctx.config["horizons"]:
        files += [data.feature_file(h), data.gfs_file(h)]
    return files


def table_columns(h: int) -> pd.DataFrame:
    """An empty frame with the columns of the model table of D+h (the stage-6 table joined with
    its GFS table): the schema only, no row is read."""
    base = pq.read_schema(data.feature_file(h)).names
    gfs = [c for c in pq.read_schema(data.gfs_file(h)).names if c not in ("t0_utc",)]
    cols = list(dict.fromkeys([c for c in base if c != data.INDEX]
                              + [c for c in gfs if c != data.INDEX]))
    return pd.DataFrame(columns=cols)


def expected_inputs(files: dict[str, Path], horizons) -> dict:
    """The stage-9 input that the registered rule gives on the committed checks (R7-04 steps 3
    and 4): `decide_input` per target x horizon, with the 05e set or the full pool."""
    from solarpv.selection.common import pool
    from solarpv.selection.run import decide_input

    inner = pd.read_csv(files["inner_train_check"])
    val = pd.read_csv(files["validation_check"])
    passes = {(t, int(h)): bool(g["passes"].all()) for (t, h), g in
              inner.groupby(["target", "horizon"])}
    flags = {(t, int(h)): bool(g["red_flag"].any()) for (t, h), g in
             val.groupby(["target", "horizon"])}
    out = {}
    for t in TARGETS:
        for h in horizons:
            source, reason = decide_input(passes[(t, h)], flags.get((t, h)))
            if source == "05e":
                rec = json.loads(files[f"05e_{t}_D{h}"].read_text(encoding="utf-8"))
                feats = list(rec["recommended"]["features"])
            elif source == "full pool":
                feats = pool(table_columns(h), h)
            else:
                raise RunnerError(f"{t}_D{h}: the stage-7 decision is {source!r} ({reason})")
            out[f"{t}_D{h}"] = {"source": source, "reason": reason, "features": feats}
    return out


def code_changes(commit: str) -> list[str]:
    """The stage-7 code files changed between the selection commit and HEAD."""
    proc = checks._git("diff", "--name-only", commit.split("+")[0], "HEAD", "--", *CODE_PATHS)
    if proc.returncode != 0:
        return [f"git diff failed: {proc.stderr.strip()[:200]}"]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def run_select(ctx: Context) -> Result:
    files = decided_files()
    horizons = [int(h) for h in ctx.config["horizons"]]
    problems: list[str] = []
    record: dict = {"decision": "D-30: verify and reuse the stage-7 selection; nothing is rerun",
                    "rows_read": 0}

    # 1. tracked and unmodified
    tracked = [checks.tracked_unmodified(p) for p in files.values()]
    record["tracked"] = tracked
    problems += [f"{r['path']}: {r['detail']}" for r in tracked if not r["ok"]]

    # 2. the tables of the stage-7 run logs (X11)
    provenance = checks.stage7_provenance(files["manifest"])
    record["tables"] = provenance
    problems += [f"{r['path']}: {r['detail']}" for r in provenance if not r["ok"]]

    # 3. the manifest: train only, one validation look
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    record["manifest"] = {k: manifest.get(k) for k in
                          ("git_commit", "run_log", "validation_run_log",
                           "splits_read_for_selection", "validation_read")}
    if list(manifest.get("splits_read_for_selection") or []) != ["train"]:
        problems.append("the selection manifest does not record selection on train only")
    if manifest.get("validation_read") != "S8 only":
        problems.append("the selection manifest does not record the single validation look")

    # 4. model_inputs.json against the registered rule
    actual = json.loads(files["model_inputs"].read_text(encoding="utf-8"))
    expected = expected_inputs(files, horizons)
    rows = []
    for key, exp in expected.items():
        got = actual.get(key)
        same = (got is not None and got.get("source") == exp["source"]
                and list(got.get("features", [])) == exp["features"]
                and int(got.get("n_features", -1)) == len(exp["features"]))
        rows.append({"key": key, "source": exp["source"], "n_features": len(exp["features"]),
                     "matches": same})
        if not same:
            problems.append(f"{key}: model_inputs.json differs from the registered rule on the "
                            "committed checks")
    record["inputs"] = rows

    # 5. the stage-7 code since the selection commit (reported)
    record["code_changed_since_selection"] = code_changes(str(manifest.get("git_commit", "")))
    if problems:
        raise RunnerError("the stage-7 selection does not verify: " + "; ".join(problems))

    digest = sha256(files["model_inputs"])
    record["model_inputs_sha256"] = digest
    sets = ctx.out / "sets"
    sets.mkdir()
    outputs = []
    for name, source in files.items():
        if name == "model_inputs" or name.startswith("05e_"):
            copy = sets / source.name
            shutil.copyfile(source, copy)
            if sha256(copy) != sha256(source):
                raise RunnerError(f"the copy of {source.name} differs from its source")
            outputs.append(copy)
    record["copies"] = {p.name: sha256(p) for p in outputs}
    out = ctx.out / "select.json"
    out.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8",
                   newline="\n")
    outputs.append(out)
    n = {r["key"]: r["n_features"] for r in rows}
    ctx.log.note(f"select: verified, nothing rerun; model_inputs.json {digest[:12]}; {n}; stage-7 "
                 f"code changed since {record['manifest']['git_commit']}: "
                 f"{record['code_changed_since_selection'] or 'none'}")
    summary = ["select (D-30): the stage-7 selection verified, nothing rerun; "
               f"model_inputs.json {digest[:12]}"]
    summary += [f"  {r['key']:<12} {r['source']:<9} {r['n_features']:>3} features" for r in rows]
    rel = {k: v.resolve().relative_to(REPO_ROOT).as_posix() if v.resolve().is_relative_to(
        REPO_ROOT) else v.as_posix() for k, v in files.items()}
    return Result(outputs, values={"model_inputs_sha256": digest, "inputs": n,
                                   "files": rel},
                  summary=summary)

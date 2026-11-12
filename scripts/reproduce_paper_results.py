"""Reproduce the development state of the project from the raw data, in a fresh checkout (gate
G-1, docs/stage9_plan.md §20.5; M32-M34, M36).

    python scripts/reproduce_paper_results.py --plan
    python scripts/reproduce_paper_results.py --dev --workdir C:/Users/<user>/repro_g1/r1
    python scripts/reproduce_paper_results.py --dev --workdir <same> --from features
    options: --skip-neural (no neural dev run), --rerun-selection (the train part of stage 7,
             about 95 minutes, without the validation look), --with-eda (stages 3 and 3b)

What it does:
- `git clone` of this repository at its HEAD (the code paths must be committed) into the empty
  folder `--workdir`, outside this checkout. Every command runs in the clone with
  `PYTHONPATH=<clone>/src`, so the clone's code is imported, not the editable install of this
  checkout (X55); the import is asserted before the first step.
- Inputs that the clone cannot produce are given to it: `Raw data/` read in place
  (`SOLAR_RAW_DIR`, verified against `00_data_audit/raw_checksums.sha256`), the GFS point
  extracts of stage 6b (`data/external/gfs_1p00/`, copied; downloading them again is an external
  service) and `configs/site.private.yaml` (copied into the clone's ignored `configs/`, removed
  at the end).
- The stages run in the order of `docs/data_processing_record.md` §19. After each stage every
  file it wrote is compared with the reference of this checkout by content (M33): parquet and
  CSV files read and compared exactly (columns of times and durations left out), JSON and YAML
  compared as data without the volatile keys (commit, run log, times, versions, the hashes of
  parquet bytes), NumPy arrays exactly; figures, logs and notebooks are listed, not compared. A
  difference of content in a table or an array fails the reproduction; other differences are
  listed for review. The clone is then put back to the reference state of that stage (tracked
  files from git, data files copied), so the next stage runs on verified inputs and the hash
  chain of the manifests holds even when a newer parquet writer gives other bytes (OP-60).
- The selection is verified by the runner's `select` (D-30); `--rerun-selection` also reruns its
  train part and compares the sets.
- Two development runs of the runner, with the invocations of the two reference runs:
  `select,baselines,xgb,evaluate,explain,report` against `20261112T063719Z_dev`, and
  `select,baselines,bilstm,tft96,tft168,tftbase,patchtst,evaluate,explain,report` against
  `20261111T160714Z_dev`. Smoke and dev never read a test row; nothing here reads one.
- Both environments are recorded and compared with the locks (M34). Nothing is installed.

The record goes to `outputs/g1_audit/reproduction_<UTC stamp>/` of this checkout: `record.json`
(the plan, the environments, every step with its time, exit code and comparison), one log per
step, and `report.md`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solarpv.config.cli import parser  # noqa: E402
from solarpv.config.provenance import find_git, sha256  # noqa: E402

RAW = REPO / "Raw data"
GFS_EXTRACTS = "data/external/gfs_1p00"
PRIVATE_SITE = "configs/site.private.yaml"
REFERENCE_RUNS = {"dev_xgb": "outputs/runs/20261112T063719Z_dev",
                  "dev_neural": "outputs/runs/20261111T160714Z_dev"}
XGB_STEPS = "select,baselines,xgb,evaluate,explain,report"
NEURAL_STEPS = "select,baselines,bilstm,tft96,tft168,tftbase,patchtst,evaluate,explain,report"
# JSON / YAML keys whose values change from run to run and say nothing about the result
VOLATILE_KEYS = {"git_commit", "commit", "run_log", "validation_run_log", "started_utc",
                 "started", "finished", "finished_utc", "created", "created_utc", "duration_s",
                 "duration", "elapsed_s", "seconds", "versions", "python", "platform", "run_id",
                 "fingerprint", "utc_time", "host", "hostname", "worker_s", "gpu_s", "time_s",
                 "solarpv_file", "environment", "invocation", "invocations", "command"}
VOLATILE_COLUMN = re.compile(r"(datetime|duration|elapsed|seconds|_time$|^time|started|"
                             r"finished|run_id|commit)", re.I)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NOT_COMPARED = (".png", ".log", ".ipynb", ".pt", ".db", ".zip", ".sha256", ".html")
TABLES = (".parquet", ".csv", ".npz")


@dataclass
class Step:
    name: str
    args: list[str]  # after the interpreter
    roots: list[str]  # files and folders the step writes (relative to the checkout)
    option: str | None = None  # the flag that turns it on (None: always)
    reference: str | None = None  # for the runner steps: the reference run
    note: str = ""
    reads_validation: bool = False


def plan(options: dict) -> list[Step]:
    steps = [
        Step("checksums", ["scripts/make_checksums.py", "--verify", str(RAW)], [],
             note="the raw files against 00_data_audit/raw_checksums.sha256"),
        Step("ingest", ["-m", "solarpv.data.ingest"], ["data/processed/01_ingest"]),
        Step("clock", ["scripts/estimate_clock_drift.py"],
             ["configs/clock_offsets.csv", "00_data_audit/clock_drift_2026-10-14"]),
        Step("preprocess", ["-m", "solarpv.data.preprocess"], ["data/processed/04_preprocess"]),
        Step("clean", ["-m", "solarpv.data.clean"],
             ["data/processed/05_clean", "outputs/05_cleaning"]),
        Step("eda", ["-m", "solarpv.eda"], ["outputs/03_eda"], option="with_eda"),
        Step("quality", ["-m", "solarpv.eda.quality"], ["outputs/03b_quality"],
             option="with_eda"),
        Step("orientation", ["00_data_audit/orientation_2026-10-20/orientation_check.py"],
             ["00_data_audit/orientation_2026-10-20", "configs/panel_orientation.yaml"]),
        Step("features", ["-m", "solarpv.features.build"],
             ["data/processed/06_features", "outputs/06_features",
              "configs/derived_constants.yaml"]),
        Step("decide_k", ["-m", "solarpv.features.decide_k"],
             ["configs/target.yaml", "outputs/06_features"], reads_validation=True,
             note="D-11 criterion 3 reads validation; the decision must come out unchanged"),
        Step("features_final", ["-m", "solarpv.features.build"],
             ["data/processed/06_features", "outputs/06_features",
              "configs/derived_constants.yaml"], note="again, with the D-11 decision"),
        Step("gfs", ["-m", "solarpv.nwp.gfs"], ["data/processed/06b_gfs", "outputs/06b_gfs"],
             note="from the copied point extracts; nothing is downloaded"),
        Step("nwp_check", ["-m", "solarpv.nwp.check"],
             ["configs/nwp_decision.yaml", "outputs/06b_gfs"], reads_validation=True,
             note="D-12 reads validation; the decision must come out unchanged"),
        Step("evaluate", ["-m", "solarpv.evaluation.evaluate"], ["outputs/07_evaluation/val"],
             reads_validation=True, note="the stage-8B references on validation"),
        Step("fixture", ["-m", "solarpv.evaluation.evaluate", "--fixture"],
             ["outputs/07_evaluation_fixture"],
             note="the submitted LSTM file with its own truth (T-15)"),
        Step("selection", ["-m", "solarpv.selection.run", "--skip-validation"],
             ["outputs/07_feature_selection", "configs/k_history_decision.yaml"],
             option="rerun_selection", note="train only; the validation look is not repeated"),
        Step("sequence_data", ["-m", "solarpv.models.sequence_data"],
             ["outputs/09_sequence_data"], reads_validation=True),
        Step("dev_xgb", ["-m", "solarpv.pipeline", "run", "--mode", "dev", "--only",
                         XGB_STEPS], ["outputs/runs"], reference=REFERENCE_RUNS["dev_xgb"],
             reads_validation=True),
        Step("dev_neural", ["-m", "solarpv.pipeline", "run", "--mode", "dev", "--only",
                            NEURAL_STEPS, "--keep-going"], ["outputs/runs"],
             option="neural", reference=REFERENCE_RUNS["dev_neural"], reads_validation=True),
    ]
    return [s for s in steps if s.option is None or options.get(s.option)]


# ── comparison ──────────────────────────────────────────────────────────────


def _strip(obj, key=""):
    """JSON / YAML data without the volatile keys and without hashes of parquet bytes."""
    if isinstance(obj, dict):
        return {k: _strip(v, k) for k, v in obj.items()
                if k not in VOLATILE_KEYS and not str(k).endswith(".parquet")
                and "parquet" not in str(k)}
    if isinstance(obj, list):
        return [_strip(v, key) for v in obj]
    if isinstance(obj, str) and HEX64.match(obj) and "parquet" in key:
        return "<parquet sha256>"
    return obj


def _diff_paths(a, b, path="", out=None, limit=8):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'reference' if k in a else 'reproduction'}")
            else:
                _diff_paths(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} != {str(b)[:60]}")
    return out


def _frame_diff(a, b) -> str | None:
    import pandas as pd

    drop = [c for c in a.columns if VOLATILE_COLUMN.search(str(c))]
    a, b = a.drop(columns=drop, errors="ignore"), b.drop(columns=drop, errors="ignore")
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))[:8]}"
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError as exc:
        return " ".join(str(exc).split())[:300]
    return None


def compare_file(reference: Path, reproduced: Path) -> dict:
    """One row: kind, equal (True / False / None for not compared), detail."""
    suffix = reproduced.suffix.lower()
    if not reference.exists():
        return {"kind": "new", "equal": None, "detail": "no reference file"}
    if reproduced.name.endswith(NOT_COMPARED) or suffix in NOT_COMPARED:
        return {"kind": "not compared", "equal": None, "detail": suffix}
    same_bytes = sha256(reference) == sha256(reproduced)
    if same_bytes:
        return {"kind": "bytes", "equal": True, "detail": "identical bytes"}
    try:
        if suffix == ".parquet":
            import pandas as pd

            d = _frame_diff(pd.read_parquet(reference), pd.read_parquet(reproduced))
            return {"kind": "table", "equal": d is None,
                    "detail": d or "same content, other bytes (parquet writer)"}
        if suffix == ".csv":
            import pandas as pd

            d = _frame_diff(pd.read_csv(reference), pd.read_csv(reproduced))
            return {"kind": "table", "equal": d is None, "detail": d or "same content"}
        if suffix == ".npz":
            import numpy as np

            with np.load(reference) as ra, np.load(reproduced) as rb:
                if sorted(ra.files) != sorted(rb.files):
                    return {"kind": "array", "equal": False, "detail": "other arrays"}
                bad = [k for k in ra.files if not np.array_equal(ra[k], rb[k], equal_nan=True)]
            return {"kind": "array", "equal": not bad, "detail": f"differ: {bad[:5]}" if bad
                    else "same arrays"}
        if suffix in (".json", ".yaml", ".yml"):
            import yaml

            load = (lambda p: json.loads(p.read_text(encoding="utf-8"))) if suffix == ".json" \
                else (lambda p: yaml.safe_load(p.read_text(encoding="utf-8")))
            a, b = _strip(load(reference)), _strip(load(reproduced))
            diffs = _diff_paths(a, b)
            return {"kind": "data", "equal": not diffs or None,
                    "detail": "; ".join(diffs) if diffs else "same data (volatile keys left out)"}
        ta = reference.read_text(encoding="utf-8").replace("\r\n", "\n")
        tb = reproduced.read_text(encoding="utf-8").replace("\r\n", "\n")
        return {"kind": "text", "equal": None if ta != tb else True,
                "detail": "text differs (review)" if ta != tb else "same text"}
    except Exception as exc:  # a file that cannot be read is a finding, not a crash
        return {"kind": "error", "equal": False, "detail": f"{type(exc).__name__}: {exc}"[:300]}


def written_files(clone: Path, roots: list[str], since: float) -> list[str]:
    """Files under the roots of a step that it created or modified."""
    out = []
    for root in roots:
        p = clone / root
        files = [p] if p.is_file() else (sorted(p.rglob("*")) if p.exists() else [])
        for f in files:
            if f.is_file() and f.stat().st_mtime >= since - 1:
                out.append(f.relative_to(clone).as_posix())
    return out


def failing(rows: list[dict]) -> list[dict]:
    """Rows that fail the reproduction: a table or an array whose content differs, or an error.
    A data file (JSON / YAML) under configs/ is a decision and must not differ either."""
    bad = []
    for r in rows:
        if r["equal"] is False:
            bad.append(r)
        elif r["equal"] is None and r["kind"] == "data" and r["path"].startswith("configs/"):
            bad.append(r)
    return bad


# ── the clone ───────────────────────────────────────────────────────────────


def git(*args, cwd=REPO, check=True) -> subprocess.CompletedProcess:
    return subprocess.run([find_git(), *args], cwd=cwd, capture_output=True, text=True,
                          check=check)


def code_is_committed() -> list[str]:
    status = git("status", "--porcelain", "--untracked-files=normal", "--", "src", "scripts",
                 "configs", "pyproject.toml", "environment.yml", "00_data_audit").stdout
    return [line for line in status.splitlines() if line.strip()]


def make_clone(workdir: Path) -> str:
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"{workdir} exists and is not empty: a reproduction starts in an empty "
                         "folder (use --from to continue one)")
    if workdir.resolve().is_relative_to(REPO.resolve()):
        raise SystemExit("the work folder must be outside this checkout")
    head = git("rev-parse", "HEAD").stdout.strip()
    workdir.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "--quiet", "--no-hardlinks", str(REPO), str(workdir))
    git("checkout", "--quiet", head, cwd=workdir)
    shutil.copytree(REPO / GFS_EXTRACTS, workdir / GFS_EXTRACTS)
    return head


def clone_env(clone: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(clone / "src")] + [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["SOLAR_RAW_DIR"] = str(RAW)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    return env


def assert_clone_import(clone: Path) -> str:
    proc = subprocess.run([sys.executable, "-c", "import solarpv, pathlib; "
                           "print(pathlib.Path(solarpv.__file__).resolve())"], cwd=clone,
                          env=clone_env(clone), capture_output=True, text=True, check=True)
    imported = Path(proc.stdout.strip())
    if not imported.is_relative_to((clone / "src").resolve()):
        raise SystemExit(f"the clone imports solarpv from {imported} (X55)")
    return str(imported)


def restore_reference(clone: Path, roots: list[str]) -> None:
    """Put the roots of a stage back to the reference: tracked files from git, new files
    removed, ignored data files copied from this checkout."""
    tracked = [r for r in roots if r != "outputs/runs"]
    if not tracked:
        return
    git("checkout", "--quiet", "--", *tracked, cwd=clone, check=False)
    git("clean", "-fdq", "--", *tracked, cwd=clone, check=False)
    for root in tracked:
        src = REPO / root
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file() and git("check-ignore", "-q", f.relative_to(REPO).as_posix(),
                                       check=False).returncode == 0:
                    dst = clone / f.relative_to(REPO)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)


def new_run(clone: Path, before: set[str]) -> Path | None:
    runs = clone / "outputs" / "runs"
    new = sorted(p for p in runs.iterdir() if p.is_dir() and p.name not in before)
    return new[-1] if new else None


def compare_run(clone_run: Path, reference: Path) -> list[dict]:
    rows = []
    for f in sorted(clone_run.rglob("*")):
        if not f.is_file() or any(part.startswith(".") for part in f.relative_to(clone_run).parts):
            continue
        rel = f.relative_to(clone_run).as_posix()
        if rel.startswith(("invocations/", "logs/")):
            continue
        row = compare_file(reference / rel, f)
        rows.append({"path": f"<run>/{rel}", **row})
    return rows


# ── the environments (M34) ─────────────────────────────────────────────────


def environments() -> dict:
    from importlib import metadata

    import yaml

    lock = yaml.safe_load((REPO / "environment.lock.yml").read_text(encoding="utf-8")) \
        if (REPO / "environment.lock.yml").exists() else {}
    pins = {}
    for dep in lock.get("dependencies", []) if lock else []:
        if isinstance(dep, str) and "=" in dep:
            name, _, version = dep.partition("=")
            pins[name.strip().lower()] = version.split("=")[0].strip()
        elif isinstance(dep, dict):
            for item in dep.get("pip", []):
                name, _, version = item.partition("==")
                pins[name.strip().lower()] = version.strip()
    installed = {d.metadata["Name"].lower(): d.version for d in metadata.distributions()
                 if d.metadata["Name"]}
    differs = {name: {"lock": v, "installed": installed.get(name)}
               for name, v in sorted(pins.items())
               if name in installed and installed[name] != v}
    gpu = {}
    gpu_lock = REPO / "requirements-gpu.lock.txt"
    gpu_py = Path(sys.prefix).parent / "solar-gpu" / "python.exe"
    if gpu_py.exists():
        out = subprocess.run([str(gpu_py), "-m", "pip", "list", "--format=json",
                              "--disable-pip-version-check"], capture_output=True, text=True,
                             timeout=300)
        try:
            have = {p["name"].lower(): p["version"] for p in json.loads(out.stdout)}
        except json.JSONDecodeError:
            have = {}
        want = {}
        if gpu_lock.exists():
            for line in gpu_lock.read_text(encoding="utf-8").splitlines():
                line = line.split("#")[0].split(";")[0].strip()
                if "==" in line:
                    name, _, version = line.partition("==")
                    want[name.strip().lower()] = version.split(" ")[0].strip()
        gpu = {"python": str(gpu_py), "pinned": len(want),
               "differs": {n: {"lock": v, "installed": have.get(n)} for n, v in sorted(want.items())
                           if have.get(n) != v and not (have.get(n) or "").startswith(v)}}
    return {"solar": {"python": sys.version.split()[0], "executable": sys.executable,
                      "pinned": len(pins), "differs_from_lock": differs},
            "solar_gpu": gpu, "installed_anything": False}


# ── the run ─────────────────────────────────────────────────────────────────


def run(args) -> int:
    options = {"with_eda": args.with_eda, "rerun_selection": args.rerun_selection,
               "neural": not args.skip_neural}
    steps = plan(options)
    if args.plan:
        for s in steps:
            flag = " [reads validation]" if s.reads_validation else ""
            print(f"{s.name:<15} python {' '.join(s.args)}{flag}")
            if s.note:
                print(f"{'':<15} {s.note}")
        return 0
    if not args.dev:
        raise SystemExit("nothing to do: --plan or --dev")
    if not args.workdir:
        raise SystemExit("--dev needs --workdir (an empty folder outside this checkout)")
    workdir = Path(args.workdir).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    record_dir = REPO / "outputs" / "g1_audit" / f"reproduction_{stamp}"
    record_dir.mkdir(parents=True)
    record = {"stamp": stamp, "workdir": str(workdir), "options": options,
              "plan": [s.name for s in steps], "steps": []}
    if args.from_:
        if not (workdir / ".git").exists():
            raise SystemExit(f"--from needs an existing reproduction in {workdir}")
        names = [s.name for s in steps]
        if args.from_ not in names:
            raise SystemExit(f"--from {args.from_}: not a step of the plan {names}")
        steps = steps[names.index(args.from_):]
        record["commit"] = git("rev-parse", "HEAD", cwd=workdir).stdout.strip()
    else:
        dirty = code_is_committed()
        if dirty:
            raise SystemExit(f"commit the code first; the clone takes HEAD: {dirty[:5]}")
        record["commit"] = make_clone(workdir)
    shutil.copy2(REPO / PRIVATE_SITE, workdir / PRIVATE_SITE)
    try:
        record["imported"] = assert_clone_import(workdir)
        record["environments"] = environments()
        env = clone_env(workdir)
        ok = True
        for step in steps:
            before_runs = {p.name for p in (workdir / "outputs" / "runs").iterdir()} \
                if (workdir / "outputs" / "runs").exists() else set()
            started = time.time()
            log = record_dir / f"{step.name}.log"
            with log.open("w", encoding="utf-8", newline="\n") as f:
                proc = subprocess.run([sys.executable, *step.args], cwd=workdir, env=env,
                                      stdout=f, stderr=subprocess.STDOUT, text=True)
            entry = {"step": step.name, "command": "python " + " ".join(step.args),
                     "exit": proc.returncode, "seconds": round(time.time() - started, 1),
                     "reads_validation": step.reads_validation}
            if step.reference:
                run_dir = new_run(workdir, before_runs)
                entry["run"] = run_dir.name if run_dir else None
                rows = compare_run(run_dir, REPO / step.reference) if run_dir else []
            else:
                rows = [{"path": rel, **compare_file(REPO / rel, workdir / rel)}
                        for rel in written_files(workdir, step.roots, started)]
            entry["compared"] = rows
            entry["failing"] = failing(rows)
            entry["counts"] = {k: sum(1 for r in rows if r["kind"] == k)
                               for k in sorted({r["kind"] for r in rows})}
            record["steps"].append(entry)
            (record_dir / "record.json").write_text(json.dumps(record, indent=1),
                                                    encoding="utf-8", newline="\n")
            print(f"{step.name}: exit {proc.returncode}, {entry['seconds']} s, "
                  f"{len(rows)} files, {len(entry['failing'])} failing", flush=True)
            if proc.returncode != 0 or entry["failing"]:
                ok = False
                if proc.returncode != 0:
                    break  # a later stage would run on missing inputs
            restore_reference(workdir, step.roots)
        record["result"] = "reproduced" if ok and len(record["steps"]) == len(steps) \
            else "NOT reproduced"
    finally:
        site = workdir / PRIVATE_SITE
        if site.exists():
            site.unlink()  # T-45: the private file does not stay in the clone
        record["private_site_removed"] = not site.exists()
        (record_dir / "record.json").write_text(json.dumps(record, indent=1), encoding="utf-8",
                                                newline="\n")
        (record_dir / "report.md").write_text(report(record), encoding="utf-8", newline="\n")
    print(f"{record['result']}; record in {record_dir.relative_to(REPO)}")
    return 0 if record["result"] == "reproduced" else 1


def report(record: dict) -> str:
    lines = [f"# Reproduction {record['stamp']}", "",
             f"- commit `{record.get('commit', '?')[:10]}`, work folder `{record['workdir']}`",
             f"- options: {record['options']}",
             f"- solarpv imported from `{record.get('imported', '?')}`",
             f"- result: **{record.get('result', 'interrupted')}**", "",
             "| step | exit | seconds | files | compared equal | failing | listed |",
             "|---|---|---|---|---|---|---|"]
    for s in record["steps"]:
        rows = s["compared"]
        equal = sum(1 for r in rows if r["equal"] is True)
        listed = sum(1 for r in rows if r["equal"] is None)
        lines.append(f"| {s['step']} | {s['exit']} | {s['seconds']} | {len(rows)} | {equal} | "
                     f"{len(s['failing'])} | {listed} |")
    lines += ["", "## Failing and listed files", ""]
    for s in record["steps"]:
        for r in s["compared"]:
            if r["equal"] is not True and r["kind"] not in ("not compared", "new"):
                lines.append(f"- {s['step']}: `{r['path']}` ({r['kind']}): {r['detail']}")
    env = record.get("environments", {})
    if env:
        lines += ["", "## Environments (nothing installed)", "",
                  f"- solar: Python {env['solar']['python']}, {env['solar']['pinned']} pins in "
                  f"`environment.lock.yml`, differing: {env['solar']['differs_from_lock']}",
                  f"- solar-gpu: {env.get('solar_gpu', {}).get('pinned', 0)} pins, differing: "
                  f"{env.get('solar_gpu', {}).get('differs', {})}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = parser("python scripts/reproduce_paper_results.py", __doc__)
    p.add_argument("--plan", action="store_true", help="print the steps; write nothing")
    p.add_argument("--dev", action="store_true", help="reproduce the development state")
    p.add_argument("--workdir", help="an empty folder outside this checkout for the clone")
    p.add_argument("--from", dest="from_", help="continue a reproduction from this step")
    p.add_argument("--skip-neural", action="store_true", help="no neural development run")
    p.add_argument("--rerun-selection", action="store_true",
                   help="also rerun the train part of stage 7 (about 95 minutes)")
    p.add_argument("--with-eda", action="store_true", help="also stages 3 and 3b")
    return run(p.parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())

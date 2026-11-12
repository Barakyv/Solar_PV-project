"""Preflight of every run, and the `check` command: environment, data, gates (docs/stage9_plan.md
§3.1, §10).

- Environment: Python, the versions of the packages, CPU, memory, disk, GPU. Nothing is
  imported to read a version (importlib.metadata), so `check` stays fast.
- Data (M4): every input the runner reads is compared with the SHA-256 recorded by the stage
  that produced it (the stage manifests), and git-tracked inputs must be unmodified. A missing
  or changed file is an error, never a default.
- Stage-7 provenance (X11): the tables named in the stage-7 run log (the log of
  `selection_manifest.json`) must still be the tables the feature sets were selected on: the
  SHA-256 prefix written in that log must match the current file.
- Pending decisions: a decision file that a later step reads (D-15, `configs/
  monotone_decision.yaml`) is reported as pending while it does not exist; the step that needs it
  refuses to run. Once it exists, it must be tracked and unmodified like any other input.
- Gates: the commit and the state of the tree; `final` needs the tag `g1-frozen` on HEAD, a clean
  tree and the private site file (D-08).
"""

import json
import os
import platform
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

from solarpv.config.constants import MONOTONE_DECISION_FILE, MOS_DECISION_FILE, XGB_THREADS
from solarpv.config.paths import OUTPUTS_DIR, PROCESSED_DIR, REPO_ROOT
from solarpv.config.provenance import find_git, git_commit, sha256
from solarpv.config.site import load_site
from solarpv.pipeline.config import FROZEN_TAG

# distribution names; the first group is needed by the steps of W1 and W2
REQUIRED = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "pvlib",
            "PyYAML", "xgboost", "optuna", "SQLAlchemy")
OPTIONAL = ("shap", "matplotlib", "psutil", "torch", "lightning",
            "pytorch-forecasting", "neuralforecast")
STAGE_MANIFESTS = (PROCESSED_DIR / "06_features" / "features_manifest.json",
                   PROCESSED_DIR / "06b_gfs" / "gfs_manifest.json")
SELECTION_MANIFEST = OUTPUTS_DIR / "07_feature_selection" / "selection_manifest.json"
MONOTONE_DECISION = MONOTONE_DECISION_FILE
MOS_DECISION = MOS_DECISION_FILE
TRACKED_INPUTS = (OUTPUTS_DIR / "07_feature_selection" / "model_inputs.json",
                  SELECTION_MANIFEST,
                  REPO_ROOT / "configs" / "split_manifest.yaml",
                  MONOTONE_DECISION, MOS_DECISION)
# tracked inputs that may not exist yet: path -> what to run (the step that reads it refuses)
PENDING_DECISIONS = {MONOTONE_DECISION: "D-15 not decided yet: run python -m "
                                        "solarpv.models.xgb_checks d15 (the xgb step refuses "
                                        "without it)",
                     MOS_DECISION: "L-09 not decided yet: run python -m "
                                   "solarpv.models.xgb_checks l09 --run <dev run> (the xgb "
                                   "step refuses without it)"}
INPUT_LINE = re.compile(r"\bINPUT\s+(\S+)\s+sha256=([0-9a-f]{8,64})\b")


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ── environment ─────────────────────────────────────────────────────────────


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _memory_gb() -> dict | None:
    try:
        import psutil
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / 2**30, 1), "available_gb": round(vm.available / 2**30, 1)}


def _gpu() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, memory, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "memory": memory, "driver": driver})
    return gpus


def environment() -> dict:
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {d: _version(d) for d in REQUIRED + OPTIONAL},
        "cpu_count": os.cpu_count(),
        "xgb_threads": XGB_THREADS,
        "memory": _memory_gb(),
        "disk_free_gb": round(disk.free / 2**30, 1),
        "gpu": _gpu(),
        "gpu_env": _gpu_env(),
    }


def _gpu_env() -> dict:
    """D-29: where `solar-gpu` is. Its full checks (versions, CUDA, sm_61) run in every neural
    step before any training (`pipeline.neural.gpu_report`)."""
    from solarpv.pipeline.neural import GPU_ENV_LOCK, gpu_python

    exe = gpu_python()
    return {"python": str(exe), "exists": exe.exists(), "lock": GPU_ENV_LOCK.exists()}


def environment_problems(env: dict) -> tuple[list[str], list[str]]:
    errors = [f"package {d} is missing" for d in REQUIRED if env["packages"].get(d) is None]
    warnings = []
    gpu_env = env.get("gpu_env") or {}
    if gpu_env and not gpu_env.get("exists"):
        warnings.append(f"no GPU environment solar-gpu at {gpu_env.get('python')}: the neural "
                        "steps will refuse to run (D-29)")
    if env["cpu_count"] and env["cpu_count"] < XGB_THREADS:
        warnings.append(f"{env['cpu_count']} CPUs, fewer than XGB_THREADS = {XGB_THREADS}: "
                        "XGBoost results stay comparable but run slower (RH-03)")
    return errors, warnings


# ── data ────────────────────────────────────────────────────────────────────


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([find_git(), *args], cwd=REPO_ROOT, capture_output=True, text=True)


def stage7_provenance(manifest=SELECTION_MANIFEST) -> list[dict]:
    """X11: one row per table read by the stage-7 selection run. The SHA-256 prefix that run's
    log wrote for the table must match the current file, or the feature sets of
    `model_inputs.json` were chosen on other tables. The validation look's log is checked too
    when it names inputs (the run of 2026-11-03 named none)."""
    manifest = Path(manifest)
    source = "stage-7 provenance"
    if not manifest.exists():
        return [{"path": _rel(manifest), "source": source, "ok": False,
                 "detail": "selection manifest missing: run stage 7"}]
    record = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for key, required in (("run_log", True), ("validation_run_log", False)):
        rel_log = record.get(key)
        if not rel_log:
            if required:
                rows.append({"path": _rel(manifest), "source": source, "ok": False,
                             "detail": f"the manifest names no {key}"})
            continue
        log = REPO_ROOT / rel_log
        if not log.exists():
            rows.append({"path": rel_log, "source": source, "ok": False,
                         "detail": "stage-7 run log missing"})
            continue
        found = INPUT_LINE.findall(log.read_text(encoding="utf-8"))
        if not found and required:
            rows.append({"path": rel_log, "source": source, "ok": False,
                         "detail": "the stage-7 run log names no input"})
        for rel, prefix in found:
            path = REPO_ROOT / rel
            if not path.exists():
                rows.append({"path": rel, "source": f"{source} ({log.name})", "ok": False,
                             "detail": "missing"})
                continue
            actual = sha256(path)
            ok = actual.startswith(prefix)
            rows.append({"path": rel, "source": f"{source} ({log.name})", "ok": ok,
                         "detail": "selected on this table" if ok else
                         f"sha256 {actual[:len(prefix)]} differs from {prefix} in the stage-7 "
                         "log: the feature sets were selected on another table; rerun stage 7"})
    return rows


def data_check(manifests=STAGE_MANIFESTS, tracked=TRACKED_INPUTS,
               selection=SELECTION_MANIFEST, pending=None) -> list[dict]:
    """One row per input: where its expected state comes from, and whether it holds. A row with
    `blocking: False` (a pending decision) is reported but does not stop a run."""
    pending = PENDING_DECISIONS if pending is None else pending
    rows = []
    for manifest in manifests:
        manifest = Path(manifest)
        if not manifest.exists():
            rows.append({"path": _rel(manifest), "source": "stage manifest", "ok": False,
                         "detail": "manifest missing: run the stage"})
            continue
        recorded = json.loads(manifest.read_text(encoding="utf-8")).get("outputs", {})
        for rel, expected in recorded.items():
            path = REPO_ROOT / rel
            expected = expected if isinstance(expected, str) else expected.get("sha256")
            if not path.exists():
                rows.append({"path": rel, "source": _rel(manifest), "ok": False,
                             "detail": "missing"})
                continue
            actual = sha256(path)
            rows.append({"path": rel, "source": _rel(manifest), "ok": actual == expected,
                         "detail": "sha256 matches" if actual == expected
                         else f"sha256 {actual[:12]} differs from the manifest {expected[:12]}"})
    for path in tracked:
        path = Path(path)
        rel = _rel(path)
        if not path.exists():
            waiting = {Path(p).resolve(): text for p, text in pending.items()}.get(path.resolve())
            rows.append({"path": rel, "source": "git", "ok": False,
                         "detail": waiting or "missing", "blocking": waiting is None})
            continue
        known = _git("ls-files", "--error-unmatch", "--", rel)
        changed = _git("status", "--porcelain", "--", rel)
        if known.returncode != 0 or changed.returncode != 0:
            rows.append({"path": rel, "source": "git", "ok": False,
                         "detail": "not tracked by git, or git unavailable"})
            continue
        clean = changed.stdout.strip() == ""
        rows.append({"path": rel, "source": "git", "ok": clean,
                     "detail": "unmodified" if clean else "modified since the last commit"})
    if selection is not None:
        rows += stage7_provenance(selection)
    return rows


# ── gates ───────────────────────────────────────────────────────────────────


def git_state() -> dict:
    """`tree_clean`: no tracked file modified anywhere, and no untracked file in the code paths
    (`git_commit()` not '+dirty'). Untracked outputs, such as the folders of earlier runs, do not
    count: they do not change the code that runs (X9 of docs/stage9_plan.md)."""
    commit = git_commit()
    tags = _git("tag", "--points-at", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    clean = (status.stdout.strip() == "" and not commit.endswith("+dirty")
             if status.returncode == 0 else None)
    return {
        "commit": commit,
        "tags_on_head": tags.stdout.split() if tags.returncode == 0 else None,
        "tree_clean": clean,
        "untracked_outputs": len([line for line in _git("status", "--porcelain").stdout
                                  .splitlines() if line.startswith("??")]),
    }


def gate_problems(mode: str, state: dict, site_private: bool) -> tuple[list[str], list[str]]:
    """Errors that stop a run of this mode, and warnings that are only recorded."""
    errors, warnings = [], []
    if state["commit"] == "unknown":
        (errors if mode == "final" else warnings).append("git commit unknown")
    if mode == "final":
        if FROZEN_TAG not in (state["tags_on_head"] or []):
            errors.append(f"HEAD does not carry the tag {FROZEN_TAG!r} (final runs only after "
                          "gate G-1, D-08)")
        if state["tree_clean"] is not True:
            errors.append("the working tree is not clean: a tracked file is modified or "
                          "uncommitted code exists")
        if not site_private:
            errors.append("configs/site.private.yaml is missing: final results need the true "
                          "site")
    else:
        if state["commit"].endswith("+dirty"):
            warnings.append("uncommitted code or configuration: this run is not reproducible "
                            "from a commit, and --resume will not trust it")
        if not site_private:
            warnings.append("placeholder site: results are not valid for the study")
    return errors, warnings


def final_step_gate(expected_commit: str | None) -> list[str]:
    """X38: the gate of a final run, checked again before every step. `HEAD` must still carry
    the tag and be the commit the invocation's preflight passed on, and the tree must be clean;
    a commit or a code edit made during the run stops the next step."""
    state = git_state()
    errors, _ = gate_problems("final", state, load_site().is_private)
    if expected_commit and state["commit"] != expected_commit:
        errors.append(f"HEAD is {state['commit']}, not {expected_commit} of the preflight: "
                      "nothing may be committed during a final run")
    return errors


def tracked_unmodified(path: Path) -> dict:
    """One row: the file exists, is tracked by git and unmodified since the last commit."""
    path = Path(path)
    rel = _rel(path)
    if not path.exists():
        return {"path": rel, "ok": False, "detail": "missing"}
    known = _git("ls-files", "--error-unmatch", "--", rel)
    changed = _git("status", "--porcelain", "--", rel)
    if known.returncode != 0 or changed.returncode != 0:
        return {"path": rel, "ok": False, "detail": "not tracked by git, or git unavailable"}
    clean = changed.stdout.strip() == ""
    return {"path": rel, "ok": clean,
            "detail": "tracked, unmodified" if clean else "modified since the last commit"}


def preflight(mode: str) -> dict:
    """Everything `check` prints, as one record."""
    env = environment()
    data = data_check()
    state = git_state()
    site_private = load_site().is_private
    env_errors, env_warnings = environment_problems(env)
    gate_errors, gate_warnings = gate_problems(mode, state, site_private)
    data_errors = [f"{r['path']}: {r['detail']}" for r in data
                   if not r["ok"] and r.get("blocking", True)]
    data_warnings = [f"{r['path']}: {r['detail']}" for r in data
                     if not r["ok"] and not r.get("blocking", True)]
    return {"mode": mode, "environment": env, "data": data, "git": state,
            "site_private": site_private,
            "errors": env_errors + data_errors + gate_errors,
            "warnings": env_warnings + data_warnings + gate_warnings}


def format_preflight(report: dict) -> list[str]:
    env, state = report["environment"], report["git"]
    pk = env["packages"]
    lines = [
        f"mode: {report['mode']}",
        f"python {env['python']} on {env['platform']}",
        "packages: " + ", ".join(f"{k} {v}" for k, v in pk.items() if v),
        "absent: " + (", ".join(k for k, v in pk.items() if v is None) or "none"),
        f"CPUs {env['cpu_count']} (XGB_THREADS {env['xgb_threads']}); memory {env['memory']}; "
        f"disk free {env['disk_free_gb']} GB",
        "GPU: " + ("; ".join(f"{g['name']}, {g['memory']}, driver {g['driver']}"
                             for g in env["gpu"]) if env["gpu"] else "none found"),
        "GPU environment (D-29): " + ((env.get("gpu_env") or {}).get("python", "")
                                      if (env.get("gpu_env") or {}).get("exists")
                                      else "not found"),
        f"git: {state['commit']}; tags on HEAD {state['tags_on_head']}; "
        f"clean tree {state['tree_clean']}",
        f"site: {'private (true coordinates)' if report['site_private'] else 'placeholder'}",
        f"data: {sum(r['ok'] for r in report['data'])} of {len(report['data'])} inputs as "
        "recorded",
    ]
    lines += [f"  {'FAIL' if r.get('blocking', True) else 'PENDING'} {r['path']}: {r['detail']}"
              for r in report["data"] if not r["ok"]]
    lines += [f"warning: {w}" for w in report["warnings"]]
    lines += [f"ERROR: {e}" for e in report["errors"]]
    lines.append("check passed" if not report["errors"] else "check FAILED")
    return lines

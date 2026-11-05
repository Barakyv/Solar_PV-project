"""Run folders, step selection, resume, status (docs/stage9_plan.md §3, §10.3).

Layout of one run, `outputs/runs/<run_id>/`:
  run.json                  the state of every step; one entry per invocation
  config.yaml               the resolved configuration (mode, windows, slices, budgets)
  invocations/<stamp>.json  the preflight of each invocation (environment, data, git)
  logs/run_<stamp>.log      the log of each invocation; logs/<step>_<stamp>.log per step
  <step>/                   a completed step: its outputs and step.json (hashes, fingerprint)
  <step>.failed-<stamp>/    a failed try, with its step.json; never read as complete
  <step>.superseded-<stamp>/  a completed step replaced by --force; kept, not deleted

A step runs in `.<step>.tmp/` and is renamed to `<step>/` only after its outputs are hashed and
its step.json is written (X4). Nothing is written outside the run folder (X8), except the
test-access log of a final test evaluation.
"""

import hashlib
import json
import os
import statistics
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml

from solarpv.config.paths import OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import git_commit, sha256
from solarpv.config.runlog import RunLog
from solarpv.pipeline import checks
from solarpv.pipeline.config import SPEC, STEP_ORDER, Window, resolve
from solarpv.pipeline.steps import IMPLEMENTED, Context, RunnerError

RUNS_DIR = OUTPUTS_DIR / "runs"
BUDGET_FAMILY = {"xgb": "xgb", "bilstm": "bilstm", "tft96": "tft", "tft168": "tft",
                 "tftbase": "tft", "patchtst": "patchtst"}


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(t: datetime | None = None) -> str:
    return (t or _now()).strftime("%Y%m%dT%H%M%SZ")


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_json(path: Path, data: dict) -> None:
    """Atomic: a reader never sees half a record."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── selection ───────────────────────────────────────────────────────────────


def select_steps(only: list[str] | None = None, from_: str | None = None,
                 to: str | None = None, order: tuple[str, ...] = STEP_ORDER) -> list[str]:
    """The steps to run, in the canonical order. `only` and `from_`/`to` exclude each other."""
    if only and (from_ or to):
        raise ValueError("use either --only or --from/--to")
    named = list(only or []) + [s for s in (from_, to) if s]
    unknown = [s for s in named if s not in order]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; the steps are {list(order)}")
    if only:
        return [s for s in order if s in set(only)]
    first = order.index(from_) if from_ else 0
    last = order.index(to) if to else len(order) - 1
    if first > last:
        raise ValueError(f"--from {from_} comes after --to {to}")
    return list(order[first:last + 1])


def check_implemented(steps: list[str], implemented=IMPLEMENTED) -> None:
    missing = [s for s in steps if s not in implemented]
    if missing:
        later = ", ".join(f"{s} ({SPEC[s].work_item})" if s in SPEC else s for s in missing)
        raise ValueError(f"not implemented yet: {later}. Implemented: {sorted(implemented)}")


# ── records ─────────────────────────────────────────────────────────────────


def step_config(step: str, config: dict) -> dict:
    family = BUDGET_FAMILY.get(step)
    return {"windows": config["windows"], "targets": config["targets"],
            "horizons": config["horizons"], "seeds": config["seeds"],
            "allow_test": config["allow_test"], "xgb_threads": config["xgb_threads"],
            "budget": config["budgets"].get(family) if family else None}


def fingerprint(step: str, mode: str, cfg: dict, commit: str, inputs: dict) -> str:
    blob = json.dumps({"step": step, "mode": mode, "config": cfg, "commit": commit,
                       "inputs": inputs}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def new_run_id(mode: str, runs_dir: Path) -> str:
    base = f"{_stamp()}_{mode}"
    run_id, n = base, 1
    while (runs_dir / run_id).exists():
        n += 1
        run_id = f"{base}_{n}"
    return run_id


def completed_records(run_dir: Path, state: dict) -> dict[str, dict]:
    out = {}
    for step, entry in state["steps"].items():
        record_path = run_dir / entry.get("folder", "") / "step.json"
        if entry.get("status") == "ok" and record_path.exists():
            out[step] = read_json(record_path)
    return out


def outputs_intact(run_dir: Path, record: dict) -> bool:
    folder = run_dir / record["folder"]
    return all((folder / name).exists() and sha256(folder / name) == digest
               for name, digest in record["outputs"].items())


def _move_aside(path: Path, kind: str) -> Path:
    target = path.with_name(f"{path.name.lstrip('.').removesuffix('.tmp')}.{kind}-{_stamp()}")
    n = 1
    while target.exists():
        n += 1
        target = target.with_name(f"{target.name}_{n}")
    path.rename(target)
    return target


# ── the runner ──────────────────────────────────────────────────────────────


class Runner:
    def __init__(self, runs_dir: Path = RUNS_DIR, implemented=IMPLEMENTED,
                 preflight=checks.preflight, echo: bool = True):
        self.runs_dir = Path(runs_dir)
        self.implemented = implemented
        self.preflight = preflight
        self.echo = echo

    def say(self, text: str = "") -> None:
        if self.echo:
            print(text, flush=True)

    # ── planning ────────────────────────────────────────────────────────────

    def _open(self, mode: str, config: dict, resume: str | None) -> tuple[str, Path, dict]:
        if resume is None:
            run_id = new_run_id(mode, self.runs_dir)
            return run_id, self.runs_dir / run_id, {
                "run_id": run_id, "mode": mode, "created_utc": _now().isoformat(),
                "config_sha256": config_sha256(config), "steps": {}, "invocations": []}
        run_dir = self.runs_dir / resume
        if not (run_dir / "run.json").exists():
            raise ValueError(f"no run {resume!r} in {_rel(self.runs_dir)}")
        state = read_json(run_dir / "run.json")
        if state["mode"] != mode:
            raise ValueError(f"run {resume} is a {state['mode']} run, not {mode}")
        if state["config_sha256"] != config_sha256(config):
            raise ValueError(f"run {resume} was created with another configuration "
                             "(slices, windows or budgets); start a new run")
        return resume, run_dir, state

    def dry_run(self, mode: str, steps: list[str], config: dict, run_id: str,
                state: dict) -> int:
        resuming = bool(state["invocations"])
        self.say(f"DRY RUN (nothing is written): mode {mode}, "
                 f"{'resume ' + run_id if resuming else 'a new run'}")
        for w in config["windows"]:
            self.say(f"  window {w['split']}: {w['label']} [{w['start']} .. {w['end']})")
        self.say(f"  targets {config['targets']}, horizons {config['horizons']}, "
                 f"seeds {config['seeds'] or 'from the budget'}")
        for step in steps:
            spec = SPEC[step]
            done = state["steps"].get(step, {}).get("status") == "ok"
            budget = step_config(step, config)["budget"]
            self.say(f"  {step:<10} {spec.purpose}; "
                     f"{'completed: skipped if its fingerprint matches' if done else 'runs'}"
                     f"{'; budget ' + json.dumps(budget) if budget else ''}; "
                     f"estimate {self.estimate(step, mode)}")
        return 0

    def estimate(self, step: str, mode: str) -> str:
        """Median duration of the completed tries of this step and mode in earlier runs."""
        durations = []
        for record in self.runs_dir.glob(f"*/{step}/step.json"):
            data = read_json(record)
            if data.get("mode") == mode and data.get("status") == "ok":
                durations.append(float(data["duration_s"]))
        if not durations:
            return "unknown (no earlier run)"
        return f"{statistics.median(durations):.0f} s (median of {len(durations)})"

    # ── running ─────────────────────────────────────────────────────────────

    def run(self, mode: str, steps: list[str], *, resume: str | None = None,
            force: bool = False, keep_going: bool = False, dry_run: bool = False,
            targets=None, horizons=None, seeds=None, allow_test: bool = False,
            command: str = "") -> int:
        config = resolve(mode, targets, horizons, seeds, allow_test)
        check_implemented(steps, self.implemented)
        run_id, run_dir, state = self._open(mode, config, resume)  # no side effect
        if dry_run:
            return self.dry_run(mode, steps, config, run_id, state)
        report = self.preflight(mode)
        for line in checks.format_preflight(report):
            self.say(line)
        if report["errors"]:
            self.say("run refused: the preflight failed")
            return 2
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "invocations").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
                                   newline="\n")
        started = _now()
        stamp = _stamp(started)
        write_json(run_dir / "invocations" / f"{stamp}.json", report)
        invocation = {"started_utc": started.isoformat(), "command": command,
                      "commit": report["git"]["commit"], "steps": steps, "force": force,
                      "keep_going": keep_going, "preflight": f"invocations/{stamp}.json",
                      "results": {}}
        state["invocations"].append(invocation)
        write_json(run_dir / "run.json", state)
        windows = [Window(**w) for w in config["windows"]]
        summary: list[str] = []
        failed_any = False
        with RunLog(f"runs/{run_id}", command, path=run_dir / "logs" / f"run_{stamp}.log",
                    echo=self.echo) as log:
            invocation["log"] = log.relative_path
            log.note(f"run {run_id}, mode {mode}, steps {steps}")
            for w in report["warnings"]:
                log.warn(w)
            for step in steps:
                outcome, lines = self._step(step, mode, config, windows, run_id, run_dir,
                                            state, force, command, log)
                invocation["results"][step] = outcome
                write_json(run_dir / "run.json", state)
                summary += lines
                if outcome.startswith("failed") or outcome.startswith("refused"):
                    failed_any = True
                    if not keep_going:
                        log.note(f"stopped after {step} ({outcome}); --keep-going would go on")
                        break
            invocation["status"] = "failed" if failed_any else "ok"
            write_json(run_dir / "run.json", state)
            log.value("status", invocation["status"])
        self.say()
        self.say(f"run {run_id} ({_rel(run_dir)}): {invocation['status']}")
        for step, outcome in invocation["results"].items():
            self.say(f"  {step}: {outcome}")
        for line in summary:
            self.say(line)
        return 1 if failed_any else 0

    def _step(self, step, mode, config, windows, run_id, run_dir, state, force, command,
              log) -> tuple[str, list[str]]:
        impl = self.implemented[step]
        completed = completed_records(run_dir, state)
        failed = [s for s, e in state["steps"].items() if e.get("status") == "failed"]
        cfg = step_config(step, config)
        tmp = run_dir / f".{step}.tmp"
        if tmp.exists():  # left by an interrupted try
            log.warn(f"{_rel(tmp)} left by an interrupted try, moved to "
                     f"{_move_aside(tmp, 'interrupted').name}")
        tmp.mkdir()
        ctx = Context(run_id=run_id, run_dir=run_dir, mode=mode, config=config, windows=windows,
                      out=tmp, log=log, completed=completed, failed=failed, command=command)
        try:
            inputs = {_rel(p): sha256(p) for p in impl.inputs(ctx)}
        except Exception as exc:  # e.g. a modified prediction file of an earlier step
            tmp.rmdir()
            log.warn(f"{step}: inputs refused: {exc}")
            return f"refused ({exc})", []  # the step's state is left as it was
        commit = git_commit()
        fp = fingerprint(step, mode, cfg, commit, inputs)
        if step in completed and not force:
            record = completed[step]
            intact = outputs_intact(run_dir, record)
            tmp.rmdir()
            if record["fingerprint"] == fp and intact and not commit.endswith("+dirty"):
                log.note(f"{step}: up to date (fingerprint {fp[:12]}), skipped")
                return "skipped (up to date)", record.get("summary", [])
            reason = ("its outputs were modified" if not intact
                      else "it ran from uncommitted code (+dirty), which a resume does not "
                           "trust" if commit.endswith("+dirty")
                      else "it was completed with other code, configuration or inputs")
            log.warn(f"{step}: {reason}; use --force to rerun it")
            return f"refused ({reason}; use --force)", []
        if (run_dir / step).exists():  # completed (--force) or left by a refused state
            moved = _move_aside(run_dir / step, "superseded")
            log.note(f"{step}: the earlier folder moved to {moved.name}")
        started = _now()
        t0 = time.perf_counter()
        step_log_path = run_dir / "logs" / f"{step}_{_stamp(started)}.log"
        record = {"step": step, "run_id": run_id, "mode": mode, "started_utc":
                  started.isoformat(), "commit": commit, "fingerprint": fp, "config": cfg,
                  "inputs": inputs}
        try:
            with RunLog(f"runs/{run_id}/{step}", command, path=step_log_path,
                        echo=self.echo) as slog:
                record["log"] = slog.relative_path
                slog.inputs([REPO_ROOT / p for p in inputs])
                ctx.log = slog
                result = impl.run(ctx)
                outside = [p for p in result.outputs
                           if not Path(p).resolve().is_relative_to(tmp.resolve())]
                if outside:
                    raise RunnerError(f"{step} wrote outside its folder: {outside}")
                slog.outputs(result.outputs)
                warnings = list(slog.warnings)
        except Exception as exc:
            folder = _move_aside(tmp, "failed")
            record.update({"status": "failed", "folder": folder.name,
                           "duration_s": round(time.perf_counter() - t0, 1),
                           "error": "".join(traceback.format_exception_only(exc)).strip()})
            write_json(folder / "step.json", record)
            state["steps"][step] = {"status": "failed", "folder": folder.name,
                                    "error": record["error"]}
            log.warn(f"{step} failed: {record['error']}")
            return f"failed ({type(exc).__name__}: {exc})", []
        record.update({
            "status": "ok", "folder": step, "finished_utc": _now().isoformat(),
            "duration_s": round(time.perf_counter() - t0, 1),
            "outputs": {Path(p).resolve().relative_to(tmp.resolve()).as_posix(): sha256(p)
                        for p in result.outputs},
            "values": result.values, "notes": result.notes, "summary": result.summary,
            "warnings": warnings})
        write_json(tmp / "step.json", record)
        tmp.rename(run_dir / step)
        state["steps"][step] = {"status": "ok", "folder": step, "fingerprint": fp,
                                "commit": commit, "finished_utc": record["finished_utc"],
                                "duration_s": record["duration_s"]}
        log.note(f"{step}: ok in {record['duration_s']} s, {len(result.outputs)} outputs")
        return "ok", result.summary

    # ── status ──────────────────────────────────────────────────────────────

    def status(self, run_id: str | None = None) -> int:
        if run_id is None:
            runs = sorted(p.parent for p in self.runs_dir.glob("*/run.json"))
            if not runs:
                self.say(f"no runs in {_rel(self.runs_dir)}")
            for run_dir in runs:
                state = read_json(run_dir / "run.json")
                steps = ", ".join(f"{s} {e['status']}" for s, e in state["steps"].items())
                self.say(f"{state['run_id']}  {state['mode']:<5}  "
                         f"{len(state['invocations'])} invocation(s)  {steps or 'no step'}")
            return 0
        run_dir = self.runs_dir / run_id
        if not (run_dir / "run.json").exists():
            self.say(f"no run {run_id!r}")
            return 1
        state = read_json(run_dir / "run.json")
        records = completed_records(run_dir, state)
        self.say(f"run {run_id}: mode {state['mode']}, created {state['created_utc']}, "
                 f"{len(state['invocations'])} invocation(s)")
        bad = 0
        for step, entry in state["steps"].items():
            if step in records:
                intact = outputs_intact(run_dir, records[step])
                bad += not intact
                rec = records[step]
                self.say(f"  {step:<10} ok in {rec['duration_s']} s at {rec['commit']}, "
                         f"fingerprint {rec['fingerprint'][:12]}, outputs "
                         f"{'as recorded' if intact else 'MODIFIED'}")
            else:
                self.say(f"  {step:<10} {entry['status']} ({entry.get('error', '')})")
        return 1 if bad else 0

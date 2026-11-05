"""Run log: a written record of every run of a stage command (user request, 2026-10-20).

Each run writes ``outputs/<stage>/runs/<UTC timestamp>.log`` and echoes the same lines to the
console. The log holds:
  header   command, git commit, start time, Python and library versions
  inputs   path and sha256 of every input file
  steps    one line when a step starts and one when it ends, with its duration
  notes    intermediate results and observations (counts, checks, decisions taken)
  values   named numbers, also collected in ``RunLog.values`` for the stage manifest
  warnings anything unexpected that did not stop the run
  outputs  path and sha256 of every file written
  footer   status (ok / failed with the traceback) and total duration
Lines are flushed as they are written, so a run that crashes still leaves its log up to the
failure. The human-written summary of each run goes to docs/run_journal.md.

    with RunLog("06_features", "python -m solarpv.features.build") as log:
        log.inputs([path_a, path_b])
        with log.step("targets"):
            log.note("theo_kw > 50 kW on 15,112 daytime hours")
            log.value("mean_daytime_k", 0.71)
        log.outputs([out_file])
"""

import json
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from solarpv.config.paths import OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import git_commit, sha256

LIBRARIES = ("numpy", "pandas", "pvlib", "xgboost", "sklearn")


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except ImportError:
                continue
        out[name] = getattr(module, "__version__", "?")
    return out


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _open_new(path: Path):
    """Create `path`, or `<stem>_2`, `<stem>_3`, ... if it exists: two runs that start in the
    same second never share a file (X5 of docs/stage9_plan.md)."""
    candidate, n = path, 1
    while True:
        try:
            return candidate, candidate.open("x", encoding="utf-8", newline="\n")
        except FileExistsError:
            n += 1
            candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")


class RunLog:
    """Written record of one run; use as a context manager (see the module docstring).

    The file is `<out_dir or outputs/<stage>>/runs/<UTC second>.log`, or `path` if given."""

    def __init__(self, stage: str, command: str, out_dir: Path | None = None,
                 echo: bool = True, path: Path | None = None):
        self.stage = stage
        self.command = command
        self.echo = echo
        self.values: dict = {}
        self.warnings: list[str] = []
        self._logged: dict[str, str] = {}
        self.started = datetime.now(UTC)
        self._t0 = time.perf_counter()
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        if path is None:
            path = (out_dir or OUTPUTS_DIR / stage) / "runs" / f"{stamp}.log"
        self.dir = Path(path).parent
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path, self._file = _open_new(Path(path))
        self._write("RUN", f"stage {stage}")
        self._write("RUN", f"command: {command}")
        self._write("RUN", f"git commit: {git_commit()}")
        self._write("RUN", f"started (UTC): {self.started.isoformat(timespec='seconds')}")
        self._write("RUN", f"versions: {json.dumps(_versions())}")

    # ── writing ────────────────────────────────────────────────────────────

    def _write(self, kind: str, text: str) -> None:
        elapsed = time.perf_counter() - self._t0
        line = f"[{elapsed:8.1f}s] {kind:<7} {text}"
        self._file.write(line + "\n")
        self._file.flush()
        if self.echo:
            print(line, flush=True)

    def note(self, text: str) -> None:
        """An intermediate result or observation, in words."""
        self._write("NOTE", text)

    def value(self, name: str, value) -> None:
        """A named number for the manifest; also written to the log."""
        self.values[name] = value
        self._write("VALUE", f"{name} = {value}")

    def warn(self, text: str) -> None:
        """Something unexpected that does not stop the run."""
        self.warnings.append(text)
        self._write("WARN", text)

    def inputs(self, paths) -> dict:
        """Log and return the sha256 of every input file."""
        out = {}
        for p in paths:
            digest = sha256(Path(p))
            out[_rel(p)] = digest
            self._write("INPUT", f"{_rel(p)} sha256={digest[:16]}")
        return out

    def outputs(self, paths) -> dict:
        """Log and return the sha256 of every output file; a file already logged with the same
        hash is not written again."""
        out = {}
        for p in paths:
            digest = sha256(Path(p))
            out[_rel(p)] = digest
            if self._logged.get(_rel(p)) != digest:
                self._write("OUTPUT", f"{_rel(p)} sha256={digest[:16]}")
            self._logged[_rel(p)] = digest
        return out

    @contextmanager
    def step(self, name: str):
        """Mark the start and end (with duration) of one step of the run."""
        start = time.perf_counter()
        self._write("STEP", f"{name} ...")
        try:
            yield
        except BaseException as exc:
            self._write("STEP", f"{name} failed after {time.perf_counter() - start:.1f}s "
                                f"({type(exc).__name__})")
            raise
        self._write("STEP", f"{name} done in {time.perf_counter() - start:.1f}s")

    # ── closing ────────────────────────────────────────────────────────────

    def close(self, status: str = "ok") -> None:
        if self._file.closed:
            return
        total = time.perf_counter() - self._t0
        self._write("RUN", f"warnings: {len(self.warnings)}")
        self._write("RUN", f"status: {status}; duration {total:.1f}s")
        self._file.close()

    @property
    def relative_path(self) -> str:
        return _rel(self.path)

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            for line in traceback.format_exception(exc_type, exc, tb):
                for part in line.rstrip().splitlines():
                    self._write("ERROR", part)
            self.close(f"failed ({exc_type.__name__})")
            return False  # re-raise
        self.close("ok")
        return False

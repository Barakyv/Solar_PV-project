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


class RunLog:
    """Written record of one run; use as a context manager (see the module docstring)."""

    def __init__(self, stage: str, command: str, out_dir: Path | None = None,
                 echo: bool = True):
        self.stage = stage
        self.command = command
        self.echo = echo
        self.values: dict = {}
        self.warnings: list[str] = []
        self.started = datetime.now(UTC)
        self._t0 = time.perf_counter()
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        self.dir = (out_dir or OUTPUTS_DIR / stage) / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{stamp}.log"
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
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
        """Log and return the sha256 of every output file."""
        out = {}
        for p in paths:
            digest = sha256(Path(p))
            out[_rel(p)] = digest
            self._write("OUTPUT", f"{_rel(p)} sha256={digest[:16]}")
        return out

    @contextmanager
    def step(self, name: str):
        """Mark the start and end (with duration) of one step of the run."""
        start = time.perf_counter()
        self._write("STEP", f"{name} ...")
        yield
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

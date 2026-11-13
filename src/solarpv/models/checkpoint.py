"""Checkpoints and resume inside the long steps (docs/stage9_plan.md §23, D-38, X94).

A step that is interrupted (a reboot, a driver reset, a process killed) continues from where it
stopped instead of from its first line. The unit of a checkpoint is an epoch of a fit, a finished
Optuna trial, and a finished model (a target x seed of a neural step, a target x horizon x seed of
`xgb`). Imported in both environments (`solar` and `solar-gpu`): the standard library at import
time; `torch` and `optuna` only inside the functions that need them.

The rules (§23.2):

- The folder is `<run folder>/checkpoints/<step>/`, outside the step's temporary folder, so that a
  resume finds it after the temporary folder was moved aside. Every file is written atomically (a
  `.part` file, `fsync`, `os.replace`): a file that exists is complete.
- `manifest.json` binds the folder to a run, a step, a mode and the fingerprint of the step (its
  code commit, configuration and the SHA-256 of every input). A folder of another identity is
  moved aside (`<step>.stale-<time>`) and the step starts again; a tree with uncommitted code never
  reads a checkpoint (`trust=False`).
- An Optuna study is not edited: it is replayed from its finished trials (`Study.add_trials`) and
  the pickled sampler, so that the next trial draws the parameters an uninterrupted study would
  draw, and a trial that was running when the process died is run again.
- The pickles are files this code wrote, in a folder of the run; nothing here reads a file that
  came from outside.
- `SOLARPV_TEST_FAULT=<point>` ends the process at that point (the tests use it to interrupt a
  worker); with the variable unset `fault` does nothing.
"""

import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

VERSION = 1
MANIFEST = "manifest.json"
UNIT_FILE = "unit.json"
PART = ".part"
FAULT_VARIABLE = "SOLARPV_TEST_FAULT"
FAULT_EXIT_CODE = 86
# the seconds between two checkpoints of a fit in flight (R-G1-33: the state of a large BiLSTM is
# 83 MB and takes 0.44 s to write and sync, more than its epochs; a TFT epoch takes 5 s, its state
# 1.5 s). 0 writes after every epoch. The environment variable changes it for one run.
FIT_INTERVAL_S = 60.0
INTERVAL_VARIABLE = "SOLARPV_CHECKPOINT_INTERVAL_S"
REPLACE_TRIES, REPLACE_WAIT_S = 5, 0.4  # R-G1-28: an antivirus or an indexer may hold a file


class CheckpointError(RuntimeError):
    """A checkpoint breaks its own rules (a stale or damaged one is dropped, not an error)."""


def fault(point: str) -> None:
    """Test support: end the process now (exit code 86, nothing flushed but stdout) when the
    environment names this point. Does nothing in production, where the variable is unset."""
    if os.environ.get(FAULT_VARIABLE) == point:
        print(f"[fault] {point}", flush=True)
        sys.stdout.flush()
        os._exit(FAULT_EXIT_CODE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, write) -> None:
    """`write(binary file)` into `<name>.part`, flushed and synced, then renamed over `path`."""
    path = Path(path)
    tmp = path.with_name(path.name + PART)
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except OSError:  # on Windows another process may hold the target for a moment
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_S)


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _json_bytes(data) -> bytes:
    return (json.dumps(data, indent=2, default=str) + "\n").encode("utf-8")


class Store:
    """The checkpoint folder of one step. `root=None` is a store that does nothing (a step that
    was started without checkpoints, for example by a test)."""

    def __init__(self, root: Path | None, identity: dict | None = None, trust: bool = True,
                 say=print, warn=None):
        self.root = None if root is None else Path(root)
        self.identity = dict(identity or {})
        self.say = say
        self.warn = warn if warn is not None else say  # a folder that is not used, a damaged file
        self.envs: list[dict] = []  # every software and GPU that wrote into the folder (R-G1-36)
        self.enabled = self.root is not None
        self.events: list[dict] = []  # what this try restored (the provenance of §23.2 rule 5)
        self.stale: Path | None = None
        self.data: dict[str, str] = {}
        self.had_checkpoint = False
        if not self.enabled:
            return
        manifest = self.root / MANIFEST
        if self.root.exists():
            reason = None
            if not manifest.exists():
                reason = "a folder without a manifest" if any(self.root.iterdir()) else None
            else:
                try:
                    old = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, ValueError):  # R-G1-30: unreadable: another identity
                    old = None
                if old is None:
                    reason = "its manifest cannot be read"
                elif old.get("version") != VERSION or old.get("identity") != self.identity:
                    reason = "it belongs to another step fingerprint (other code, inputs or run)"
                elif not trust:
                    reason = "the tree has uncommitted code, which a resume does not trust"
                else:
                    self.data = dict(old.get("data", {}))
                    self.envs = list(old.get("environments") or [])
                    self.events = []
            if reason is not None:
                self.stale = self.root.with_name(f"{self.root.name}.stale-{_stamp()}")
                n = 1
                while self.stale.exists():
                    n += 1
                    self.stale = self.stale.with_name(f"{self.stale.name}_{n}")
                os.replace(self.root, self.stale)
                self.warn(f"checkpoint: {self.root.name} not used ({reason}); moved to "
                          f"{self.stale.name}, the step starts from its first line")
        self.root.mkdir(parents=True, exist_ok=True)
        for part in self.root.rglob(f"*{PART}"):
            part.unlink()  # a write that was cut off
        self.had_checkpoint = (self.root / MANIFEST).exists()
        if not self.had_checkpoint:
            self._write_manifest()

    # ── the manifest ────────────────────────────────────────────────────────

    def _write_manifest(self) -> None:
        write_atomic(self.root / MANIFEST, lambda f: f.write(_json_bytes({
            "version": VERSION, "identity": self.identity, "data": self.data,
            "environments": self.envs,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})))

    def bind(self, name: str, digest: str) -> None:
        """Bind the data of one part of the step (a target's arrays) to its content digest. A
        checkpoint of that part made from other data is dropped (the build is meant to be
        deterministic, so this is a guard, not a path)."""
        if not self.enabled:
            return
        old = self.data.get(name)
        if old is not None and old != digest:
            self.warn(f"checkpoint: the data of {name} changed since the checkpoint; its "
                      "checkpoints are dropped")
            self.drop_matching(f"_{name}_seed")
        if old != digest:
            self.data[name] = digest
            self._write_manifest()

    def environment(self, info: dict) -> None:
        """Record the software and the GPU of this try. A resume on another stack (another GPU,
        driver, torch or cuDNN) is a valid run but not claimed to equal an uninterrupted one
        (A-20, R-G1-31): the difference is a warning and an event of the step's record."""
        if not self.enabled:
            return
        others = [e for e in self.envs if e != info]
        if others:  # in every try that finds more than one: the step that completes says it
            self.events.append({"kind": "environment_differs", "made_on": others, "now": info})
            self.warn(f"checkpoint: part of the folder was made on another environment "
                      f"({others}; now {info}); the continued step is valid, and not claimed "
                      "to equal an uninterrupted one")
        if info not in self.envs:
            self.envs.append(info)
            self._write_manifest()

    def note(self, kind: str, **fields) -> None:
        """Record what this try restored; the step puts the list in its record."""
        event = {"kind": kind, **fields}
        self.events.append(event)
        self.say("checkpoint: restored " + ", ".join(f"{k} {v}" for k, v in event.items()))

    # ── files ───────────────────────────────────────────────────────────────

    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return self.enabled and (self.root / name).exists()

    def drop(self, *names: str) -> None:
        if not self.enabled:
            return
        for name in names:
            p = self.root / name
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    def drop_matching(self, token: str) -> None:
        """Delete every file or folder of the store whose name contains `token`."""
        if self.enabled:
            for p in list(self.root.iterdir()):
                if token in p.name and p.name != MANIFEST:
                    self.drop(p.name)

    def save_pickle(self, name: str, obj) -> None:
        if self.enabled:
            write_atomic(self.root / name, lambda f: pickle.dump(obj, f, protocol=4))

    def load_pickle(self, name: str):
        """The object, or None when there is none or it cannot be read (it is then deleted)."""
        if not self.has(name):
            return None
        try:
            with open(self.root / name, "rb") as f:
                return pickle.load(f)
        except Exception as exc:  # noqa: BLE001 - a damaged checkpoint is dropped, never trusted
            self.warn(f"checkpoint: {name} cannot be read ({type(exc).__name__}); it is deleted")
            self.drop(name)
            return None

    def save_torch(self, name: str, obj) -> None:
        if self.enabled:
            import torch

            write_atomic(self.root / name, lambda f: torch.save(obj, f))

    def load_torch(self, name: str):
        if not self.has(name):
            return None
        import torch

        try:
            return torch.load(self.root / name, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            self.warn(f"checkpoint: {name} cannot be read ({type(exc).__name__}); it is deleted")
            self.drop(name)
            return None

    # ── units: a finished model, with its files and their hashes ────────────

    def save_unit(self, name: str, files: dict[str, Path], record: dict) -> None:
        """`files`: relative name -> the file to keep. The unit counts only once `unit.json`, which
        is written last, exists."""
        if not self.enabled:
            return
        folder = self.root / f"unit_{name}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        hashes = {}
        for rel, src in files.items():
            dst = folder / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(dst, lambda f, src=src: f.write(Path(src).read_bytes()))
            hashes[rel] = sha256_file(dst)
        write_atomic(folder / UNIT_FILE, lambda f: f.write(_json_bytes(
            {"unit": name, "files": hashes, "record": record})))

    def restore_unit(self, name: str, dest: Path) -> dict | None:
        """Copy the unit's files to `dest` (each checked against its hash) and return its record;
        None when there is no complete unit (a damaged one is deleted)."""
        if not self.has(f"unit_{name}/{UNIT_FILE}"):
            return None
        folder = self.root / f"unit_{name}"
        try:
            meta = json.loads((folder / UNIT_FILE).read_text(encoding="utf-8"))
            for rel, digest in meta["files"].items():
                if sha256_file(folder / rel) != digest:
                    raise CheckpointError(f"{rel}: SHA-256 differs from the unit record")
        except Exception as exc:  # noqa: BLE001
            self.warn(f"checkpoint: unit {name} is damaged ({exc}); it is deleted")
            shutil.rmtree(folder, ignore_errors=True)
            return None
        dest = Path(dest)
        for rel in meta["files"]:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / rel, dest / rel)
        return meta["record"]


def open_store(root: Path | None, identity: dict | None, trust: bool = True, say=print,
               warn=None) -> Store:
    return Store(root, identity, trust=trust, say=say, warn=warn)


# ── Optuna studies: replayed, not edited ────────────────────────────────────


def study_state(study) -> dict:
    """The finished trials of a study and its sampler (with its random state)."""
    import optuna

    return {"trials": study.get_trials(deepcopy=True, states=(optuna.trial.TrialState.COMPLETE,)),
            "sampler": study.sampler}


def replayed_study(storage, name: str, seed: int, state: dict | None = None,
                   direction: str = "minimize"):
    """A new study in `storage` that holds the finished trials of `state` (their numbers kept) and
    continues with the sampler of `state`, or with a fresh TPE sampler of `seed`."""
    import optuna

    sampler = state["sampler"] if state else optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(storage=storage, sampler=sampler, study_name=name,
                                direction=direction)
    if state and state["trials"]:
        study.add_trials(state["trials"])
        if [t.number for t in study.trials] != list(range(len(state["trials"]))):
            raise CheckpointError(f"{name}: the replayed trials did not keep their numbers")
    return study


def study_callback(store: Store, key: str, point: str, extra=None):
    """An Optuna callback: after every finished trial the study is saved under `key` (with
    `extra()` if given), then the fault point `<point>:<trial number>` is offered."""
    def callback(study, trial) -> None:
        if store.enabled:
            state = study_state(study)
            if extra is not None:
                state["extra"] = extra()
            store.save_pickle(key, state)
        fault(f"{point}:{trial.number}")

    return callback


# ── what a checkpoint folder holds (status) ─────────────────────────────────


def describe(root: Path) -> list[str]:
    """Lines for `status`: the identity, the finished units and the fit in flight."""
    root = Path(root)
    lines = []
    try:
        meta = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"{root.name}: no readable manifest"]
    units = sorted([p.name.removeprefix("unit_") for p in root.glob("unit_*")
                    if (p / UNIT_FILE).exists()]
                   + [p.name.removeprefix("unit_").removesuffix(".pkl")
                      for p in root.glob("unit_*.pkl")])
    studies = sorted(p.name for p in root.glob("study_*"))
    fits = sorted(p.name for p in root.glob("fit_*"))
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**20
    lines.append(f"{root.name}: {len(units)} finished unit(s), {len(studies)} study checkpoint(s), "
                 f"{len(fits)} fit checkpoint(s), {size:.1f} MB, fingerprint "
                 f"{str(meta.get('identity', {}).get('fingerprint', ''))[:12]}")
    if units:
        lines.append("  units: " + ", ".join(units))
    return lines


def remove_all(root: Path, step: str) -> tuple[int, float]:
    """After a step succeeded: delete its checkpoint folder and the stale ones of earlier tries.
    Returns (files, MB)."""
    root = Path(root)
    n, size = 0, 0
    for folder in [root / step, *sorted(root.glob(f"{step}.stale-*"))]:
        if folder.exists():
            files = [f for f in folder.rglob("*") if f.is_file()]
            n += len(files)
            size += sum(f.stat().st_size for f in files)
            shutil.rmtree(folder, ignore_errors=True)
    if root.exists() and not any(root.iterdir()):
        root.rmdir()
    return n, size / 2**20

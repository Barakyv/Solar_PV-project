"""`python -m solarpv.pipeline archive <run_id>` (docs/stage9_plan.md §19.5, D-31).

One zip of a whole run folder, including the files that git ignores (prediction files, arrays,
Optuna studies, models), with the SHA-256 and size of every file in `archive_manifest.json`
inside the zip, and the zip's own SHA-256 in `<zip>.sha256` next to it. `--verify <zip>` checks
every member against the manifest. The zip goes to `outputs/archives/` (ignored by git); copying
it off the VDI, whose persistence after logoff is unknown (§1.1), is the user's step. Temporary
folders of interrupted steps are not packed; a file named like the private site file is refused
(it never belongs in a run folder, T-45).
"""

import hashlib
import io
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from solarpv.config.paths import OUTPUTS_DIR
from solarpv.config.provenance import git_commit, sha256

ARCHIVES_DIR = OUTPUTS_DIR / "archives"
MANIFEST = "archive_manifest.json"


class ArchiveError(RuntimeError):
    pass


def _files(run_dir: Path) -> list[Path]:
    out = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir)
        if any(part.startswith(".") and part.endswith(".tmp") for part in rel.parts):
            continue
        if "site.private" in p.name:
            raise ArchiveError(f"{rel}: the private site file is never archived (T-45)")
        out.append(p)
    return out


def archive(run_dir: Path, dest: Path = ARCHIVES_DIR) -> Path:
    run_dir, dest = Path(run_dir).resolve(), Path(dest).resolve()
    if not (run_dir / "run.json").exists():
        raise ArchiveError(f"{run_dir} is not a run folder (no run.json)")
    if dest.is_relative_to(run_dir):
        raise ArchiveError("the archive cannot be written inside the run folder")
    state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    files = _files(run_dir)
    manifest = {"run_id": state["run_id"], "mode": state["mode"],
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "archived_at_commit": git_commit(), "n_files": len(files),
                "files": {p.relative_to(run_dir).as_posix(): {"sha256": sha256(p),
                                                              "bytes": p.stat().st_size}
                          for p in files}}
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = dest / f"{state['run_id']}_{stamp}.zip"
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
        for p in files:
            z.write(p, f"{state['run_id']}/{p.relative_to(run_dir).as_posix()}")
    os.replace(tmp, path)
    digest = sha256(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n",
                                                     encoding="utf-8", newline="\n")
    problems = verify(path)
    if problems:
        raise ArchiveError(f"the new archive does not verify: {problems[:5]}")
    return path


def verify(path: Path) -> list[str]:
    """Problems of an archive: members against the manifest, and the zip against its .sha256."""
    path = Path(path)
    problems = []
    side = path.with_name(path.name + ".sha256")
    if side.exists():
        expected = side.read_text(encoding="utf-8").split()[0]
        if sha256(path) != expected:
            problems.append(f"{path.name}: SHA-256 differs from {side.name}")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read(MANIFEST).decode("utf-8"))
        prefix = f"{manifest['run_id']}/"
        members = {n[len(prefix):]: n for n in z.namelist() if n.startswith(prefix)}
        for rel, info in manifest["files"].items():
            if rel not in members:
                problems.append(f"{rel}: missing from the archive")
                continue
            h = hashlib.sha256()
            with z.open(members[rel]) as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != info["sha256"]:
                problems.append(f"{rel}: SHA-256 differs from the manifest")
        extra = sorted(set(members) - set(manifest["files"]))
        problems += [f"{rel}: in the archive but not in the manifest" for rel in extra]
    return problems


def describe(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(io.TextIOWrapper(z.open(MANIFEST), encoding="utf-8").read())
    total = sum(f["bytes"] for f in manifest["files"].values())
    return (f"{Path(path).name}: run {manifest['run_id']} ({manifest['mode']}), "
            f"{manifest['n_files']} files, {total / 2**20:.1f} MB before compression, "
            f"{Path(path).stat().st_size / 2**20:.1f} MB packed")

"""Provenance for stage manifests: code version and file hashes."""

import hashlib
import shutil
import subprocess
from pathlib import Path

from solarpv.config.paths import REPO_ROOT


def git_commit() -> str:
    """Short hash of HEAD, with '+dirty' if tracked files are modified; 'unknown' without git."""
    git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    try:
        head = subprocess.run([git, "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run([git, "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

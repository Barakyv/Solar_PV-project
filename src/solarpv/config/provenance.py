"""Provenance for stage manifests: code version and file hashes."""

import hashlib
import shutil
import subprocess
from pathlib import Path

from solarpv.config.paths import REPO_ROOT

# what produces the outputs; a modified output file does not make the code dirty
CODE_PATHS = ("src", "scripts", "configs", "pyproject.toml", "environment.yml")
# git off PATH: the laptop's system install, the college machine's per-user PortableGit (RH-02)
GIT_FALLBACKS = (Path(r"C:\Program Files\Git\cmd\git.exe"),
                 Path.home() / "PortableGit" / "cmd" / "git.exe")


def find_git() -> str:
    """git on PATH, else the first fallback that exists; 'git' (which then fails) otherwise."""
    found = shutil.which("git")
    if found:
        return found
    return next((str(p) for p in GIT_FALLBACKS if p.exists()), "git")


def git_commit() -> str:
    """Short hash of HEAD, with '+dirty' if code or configuration differs from it: a modified
    tracked file, or an untracked file that is not ignored (a new module that is not committed
    would otherwise run under a clean commit, X6 of docs/stage9_plan.md); 'unknown' without
    git."""
    git = find_git()
    try:
        head = subprocess.run([git, "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run([git, "status", "--porcelain", "--untracked-files=normal", "--",
                                *CODE_PATHS],
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


def text_sha256(path: Path) -> str:
    """Hash of a text file with CRLF read as LF, so a git checkout (eol=lf in .gitattributes)
    does not change it."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

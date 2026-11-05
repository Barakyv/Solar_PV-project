"""Provenance: the manifests record the real commit (RH-03).

On the college machine git is a per-user PortableGit that is not on PATH; `git_commit()` then
returned 'unknown' for every manifest. It now falls back to the known install locations.
"""

import subprocess
from pathlib import Path

from solarpv.config import provenance
from solarpv.config.paths import REPO_ROOT


def test_find_git_falls_back_when_git_is_not_on_path(monkeypatch, tmp_path):
    exe = tmp_path / "PortableGit" / "cmd" / "git.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    monkeypatch.setattr(provenance, "GIT_FALLBACKS", (tmp_path / "missing.exe", exe))
    assert provenance.find_git() == str(exe)
    monkeypatch.setattr(provenance, "GIT_FALLBACKS", (tmp_path / "missing.exe",))
    assert provenance.find_git() == "git"


def test_untracked_code_makes_the_commit_dirty(monkeypatch, tmp_path):
    """X6 (T-54): a new module that is not committed must not run under a clean commit; an
    ignored file does not count."""
    git = provenance.find_git()

    def g(*args):
        subprocess.run([git, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                        *args], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    g("init", "-q")
    g("add", ".")
    g("commit", "-q", "-m", "init")
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    assert not provenance.git_commit().endswith("+dirty")
    (tmp_path / "src" / "cache.pyc").write_bytes(b"")
    assert not provenance.git_commit().endswith("+dirty")
    (tmp_path / "src" / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert provenance.git_commit().endswith("+dirty")


def test_git_commit_is_known_in_the_repository():
    if not (Path(REPO_ROOT) / ".git").exists():
        return
    assert provenance.git_commit() != "unknown"

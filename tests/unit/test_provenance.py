"""Provenance: the manifests record the real commit (RH-03).

On the college machine git is a per-user PortableGit that is not on PATH; `git_commit()` then
returned 'unknown' for every manifest. It now falls back to the known install locations.
"""

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


def test_git_commit_is_known_in_the_repository():
    if not (Path(REPO_ROOT) / ".git").exists():
        return
    assert provenance.git_commit() != "unknown"

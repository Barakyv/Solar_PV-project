"""Shared helpers for the audit of the original project backup (read-only).

The backup (`old files/solar_ds_capstone/`, added by the user on 2026-11-01) is an older working
copy of the original project, from about January 2026. The project went through several
iterations after it, up to the v6 submission (`submission_evidence/`). It is therefore evidence of
a step on the way, never of the final pipeline: where it disagrees with the submission, the
submission prevails (PROJECT_MASTER.md §0, rule 5).

Its `.git` folder is partial (no HEAD, no config, one commit, 2 of the trees), but it keeps 483
loose blobs. Blobs are read here with zlib only; no git command, notebook, pickle or model file
from the backup is ever executed or unpickled.
"""

import hashlib
import json
import zlib
from pathlib import Path

from solarpv.config.paths import REPO_ROOT

BACKUP = REPO_ROOT / "old files" / "solar_ds_capstone"
GIT_OBJECTS = BACKUP / ".git" / "objects"
RECOVERED = REPO_ROOT / "data" / "external" / "original_backup_recovered"  # git-ignored
SUBMISSION = REPO_ROOT / "submission_evidence"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def loose_objects():
    """Yield (sha, kind, body) for every loose object of the partial .git folder."""
    for path in sorted(GIT_OBJECTS.glob("??/*")):
        raw = zlib.decompress(path.read_bytes())
        head, _, body = raw.partition(b"\0")
        yield path.parent.name + path.name, head.split()[0].decode(), body


def worktree_files():
    """Every file of the backup outside .git, as {blob sha: relative path}."""
    out = {}
    for path in sorted(BACKUP.rglob("*")):
        if path.is_file() and ".git" not in path.relative_to(BACKUP).parts:
            out[blob_sha(path.read_bytes())] = path.relative_to(BACKUP).as_posix()
    return out


def kind_of(body: bytes) -> str:
    if body[:4] == b"PAR1":
        return "parquet"
    if body[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if body[:16] == b"SQLite format 3\x00":
        return "sqlite"
    if body[:2] == b"PK":
        return "zip (keras / torch)"
    head = body.lstrip()[:1]
    if head == b"{" and b'"cells"' in body[:20000]:
        return "notebook"
    if head in (b"{", b"["):
        return "json"
    try:
        body[:4000].decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "binary"


def notebook_title(body: bytes) -> str:
    nb = json.loads(body.decode("utf-8"))
    for cell in nb.get("cells", []):
        text = "".join(cell.get("source", "")).strip()
        for line in text.splitlines():
            line = line.strip().lstrip("#").strip()
            if line and not set(line) <= set("=-# ") and not line.startswith("<"):
                return line[:120]
    return ""


def recovered(sha_prefix: str) -> Path:
    """Path of a recovered blob (written by inventory.py)."""
    hits = sorted(RECOVERED.glob(sha_prefix + "*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"blob {sha_prefix} not recovered; run inventory.py first")
    return hits[0]


def notebook_code(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c.get("source", "")) for c in nb["cells"] if c["cell_type"] == "code"]

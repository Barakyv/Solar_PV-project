"""Pack the project for another machine, and verify the package after it is unpacked there.

    python scripts/handoff_pack.py           # source: handoff/solar_handoff_<date>_<sha>.zip
    python scripts/handoff_pack.py --verify  # target, in the unpacked solar/ folder

The package is the working copy as it is: the git history (.git), every tracked and untracked
file, and the data the next stages read (Raw data/, data/processed/, data/external/gfs_1p00/).
Left out: caches, worktrees, the old backup (old files/solar_ds_capstone/, 2 GB, audited in
00_data_audit/original_backup_2026-11-01/), the blobs recovered from it, and the true site
coordinates (configs/site.private.yaml: copied by hand, only with the user's approval, Q-16).

HANDOFF_MANIFEST.sha256 lists every packed file outside .git with its SHA-256 (the format of
`sha256sum -c`); HANDOFF_INFO.json records the commit, the untracked files and the counts. Git
checks its own objects (`git fsck`). Standard library only, so --verify runs before the conda
environment exists. Hand-off instructions: docs/HANDOFF.md.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "handoff"
MANIFEST = "HANDOFF_MANIFEST.sha256"
INFO = "HANDOFF_INFO.json"
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}
SKIP_PATHS = {
    ".claude/worktrees",
    "old files/solar_ds_capstone",
    "data/external/original_backup_recovered",
    "handoff",
    "src/solarpv.egg-info",
    "configs/site.private.yaml",
    ".env",
    MANIFEST,
    INFO,
}
STORED = {".parquet", ".png", ".zip", ".gz", ".pack", ".xlsx", ".docx", ".pdf"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def files() -> list[str]:
    """Every file to pack, as a POSIX path relative to the project root, in a fixed order."""
    out = []
    for folder, dirs, names in os.walk(ROOT):
        rel = Path(folder).relative_to(ROOT).as_posix()
        rel = "" if rel == "." else rel + "/"
        dirs[:] = sorted(d for d in dirs
                         if d not in SKIP_NAMES and rel + d not in SKIP_PATHS)
        out += [rel + n for n in sorted(names) if rel + n not in SKIP_PATHS]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack() -> int:
    commit = git("rev-parse", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    modified = git("status", "--porcelain", "--untracked-files=no").splitlines()
    names = files()
    lines, n_bytes = [], 0
    for rel in names:
        if not rel.startswith(".git/"):
            lines.append(f"{sha256(ROOT / rel)}  {rel}")
        n_bytes += (ROOT / rel).stat().st_size
    info = {
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "modified_tracked_files": modified,
        "untracked_files": untracked,
        "n_files": len(names),
        "n_files_in_manifest": len(lines),
        "bytes_before_compression": n_bytes,
        "left_out": sorted(SKIP_PATHS - {MANIFEST, INFO}) + sorted(SKIP_NAMES),
        "site_private_present_on_source": (ROOT / "configs/site.private.yaml").exists(),
    }
    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / f"solar_handoff_{dt.date.today():%Y-%m-%d}_{commit[:7]}.zip"
    part = target.with_suffix(".zip.part")
    with zipfile.ZipFile(part, "w", allowZip64=True) as z:
        for rel in names:
            kind = zipfile.ZIP_STORED if Path(rel).suffix.lower() in STORED \
                else zipfile.ZIP_DEFLATED
            z.write(ROOT / rel, f"solar/{rel}", compress_type=kind)
        z.writestr(f"solar/{MANIFEST}", "\n".join(lines) + "\n")
        z.writestr(f"solar/{INFO}", json.dumps(info, indent=2) + "\n")
    with zipfile.ZipFile(part) as z:
        bad = z.testzip()
    if bad is not None:
        print(f"CRC error in the package: {bad}")
        return 1
    part.replace(target)
    print(f"{target}")
    print(f"commit {commit[:7]}, {len(names):,} files, {n_bytes / 1e9:.2f} GB before compression,"
          f" {target.stat().st_size / 1e9:.2f} GB packed; {len(untracked)} untracked files"
          f" included; {len(modified)} modified tracked files")
    return 0


def verify() -> int:
    manifest = ROOT / MANIFEST
    if not manifest.exists():
        print(f"{MANIFEST} not found: run this in the unpacked solar/ folder")
        return 1
    missing, changed, n = [], [], 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        n += 1
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != digest:
            changed.append(rel)
    info = json.loads((ROOT / INFO).read_text(encoding="utf-8"))
    print(f"{n:,} files checked against the package of {info['created']} "
          f"(commit {info['commit'][:7]}): {len(missing)} missing, {len(changed)} changed")
    for rel in (missing + changed)[:20]:
        print(f"  {rel}")
    head = git("rev-parse", "HEAD")
    print(f"git HEAD {head[:7]}" + ("" if head == info["commit"] else
                                     f" (the package was made at {info['commit'][:7]})"))
    if not (ROOT / "configs/site.private.yaml").exists():
        print("configs/site.private.yaml is absent: the code uses the anonymised placeholder site"
              " (configs/site.example.yaml); see docs/HANDOFF.md")
    return 1 if missing or changed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--verify", action="store_true", help="check the unpacked files")
    return verify() if p.parse_args().verify else pack()


if __name__ == "__main__":
    sys.exit(main())

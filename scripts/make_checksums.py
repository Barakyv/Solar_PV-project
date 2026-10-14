"""Stage 0: SHA-256 checksums of the raw data, and verification of a backup copy.

Usage:
    python scripts/make_checksums.py                    # write 00_data_audit/raw_checksums.sha256
    python scripts/make_checksums.py --verify <folder>  # check a backup copy against that file
"""

import argparse
import hashlib
import sys
from pathlib import Path

from solarpv.config.paths import AUDIT_DIR, RAW_DIR

CHECKSUM_FILE = AUDIT_DIR / "raw_checksums.sha256"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_checksums(raw_dir: Path) -> int:
    files = sorted(p for p in raw_dir.iterdir() if p.is_file())
    lines = [f"{sha256(p)}  {p.name}" for p in files]
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums to {CHECKSUM_FILE}")
    return 0


def verify(copy_dir: Path) -> int:
    expected = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    bad = []
    for name, digest in expected.items():
        p = copy_dir / name
        if not p.exists():
            bad.append(f"missing: {name}")
        elif sha256(p) != digest:
            bad.append(f"mismatch: {name}")
    for b in bad:
        print(b)
    print(f"{len(expected) - len(bad)}/{len(expected)} files OK in {copy_dir}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", type=Path, help="backup folder to verify")
    args = parser.parse_args()
    return verify(args.verify) if args.verify else write_checksums(RAW_DIR)


if __name__ == "__main__":
    sys.exit(main())

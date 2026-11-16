# ruff: noqa: E501  (docstring and assert messages; see test_stage11_part_c)
"""T-193 (X119): no tracked text file holds the site coordinates (T-45, Q-09).

The true coordinates live only in `configs/site.private.yaml`, which git ignores. Before 2026-11-16 the guard
covered the audit of the original backup and the report; two working documents still held the coordinates (X119).
This test reads the private file when it exists on this machine and scans every tracked text file for either
coordinate written with four to six decimals. It never prints a coordinate: a failure names the files only.
"""

import re
import subprocess

import pytest
import yaml

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import find_git

PRIVATE = REPO_ROOT / "configs" / "site.private.yaml"
TEXT = {".md", ".csv", ".json", ".yaml", ".yml", ".py", ".ipynb", ".txt", ".log", ".html", ".toml", ".cfg"}


def _coordinates(node) -> list[float]:
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).lower() in ("latitude", "longitude") and isinstance(v, int | float):
                out.append(float(v))
            out += _coordinates(v)
    return out


def test_t193_no_tracked_text_file_holds_the_site_coordinates():
    if not PRIVATE.exists():
        pytest.skip("configs/site.private.yaml is not on this machine")
    values = _coordinates(yaml.safe_load(PRIVATE.read_text(encoding="utf-8")))
    assert len(values) >= 2, "the private file names a latitude and a longitude"
    stems = set()
    for v in values:
        stems.add(f"{v:.6f}"[:-2])  # four decimals, truncated
        stems.add(f"{v:.4f}")  # four decimals, rounded
    rx = re.compile(r"(?<![\d.])(?:" + "|".join(re.escape(s) for s in stems) + r")\d{0,2}(?!\d)")
    files = subprocess.run([find_git(), "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    hits = []
    for rel in files:
        path = REPO_ROOT / rel
        if path.suffix.lower() not in TEXT or not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        if rx.search(path.read_bytes().decode("utf-8", errors="ignore")):
            hits.append(rel)
    assert not hits, f"tracked files hold the site coordinates (replace them by a pointer to the private file): {hits}"

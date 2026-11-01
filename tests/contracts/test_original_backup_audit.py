"""T-45: the audit of the original backup supports F-26, F-27 and F-28, and writes no coordinate.

The audit outputs are tracked in git (00_data_audit/original_backup_2026-11-01/), so these checks
run without the backup itself; the coordinate guard reads the backup's request cache and the
private site file when they exist on this machine.
"""

import json
import re

import pytest

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, REPO_ROOT

HERE = AUDIT_DIR / "original_backup_2026-11-01"
CACHE = REPO_ROOT / "old files" / "solar_ds_capstone" / "04_feature_engineering" / ".cache" \
    / "http_cache.sqlite"
REVIEW_NB = REPO_ROOT / "notebooks" / "00b_original_backup_review.ipynb"


def _load(name: str) -> dict:
    path = HERE / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_selection_evidence_supports_f26_and_f27():
    e = _load("selection_evidence.json")
    for target, nb in e["base_notebooks"].items():
        assert nb["reads_whole_table"] and not nb["row_filter_by_split_or_date"], target  # F-26
        assert e["fed_submitted_05e_D1"][target]["identical_sets"], target
    assert not e["base_notebooks"]["direct"]["per_inverter_dc_forbidden"]  # F-27
    assert e["base_notebooks"]["indirect"]["per_inverter_dc_forbidden"]
    assert e["direct_shap"]["dc_share_of_total"] > 0.9
    top2 = {row["feature"] for row in e["direct_shap"]["top5"][:2]}
    assert top2 == {"s01_inv1_DC_kW", "s01_inv2_DC_kW"}
    for name, start in e["submitted_05e_start_sets"].items():
        assert start["subset_of_D1"], name  # the base stage ran once
        assert all("_lag_" in f for f in start["dropped_from_D1"]), name


def test_weather_source_supports_f28():
    w = _load("weather_source.json")
    for requests in w["requests"].values():
        for r in requests:
            assert r["endpoint"].endswith("/v1/archive")
            assert r["parameters"]["models"] == "gfs_seamless"
            assert not {"latitude", "longitude"} & set(r["parameters"])
    skill = w["skill"]
    assert set(skill["daily_r"]) == {"train_and_val", "train", "val"}  # the test split unread
    both = skill["daily_r"]["train_and_val"]
    assert abs(both["original"] - both["gfs_d1"]) < 0.05  # about D+1 skill, not an observation
    best = skill["hourly_r_by_shift"][str(skill["best_shift_h"])]
    assert skill["best_shift_h"] == -1  # hour-end labels (N7)
    for shift in skill["original_join_shifts_h"]:
        assert skill["hourly_r_by_shift"][str(shift)] < best / 2


def test_no_coordinate_is_written_out():
    secrets = set()
    if CACHE.exists():
        secrets |= set(re.findall(rb"(?:latitude|longitude)=([0-9]+\.[0-9]+)",
                                  CACHE.read_bytes()))
    private = CONFIGS_DIR / "site.private.yaml"
    if private.exists():
        secrets |= {v.encode() for v in re.findall(r"(?:latitude|longitude):\s*([0-9]+\.[0-9]+)",
                                                   private.read_text(encoding="utf-8"))}
    if not secrets:
        pytest.skip("no private coordinates on this machine")
    files = [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [REVIEW_NB] if REVIEW_NB.exists() else []
    for path in files:
        data = path.read_bytes()
        for secret in secrets:
            assert secret not in data, path.name

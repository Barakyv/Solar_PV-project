"""The data-processing record stays complete and current (docs/data_processing_record.md §0).

Fails when: the generated numbers are stale; a processed-data stage has no section in the
record; a cleaning rule is not described; OP ids are missing, duplicated or out of order.
"""

import importlib.util
import re

import pytest
import yaml

from solarpv.config.paths import CONFIGS_DIR, PROCESSED_DIR, REPO_ROOT

RECORD = REPO_ROOT / "docs" / "data_processing_record.md"
NUMBERS = REPO_ROOT / "docs" / "data_record_numbers.md"
# processed-data directory -> heading text the record must contain
STAGE_SECTIONS = {
    "01_ingest": "(stage 2,",
    "04_preprocess": "(stage 4,",
    "05_clean": "(stage 5,",
    "06_features": "(stage 6,",
    "06b_gfs": "(stage 6b,",
}


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_data_record", REPO_ROOT / "scripts" / "build_data_record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> str:
    return RECORD.read_text(encoding="utf-8")


def test_numbers_file_is_current():
    builder = _builder()
    missing = [k for k, p in builder.SOURCES.items() if not p.exists()]
    if missing:
        pytest.skip(f"stage outputs not present: {missing}")
    current = NUMBERS.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert current == builder.build(), "run: python scripts/build_data_record.py"


def test_every_processed_stage_has_a_section(record):
    stages = sorted(p.name for p in PROCESSED_DIR.glob("[0-9][0-9]_*") if p.is_dir()
                    and any(p.iterdir())) if PROCESSED_DIR.exists() else []
    unknown = [s for s in stages if s not in STAGE_SECTIONS]
    assert not unknown, f"new stage(s) {unknown}: add a section to the record and to this test"
    for stage in stages:
        assert STAGE_SECTIONS[stage] in record, stage


def test_every_cleaning_rule_is_described(record):
    policy = yaml.safe_load((CONFIGS_DIR / "cleaning_policy.yaml").read_text(encoding="utf-8"))
    rules = ["R1 range", "R2 offset", "R3 frozen", "R4 night zero", "R5 night stale",
             "R6 dead DC", "R7 scale fault", "F1 `downtime", "F2 `curtail", "F3 `derate"]
    for rule in rules:
        assert rule in record, rule
    for key in ("range", "range_margin_minutes", "frozen_min_minutes", "frozen_margin_minutes",
                "night", "dc_dead", "scale_fault", "downtime", "curtailment", "derate",
                "hourly"):
        assert key in policy  # the record's rule table must follow the policy's sections


def test_operation_ids_are_complete_and_unique(record):
    defined = re.findall(r"(?:\*\*|\| )OP-(\d{2})\b(?= —|\s\|)", record)
    numbers = [int(n) for n in defined]
    assert len(numbers) >= 45, "OP pattern found too few entries"  # 45 documented on 2026-11-01
    assert numbers == list(range(1, len(numbers) + 1)), numbers
    for ref in set(re.findall(r"OP-(\d{2})", record)):
        assert int(ref) <= len(numbers), f"OP-{ref} referenced but not defined"


def test_numbers_sections_referenced_exist(record):
    numbers = NUMBERS.read_text(encoding="utf-8")
    for ref in sorted(set(re.findall(r"\bN(\d{1,2})\b", record))):  # N1 ... N11
        assert f"## N{ref}." in numbers, f"N{ref}"

"""T-72: the figure book is complete and current (docs/figures_plan.md, OP-56).

Fails when: a registered figure was not built or was changed after its build; an input changed
since the build; a takeaway does not re-render from its stored facts; the catalog is stale; a
completed stage has no process figure; a process figure reads the submission's evidence (except
the evaluation-code check); a correction figure is missing from the defense notes.
"""

import json

import pytest

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import sha256
from solarpv.figures.__main__ import (
    CATALOG,
    MANIFEST,
    catalog_text,
    input_hash,
    render_takeaway,
)
from solarpv.figures.registry import load_all

# stage -> minimum number of process figures; update when a stage is completed
REQUIRED = {"0": 1, "1": 3, "2": 3, "4": 3, "5": 2, "6": 5, "6b": 3, "7": 2, "8": 3, "9": 6}
SUBMISSION_READERS = {"S08-02"}  # evaluation code checked on the surviving forecasts (T-15)
DEFENSE = REPO_ROOT / "docs" / "defense_notes.md"


@pytest.fixture(scope="module")
def specs():
    return load_all()


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.exists(), "run: python -m solarpv.figures build"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_completed_stage_has_process_figures(specs):
    for stage, n in REQUIRED.items():
        got = sum(s.stage == stage and s.kind == "process" for s in specs)
        assert got >= n, f"stage {stage}: {got} process figures, expected at least {n}"


def test_every_figure_is_built_and_unchanged(specs, manifest):
    for s in specs:
        e = manifest["figures"].get(s.id)
        assert e, f"{s.id} not built: python -m solarpv.figures build --only {s.id}"
        path = REPO_ROOT / e["file"]
        assert path.exists(), e["file"]
        assert sha256(path) == e["file_sha256"], f"{s.id} changed after its build"
        assert set(e["inputs"]) == set(s.inputs), f"{s.id}: inputs changed, rebuild"


def test_inputs_unchanged_since_build(specs, manifest):
    missing = [rel for s in specs for rel in s.inputs if not (REPO_ROOT / rel).exists()]
    if missing:
        pytest.skip(f"stage outputs not present on this machine: {sorted(set(missing))[:3]}")
    stale = [s.id for s in specs for rel, h in manifest["figures"][s.id]["inputs"].items()
             if input_hash(rel) != h]
    assert not stale, f"inputs changed since the build, rebuild: {stale}"


def test_takeaways_render_from_facts(specs, manifest):
    for s in specs:
        e = manifest["figures"][s.id]
        assert render_takeaway(s.takeaway, e["facts"]) == e["takeaway"], s.id


def test_catalog_is_current(manifest):
    assert CATALOG.read_text(encoding="utf-8") == catalog_text(manifest), \
        "run: python -m solarpv.figures catalog"


def test_process_figures_do_not_read_the_submission(specs):
    for s in specs:
        if s.kind == "process" and s.id not in SUBMISSION_READERS:
            bad = [r for r in s.inputs if r.startswith("submission_evidence")
                   or "original_backup" in r]
            assert not bad, f"{s.id} is a process figure but reads {bad}"


def test_defense_notes_use_every_correction_figure(specs):
    text = DEFENSE.read_text(encoding="utf-8")
    for s in specs:
        if s.kind == "correction":
            assert s.id in text and s.filename in text, f"{s.id} missing in defense_notes.md"

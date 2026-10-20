"""notebooks/04b_features_target.ipynb is built from the current stage-6 outputs, every code cell
was executed without error and is followed by a markdown note (docs/stage6_plan.md, S7).

The notebook records the sha256 of the files its notes are formatted from; all of them are in
git, so this test runs without the data.
"""

import pytest

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import text_sha256

NOTEBOOK = REPO_ROOT / "notebooks" / "04b_features_target.ipynb"
SOURCES = {
    "features_manifest.json": REPO_ROOT / "outputs" / "06_features" / "features_manifest.json",
    "target.yaml": REPO_ROOT / "configs" / "target.yaml",
    "derived_constants.yaml": REPO_ROOT / "configs" / "derived_constants.yaml",
    "orientation_summary.json": REPO_ROOT / "00_data_audit" / "orientation_2026-10-20"
    / "orientation_summary.json",
    "quality_manifest.json": REPO_ROOT / "outputs" / "03b_quality" / "quality_manifest.json",
}


def test_notebook_is_current_and_every_code_cell_has_a_note():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    recorded = nb.metadata["solarpv"]["sources_sha256"]
    assert set(recorded) == set(SOURCES)
    for name, path in SOURCES.items():
        assert recorded[name] == text_sha256(path), \
            f"{name} changed; run: python scripts/make_features_notebook.py"
    cells = nb.cells
    code = [i for i, c in enumerate(cells) if c.cell_type == "code"]
    assert len(code) >= 10
    for i in code:
        assert i + 1 < len(cells) and cells[i + 1].cell_type == "markdown", f"cell {i}"
        assert cells[i].outputs, f"cell {i} was not executed"
        assert not any(o.get("output_type") == "error" for o in cells[i].outputs), f"cell {i}"

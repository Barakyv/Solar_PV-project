"""The stage notebooks built by scripts/_notebook.py are current and complete (user request,
2026-11-01: notebooks for every completed stage, as in the original project, to present live).

For each notebook: every source its notes were formatted from still has the recorded sha256
(sources under data/ are checked when present), every code cell was executed without error and
is followed by a markdown note.
"""

import pytest

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import text_sha256

NOTEBOOKS = ["00b_original_backup_review", "01_data_ingest", "03_preprocessing", "04_cleaning",
             "04c_nwp_gfs", "07_evaluation"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_current_and_every_code_cell_has_a_note(name):
    nbformat = pytest.importorskip("nbformat")
    path = REPO_ROOT / "notebooks" / f"{name}.ipynb"
    assert path.exists(), f"run: python scripts/make_{name.split('_', 1)[1]}_notebook.py"
    nb = nbformat.read(path, as_version=4)
    meta = nb.metadata["solarpv"]
    for rel, digest in meta["sources_sha256"].items():
        source = REPO_ROOT / rel
        if not source.exists():
            assert rel.startswith("data/"), f"{rel} is missing"
            continue
        assert text_sha256(source) == digest, f"{rel} changed; run: python {meta['builder']}"
    cells = nb.cells
    code = [i for i, c in enumerate(cells) if c.cell_type == "code"]
    assert len(code) >= 5
    for i in code:
        assert i + 1 < len(cells) and cells[i + 1].cell_type == "markdown", f"cell {i}"
        assert cells[i].outputs, f"cell {i} was not executed"
        assert not any(o.get("output_type") == "error" for o in cells[i].outputs), f"cell {i}"

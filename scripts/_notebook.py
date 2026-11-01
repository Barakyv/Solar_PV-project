"""Shared helper of the notebook builders added on 2026-11-01 (scripts/make_*_notebook.py).

Each builder lists its cells and the files its notes are formatted from. `write_notebook` checks
that every code cell is followed by a markdown note (user request, 2026-10-20), records the
sha256 of every source file under the notebook metadata, executes the notebook and writes it.
tests/contracts/test_stage_notebooks.py checks the result.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import text_sha256

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell


def write_notebook(path: Path, cells: list, sources: list[Path], timeout: int = 900) -> None:
    for i, cell in enumerate(cells):
        if cell.cell_type == "code":
            assert i + 1 < len(cells) and cells[i + 1].cell_type == "markdown", \
                f"{path.name}: code cell {i} has no note after it"
    recorded = {Path(p).resolve().relative_to(REPO_ROOT).as_posix(): text_sha256(Path(p))
                for p in sources}
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "solarpv": {"sources_sha256": recorded,
                    "builder": f"scripts/make_{path.stem.split('_', 1)[1]}_notebook.py"}})
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    path.write_bytes(nbformat.writes(nb).encode("utf-8"))
    print("wrote", path.relative_to(REPO_ROOT))

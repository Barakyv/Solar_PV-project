"""Shared helper of the notebook builders added on 2026-11-01 (scripts/make_*_notebook.py).

Each builder lists its cells and the files its notes are formatted from. `write_notebook` checks
that every code cell is followed by a markdown note (user request, 2026-10-20), records the
sha256 of every source file under the notebook metadata, executes the notebook and writes it.
tests/contracts/test_stage_notebooks.py checks the result.

The notebook runs in a Jupyter kernel (nbclient) wherever one can start. On the college machine
libzmq cannot create its poller and aborts any kernel (RH-04); there the cells run in an IPython
shell in this process, with the same outputs, and the metadata records which executor ran.
"""

import base64
import subprocess
import sys
from contextlib import chdir
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import text_sha256

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell
ZMQ_PROBE = ("import zmq; s = zmq.Context().socket(zmq.PAIR); "
             "s.bind_to_random_port('tcp://127.0.0.1'); s.close(0)")


def kernel_available() -> bool:
    """libzmq aborts the whole process when it fails, so the check runs in a child process."""
    probe = subprocess.run([sys.executable, "-c", ZMQ_PROBE], capture_output=True, timeout=120)
    return probe.returncode == 0


def _output(kind: str, captured: dict, **kw):
    data = {k: base64.b64encode(v).decode("ascii") if isinstance(v, bytes) else v
            for k, v in captured["data"].items()}
    return nbformat.v4.new_output(kind, data=data, metadata=captured.get("metadata") or {}, **kw)


def execute_in_process(nb, cwd: Path) -> None:
    """Run the code cells in an IPython shell in this process, without a kernel or zmq, and store
    their outputs as nbclient does: stdout and stderr, the display calls, then the cell's value.
    A failing cell raises, with its traceback."""
    from IPython.core.displayhook import CapturingDisplayHook
    from IPython.core.displaypub import CapturingDisplayPublisher
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=config)
    count = 0
    with chdir(cwd):
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            values = []
            saved = shell.display_pub, shell.display_trap.hook
            shell.display_pub = CapturingDisplayPublisher()
            shell.display_trap.hook = CapturingDisplayHook(shell=shell, outputs=values)
            try:
                with capture_output(display=False) as cap:
                    result = shell.run_cell(cell.source)
            finally:
                displays = shell.display_pub.outputs
                shell.display_pub, shell.display_trap.hook = saved
            if not result.success:
                raise RuntimeError(f"code cell {count} failed:\n{cap.stdout}{cap.stderr}")
            cell.outputs = [nbformat.v4.new_output("stream", name=name, text=text)
                            for name, text in (("stdout", cap.stdout), ("stderr", cap.stderr))
                            if text]
            cell.outputs += [_output("display_data", d) for d in displays]
            cell.outputs += [_output("execute_result", v, execution_count=count) for v in values]
            cell.execution_count = count


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
    if kernel_available():
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        nb.metadata["solarpv"]["executor"] = "nbclient, python3 kernel"
    else:
        print("no Jupyter kernel can start here (libzmq, RH-04): running the cells in process")
        execute_in_process(nb, path.parent)
        nb.metadata["solarpv"]["executor"] = "in-process IPython shell (RH-04)"
    path.write_bytes(nbformat.writes(nb).encode("utf-8"))
    print("wrote", path.relative_to(REPO_ROOT))

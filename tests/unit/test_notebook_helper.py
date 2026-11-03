"""The in-process executor of scripts/_notebook.py (RH-04) stores the outputs a kernel would:
stdout, display calls and the cell's value, in the nbformat schema, and stops on a failing cell.
"""

import importlib.util

import pytest

from solarpv.config.paths import REPO_ROOT


def _helper():
    spec = importlib.util.spec_from_file_location(
        "_notebook", REPO_ROOT / "scripts" / "_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_in_process_outputs_follow_the_notebook_schema(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nb_helper = _helper()
    (tmp_path / "x.txt").write_text("from the notebook folder", encoding="utf-8")
    nb = nbformat.v4.new_notebook(cells=[
        nb_helper.code("import pandas as pd\nfrom IPython.display import display\nprint('hi')\n"
                       "display(pd.DataFrame({'a': [1]}))\nopen('x.txt').read()"),
        nb_helper.md("note"),
        nb_helper.code("y = 1"),
        nb_helper.md("note")])
    nb_helper.execute_in_process(nb, tmp_path)
    first, second = nb.cells[0], nb.cells[2]
    assert [o.output_type for o in first.outputs] == ["stream", "display_data", "execute_result"]
    assert first.outputs[0].text == "hi\n"
    assert "text/html" in first.outputs[1].data
    assert first.outputs[2].data["text/plain"] == "'from the notebook folder'"
    assert (first.execution_count, second.execution_count) == (1, 2)
    assert second.outputs == []
    nbformat.validate(nb)


def test_in_process_failure_raises_with_the_traceback(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nb_helper = _helper()
    nb = nbformat.v4.new_notebook(cells=[nb_helper.code("1 / 0"), nb_helper.md("note")])
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        nb_helper.execute_in_process(nb, tmp_path)

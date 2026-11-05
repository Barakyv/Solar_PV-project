"""The in-process executor of scripts/_notebook.py (RH-04) stores the outputs a kernel would:
stdout, display calls and the cell's value, in the nbformat schema, and stops on a failing cell.
A Jupyter kernel can start, on the college machine through the long TMP of RH-05.
"""

import importlib.util
import os

import pytest

from solarpv.config.paths import REPO_ROOT


def _helper():
    pytest.importorskip("nbclient")
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


def test_zmq_tmp_is_too_long_for_an_af_unix_address(monkeypatch):
    nb_helper = _helper()
    for base in ("C:\\a", "C:\\Users\\someone\\AppData\\Local", "C:\\" + "x" * 120):
        monkeypatch.setenv("LOCALAPPDATA", base)
        path = str(nb_helper.zmq_tmp_dir())
        assert len(path) > nb_helper.SUN_PATH and path.isascii(), path


def test_enable_kernel_tries_the_long_tmp_first_and_restores_tmp_if_it_fails(monkeypatch,
                                                                             tmp_path):
    nb_helper = _helper()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("TMP", "plain")
    seen = []

    def probe(results):
        def kernel_available():
            seen.append(os.environ["TMP"])
            return results[len(seen) - 1]
        return kernel_available

    monkeypatch.setattr(nb_helper, "kernel_available", probe([True]))
    assert nb_helper.enable_kernel() and os.environ["TMP"] == str(nb_helper.zmq_tmp_dir())
    assert seen == [str(nb_helper.zmq_tmp_dir())]
    for results, ok in (([False, True], True), ([False, False], False)):
        monkeypatch.setenv("TMP", "plain")
        seen.clear()
        monkeypatch.setattr(nb_helper, "kernel_available", probe(results))
        assert nb_helper.enable_kernel() is ok and os.environ["TMP"] == "plain"
        assert seen == [str(nb_helper.zmq_tmp_dir()), "plain"]


def test_clean_zmq_tmp_removes_only_empty_folders(monkeypatch, tmp_path):
    nb_helper = _helper()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    root = nb_helper.zmq_tmp_dir()
    (root / "x1.0").mkdir(parents=True)
    (root / "x1.1").mkdir()
    (root / "x1.1" / "kept.txt").write_text("not libzmq's", encoding="utf-8")
    nb_helper.clean_zmq_tmp()
    assert sorted(p.name for p in root.iterdir()) == ["x1.1"]


def test_a_zmq_socket_can_be_created_here(monkeypatch):
    """RH-05: on the college machine a zmq socket aborts the process unless TMP is long."""
    pytest.importorskip("zmq")
    nb_helper = _helper()
    monkeypatch.setenv("TMP", os.environ.get("TMP", ""))  # restored after the test
    assert nb_helper.enable_kernel()
    nb_helper.clean_zmq_tmp()

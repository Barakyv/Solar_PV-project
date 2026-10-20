"""Run log (solarpv.config.runlog): every run leaves a complete, readable record."""

import pytest

from solarpv.config.runlog import RunLog


def _lines(log: RunLog) -> list[str]:
    return log.path.read_text(encoding="utf-8").splitlines()


def test_a_successful_run_is_fully_recorded(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("x", encoding="utf-8")
    with RunLog("test_stage", "python -m demo", out_dir=tmp_path, echo=False) as log:
        hashes = log.inputs([src])
        with log.step("step one"):
            log.note("an intermediate observation")
            log.value("rows", 42)
        log.warn("something odd")
        log.outputs([src])
    text = "\n".join(_lines(log))
    for expected in ("command: python -m demo", "git commit:", "versions:", "INPUT",
                     "step one ...", "step one done", "NOTE    an intermediate observation",
                     "VALUE   rows = 42", "WARN    something odd", "OUTPUT", "warnings: 1",
                     "status: ok"):
        assert expected in text, expected
    assert log.values == {"rows": 42}
    assert len(next(iter(hashes.values()))) == 64  # full sha256 returned to the caller
    assert log.path.parent == tmp_path / "runs"


def test_a_failed_run_keeps_its_log_and_the_traceback(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with RunLog("test_stage", "python -m demo", out_dir=tmp_path, echo=False) as log:
            log.note("before the failure")
            1 / 0  # noqa: B018
    text = "\n".join(_lines(log))
    assert "before the failure" in text
    assert "ERROR" in text and "ZeroDivisionError" in text
    assert "status: failed (ZeroDivisionError)" in text

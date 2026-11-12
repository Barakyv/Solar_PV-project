"""Gate G-1 (W6, docs/stage9_plan.md §20).

T-95  X55: the runner refuses a working directory inside another checkout of the project (an
      editable install would run this checkout's code there); outside any checkout, and in this
      checkout, nothing is refused
T-96  X56: a modified tracked document, not only code, makes the per-step gate of a final run
      refuse; untracked outputs (the run folders, the test-access log) do not; the runbook says
      that no tracked file may change during a final run
T-99  X64: a manifest names a raw file `Raw data/<file>` also when SOLAR_RAW_DIR is outside the
      checkout
"""

import subprocess
from types import SimpleNamespace

from solarpv.config.paths import DOCS_DIR, REPO_ROOT
from solarpv.pipeline import checks
from solarpv.pipeline.__main__ import main as runner_main


def make_checkout(root):
    (root / "src" / "solarpv").mkdir(parents=True)
    (root / "src" / "solarpv" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='solarpv'\n", encoding="utf-8")
    return root


# ── T-95 (X55) ──────────────────────────────────────────────────────────────


def test_another_checkout_is_refused(tmp_path, monkeypatch, capsys):
    other = make_checkout(tmp_path / "clone")
    (other / "outputs").mkdir()
    assert checks.checkout_of(other / "outputs") == other.resolve()
    problem = checks.checkout_problem(other / "outputs")
    assert problem and "X55" in problem and str(other / "src") in problem
    monkeypatch.chdir(other)
    assert runner_main(["status"]) == 2
    assert "X55" in capsys.readouterr().err
    assert runner_main(["list"]) == 0  # listing the steps touches nothing


def test_this_checkout_and_no_checkout_are_accepted(tmp_path):
    assert checks.checkout_problem(REPO_ROOT) is None
    assert checks.checkout_problem(REPO_ROOT / "outputs") is None
    assert checks.checkout_of(tmp_path) is None
    assert checks.checkout_problem(tmp_path) is None


# ── T-96 (X56) ──────────────────────────────────────────────────────────────


def fake_git(status_lines: list[str], tag: bool = True):
    def run(*args):
        if args[:1] == ("tag",):
            return SimpleNamespace(returncode=0, stdout="g1-frozen\n" if tag else "")
        if args[:1] == ("status",):
            untracked = "--untracked-files=no" not in args
            lines = [s for s in status_lines if untracked or not s.startswith("??")]
            return SimpleNamespace(returncode=0, stdout="\n".join(lines))
        return SimpleNamespace(returncode=0, stdout="")
    return run


def test_a_modified_document_stops_a_final_step(monkeypatch):
    monkeypatch.setattr(checks, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(checks, "load_site", lambda: SimpleNamespace(is_private=True))
    monkeypatch.setattr(checks, "_git", fake_git([]))
    assert checks.final_step_gate("abc1234") == []
    # untracked outputs: a run folder and the test-access log do not stop the run
    monkeypatch.setattr(checks, "_git", fake_git(
        ["?? outputs/runs/20261113T000000Z_final/",
         "?? outputs/07_evaluation/test_access_log.csv"]))
    assert checks.final_step_gate("abc1234") == []
    # a document edited during the run (the run journal) stops the next step
    monkeypatch.setattr(checks, "_git", fake_git([" M docs/run_journal.md"]))
    errors = checks.final_step_gate("abc1234")
    assert errors and "not clean" in errors[0]


def test_the_runbook_says_no_tracked_file_may_change():
    text = " ".join((DOCS_DIR / "runbook_final.md").read_text(encoding="utf-8").split())
    assert "no tracked file may change" in text
    assert "documents included" in text


def test_a_raw_file_outside_the_checkout_is_named_as_in_the_checkout(tmp_path, monkeypatch):
    """T-99, X64: with SOLAR_RAW_DIR outside the checkout (a reproduction in a fresh clone) a
    manifest still names a raw file `Raw data/<file>`; files of the checkout keep their relative
    path."""
    from solarpv.config import paths

    raw = tmp_path / "elsewhere" / "Raw data"
    raw.mkdir(parents=True)
    (raw / "2112.CSV").write_text("x", encoding="utf-8")
    monkeypatch.setattr(paths, "RAW_DIR", raw)
    assert paths.recorded_path(raw / "2112.CSV") == "Raw data/2112.CSV"
    assert paths.recorded_path(REPO_ROOT / "configs" / "clock_offsets.csv") \
        == "configs/clock_offsets.csv"
    monkeypatch.setattr(paths, "RAW_DIR", REPO_ROOT / "Raw data")
    assert paths.recorded_path(REPO_ROOT / "Raw data" / "2112.CSV") == "Raw data/2112.CSV"


def test_the_real_tree_check_covers_documents():
    """The gate's status call lists modified tracked files anywhere in the tree (no path filter):
    run it on the real repository and check the arguments it uses."""
    status = checks._git("status", "--porcelain", "--untracked-files=no")
    assert status.returncode == 0
    git = checks.find_git()
    full = subprocess.run([git, "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert status.stdout == full.stdout

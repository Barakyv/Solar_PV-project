"""T-103 (D-36): after the tag `g1-frozen`, only code that reads the frozen outputs of a finished
final run may change (figures, `make_*` scripts, notebooks, documents, tests, outputs); any other
path reopens G-1.

The classification is always tested; the check of the real history is active once the tag exists.
"""

import pytest

from solarpv.pipeline import checks


def test_the_paths_that_reopen_g1_are_named():
    ok = ["src/solarpv/figures/s09_w6.py", "scripts/make_models_notebook.py",
          "notebooks/06_models.ipynb", "docs/errata.md", "tests/unit/test_g1.py",
          "outputs/runs/20261113T000000Z_final/step.json", "PROJECT_MASTER.md", "CLAUDE.md",
          ".claude/skills/task-figures/SKILL.md"]
    assert checks.after_tag_problems(ok) == []
    bad = ["src/solarpv/models/xgb_sota.py", "src/solarpv/pipeline/analyses.py",
           "configs/target.yaml", "scripts/reproduce_paper_results.py", "pyproject.toml",
           "environment.lock.yml", "src\\solarpv\\evaluation\\evaluate.py"]
    assert checks.after_tag_problems(ok + bad) == bad


def test_the_figures_folder_is_the_only_part_of_the_package_that_may_change():
    assert checks.after_tag_problems(["src/solarpv/figures/registry.py"]) == []
    for p in ("src/solarpv/figures_extra/x.py", "src/solarpv/config/cli.py",
              "src/solarpv/pipeline/report.py"):
        assert checks.after_tag_problems([p]) == [p]


def test_after_the_tag_only_allowed_paths_changed():
    tag = checks._git("tag", "--list", checks.FROZEN_TAG).stdout.strip()
    if not tag:
        pytest.skip("g1-frozen does not exist yet: the check starts at the tag")
    assert checks.after_tag_changes() == [], "G-1 is reopened by a change outside D-36"

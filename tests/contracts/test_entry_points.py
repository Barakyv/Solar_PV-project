"""T-98, X62 (docs/stage9_plan.md §20.3): every entry point parses its arguments before any work.

On 2026-11-12 `python -m solarpv.data.clean --help` cleaned the data again and rewrote the stage-5
files (run journal R9-26): the modules had no argument parsing. Now:
- statically, every entry point of `src/solarpv` and `scripts/` parses its arguments (argparse,
  or `solarpv.config.cli.no_options`), and the parse is the first thing it does: the first
  statement of `main`, or, for scripts that run at import, the first statement after the
  imports;
- behaviourally, only after the static check has passed (so that a regression cannot run a
  stage from a test): `main(["--help"])` exits 0 and an unknown argument exits 2, and two scripts
  that run at import are started with `--help` and write nothing;
- no parser accepts an abbreviated option (`--allow` must not mean `--allow-test`).
"""

import ast
import importlib
import subprocess
import sys

import pytest

from solarpv.config.paths import REPO_ROOT

# entry points that do not need a parser: what they do without one, and why that is safe
EXEMPT = {
    "src/solarpv/config/constants.py": "prints a self-test of the constants; writes nothing",
    "src/solarpv/models/neural_worker.py": "takes exactly two positional folders, prints its "
                                           "docstring and exits 2 otherwise; the output folder "
                                           "must not exist",
}
PARSE_CALLS = {"no_options", "ArgumentParser", "parser"}
# module mains whose first statement is no_options (called with --help below)
MODULE_MAINS = ["solarpv.data.ingest", "solarpv.data.preprocess", "solarpv.data.clean",
                "solarpv.features.build", "solarpv.features.decide_k", "solarpv.nwp.gfs",
                "solarpv.nwp.check", "solarpv.eda.figures"]


def entry_points() -> list[str]:
    files = sorted((REPO_ROOT / "src" / "solarpv").rglob("*.py")) \
        + sorted((REPO_ROOT / "scripts").glob("*.py"))
    out = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        tree = ast.parse(text)
        guarded = any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)
                      for n in tree.body)
        builder = f.parent.name == "scripts" and f.name.startswith("make_")
        if guarded or f.name == "__main__.py" or builder:
            out.append(f.relative_to(REPO_ROOT).as_posix())
    return out


def _calls(node) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            names.add(fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", ""))
    return names


def parse_first(rel: str) -> str | None:
    """None when the entry point parses its arguments before any work; otherwise the reason."""
    tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
                None)
    if main is not None:
        body = [s for s in main.body if not (isinstance(s, ast.Expr)
                                             and isinstance(s.value, ast.Constant))]
        if not body:
            return "empty main"
        if _calls(body[0]) & PARSE_CALLS:
            return None
        return f"main starts with `{ast.unparse(body[0])[:60]}`, not with the argument parse"
    # a script or __main__ that works at import: the first statement after the imports
    rest = [s for s in tree.body
            if not isinstance(s, (ast.Import, ast.ImportFrom))
            and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if rest and _calls(rest[0]) & PARSE_CALLS:
        return None
    if rest and isinstance(rest[0], ast.Expr) and "main" in _calls(rest[0]):
        return None  # e.g. `sys.exit(main())` of a module whose main is checked on its own
    return f"the first statement after the imports is `{ast.unparse(rest[0])[:60]}`" \
        if rest else "nothing after the imports"


def test_every_entry_point_parses_its_arguments_first():
    points = entry_points()
    assert len(points) >= 30, points  # the scan finds the stage modules and the builders
    problems = {}
    for rel in points:
        if rel in EXEMPT:
            continue
        reason = parse_first(rel)
        if reason:
            problems[rel] = reason
    assert not problems, problems


def test_no_parser_accepts_abbreviations():
    """`allow_abbrev=False` wherever a parser has an option that starts work on the test split
    or a validation look; the shared helper sets it for all others."""
    from solarpv.config import cli

    assert cli.parser("x", None).allow_abbrev is False
    for rel in ("src/solarpv/pipeline/__main__.py", "src/solarpv/evaluation/evaluate.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "allow_abbrev=False" in text, rel
    from solarpv.pipeline.__main__ import main as runner_main

    with pytest.raises(SystemExit) as exc:
        runner_main(["run", "--mode", "smoke", "--allow", "--dry-run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("module", MODULE_MAINS)
def test_help_runs_nothing_and_unknown_arguments_fail(module, capsys):
    rel = "src/" + module.replace(".", "/") + ".py"
    assert parse_first(rel) is None  # never call a main that would run its stage
    mod = importlib.import_module(module)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        mod.main(["--bogus"])
    assert exc.value.code == 2


def test_selection_rejects_an_abbreviated_validation_flag():
    rel = "src/solarpv/selection/run.py"
    assert parse_first(rel) is None
    from solarpv.config import cli

    # guard first: with abbreviations on, `--valid` below would start the validation look
    assert cli.parser("x", None).allow_abbrev is False
    from solarpv.selection import run as selection_run

    for argv, code in ((["--help"], 0), (["--valid"], 2), (["--skip"], 2)):
        with pytest.raises(SystemExit) as exc:
            selection_run.main(argv)
        assert exc.value.code == code, argv


@pytest.mark.parametrize("args", [["scripts/make_eda_notebook.py"],
                                  ["-m", "solarpv.eda.quality"]])
def test_scripts_that_run_at_import_stop_at_help(args):
    target = args[-1] if args[0] != "-m" else "src/solarpv/eda/quality/__main__.py"
    assert parse_first(target) is None
    watched = [REPO_ROOT / "notebooks" / "02_eda.ipynb",
               REPO_ROOT / "outputs" / "03b_quality" / "quality_manifest.json"]
    before = {p: p.stat().st_mtime_ns for p in watched if p.exists()}
    proc = subprocess.run([sys.executable, *args, "--help"], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert "usage:" in proc.stdout
    assert {p: p.stat().st_mtime_ns for p in before} == before

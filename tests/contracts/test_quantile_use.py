"""T-92b: every use of the quantile forecasts as a distribution goes through the sorting (D-33).

Fails when: code that reads `quantiles_*.csv` or a quantile column (`q0.1`, ...) neither imports
`solarpv.evaluation.quantiles` nor is one of the two files allowed not to (the writer, which keeps
the raw model output, and the report, which only lists the file names); the writer sorts what it
writes; the guard itself stops finding a planted unsorted reader; or, on the real development run,
the written point forecast is not the 0.5 head, sorting leaves a row out of order or changes the
point, or the sorted quantile loss is higher than the raw one.
"""

import re

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import DEV_SEED
from solarpv.config.paths import REPO_ROOT
from solarpv.evaluation import quantiles as Q
from solarpv.figures.s09_models import NDEV, QUANTILE_STEPS

MODULE = "src/solarpv/evaluation/quantiles.py"
WRITER = "src/solarpv/pipeline/neural.py"  # writes the raw quantile columns
NAMES_ONLY = ("src/solarpv/pipeline/report.py",)  # lists the file names, reads no value
# a quantile file name ("quantiles_<model>...") or a quantile column label ("q0.1", f"q{level}")
USES_QUANTILES = re.compile(r"""quantiles_|["']q0\.\d|f["']q\{""")
IMPORTS_MODULE = re.compile(r"evaluation(\.| import )quantiles")


def sources() -> dict[str, str]:
    files = [*(REPO_ROOT / "src" / "solarpv").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")]
    return {p.relative_to(REPO_ROOT).as_posix(): p.read_text(encoding="utf-8") for p in files}


def unsorted_readers(files: dict[str, str]) -> list[str]:
    """Files that use quantiles without the sorting module and are not allowed to."""
    allowed = {MODULE, WRITER, *NAMES_ONLY}
    return sorted(name for name, text in files.items()
                  if name not in allowed and USES_QUANTILES.search(text)
                  and not IMPORTS_MODULE.search(text))


def test_every_reader_of_quantiles_imports_the_sorting_module():
    found = unsorted_readers(sources())
    assert not found, f"{found} read quantiles without solarpv.evaluation.quantiles (D-33)"


def test_the_guard_finds_a_planted_unsorted_reader():
    planted = {"src/solarpv/x.py": 'q = pd.read_csv(path / "quantiles_tft_96_seed42.csv")',
               "src/solarpv/y.py": 'lo, hi = frame["q0.1"], frame["q0.9"]',
               "src/solarpv/z.py": 'cols = [f"q{level}" for level in levels]',
               "src/solarpv/ok.py": "from solarpv.evaluation import quantiles as Q\n"
                                    'q = pd.read_csv("quantiles_tft_96_seed42.csv")',
               "src/solarpv/none.py": "QUANTILES = (0.1, 0.5, 0.9)"}
    assert unsorted_readers(planted) == ["src/solarpv/x.py", "src/solarpv/y.py",
                                         "src/solarpv/z.py"]


def test_the_writer_keeps_the_raw_model_output():
    text = sources()[WRITER]
    assert not IMPORTS_MODULE.search(text) and "ordered(" not in text, \
        "the quantile files must keep the raw output; sorting happens where they are read"


def test_the_report_only_lists_quantile_file_names():
    for name in NAMES_ONLY:
        text = sources()[name]
        hits = [m.start() for m in re.finditer("quantiles_", text)]
        assert hits, name
        # each mention sits inside a `startswith(...)` name filter (it may continue a line)
        flat = re.sub(r"\s+", " ", text)  # a filter may span lines
        hits = [m.start() for m in re.finditer("quantiles_", flat)]
        assert all("startswith" in flat[max(0, i - 120):i] for i in hits), \
            f"{name} must only filter by name"


def _merged(step: str, model: str) -> pd.DataFrame:
    base = REPO_ROOT / NDEV / step
    qf = base / f"quantiles_{model}_seed{DEV_SEED}.csv"
    pf = base / f"predictions_{model}_seed{DEV_SEED}.csv"
    if not (qf.exists() and pf.exists()):
        pytest.skip(f"{NDEV}/{step} is not present")
    q = pd.read_csv(qf, parse_dates=["timestamp"])
    p = pd.read_csv(pf, parse_dates=["timestamp"])
    m = q.merge(p[["timestamp", "target", "horizon", "y_true", "y_pred"]],
                on=["timestamp", "target", "horizon"], validate="one_to_one")
    return m[m["y_true"].notna() & m["y_pred"].notna()]


@pytest.mark.parametrize("step,model", QUANTILE_STEPS)
def test_on_the_development_run_the_point_is_the_median_head_and_sorting_keeps_it(step, model):
    m = _merged(step, model)
    assert len(m) > 1000
    assert np.allclose(m["q0.5"], m["y_pred"]), "the point forecast is not the 0.5 head"
    for _, g in m.groupby("target"):
        o = Q.ordered(g)
        assert not Q.crossed(o).any()
        assert o["y_pred"].equals(g["y_pred"]), "sorting changed the point forecast"
        y = g["y_true"].to_numpy()
        assert Q.pinball(y, o) <= Q.pinball(y, g) + 1e-9

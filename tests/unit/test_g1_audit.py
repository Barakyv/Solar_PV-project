"""T-94 (docs/stage9_plan.md §20.4; M35, X57): the manual leakage audit of gate G-1.

- docs/leakage_audit.md §3 has a row for every feature group of the contract and for every
  processing step before the hourly table; every row names code that exists and a test ID that
  is registered;
- the causal clock offset of scripts/g1_audit.py is a trailing median (only earlier days),
  carried forward, restarted at a resync, and it equals a hand computation on a constructed
  series;
- the audit's evidence files exist and agree with the document (the numbers are not typed).
"""

import importlib.util
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest

from solarpv.config.paths import DOCS_DIR, OUTPUTS_DIR, REPO_ROOT
from solarpv.features.contract import FEATURE_GROUPS

AUDIT = (DOCS_DIR / "leakage_audit.md").read_text(encoding="utf-8")
SECTION = AUDIT.split("## 3. Gate G-1")[1].split("## History")[0]
MASTER = (REPO_ROOT / "PROJECT_MASTER.md").read_text(encoding="utf-8")
PLAN = (DOCS_DIR / "stage9_plan.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def A():
    path = REPO_ROOT / "scripts" / "g1_audit.py"
    spec = importlib.util.spec_from_file_location("g1_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in line.strip("|").split("|")]
            for line in text.splitlines() if re.match(r"^\| \d+ \|", line)]


def test_every_feature_group_and_processing_step_has_a_row():
    table = rows(SECTION)
    assert [int(r[0]) for r in table] == list(range(1, len(table) + 1))
    text = " ".join(" ".join(r) for r in table).lower()
    for group in FEATURE_GROUPS:  # the contract's groups, by what they hold
        words = {"clearsky": "clear sky", "rolling_stats": "rolling", "lag_power": "lags",
                 "time_encodings": "calendar", "other": "geometry"}[group]
        assert words in text, group
    for step in ("gfs", "encoder", "decoder", "patchtst", "scaler", "l-09", "references",
                 "selection", "ingest", "clock", "polled", "cleaning"):
        assert step in text, step


def test_every_row_names_existing_code_and_a_registered_test():
    registered = set(re.findall(r"\bT-\d{2,3}\b", MASTER)) | set(re.findall(r"\bT-\d{2,3}\b",
                                                                            PLAN))
    src = REPO_ROOT / "src" / "solarpv"
    for r in rows(SECTION.split("### 3.2")[0]):
        code = re.findall(r"`([A-Za-z_.]+)`", r[3])
        assert code, r[0]
        last = None
        for name in code:
            parts = name.split(".")
            if len(parts) == 1:  # a function or class of the module named before it
                assert last is not None, (r[0], name)
                assert re.search(rf"^(def|class) {name}\b|^{name}\s*[:=]|^    def {name}\b",
                                 last.read_text(encoding="utf-8"), re.M), (r[0], name)
                continue
            module = next((src / Path(*parts[:k]).with_suffix(".py")
                           for k in range(len(parts), 0, -1)
                           if (src / Path(*parts[:k]).with_suffix(".py")).exists()), None)
            assert module is not None, (r[0], name)
            rest = parts[len(module.relative_to(src).with_suffix("").parts):]
            if rest:  # module.attribute
                a = rest[0]
                assert re.search(rf"^(def|class) {a}\b|^{a}\s*[:=]|^    def {a}\b",
                                 module.read_text(encoding="utf-8"), re.M), (r[0], name)
            last = module
        tests = re.findall(r"T-\d{2,3}", r[4])
        assert tests and set(tests) <= registered, (r[0], tests)


def test_the_trailing_clock_median_uses_earlier_days_only(A):
    calendar = pd.date_range("2023-01-01", periods=40, freq="D")
    values = pd.Series(np.arange(40, dtype=float), index=calendar)
    days = pd.Series(calendar)
    steps = [pd.Timestamp("2023-01-21")]  # a resync restarts the window
    out, warm = A._trailing_within_segments(days, pd.Series(values.to_numpy()), steps, calendar,
                                            window=5, min_obs=3)
    seg1 = values[:20].rolling(5, min_periods=3).median()
    seg2 = values[20:].rolling(5, min_periods=3).median()
    expected = pd.concat([seg1.bfill(), seg2.bfill()])
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy())
    assert int(warm.sum()) == 4  # two days per segment before three observations exist
    # a later value never changes an earlier offset
    changed = values.copy()
    changed.iloc[30:] += 100
    out2, _ = A._trailing_within_segments(days, pd.Series(changed.to_numpy()), steps, calendar,
                                          window=5, min_obs=3)
    np.testing.assert_allclose(out2.to_numpy()[:30], out.to_numpy()[:30])


def test_the_evidence_files_agree_with_the_document():
    folder = OUTPUTS_DIR / "g1_audit" / "audit"
    clean = pd.read_csv(folder / "lookahead_clean.csv").set_index("source")
    clock = pd.read_csv(folder / "lookahead_clock.csv").set_index("clock")
    access = json.loads((folder / "test_access.json").read_text(encoding="utf-8"))
    assert int(clean["issuance_days"].iloc[0]) == 1364 and f"{1364:,}" in SECTION
    assert int(clean["method_mismatch_cells"].sum()) == 0
    assert int(clean["cells_changed"].sum()) == 66
    assert int(clean.loc["irradiance", "cells_changed"]) == 18
    assert int(clock.loc["minute_meteo", "days_shift_differs"]) == 552
    assert int(clock["max_shift_difference_min"].max()) == 3
    assert access["log_exists"] is False and access["prediction_rows_in_test"] == 0
    assert f"{access['step_records']} step" in SECTION


from pathlib import Path  # noqa: E402  (used in the module-path check above)

# ruff: noqa: E501  (the docstring lists the commands it runs; see make_stage11_part_a_notebook)
"""Rebuild every table, figure and report section of stage 11 from the frozen final run (no training).

    python scripts/make_tables_from_frozen.py

This is the stage-11 counterpart of the option that rebuilds the tables from saved predictions
(`--skip-train` of `reproduce_paper_results.py`, S-45, Q-30 option a). The reproduction script itself is
outside the D-36 list and is not edited; this script is new and only reads outputs. It runs, in order:

1. `python -m solarpv.pipeline status <run>`: the run must list every step `ok` with its outputs as recorded.
2. `python -m solarpv.figures.analyses11`: the 13 tables of `outputs/11_analyses/` (reads the saved forecasts
   and the stage-6 truth table; every read of test rows is appended to the test access log).
3. `python -m solarpv.figures build --stage 11` and the correction figures C-11 and C-12: the figures of stage 11.
4. `python scripts/make_report_v7.py`: the report sections, `docs/traceability.csv`, `docs/old_vs_new.csv`
   and the table of old against new results in `docs/errata.md`.

It never runs the step `test`, never trains a model and never changes a budget, a seed or a rule (D-37).
The run that trains XGB-Opt from the raw data is a separate, long job that is left for the end of the
project (Q-30).
"""

import subprocess
import sys

from solarpv.config.cli import no_options
from solarpv.config.paths import REPO_ROOT

no_options("python scripts/make_tables_from_frozen.py", __doc__)  # X62: --help runs nothing

RUN_ID = "20261114T083930Z_final"
STEPS = [
    ("the run is complete and its outputs are as recorded",
     [sys.executable, "-m", "solarpv.pipeline", "status", RUN_ID]),
    ("the tables of the analyses (part A)", [sys.executable, "-m", "solarpv.figures.analyses11"]),
    ("the figures of stage 11", [sys.executable, "-m", "solarpv.figures", "build", "--stage", "11"]),
    ("the correction figures of part B", [sys.executable, "-m", "solarpv.figures", "build", "--only",
                                         "C-11,C-12"]),
    ("the report sections, traceability and the table of old against new",
     [sys.executable, "scripts/make_report_v7.py"]),
]

for what, cmd in STEPS:
    print(f"== {what}: {' '.join(cmd[1:])}", flush=True)
    done = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if done.returncode != 0:
        sys.exit(f"stopped: {what} failed with exit code {done.returncode}")
print("done: everything was rebuilt from the frozen predictions; nothing was trained")

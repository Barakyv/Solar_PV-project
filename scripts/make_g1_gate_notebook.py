"""Write notebooks/08_g1_gate.ipynb (gate G-1, W6 of stage 9) and execute it.

    python scripts/g1_audit.py
    python scripts/reproduce_paper_results.py --dev --workdir <empty folder> \\
        --with-eda --rerun-selection
    python scripts/make_g1_gate_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the audit
files, the reproduction record and the development run of the steps `cv` and `ablation`, never
typed. The notebook shows what the gate checked, not a new analysis.
"""

import json
import re

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import DOCS_DIR, OUTPUTS_DIR, REPO_ROOT
from solarpv.figures._data import W6_DEV_RUN
from solarpv.figures.s09_w6 import explained_reason

no_options("python scripts/make_g1_gate_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "08_g1_gate.ipynb"
AUDIT = OUTPUTS_DIR / "g1_audit" / "audit"
LEAKAGE = DOCS_DIR / "leakage_audit.md"
INCIDENT = OUTPUTS_DIR / "g1_audit" / "incident_2026-11-12" / "regenerated_vs_original_content.json"
# the latest finished record: a running reproduction writes its record step by step
RECORDS = [p for p in sorted((OUTPUTS_DIR / "g1_audit").glob("reproduction_2026*/record.json"))
           if json.loads(p.read_text(encoding="utf-8")).get("result")]
REPRO = RECORDS[-1]
REPRO_REL = REPRO.relative_to(REPO_ROOT).as_posix()
CV = REPO_ROOT / W6_DEV_RUN / "cv" / "cv_summary.csv"
AB = REPO_ROOT / W6_DEV_RUN / "ablation" / "ablation_summary.csv"
RESUME = OUTPUTS_DIR / "g1_audit" / "resume_check.json"  # scripts/g1_resume_check.py (D-38)
SOURCES = [LEAKAGE, AUDIT / "test_access.json", AUDIT / "lookahead_clean.csv",
           AUDIT / "lookahead_clock.csv", AUDIT / "lookahead_polled.csv", INCIDENT, REPRO, CV, AB,
           RESUME]


def table_rows(text: str, start: str, stop: str, width: int) -> pd.DataFrame:
    block = text.split(start)[1].split(stop)[0]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in block.splitlines()
            if re.match(r"^\| \d+ \|", line)]
    return pd.DataFrame(rows, columns=[f"c{i}" for i in range(width)])


LEAK = LEAKAGE.read_text(encoding="utf-8")
FAMILIES = table_rows(LEAK, "### 3.1", "### 3.2", 5)
STEPS = table_rows(LEAK, "### 3.2", "### 3.3", 5)
ACC = json.loads((AUDIT / "test_access.json").read_text(encoding="utf-8"))
CLEAN = pd.read_csv(AUDIT / "lookahead_clean.csv")
CLOCK = pd.read_csv(AUDIT / "lookahead_clock.csv")
POLLED = pd.read_csv(AUDIT / "lookahead_polled.csv")
INC = json.loads(INCIDENT.read_text(encoding="utf-8"))
REC = json.loads(REPRO.read_text(encoding="utf-8"))
assert REC.get("result"), f"{REPRO} is not a finished reproduction record"
UNEXPLAINED = [(s["step"], r["path"]) for s in REC["steps"] for r in s["failing"]
               if not explained_reason(s["step"], r["path"])]
N_FAIL = sum(len(s["failing"]) for s in REC["steps"])
CVS = pd.read_csv(CV)
ABS = pd.read_csv(AB)
GFS = ABS[ABS["mode"] == "no_nwp_gfs"]["mean_delta_pp"]
GAP = -CVS["validation_minus_cv_pp"]
RES = json.loads(RESUME.read_text(encoding="utf-8"))
RES_BROKEN = [r for r in RES["configs"] if not r["first_design"]["equal"]]
RES_REPLAY = [r["replay_seconds"] / r["seconds_of_the_epochs_done"] for r in RES["configs"]
              if r["interlayer_dropout"]]
# claims of the notes below, asserted here and not typed
assert all(r["final"]["equal"] for r in RES["configs"]), "a continued fit differs"
assert RES_BROKEN and all(r["interlayer_dropout"] for r in RES_BROKEN), "X95 changed"
assert all(r["first_design"]["equal"] for r in RES["configs"] if not r["interlayer_dropout"])
assert (CLEAN["test_days_affected"] == 0).all(), "a cleaned cell changed on a test day"
assert (CLEAN["method_mismatch_cells"] == 0).all(), "cleaning differs from the stage-5 table"
assert (GAP > 0).all(), "validation is not below the CV mean everywhere"
_single = ABS[ABS["mode"] != "minimal"]
_top = _single.loc[_single.groupby(["target", "horizon"])["mean_delta_pp"].idxmax(), "mode"]
assert (_top == "no_nwp_gfs").all(), "removing the GFS forecasts is not the largest change"

cells = [
    md("""# 08 — Gate G-1: the end-to-end audit before the final run (W6 of stage 9)

**Purpose.** Before the one final training run, an audit of the whole pipeline: when each input
is known against the issuance time, whether the test split touched any decision, whether the
pipeline reproduces from the raw data, and the two analyses of v6 that train models.

**The gate** (`PROJECT_MASTER.md` §7) asks for nine things: the assumptions verified or
declared, the full test suite green, a manual leakage audit, a test split that touched no
decision, a reproduction from scratch, an external review, the smoke and dry runs, the scope
register, and the tag `g1-frozen`.

**Rules.**
- The reproduction is a consistency check of the corrected pipeline, not a match to the submitted
  report (D-34).
- Nothing is changed after the tag unless it only reads frozen outputs (D-36).

**Documented in** OP-60 to OP-62, `docs/leakage_audit.md` §3, `docs/stage9_plan.md` §20 and §21,
run journal R9-25 to R9-32, errata X55 to X75, tests T-93 to T-103 and the figures S09-34 to
S09-41."""),
    md("## 1. The manual leakage audit: when is each input known?"),
    code("""import re

import pandas as pd

import json

from solarpv.config.paths import DOCS_DIR, OUTPUTS_DIR, REPO_ROOT

text = (DOCS_DIR / "leakage_audit.md").read_text(encoding="utf-8")
block = text.split("### 3.1")[1].split("### 3.2")[0]
rows = [[c.strip() for c in line.strip("|").split("|")] for line in block.splitlines()
        if re.match(r"^\\| \\d+ \\|", line)]
families = pd.DataFrame(rows, columns=["#", "family", "known at", "code", "test"])
families[["#", "family", "test"]]"""),
    md(f"""**What it shows.** One row for each of the {len(FAMILIES)} families of inputs, with the
test that proves its rule: the measured lags and rolling statistics, the imputation, the masks, the
calendar and geometry, the GFS run, the encoders and decoders of the sequence models, PatchTST,
the scalers, the L-09 correction, the references, the selection, the tuning and the k history.

**What it means.** No measured input reads an hour that ends after T0. The only information about
the forecast days comes from deterministic functions of time and from the GFS run initialised on
the previous day, every file of which was uploaded before T0 (checked row by row, T-17). The
original audit was a list of forbidden names; it could not state this."""),
    md("## 2. The steps before the hourly table (X57)"),
    code("""block = text.split("### 3.2")[1].split("### 3.3")[0]
rows = [[c.strip() for c in line.strip("|").split("|")] for line in block.splitlines()
        if re.match(r"^\\| \\d+ \\|", line)]
pd.DataFrame(rows, columns=["#", "step", "look-ahead", "measured", "reading"])[
    ["#", "step", "reading"]]"""),
    md(f"""**What it shows.** The {len(STEPS)} steps that build the hourly table, which no earlier
audit had examined. Three look at data after T0 when they prepare an hour before it: the clock
correction (a centred median), the polled-source reconstruction (interpolation towards the next
poll) and the cleaning (frozen runs, margins and a centred window).

**What it means.** Each was measured, not assumed (next cell), and declared in
`docs/deviations.md`. The user accepted them (D-35)."""),
    code("""audit = OUTPUTS_DIR / "g1_audit" / "audit"
clean = pd.read_csv(audit / "lookahead_clean.csv")
clock = pd.read_csv(audit / "lookahead_clock.csv")
polled = pd.read_csv(audit / "lookahead_polled.csv")
display(clean[["source", "issuance_days", "cells_changed", "within_6h_of_t0", "days_affected",
               "test_days_affected", "method_mismatch_cells"]])
display(clock[["clock", "days", "days_shift_differs", "share_pct", "max_shift_difference_min"]])
polled[["source", "polled_issuance_days", "max_bound_on_last_hour_mean"]]"""),
    md(f"""**What it shows.** Cleaning: for every one of the {int(CLEAN['issuance_days'].iloc[0]):,}
issuance days the stage-4 minutes were cleaned up to T0 and with the next day, and the source
values of the 48 hours before T0 compared. {int(CLEAN['cells_changed'].sum())} of
{int(CLEAN['cells'].sum()):,} cells change, on {int(CLEAN['days_affected'].max())} days at most per
series, none of them on a test day. Clock: a trailing median would change the whole-minute
shift on {CLOCK['share_pct'].max():.0f} % of the days, by at most
{int(CLOCK['max_shift_difference_min'].max())} minutes. Polled source: over
{int(POLLED['polled_issuance_days'].max())} issuance days, the interpolation towards the next poll
changes the mean of the last source hour by {POLLED['max_bound_on_last_hour_mean'].max():.2f} at
most (the unit of the series, °C for a temperature).

**What it means.** The clock correction is the only look-ahead of size, and it measures the
logger's clock, not the weather or the plant. The method check (`method_mismatch_cells`,
{int(CLEAN['method_mismatch_cells'].sum())}) shows that the cleaning with the next day equals the
stage-5 table."""),
    md("## 3. The test split touched no decision"),
    code("""import json

access = json.loads((audit / "test_access.json").read_text(encoding="utf-8"))
pd.DataFrame(access["per_run"])[["run", "step_records", "test_access_entries", "prediction_files",
                                 "rows_at_or_after_test_start", "last_prediction_hour"]]"""),
    md(f"""**What it shows.** Every run of the runner so far: {ACC['runs']} runs,
{ACC['step_records']} step records, {ACC['prediction_files']} forecast files. The test-access log
{'exists' if ACC['log_exists'] else 'does not exist'}, and no step recorded a test access.

**What it means.** {ACC['prediction_rows_in_test']} forecast hours fall in the test split: the
latest hour any run forecast is before {ACC['test_start'][:10]}. The test split is read once, by
the final run (phase 1 forecasts it, phase 2 scores it, D-32), and every read is logged."""),
    md("## 4. The incident of 2026-11-12 (OP-60) and what it led to"),
    code("""incident = json.loads((OUTPUTS_DIR / "g1_audit" / "incident_2026-11-12"
                      / "regenerated_vs_original_content.json").read_text(encoding="utf-8"))
pd.DataFrame([{"file": k.split("/", 2)[-1], "rows": v["shape_new"][0], "columns": v["shape_new"][1],
               "content equal": v["content_equal"], "writer before": v["created_by_orig"],
               "writer after": v["created_by_new"]} for k, v in incident.items()])"""),
    md(f"""**What it shows.** Reading the options of the stage modules ran them: stages 4 and 5 were
rewritten on the college machine. The four regenerated tables are compared with the originals,
extracted from the hand-off package and checked against its hashes.

**What it means.** All {len(INC)} tables have exactly the original content ({sum(v['shape_new'][0]
* v['shape_new'][1] for v in INC.values()):,} cells); only the parquet writer's bytes differ. The
originals were restored, and every entry point now parses its arguments before any work (X62,
T-98)."""),
    md("## 5. The reproduction from the raw data"),
    code("""REPRO = REPO_ROOT / "@REPRO@"
repro = json.loads(REPRO.read_text(encoding="utf-8"))
pd.DataFrame([{"step": s["step"], "exit": s["exit"], "seconds": s["seconds"],
               "files": len(s["compared"]), "failing": len(s["failing"])}
              for s in repro["steps"]])""".replace("@REPRO@", REPRO_REL)),
    md(f"""**What it shows.** {len(REC['steps'])} steps of a fresh clone, from the checksums of the
raw files to the two development runs of the runner (all six models), each stage's files compared
with this checkout's by content. Result recorded by the script: **{REC.get('result')}**, because
{N_FAIL} files differ.

**What it means.** {N_FAIL - len(UNEXPLAINED)} of the {N_FAIL} are explained by a documented fix:
the decision checks of stage 6 and 6b were run on the laptop and give other numbers on this machine
with the same decisions (X66), and the reports of the neural development run differ by the warning
that the step now emits (X45) and by row order. {len(UNEXPLAINED)} are unexplained. By D-34 this
is a consistency check, and the explained differences are listed and never made to pass by a wider
rule."""),
    md("## 6. The two analyses that train models (D-36)"),
    code("""RUN = REPO_ROOT / "@RUN@"
cv = pd.read_csv(RUN / "cv" / "cv_summary.csv")
display(cv[["target", "horizon", "cv_mean", "cv_std", "validation_uncorrected",
            "validation_minus_cv_pp"]].round(2))
ab = pd.read_csv(RUN / "ablation" / "ablation_summary.csv")
ab[ab["mode"].isin(["no_nwp_gfs", "no_history", "minimal"])][
    ["target", "horizon", "mode", "mean_delta_pp", "margin_pp", "verdict"]].round(2)"""
         .replace("@RUN@", W6_DEV_RUN)),
    md(f"""**What it shows.** S-38: the nRMSE in kW of five expanding date folds of the train rows,
for the XGB models of the development run, against their validation value. S-41: the change in
validation error when a feature group is removed and the model refitted (a subset of modes).

**What it means.** The fold means range from {CVS['cv_mean'].min():.1f} to
{CVS['cv_mean'].max():.1f} %, and the validation value is {GAP.min():.1f} to {GAP.max():.1f} pp
below the CV mean in every case; this is reported and not explained. Removing the GFS forecasts
raises the error by {GFS.min():.2f} to {GFS.max():.2f} pp, the largest change of any mode except
`minimal`, which keeps only the time encodings.
These are single-seed development values and the verdicts are provisional; the final run repeats
them with three seeds and the pooled seed std."""),
    md("## 7. Checkpoints and resume inside the long steps (D-38, X95)"),
    code("""resume = json.loads((OUTPUTS_DIR / "g1_audit" / "resume_check.json").read_text("utf-8"))
pd.DataFrame([{"configuration": r["name"],
               "first design: largest weight difference": r["first_design"]["max_abs_weight_diff"],
               "final: largest weight difference": r["final"]["max_abs_weight_diff"],
               "dropout between layers": r["interlayer_dropout"]}
              for r in resume["configs"]])"""),
    md(f"""**What it shows.** For each of {len(RES['configs'])} configurations of the BiLSTM and the
TFT, a fit of {RES['epochs']} epochs is continued from the checkpoint after epoch
{RES['resumed_after_epoch']}, and its weights are compared with those of the uninterrupted fit
(`scripts/g1_resume_check.py`, synthetic windows, on {RES['gpu']}). The first design restored every
random state of torch; the final one also runs the earlier training passes again.

**What it means.** The final design is equal to the uninterrupted fit in
{sum(r['final']['equal'] for r in RES['configs'])} of {len(RES['configs'])} configurations. The
first design failed in the {len(RES_BROKEN)} that have dropout between their LSTM layers (weights up
to {max(r['first_design']['max_abs_weight_diff'] for r in RES_BROKEN):.1e} apart): cuDNN keeps a
dropout state that torch's random state does not hold (X95). The replay costs
{100 * min(RES_REPLAY):.0f} to {100 * max(RES_REPLAY):.0f} % of the seconds of the epochs it
replaces. A worker that is killed and started again, and the xgb step, continue to the files of an
uninterrupted run (T-116, T-118); this table is the check of the fit itself."""),
    md("""## 8. Where the gate stands

The checklist, with the state of every item read from its evidence file, is figure S09-34 of the
figure book (`docs/figure_sheets/S09-W6.md`). What remains is outside this notebook: the
external review, the user's approval of the tag, the tag `g1-frozen`, and the final run by
`docs/runbook_final.md` in a plain PowerShell window of the Horizon desktop."""),
]

write_notebook(NB, cells, SOURCES)

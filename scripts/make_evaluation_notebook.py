"""Write notebooks/07_evaluation.ipynb (stage 8: the evaluation layer) and execute it.

    python -m solarpv.evaluation.evaluate --fixture
    python -m solarpv.evaluation.evaluate
    python scripts/make_evaluation_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-8
outputs (fixture and validation) and from the submitted benchmark, never typed. Only the
validation split is evaluated; the notebook shows that the test split is locked.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.paths import EVIDENCE_DIR, OUTPUTS_DIR, REPO_ROOT

NB = REPO_ROOT / "notebooks" / "07_evaluation.ipynb"
VAL = OUTPUTS_DIR / "07_evaluation" / "val"
FIX = OUTPUTS_DIR / "07_evaluation_fixture" / "benchmark_summary.csv"
OLD = EVIDENCE_DIR / "01_benchmark" / "benchmark_summary.csv"
SOURCES = [FIX, OLD, VAL / "manifest.json", VAL / "benchmark_summary.csv",
           VAL / "benchmark_views.csv", VAL / "dm_tests.csv", VAL / "dm_tests_kw.csv",
           VAL / "seasonal_metrics.csv"]

old = pd.read_csv(OLD)
fix = pd.read_csv(FIX)
t15 = fix.merge(old[old["model"] == "lstm_sota"], on=["model", "target", "horizon"],
                suffixes=("_new", "_old"))
t15["diff"] = (t15["nRMSE_pct_new"] - t15["nRMSE_pct_old"]).abs()
B = pd.read_csv(VAL / "benchmark_summary.csv")
d1 = B[B["horizon"] == 1].set_index(["model", "target"])["nRMSE_kW_pct"]
VW = pd.read_csv(VAL / "benchmark_views.csv")
vw = VW[(VW["model"] == "climatology") & (VW["target"] == "direct") & (VW["horizon"] == 1)] \
    .set_index("view")["nRMSE_kW_pct"]
DM = pd.read_csv(VAL / "dm_tests.csv")
DMK = pd.read_csv(VAL / "dm_tests_kw.csv")
_key = ["target", "horizon", "model_a", "model_b"]
_direct = DM[DM["target"] == "direct"].set_index(_key)["dm_stat_hln"]
SAME_DIRECT = float((DMK[DMK["target"] == "direct"].set_index(_key)["dm_stat_hln"]
                     - _direct).abs().max())
MAN = json.loads((VAL / "manifest.json").read_text(encoding="utf-8"))

cells = [
    md("""# 07 — The evaluation layer (stage 8)

**Purpose.** One evaluation code for every model of the project: metrics, confidence intervals,
Diebold–Mariano tests, seasons, operating conditions, the comparison of the two routes, and the
reference forecasts that every model must beat.

**Two parts.**
- **Part A** proves that the code computes what the original computed (T-15), on the only
  predictions of the original that survived.
- **Part B** moves the truth to the corrected feature tables and adds the references.

**Rules.**
- The truth comes from the stage-6 tables, never from the prediction file.
- Every model is scored on every daytime hour that has a truth (C-35).
- The two routes are compared on the same hours (C-34).
- **The test split is locked** until the final evaluation after gate G-1.

**Documented in** OP-30, OP-40, OP-41 and OP-54 (the kW statistics), `docs/stage8b_plan.md`,
run journal R8-01 to R8-03 and R9-11, tests T-03, T-04, T-12, T-15, T-38 to T-41 and T-65."""),
    md("## 1. Part A: the code reproduces the original numbers (T-15)"),
    code("""import pandas as pd

from solarpv.config.paths import EVIDENCE_DIR, OUTPUTS_DIR

old = pd.read_csv(EVIDENCE_DIR / "01_benchmark" / "benchmark_summary.csv")
new = pd.read_csv(OUTPUTS_DIR / "07_evaluation_fixture" / "benchmark_summary.csv")
t15 = new.merge(old[old["model"] == "lstm_sota"], on=["model", "target", "horizon"],
                suffixes=("_rebuild", "_submitted"))
t15[["target", "horizon", "n_rebuild", "n_submitted", "nRMSE_pct_rebuild",
     "nRMSE_pct_submitted"]]"""),
    md(f"""**What it shows.** The surviving BiLSTM predictions of the original, evaluated by the new
code with the original's conventions (its own truth, its daytime mask and its MEAN_K), next to the
submitted benchmark.

**What it means.** The numbers agree to at most {t15['diff'].max():.4f} points of nRMSE, on the
same number of hours. The evaluation code is right; every later difference from the report comes
from the data and the corrections, not from the metric code."""),
    md("## 2. The test split is locked"),
    code("""from solarpv.evaluation import evaluate

try:
    evaluate.run([], OUTPUTS_DIR / "scratch", mean_k=0.788157, truth=pd.DataFrame(),
                 split="test")
except PermissionError as error:
    print("refused:", error)"""),
    md("""**What it shows.** Asking the evaluation for the test split is refused unless the caller
passes `allow_test=True`. That flag is used once, in the final evaluation after the code is
frozen (`g1-frozen`), and every use is written to `outputs/07_evaluation/test_access_log.csv`.

**What it means.** No decision of the project, from the thresholds of stage 5 to the features of
stage 7 and the tuning of stage 9, can see the test results. In the original, test results were
read throughout development."""),
    md("## 3. Part B: the references on validation"),
    code("""bench = pd.read_csv(OUTPUTS_DIR / "07_evaluation" / "val" / "benchmark_summary.csv")
bench.pivot_table(index=["model", "target"], columns="horizon",
                  values="nRMSE_kW_pct").round(2)"""),
    md(f"""**What it shows.** nRMSE in % of the 3,449 kW capacity on the {int(B['n'].iloc[0]):,}
daytime validation hours, for the three reference forecasts:
- **Persistence:** the same hour of the last day known at T0.
- **Climatology:** the train mean of each month and hour.
- **Clear sky:** the clear-sky power times the mean daytime k.

**What it means.** At D+1 for the direct target:
- climatology {d1[('climatology', 'direct')]:.2f} %;
- clear sky {d1[('clearsky_k', 'direct')]:.2f} %;
- persistence {d1[('persistence', 'direct')]:.2f} %.

In this desert climate persistence is the weakest reference, and climatology the strongest. A
model that does not beat climatology adds nothing; that is the bar for stage 9. The errors of
both routes are given in kW, so the routes compare directly."""),
    code("""views = pd.read_csv(OUTPUTS_DIR / "07_evaluation" / "val" / "benchmark_views.csv")
views[(views["horizon"] == 1)].pivot_table(index=["model", "target"], columns="view",
                                           values="nRMSE_kW_pct").round(2)"""),
    md(f"""**What it shows.** The same scores on subsets of hours (D-21, D-22): all hours, normal
operation only (no stop, curtailment or derate), and by data source.

**What it means.** For climatology at D+1 (direct), excluding the op-limited hours lowers the
error from {vw['all']:.2f} % to {vw['normal_operation']:.2f} %. Part of every error is the plant,
not the weather; the main result uses all hours, and this view shows the share. The views of the
2025 source (polled, mixed) do not appear on validation: that source exists only in the test
months."""),
    code("""dm = pd.read_csv(OUTPUTS_DIR / "07_evaluation" / "val" / "dm_tests.csv")
dm[["target", "horizon", "model_a", "model_b", "dm_stat_hln", "p_value_hln",
    "pct_of_mde", "significant"]].round(3)"""),
    md(f"""**What it shows.** Diebold–Mariano tests against persistence, with the Harvey–Leybourne–
Newbold correction and Bonferroni, and the share of the minimum detectable effect.

**What it means.** {int(DM['significant'].sum())} of {len(DM)} comparisons are significant at the
corrected level. The original report described HLN and Bonferroni but computed neither (C-15);
here both are computed, and the uncorrected value is reported next to them."""),
    code("""dm_kw = pd.read_csv(OUTPUTS_DIR / "07_evaluation" / "val" / "dm_tests_kw.csv")
dm_kw[["target", "horizon", "model_a", "model_b", "dm_stat_hln", "p_value_hln",
       "pct_of_mde", "significant"]].round(3)"""),
    md(f"""**What it shows.** The same tests on the errors in kW against the actual AC power, for
both routes (Q-19, added on 2026-11-07). The indirect forecast is converted with the same
function as every other kW number.

**What it means.** {int(DMK['significant'].sum())} of {len(DMK)} comparisons are significant. For
the direct target the two tables agree (largest difference of the statistic
{SAME_DIRECT:.1g}), because its unit is already kW. For the indirect route the k table gives every
daytime hour the same weight, so the low-sun hours, where k is unstable and the power small, weigh
as much as noon; the kW table weighs the hours by their power. Every decision of the project is
taken in kW, so the kW table is the one stage 10 reads, and the k table stays as a secondary
statistic."""),
    code("""seasons = pd.read_csv(OUTPUTS_DIR / "07_evaluation" / "val" / "seasonal_metrics.csv")
d1 = seasons[(seasons["horizon"] == 1) & (seasons["target"] == "direct")]
d1.pivot_table(index="model", columns="season", values="nRMSE_pct").round(2)"""),
    md(f"""**What it shows.** D+1 errors of the direct target by season on validation (February to
September 2024), nRMSE in % of capacity.

**What it means.** Errors are largest in the seasons with changeable weather and smallest in the
clear summer. The same tables will be produced for every model in stages 9 to 11.

**Run recorded in** `{MAN['run_log']}` (commit {MAN['git_commit']}, truth
`{MAN['truth']}`).

**Next stages.** Stage 7 selects the features; stage 9 trains the light models and scores them
here against these references.

**Reproduce.** `python -m solarpv.evaluation.evaluate --fixture`,
`python -m solarpv.evaluation.evaluate`, then `python scripts/make_evaluation_notebook.py`."""),
]

write_notebook(NB, cells, SOURCES)

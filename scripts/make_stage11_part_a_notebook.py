# ruff: noqa: E501  (markdown notes with values cannot be wrapped; see make_quality_notebook)
"""Write notebooks/09_stage11_part_a.ipynb (stage 11, part A) and execute it.

    python -m solarpv.figures.analyses11
    python scripts/make_stage11_part_a_notebook.py

Every code cell is followed by a markdown note. Every number in the notes is read from the tables
of `outputs/11_analyses/`, never typed; each claim is asserted here before the note is written.
The notebook shows what part A computed from the frozen final run `20261114T083930Z_final`.
No model is trained and the step `test` is not run again.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import REPO_ROOT

no_options("python scripts/make_stage11_part_a_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "09_stage11_part_a.ipynb"
A = REPO_ROOT / "outputs" / "11_analyses"
TABLES = ["metrics_kw", "decomposition", "decomposition_seeds", "quintiles", "quintile_control",
          "hour_season", "seasonal_kw", "encoder_length", "patchtst_gap", "benchmark_ci_kw",
          "scatter_hist", "horizon_robustness", "literature_alignment"]
SOURCES = [A / f"{t}.csv" for t in TABLES] + [A / "manifest.json",
                                               REPO_ROOT / "docs" / "literature_review.md"]
M6 = ["xgb_sota", "lstm_sota", "tft_96", "tft_168", "tft", "patchtst"]

man = json.loads((A / "manifest.json").read_text(encoding="utf-8"))
ci = pd.read_csv(A / "benchmark_ci_kw.csv")
hr = pd.read_csv(A / "horizon_robustness.csv")
dec = pd.read_csv(A / "decomposition.csv")
dsd = pd.read_csv(A / "decomposition_seeds.csv")
qu = pd.read_csv(A / "quintiles.csv")
qc = pd.read_csv(A / "quintile_control.csv")
hs = pd.read_csv(A / "hour_season.csv")
sea = pd.read_csv(A / "seasonal_kw.csv")
enc = pd.read_csv(A / "encoder_length.csv")
pt = pd.read_csv(A / "patchtst_gap.csv")
lit = pd.read_csv(A / "literature_alignment.csv")

# claims of the notes below, asserted here and not typed
g = ci.groupby(["model", "target", "horizon", "split"])[["nRMSE_kW_pct", "ci_lo", "ci_hi"]].mean()
HALF = {s: float(((g.xs(s, level="split")["ci_hi"] - g.xs(s, level="split")["ci_lo"]) / 2)
                 .loc[M6].mean()) for s in ("val", "test")}
SIX = hr[hr["model"].isin(M6)]
VAL_DOWN = int((SIX[SIX["split"] == "val"]["D3_minus_D1_pp"] < 0).sum())
TEST_UP = int((SIX[SIX["split"] == "test"]["D3_minus_D1_pp"] > 0).sum())
assert VAL_DOWN > 6 and TEST_UP > 6
D1 = dec[dec["view"] == "all"]
DN = dec[dec["view"] == "normal_operation"]
ME_ALL = D1[(D1["split"] == "test") & D1["model"].isin(M6)].groupby("model")["me_kw"].mean()
ME_NORM = DN[(DN["split"] == "test") & DN["model"].isin(M6)].groupby("model")["me_kw"].mean()
assert (ME_ALL > 0).all() and ME_NORM[[m for m in M6 if m != "patchtst"]].abs().max() < 15
B2 = D1[D1["model"].isin(M6) & (D1["model"] != "patchtst")].groupby("split")["bias2_share"].max()
SV = dsd[dsd["view"] == "all"]["seed_variance_share"]
q = qu[(qu["model"] == "xgb_sota") & (qu["seed"] == 42) & (qu["target"] == "direct")
       & (qu["horizon"] == 1) & (qu["split"] == "test")]
qo, qf = q[q["binned_by"] == "observed"].set_index("bin"), q[q["binned_by"] == "forecast"].set_index("bin")
qk = qc[(qc["target"] == "direct") & (qc["horizon"] == 1) & (qc["split"] == "test")].set_index("bin")
assert qo.loc[1, "me_kw"] > 0 > qo.loc[5, "me_kw"] and qf.loc[5, "me_kw"] > 0
assert qk.loc[1, "me_calibrated_kw"] > 0 > qk.loc[5, "me_calibrated_kw"]
SX = sea[(sea["model"] == "xgb_sota") & (sea["target"] == "indirect") & (sea["horizon"] == 1)]
FALL = SX[SX["season"] == "Fall"].set_index("split")
assert FALL.loc["test", "nRMSE_kW_pct"] > 3 * FALL.loc["val", "nRMSE_kW_pct"]
H = hs[(hs["model"] == "xgb_sota") & (hs["split"] == "test") & (hs["target"] == "indirect")
       & (hs["horizon"] == 1) & (hs["season"] == "Fall") & (hs["hour"] == 11)].set_index("view")
assert H.loc["normal_operation", "nRMSE_kW_pct"] < 0.7 * H.loc["all", "nRMSE_kW_pct"]
ENC_T = enc[enc["split"] == "test"]
assert not enc[enc["split"] == "val"]["significant"].any()
PT_X = pt[pt["model_b"] == "xgb_sota"]
PT_C = pt[pt["model_b"] == "climatology"]
assert PT_X["significant"].all() and (PT_C[PT_C["significant"]]["diff_a_minus_b_pp"] > 0).all()
OURS = lit[lit["model"] != "literature"]
LIT = lit[lit["model"] == "literature"]["nRMSE_mean_pct"]

cells = [
    md("""# 09 - Stage 11, part A: tables and figures from the frozen final run

**Purpose.** The analyses of the submitted report that need no training, computed from the saved
forecasts of the final run `20261114T083930Z_final` (tag `g1-frozen`): the benchmark with
intervals, the horizon, the bias and variance of the error, the production quintiles, the error by
season and local hour, the encoder length of the TFT, PatchTST against XGB, and the comparison
with the literature on one normalisation.

**Rules.**
- Nothing is trained and the step `test` is not run again. The test rows are read only to score the
  frozen forecasts; each read is in `outputs/07_evaluation/test_access_log.csv`.
- Every test number is placed beside the validation number of the same view.
- No winner is declared (D-37). Four of the five neural models have one seed.

**Documented in** `docs/stage11_plan.md`, OP-66, run journal R11-01, errata C-78 to C-84 (M45-M50)
and X100-X106, tests T-169 to T-177, and the figures S11-01 to S11-17."""),
    md("## 1. What the module checked before it wrote anything"),
    code("""import json

import pandas as pd

from solarpv.config.paths import REPO_ROOT

A = REPO_ROOT / "outputs" / "11_analyses"
man = json.loads((A / "manifest.json").read_text(encoding="utf-8"))
pd.Series({"run": man["run"], "tables": len(man["rows"]),
           "input files": len(man["inputs_sha256"]),
           "max difference of nRMSE vs the run's report (pp)":
               man["max_abs_diff_vs_report_nRMSE_kW_pct"],
           "max relative error of MSE = ME^2 + Var(e)":
               man["max_rel_identity_error_bias2_plus_variance"]})"""),
    md(f"""**What it shows.** The module reads {len(man['inputs_sha256'])} prediction files, each
checked against the SHA-256 of its step record, and scores them in kW against the actual AC.

**What it means.** The nRMSE of every model, seed, route, horizon and split agrees with the run's
own report tables to {man['max_abs_diff_vs_report_nRMSE_kW_pct']:.1e} (the tables are written with
six decimals), so the new tables use the same errors as the acceptance and the test step (X100).
The decomposition MSE = ME^2 + Var(e) holds to {man['max_rel_identity_error_bias2_plus_variance']:.0e}."""),
    md("## 2. The benchmark in kW, with intervals (S-23)"),
    code("""ci = pd.read_csv(A / "benchmark_ci_kw.csv")
g = ci.groupby(["model", "target", "horizon", "split"])[["nRMSE_kW_pct", "ci_lo", "ci_hi"]].mean()
t = g.xs(("indirect", 1), level=("target", "horizon")).round(2)
t.unstack("split")"""),
    md(f"""**What it shows.** The nRMSE in kW (percent of 3,449 kW) of the indirect route at D+1, with a
95 % circular block bootstrap interval, mean over seeds, validation and test, for the six models and
the three references (`S11-01` draws all six cells).

**What it means.** The half-width of the interval of one score is {HALF['val']:.2f} pp on validation
and {HALF['test']:.2f} pp on test (mean over the six models). These intervals are not paired, so their
overlap is not a test of a difference; the paired test is S10-04."""),
    md("## 3. The horizon (S-26)"),
    code("""hr = pd.read_csv(A / "horizon_robustness.csv")
hr[hr["target"] == "indirect"].round(2).set_index(["model", "split"])[
    ["D1", "D2", "D3", "D3_minus_D1_pp"]]"""),
    md(f"""**What it shows.** The seed-mean nRMSE of every model and reference at D+1, D+2 and D+3, and
the change from D+1 to D+3 (the direct route is in the file).

**What it means.** On validation the error at D+3 is lower than at D+1 in {VAL_DOWN} of the 12 pairs of
a model and a route; on test it is higher in {TEST_UP} of 12. The sign of the horizon effect reverses
between the two periods, and its size is at most
{float(SIX['D3_minus_D1_pp'].abs().max()):.2f} pp: the order of the horizons is a property of the
period (R6b-07, R7-05), not of the horizon (`S11-02`)."""),
    md("## 4. Bias² and variance, and where the bias comes from (S-39, L-15)"),
    code("""dec = pd.read_csv(A / "decomposition.csv")
d = dec[(dec["split"] == "test") & (dec["horizon"] == 1) & (dec["target"] == "indirect")]
d.pivot_table(index="model", columns="view", values="me_kw").round(1).join(
    d[d["view"] == "all"].groupby("model")[["bias2_share", "variance_ratio"]].mean().round(3))"""),
    md(f"""**What it shows.** For the indirect route at D+1 on test: the mean error in kW on all daytime
hours and on the hours of normal operation (no curtailment, stop or derate; D-22), the bias² share of the
MSE and the variance ratio V(f)/V(y) (L-15).

**What it means.** Bias² is at most {B2['val']:.1%} of the squared error on validation and
{B2['test']:.1%} on test for the five models other than PatchTST. Every model has a positive mean error on
all hours ({ME_ALL.min():+.0f} to {ME_ALL.max():+.0f} kW on test), and for XGB, BiLSTM and the three TFTs
it is {ME_NORM[[m for m in M6 if m != 'patchtst']].min():+.0f} to
{ME_NORM[[m for m in M6 if m != 'patchtst']].max():+.0f} kW in normal operation: the positive bias of
the neural forecasts (a G-1 list point, D-37) is made on the operation-limited hours. PatchTST keeps
{ME_NORM['patchtst']:+.0f} kW. The variance between the three seeds is at most {SV.max():.1%} of the
squared error (`S11-11`). The submitted section 12.3 had no definition of "variance"; here it is the
variance of the error, and the seed split is a separate table (M46)."""),
    md("## 5. Production quintiles and the control for binning by the outcome (S-40, M45)"),
    code("""qu = pd.read_csv(A / "quintiles.csv")
qc = pd.read_csv(A / "quintile_control.csv")
q = qu[(qu["model"] == "xgb_sota") & (qu["seed"] == 42) & (qu["target"] == "direct")
       & (qu["horizon"] == 1) & (qu["split"] == "test")]
t = q.pivot(index="bin", columns="binned_by", values="me_kw").round(0)
t["calibrated control, observed bins"] = qc[(qc["target"] == "direct") & (qc["horizon"] == 1)
    & (qc["split"] == "test")].set_index("bin")["me_calibrated_kw"].round(0)
t["op-limited share, observed bins"] = q[q["binned_by"] == "observed"].set_index("bin")[
    "op_limited_share"].round(3)
t"""),
    md(f"""**What it shows.** The mean error of the XGB (seed 42, direct route, D+1, test) in five bins of
equal validation count, binned by the observed power and by the forecast; a control (the same forecast
made unbiased on validation by an isotonic regression) binned by the observed power; and the share of
operation-limited hours in each observed bin.

**What it means.** Binned by the observed power the lowest bin has {qo.loc[1, 'me_kw']:+.0f} kW and the
highest {qo.loc[5, 'me_kw']:+.0f} kW, the pattern that the submitted report read as inverter downtime and
as underestimation of the peak. Binned by the forecast it is {qf.loc[1, 'me_kw']:+.0f} and
{qf.loc[5, 'me_kw']:+.0f} kW. The control forecast, unbiased by construction, gives
{qk.loc[1, 'me_calibrated_kw']:+.0f} and {qk.loc[5, 'me_calibrated_kw']:+.0f} kW in the observed bins, so
the pattern comes with binning by the outcome (M45). {qo.loc[1, 'op_limited_share']:.0%} of the hours of
the lowest observed bin are operation-limited (`S11-12`)."""),
    md("## 6. Error by season and by local hour (S-29, S-30, L-20)"),
    code("""sea = pd.read_csv(A / "seasonal_kw.csv")
s = sea[(sea["target"] == "indirect") & (sea["horizon"] == 1)
        & sea["model"].isin(["xgb_sota", "climatology"])]
s.pivot_table(index=["model", "season"], columns="split", values="nRMSE_kW_pct").round(2).join(
    s[s["model"] == "xgb_sota"].set_index(["model", "season", "split"])["months"]
    .unstack("split").add_prefix("months "), how="left")"""),
    md(f"""**What it shows.** The nRMSE in kW by season for XGB and climatology, with the local months
that each split holds in that season.

**What it means.** The test split has no summer. Its autumn is {FALL.loc['test', 'months']} and the
validation autumn is {FALL.loc['val', 'months']}, so a season is not the same weather in the two periods
(X104). XGB's error in the autumn is {FALL.loc['val', 'nRMSE_kW_pct']:.1f} % on validation and
{FALL.loc['test', 'nRMSE_kW_pct']:.1f} % on test, and the rise of the test period is concentrated there
(`S11-04`)."""),
    code("""hs = pd.read_csv(A / "hour_season.csv")
h = hs[(hs["split"] == "test") & (hs["target"] == "indirect") & (hs["horizon"] == 1)
       & (hs["model"] == "xgb_sota") & (hs["season"] == "Fall") & (hs["n"] >= 20)]
h.pivot(index="hour", columns="view", values="nRMSE_kW_pct").round(1).join(
    h[h["view"] == "all"].set_index("hour")["n"])"""),
    md(f"""**What it shows.** XGB on test in the autumn, by local hour: the nRMSE on all hours and in normal
operation, and the number of hours.

**What it means.** At 11:00 the error is {H.loc['all', 'nRMSE_kW_pct']:.1f} % on all hours and
{H.loc['normal_operation', 'nRMSE_kW_pct']:.1f} % in normal operation: the autumn peak of the diurnal
profile (`S11-05`) is mostly the operation-limited hours, which cluster around midday; in normal operation
the profile is flat (`S11-17`). The submitted "afternoon shoulder" is not seen (M47)."""),
    md("## 7. Two paired comparisons (S-34, S-35)"),
    code("""enc = pd.read_csv(A / "encoder_length.csv")
pt = pd.read_csv(A / "patchtst_gap.csv")
display(enc[["split", "target", "horizon", "diff_a_minus_b_pp", "p_value_hln", "pct_of_mde",
             "significant"]].round(3))
pt[["split", "target", "horizon", "model_b", "diff_a_minus_b_pp", "p_value_hln", "significant"]
   ].round(3)"""),
    md(f"""**What it shows.** TFT-168 minus TFT-96 (a positive difference: the longer encoder is worse) and
PatchTST minus XGB and minus climatology, in kW, with the paired Diebold-Mariano test (HLN, Bonferroni
over the six cells of a split).

**What it means.** The encoder length: on validation the difference is
{enc[enc['split'] == 'val']['diff_a_minus_b_pp'].min():+.2f} to
{enc[enc['split'] == 'val']['diff_a_minus_b_pp'].max():+.2f} pp with no significant cell; on test it is
negative for the indirect route and positive for the direct route, and {int(enc['significant'].sum())} of
12 cells is significant. A consistent advantage of the shorter window, as the submitted report states, is
not found (M48, `S11-09`). PatchTST is worse than XGB by
{PT_X['diff_a_minus_b_pp'].min():.2f} to {PT_X['diff_a_minus_b_pp'].max():.2f} pp, significantly in every
cell, and worse than climatology by {PT_C['diff_a_minus_b_pp'].min():+.2f} to
{PT_C['diff_a_minus_b_pp'].max():+.2f} pp (significant in {int(PT_C['significant'].sum())} of 12): the
submitted bound on the value of feature engineering is not stated (M49, `S11-14`). Both comparisons have
one seed per model."""),
    md("## 8. The literature, on one normalisation (L-20, L-16)"),
    code("""lit = pd.read_csv(A / "literature_alignment.csv")
x = lit[lit["model"] != "literature"].groupby(["model", "split"])["nRMSE_mean_pct"].mean().unstack()
display(x.round(1))
lit[lit["model"] == "literature"][["source", "nRMSE_mean_pct"]]"""),
    md(f"""**What it shows.** Our D+1 nRMSE normalised by the mean observed daytime power (the
normalisation of the papers of `docs/literature_review.md`), and the papers' numbers.

**What it means.** Our six models are {OURS[OURS['model'].isin(M6)].groupby('model')['nRMSE_mean_pct'].mean().min():.0f}
to {OURS[OURS['model'].isin(M6)].groupby('model')['nRMSE_mean_pct'].mean().max():.0f} % (validation and test
together by model), the papers {LIT[LIT < 50].min():.1f} to {LIT[LIT < 50].max():.1f} % for tuned models. The
numbers are not comparable one to one: a hyper-arid site where most hours are clear, GFS instead of a 9 km
ECMWF run, hourly instead of 15-minute data, one plant instead of a mean over seven, and an eight-month test
period. No ranking against the papers is made (M50, `S11-15`)."""),
    md("""## 9. What part A did not do

- **Dust days (L-20).** The project holds no aerosol record; the definition of a dust day is Q-29 for
  the user.
- **`reproduce_paper_results.py --xgb-only` and `--skip-train` (S-45).** The script is outside the D-36
  list; Q-30 for the user.
- **The report text** (S-01, S-05, S-42 to S-44) and the final table of old against new values: part B."""),
]

write_notebook(NB, cells, SOURCES)

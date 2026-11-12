"""Write notebooks/06_models.ipynb (stage 9: the runner, XGB-dev and the neural models in
development) and execute it.

    python -m solarpv.pipeline run --mode dev --only select,baselines,xgb,evaluate,explain,report
    python -m solarpv.pipeline run --mode dev --resume 20261111T160714Z_dev \
        --only select,explain,report
    python scripts/make_models_notebook.py

In the numbering of the original project (06: the models). Every code cell is followed by a
markdown note; every number in a note is read from the outputs of the two development runs (the
XGB dev run of W5, which reproduces `20261107T135938Z_dev`, and the neural dev run of W4), never
typed, and every interpretive claim is asserted here first. No test row is read: these are
tuning-split numbers (D-10), optimistic by construction, and they decide nothing about the final
run.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import REPO_ROOT
from solarpv.pipeline.config import STEPS, TEST_STEP

no_options("python scripts/make_models_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "06_models.ipynb"
XRUN = "outputs/runs/20261112T063719Z_dev"
NRUN = "outputs/runs/20261111T160714Z_dev"
OLD_XRUN = "outputs/runs/20261107T135938Z_dev"
X, N, OLD = REPO_ROOT / XRUN, REPO_ROOT / NRUN, REPO_ROOT / OLD_XRUN
SOURCES = [X / "select" / "select.json", N / "select" / "select.json",
           X / "evaluate" / "val" / "benchmark_summary.csv",
           X / "evaluate" / "val" / "dm_tests_kw.csv",
           OLD / "evaluate" / "val" / "benchmark_summary.csv",
           N / "evaluate" / "val" / "benchmark_summary.csv",
           N / "report" / "report_training.csv",
           X / "explain" / "xgb" / "xgb_shap_checks.csv",
           X / "explain" / "xgb" / "xgb_shap_features.csv",
           X / "explain" / "xgb" / "xgb_shap_groups.csv",
           N / "explain" / "tft" / "tft_attention_summary.csv",
           N / "explain" / "tft" / "tft_attention_lag.csv",
           N / "explain" / "tft" / "tft_variable_importance.csv",
           X / "explain" / "hparam_importance.csv", N / "explain" / "hparam_importance.csv",
           X / "report" / "report_warnings.csv", N / "report" / "report_warnings.csv",
           X / "report" / "report_g1_points.csv", N / "report" / "report_g1_points.csv"]
REFS = ["persistence", "climatology", "clearsky_k"]


def jl(p):
    return json.loads(p.read_text(encoding="utf-8"))


SEL, SEL_N = jl(X / "select" / "select.json"), jl(N / "select" / "select.json")
XB = pd.read_csv(X / "evaluate" / "val" / "benchmark_summary.csv")
OLDB = pd.read_csv(OLD / "evaluate" / "val" / "benchmark_summary.csv")
NBM = pd.read_csv(N / "evaluate" / "val" / "benchmark_summary.csv")
DMK = pd.read_csv(X / "evaluate" / "val" / "dm_tests_kw.csv")
TRAIN = pd.read_csv(N / "report" / "report_training.csv")
CHK = pd.read_csv(X / "explain" / "xgb" / "xgb_shap_checks.csv")
FEA = pd.read_csv(X / "explain" / "xgb" / "xgb_shap_features.csv")
GRP = pd.read_csv(X / "explain" / "xgb" / "xgb_shap_groups.csv")
ATT = pd.read_csv(N / "explain" / "tft" / "tft_attention_summary.csv")
LAG = pd.read_csv(N / "explain" / "tft" / "tft_attention_lag.csv")
VAR = pd.read_csv(N / "explain" / "tft" / "tft_variable_importance.csv")
HPX = pd.read_csv(X / "explain" / "hparam_importance.csv")
HPN = pd.read_csv(N / "explain" / "hparam_importance.csv")
W = pd.concat([pd.read_csv(X / "report" / "report_warnings.csv"),
               pd.read_csv(N / "report" / "report_warnings.csv")], ignore_index=True)
G1 = pd.concat([pd.read_csv(X / "report" / "report_g1_points.csv"),
                pd.read_csv(N / "report" / "report_g1_points.csv")], ignore_index=True)

# ── the facts of the notes, each claim asserted ─────────────────────────────
inputs = {r["key"]: r for r in SEL["inputs"]}
assert SEL["model_inputs_sha256"] == SEL_N["model_inputs_sha256"]
key = ["model", "target", "horizon", "seed"]
SAME_OLD = float(XB.set_index(key)[["nRMSE_kW_pct", "nRMSE_pct"]].sub(
    OLDB.set_index(key)[["nRMSE_kW_pct", "nRMSE_pct"]]).abs().max().max())
assert SAME_OLD == 0.0  # the W5 dev run reproduces the W2 dev run
xk = XB[XB["model"] == "xgb_sota"].set_index(["target", "horizon"])["nRMSE_kW_pct"]
ref = XB[XB["model"].isin(REFS)].groupby(["target", "horizon"])["nRMSE_kW_pct"].min()
assert (xk < ref).all()  # below every reference at every target and horizon
assert all(xk[(t, 3)] < xk[(t, 1)] for t in ("indirect", "direct"))  # the R9-06 pattern
clim = XB[XB["model"] == "climatology"].set_index(["target", "horizon"])["nRMSE_kW_pct"]
pers = XB[XB["model"] == "persistence"].set_index(["target", "horizon"])["nRMSE_kW_pct"]
DMX = DMK[DMK["model_b"] == "xgb_sota"]
DMC = DMX[DMX["model_a"] == "climatology"]
N_HOURS = int(XB["n"].iloc[0])
assert XB["n"].nunique() == 1 and NBM["n"].nunique() == 1 and int(NBM["n"].iloc[0]) == N_HOURS
neural = NBM[~NBM["model"].isin(REFS)]
BELOW_PERS = int(sum(r.nRMSE_kW_pct < pers[(r.target, r.horizon)] for r in neural.itertuples()))
tuned = neural[neural["model"] != "patchtst"]
GAP_XGB = float(max(r.nRMSE_kW_pct - xk[(r.target, r.horizon)] for r in tuned.itertuples()))
patch = neural[neural["model"] == "patchtst"]
assert all(r.nRMSE_kW_pct > clim[(r.target, r.horizon)] for r in patch.itertuples())
allm = pd.concat([XB[XB["model"] == "xgb_sota"], neural], ignore_index=True)
BEST = {t: allm[(allm["target"] == t) & (allm["horizon"] == 1)].sort_values(
    "nRMSE_kW_pct").iloc[0] for t in ("indirect", "direct")}
BIL = TRAIN[(TRAIN["step"] == "bilstm") & (TRAIN["target"] == "indirect")].iloc[0]
f42 = FEA[FEA["seed"] == 42]
TOP = {(t, h): f42[(f42["target"] == t) & (f42["horizon"] == h)].sort_values("rank").iloc[0]
       for t in ("indirect", "direct") for h in (1, 2, 3)}
assert all(r["group"] == "nwp_gfs" for r in TOP.values())  # a GFS forecast leads everywhere
GFS = GRP[GRP["group"] == "nwp_gfs"]["share_pct"]
assert (CHK["dominance_first_over_third"] < 10).all()  # no stage-7 dominance red flag
A_ALL = ATT[ATT["block"] == "all"]
assert A_ALL["same_hour_ratio"].max() < 1.2  # no clear same-hour preference
dl = LAG[(LAG["target"] == "direct") & (LAG["block"] == "all")]
PEAK = {s: int(g.groupby((g["lag_h"] % 24).astype(int))["attention"].mean().idxmax())
        for s, g in dl.groupby("step")}
ENC = VAR[VAR["part"] == "encoder"].drop_duplicates(["step", "target", "variable"])
TOPVAR = ENC.sort_values("variable_weight").iloc[-1]
LEADERS = ENC.sort_values("variable_weight", ascending=False).groupby(["step", "target"]).head(1)
assert LEADERS["variable"].nunique() > 1  # the leaders differ between models and routes
ONLY = W[W["status"] == "recomputed only"]
assert (ONLY["kind"] == "edge_space").all() and ONLY["text"].str.contains(
    "hidden_continuous_size").all()  # X45 is the only source of a recomputed-only warning
HPXTOP = HPX[HPX["step"] == "xgb"].groupby("param")["importance_mean"].mean().sort_values()
STUDIES = HPN.drop_duplicates(["step", "target", "seed"])
TFT_TRIALS = int(STUDIES[STUDIES["step"].isin(["tft96", "tft168"])]["n_complete"].max())

cells = [
    md(f"""# 06 — The models (stage 9): the runner, XGB-dev and the neural models in development

**Purpose.** One runner (`python -m solarpv.pipeline`) trains every model of the study in
sequence, evaluates them with the evaluation layer of stage 8, explains them and writes one
report per run. This notebook shows the two development runs of stage 9, XGB-dev (W2, rerun in
W5 with the new steps) and the five neural models (W4), with what W5 added: the verified input
(`select`), the explanations (`explain`) and the run report (`report`).

**Rules.**
- Development runs read train and validation only, not one test row.
- Validation chose the hyper-parameters and the stopping rounds (D-10), so every number here is
  a tuning-split number, optimistic by construction. The final run (stage 10) and the test split
  (stage 11) come once, after gate G-1, on frozen code.
- Every model and every reference is scored on the same {N_HOURS:,} daytime validation hours per
  horizon.

**Runs.** `{XRUN}` (XGB) and `{NRUN}` (BiLSTM, TFT-96, TFT-168, TFT baseline, PatchTST).

**Documented in** `docs/stage9_plan.md` §13–§19, OP-49 to OP-58, run journal R9-01 to R9-21,
tests T-51 to T-91, `docs/runbook_final.md`."""),
    md("## 1. The runner and its steps"),
    code(f"""import json

import pandas as pd

from solarpv.config.paths import REPO_ROOT
from solarpv.pipeline.config import STEPS
from solarpv.pipeline.runner import select_steps

XD = REPO_ROOT / "{XRUN}"  # XGB-dev, with select, explain, report
ND = REPO_ROOT / "{NRUN}"  # the neural models, resumed with them
print("default sequence:", select_steps())
pd.DataFrame([(s.name, s.stage, s.device, s.purpose) for s in STEPS],
             columns=["step", "stage", "device", "purpose"])"""),
    md(f"""**What it shows.** The {len(STEPS)} steps of the runner in their order. A run executes a
selection of them (`--only`, `--from`, `--to`), and `--resume` skips a step whose fingerprint
(mode, configuration, code commit, input hashes) is unchanged.

**What it means.** The last step, `{TEST_STEP}`, is not in the default sequence: it is the second
phase of a final run and runs only with `--allow-test`, after the stage-10 acceptance has been
written (D-32, X51). In development the test split cannot be reached at all."""),
    md("## 2. The input of every model, verified (`select`, D-30)"),
    code("""sel = json.loads((XD / "select" / "select.json").read_text(
    encoding="utf-8"))
print("model_inputs.json", sel["model_inputs_sha256"][:16], "| rows read:", sel["rows_read"])
pd.DataFrame(sel["inputs"])"""),
    md(f"""**What it shows.** The `select` step reruns nothing (stage 7 is closed, R7-04 and R7-05).
It checks that the stage-7 outputs are unmodified, that the tables named in the stage-7 run logs
are the current ones, that the selection read train only with one validation look, and that
`model_inputs.json` equals the registered rule applied to the committed checks.

**What it means.** The indirect route uses the 05e sets of {inputs['indirect_D1']['n_features']},
{inputs['indirect_D2']['n_features']} and {inputs['indirect_D3']['n_features']} features, the
direct route the full pool of {inputs['direct_D1']['n_features']},
{inputs['direct_D2']['n_features']} and {inputs['direct_D3']['n_features']} (F-29). Both runs
verified the same file (SHA-256 `{SEL['model_inputs_sha256'][:16]}`), and every model step of a
run refuses another one (M25)."""),
    md("## 3. XGB-dev against the references"),
    code("""bench = pd.read_csv(XD / "evaluate" / "val" / "benchmark_summary.csv")
bench.pivot_table(index=["target", "model"], columns="horizon", values="nRMSE_kW_pct").round(2)"""),
    md(f"""**What it shows.** Validation nRMSE in % of the 3,449 kW capacity (the indirect route
converted to kW): XGB-dev (20 trials, seed 42) against the three references.

**What it means.** XGB-dev scores {xk[('indirect', 1)]:.2f} / {xk[('indirect', 2)]:.2f} /
{xk[('indirect', 3)]:.2f} % (indirect) and {xk[('direct', 1)]:.2f} / {xk[('direct', 2)]:.2f} /
{xk[('direct', 3)]:.2f} % (direct) at D+1 / D+2 / D+3, below every reference at every horizon
(climatology {clim[('direct', 1)]:.2f} %, persistence {pers[('direct', 1)]:.2f} % at direct D+1).
This run reproduces the XGB dev run of W2 (`{OLD_XRUN}`) exactly: the largest difference over
every model, target, horizon and unit is {SAME_OLD:.1f}. D+3 below D+1 is the validation-sample
pattern examined in R9-06."""),
    code("""dm = pd.read_csv(XD / "evaluate" / "val" / "dm_tests_kw.csv")
dm[dm["model_b"] == "xgb_sota"][["target", "horizon", "model_a", "p_value_hln", "pct_of_mde",
                                  "significant"]].round(4)"""),
    md(f"""**What it shows.** Diebold–Mariano tests in kW (Q-19) of XGB-dev against every reference,
with the HLN correction and Bonferroni.

**What it means.** {int(DMX['significant'].sum())} of {len(DMX)} comparisons are significant.
Against climatology the effect is {DMC['pct_of_mde'].min():.0f} to {DMC['pct_of_mde'].max():.0f}
% of the minimum detectable effect: better, with a modest margin (a point of the G-1 list,
R9-11)."""),
    md("## 4. The neural models in development"),
    code("""nb = pd.read_csv(ND / "evaluate" / "val" / "benchmark_summary.csv")
nb.pivot_table(index=["target", "model"], columns="horizon", values="nRMSE_kW_pct").round(2)"""),
    md(f"""**What it shows.** The five neural models of v6 with the dev budgets of D-17 (one third
of the v6 trials, the full epochs, seed 42), on the same {N_HOURS:,} hours.

**What it means.** {BELOW_PERS} of {len(neural)} model × target × horizon cases are below
persistence. At D+1 the best model is `{BEST['indirect']['model']}`
({BEST['indirect']['nRMSE_kW_pct']:.2f} %) in the indirect route and `{BEST['direct']['model']}`
({BEST['direct']['nRMSE_kW_pct']:.2f} %) in the direct route; the tuned neural models are at most
{GAP_XGB:.2f} points above XGB-dev, as the literature expects for well-tuned families (C1).
PatchTST sees only the target series (L-19) and stays above climatology everywhere."""),
    code("""train = pd.read_csv(ND / "report" / "report_training.csv")
train[["step", "subject", "unit_of_best", "best", "cap", "run", "gap_ratio", "value_kw",
       "final_minus_best_trial_pp"]].round(3)"""),
    md(f"""**What it shows.** The training diagnostics of every final fit, from the run report: the
best epoch (PatchTST: step) against its cap, the train/validation gap, and the final fit against
its best trial.

**What it means.** The BiLSTM indirect final fit scores {BIL['value_kw']:.2f} % against
{BIL['best_trial_value_kw']:.2f} % for its best 25-epoch trial
({BIL['final_minus_best_trial_pp']:+.2f} points): early stopping on the quantile loss does not
follow the kW objective there. This point and the others of §6 go to the G-1 list; nothing is
retuned on validation."""),
    md("## 5. What the models rely on (`explain`, D-31)"),
    code("""chk = pd.read_csv(XD / "explain" / "xgb" / "xgb_shap_checks.csv")
chk[["target", "horizon", "n_rows", "unit", "n_rounds", "rounds_in_booster",
     "additivity_relative", "contribs_max_diff", "unsliced_additivity_max",
     "dominance_first_over_third"]]"""),
    md(f"""**What it shows.** Exact TreeSHAP of every XGB booster on the
{int(CHK['n_rows'].iloc[0]):,} scored hours per horizon, with its checks.

**What it means.** The explanation adds up to the forecast within
{CHK['additivity_relative'].max():.1e} of the output scale, and XGBoost's own contributions agree
(largest difference {CHK['contribs_max_diff'].max():.1g}). The saved boosters hold 50 rounds more
than the forecast uses; explained whole they would be off by up to
{CHK['unsliced_additivity_max'].max():.0f} (k or kW), so every explanation slices the booster to
its forecast rounds first (X46). No feature dominates by the stage-7 measure (first over third
at most {CHK['dominance_first_over_third'].max():.1f}, threshold 10)."""),
    code("""feat = pd.read_csv(XD / "explain" / "xgb" / "xgb_shap_features.csv")
top = feat[(feat["seed"] == 42) & (feat["rank"] <= 3)]
top[["target", "horizon", "rank", "feature", "group", "mean_abs_kw", "mean_kw"]].round(1)"""),
    md(f"""**What it shows.** The three largest mean |SHAP| (kW) per target and horizon, and their
mean signed value.

**What it means.** A GFS forecast is the first feature everywhere: at D+1
`{TOP[('indirect', 1)]['feature']}` (indirect) and `{TOP[('direct', 1)]['feature']}` (direct). The
GFS group holds {GFS.min():.0f} to {GFS.max():.0f} % of the group attribution (the mean over
hours of |sum of SHAP over the group|, M27). SHAP describes the fitted booster on these hours; it
is not a causal effect of a feature on the plant."""),
    code("""att = pd.read_csv(ND / "explain" / "tft" / "tft_attention_summary.csv")
cols = ["step", "target", "encoder_positions", "encoder_mass_mean", "same_hour_ratio",
        "argmax_lag_h", "first_position_ratio"]
att[att["block"] == "all"][cols].round(2)"""),
    md(f"""**What it shows.** The TFT attention of the development models on validation: the share
of the encoder attention on positions a whole number of days before each step's valid hour,
against the share of such positions (ratio 1 = no preference), the strongest lag before
issuance, and the first position of the window against the same hour of the later days.

**What it means.** No model prefers the hour it forecasts: the ratio is
{A_ALL['same_hour_ratio'].min():.2f} to {A_ALL['same_hour_ratio'].max():.2f}
({int((A_ALL['same_hour_ratio'] < 1).sum())} of {len(A_ALL)} pairs below 1). The daily rhythm of
the attention is a preference for fixed clock hours of earlier days: in the direct route the
profile peaks at {PEAK['tft96']}:00 (TFT-96), {PEAK['tft168']}:00 (TFT-168) and
{PEAK['tftbase']}:00 (TFT baseline) local time, whatever hour is forecast. Every model attends
only inside its own encoder (C-55). Attention is not attribution; no agreement with SHAP is
claimed as validation (M29)."""),
    code("""var = pd.read_csv(ND / "explain" / "tft" / "tft_variable_importance.csv")
enc = var[var["part"] == "encoder"].drop_duplicates(["step", "target", "variable"])
enc.sort_values("variable_weight", ascending=False).groupby(["step", "target"]).head(2)[
    ["step", "target", "variable", "variable_weight"]].round(3)"""),
    md(f"""**What it shows.** The two largest variable-selection weights of each TFT encoder, a
measured series and its fill flag added (X48).

**What it means.** The largest is `{TOPVAR['variable']}` in {TOPVAR['step']} {TOPVAR['target']}
({TOPVAR['variable_weight']:.2f}); the leaders differ between models and routes. The weights
come from one seed and a short search; the final run gives the values of the report."""),
    code("""hp = pd.concat([pd.read_csv(XD / "explain" / "hparam_importance.csv"),
                pd.read_csv(ND / "explain" / "hparam_importance.csv")])
hp.dropna(subset=["importance_mean"]).groupby(["step", "param"])[
    ["importance_mean", "importance_std"]].mean().round(2)"""),
    md(f"""**What it shows.** fANOVA importance of the searched hyper-parameters, mean and std over
ten seeded evaluators, averaged per model (M30).

**What it means.** For XGB the most important parameter is `{HPXTOP.index[-1]}`
({HPXTOP.iloc[-1]:.2f} on average). {int(STUDIES['all_in_startup'].sum())} of {len(STUDIES)}
neural studies lie entirely inside TPE's random start-up ({TFT_TRIALS} TFT trials against 10), so
their importances describe a few random trials only."""),
    md("## 6. The run reports: warnings and the G-1 list"),
    code("""w = pd.concat([pd.read_csv(XD / "report" / "report_warnings.csv"),
               pd.read_csv(ND / "report" / "report_warnings.csv")])
w[["step", "kind", "status", "text"]]"""),
    md(f"""**What it shows.** The automatic warnings of §3.5 in the two development runs, and
whether the report reproduced them from the stored records (X49).

**What it means.** {len(W)} warnings, {int((W['status'] == 'matched').sum())} reproduced exactly;
{int((W['status'] == 'recomputed only').sum())} appears only in the recomputation: the W4 step
judged `hidden_continuous_size` by its sampled value, and the report by the value the network
used, at the edge of its effective range (X45). Every warning was read in the reviews of R9-06,
R9-17 and R9-21."""),
    code("""g1 = pd.concat([pd.read_csv(XD / "report" / "report_g1_points.csv"),
                pd.read_csv(ND / "report" / "report_g1_points.csv")])
g1.groupby("point").size().rename("facts").reset_index()"""),
    md(f"""**What it shows.** The facts of the two runs that inform the list of gate G-1, grouped by
point ({len(G1)} facts in all).

**What it means.** The report states facts and gives no verdict. The points stay a list until
G-1; no validation decision is changed because of them without the user (stage-9 plan §19.1)."""),
    md("## 7. The final run"),
    code("""from solarpv.pipeline.runner import scale_dev_to_final

rows = []
for run, step in [(XD, "xgb"), *((ND, s) for s in
                  ("bilstm", "tft96", "tft168", "tftbase", "patchtst"))]:
    folder = run / step
    rec = json.loads((folder / "step.json").read_text(encoding="utf-8"))
    ratio, upper = scale_dev_to_final(step, folder)
    rows.append({"step": step, "dev_h": rec["duration_s"] / 3600, "fits_ratio": ratio,
                 "final_h": rec["duration_s"] * ratio / 3600,
                 "upper_h": rec["duration_s"] * upper / 3600})
est = pd.DataFrame(rows)
print(f"final run: about {est['final_h'].sum():.1f} h, up to {est['upper_h'].sum():.1f} h")
est.round(2)"""),
    md("""**What it shows.** The duration of each model step of the final run, scaled from its
development run by the number of fits of the two budgets (X43), and its upper bound if every
neural fit ran to its epoch cap and every XGB trial took as long as the longest dev trial.

**What it means.** The total and the longest step decide whether the final run goes in one
sequence or in segments (`docs/runbook_final.md`, after the college IT answers Q-15). The run has
two phases (D-32): training, validation and the stage-10 acceptance first; the test split only
afterwards, by the step `test`, with every read logged.

**Reproduce.** The two run commands at the top of `scripts/make_models_notebook.py`, then
`python scripts/make_models_notebook.py`."""),
]

write_notebook(NB, cells, SOURCES)

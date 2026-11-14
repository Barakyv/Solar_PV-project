# ruff: noqa: E501  (report text with values cannot be wrapped; see make_quality_notebook)
"""Write the report sections of stage 11, part B, from the frozen final run (no training).

    python scripts/make_report_v7.py

Writes `docs/report_v7.md` (S-01, S-05, S-42 to S-44 and the correction note), `docs/traceability.csv`
(one row per number of the report: claim, value_old, value_new, file, script, verified_on),
`docs/old_vs_new.csv` and the table "Old vs new results" of `docs/errata.md`.

Every number is read from a file of `outputs/` or of `submission_evidence/` (read-only) and is
registered when it is formatted (`Facts`); a test (T-179) checks that no other number is in the text.
The test split is read only through the tables the run already wrote (`report/`, `test/`,
`outputs/11_analyses/`); the step `test` is not run and no model is trained (D-36, D-37).
"""

import csv
import json
import re

import pandas as pd

from solarpv.config.cli import no_options
from solarpv.config.paths import REPO_ROOT

no_options("python scripts/make_report_v7.py", __doc__)  # X62: --help runs nothing

SCRIPT = "scripts/make_report_v7.py"
VERIFIED_ON = "2026-11-14"
RUN = "outputs/runs/20261114T083930Z_final"
RUN_PATH = REPO_ROOT / RUN
A = REPO_ROOT / "outputs" / "11_analyses"
EV = REPO_ROOT / "submission_evidence"
M6 = ["xgb_sota", "lstm_sota", "tft_96", "tft_168", "tft", "patchtst"]
REFS = ["persistence", "climatology", "clearsky_k"]
NAME = {"xgb_sota": "XGB-Opt", "lstm_sota": "BiLSTM-Opt", "tft_96": "TFT-96", "tft_168": "TFT-168",
        "tft": "TFT baseline", "patchtst": "PatchTST", "climatology": "climatology",
        "persistence": "persistence", "clearsky_k": "clear sky x k"}
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def rd(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / rel)


def jd(rel: str):
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


class Facts:
    """Every number of the report passes through here and is written to docs/traceability.csv."""

    def __init__(self):
        self.rows: list[dict] = []
        self.tokens: set[str] = set()

    def _add(self, claim, shown, file, old=""):
        self.rows.append({"claim": claim, "value_old": old, "value_new": shown, "file": file,
                          "script": SCRIPT, "verified_on": VERIFIED_ON})
        self.tokens.update(t.rstrip(",.") for t in NUM.findall(shown))
        return shown

    def n(self, claim, value, file, fmt=".2f", old=""):
        return self._add(claim, format(value, fmt), file, old)

    def span(self, claim, lo, hi, file, fmt=".2f", old=""):
        return self._add(claim, f"{format(lo, fmt)} to {format(hi, fmt)}", file, old)

    def s(self, claim, text, file, old=""):
        return self._add(claim, text, file, old)


F = Facts()
V6 = "submission_evidence/13_report/Solar_PV_Forecasting_Report_CORRECTED_v6.docx"
PM = "PROJECT_MASTER.md"


def o(claim: str, text: str, file: str = V6) -> str:
    """A number quoted from the submitted report (or another fixed document): registered as an old value."""
    return F.s(claim, text, file, old=text)


# ── inputs ───────────────────────────────────────────────────────────────────
BT = rd(f"{RUN}/report/report_benchmark_test.csv")
BV = rd(f"{RUN}/report/report_benchmark_val.csv")
OLD = rd("submission_evidence/01_benchmark/benchmark_summary.csv")
SPLITS = rd("outputs/03_eda/tables/table_splits.csv").set_index("split")
DMT = rd(f"{RUN}/report/report_dm_kw_test.csv")
DMV = rd(f"{RUN}/report/report_dm_kw_val.csv")
RT_V = rd(f"{RUN}/evaluate/val/route_comparison.csv")
RT_T = rd(f"{RUN}/test/test/route_comparison.csv")
RT_OLD = rd("submission_evidence/03_route_comparison/route_comparison.csv")
HR = rd("outputs/11_analyses/horizon_robustness.csv")
DEC = rd("outputs/11_analyses/decomposition.csv")
DSD = rd("outputs/11_analyses/decomposition_seeds.csv")
QU = rd("outputs/11_analyses/quintiles.csv")
QC = rd("outputs/11_analyses/quintile_control.csv")
SEA = rd("outputs/11_analyses/seasonal_kw.csv")
ENC = rd("outputs/11_analyses/encoder_length.csv")
PTG = rd("outputs/11_analyses/patchtst_gap.csv")
LIT = rd("outputs/11_analyses/literature_alignment.csv")
SPREAD = rd(f"{RUN}/report/report_seed_spread.csv")
SHV = rd(f"{RUN}/explain/xgb/xgb_shap_groups.csv")
SHT = rd(f"{RUN}/test/shap_test/xgb_shap_groups_test.csv")
RANKV = rd(f"{RUN}/explain/xgb/xgb_shap_rank_stability.csv")
RANKT = rd(f"{RUN}/test/shap_test/xgb_shap_rank_stability_test.csv")
ATT = rd(f"{RUN}/explain/tft/tft_attention_summary.csv")
ABV = rd(f"{RUN}/report/report_ablation.csv")
ABT = rd(f"{RUN}/report/report_ablation_test.csv")
CVT = rd(f"{RUN}/report/report_cv_vs_test.csv")
G1 = rd(f"{RUN}/report/report_g1_points.csv")
STUDIES = rd(f"{RUN}/report/report_studies.csv")
STEPS = rd(f"{RUN}/report/report_steps.csv").set_index("step")
SHAP_OLD = rd("submission_evidence/07_shap/shap_group_importance.csv")
SEA_OLD = rd("submission_evidence/04_seasonal/seasonal_metrics.csv")
SEED_OLD = rd("submission_evidence/05_seed_variability/seed_variability_summary.csv")
DM_OLD = rd("submission_evidence/02_statistical_tests/dm_tests.csv")
MDE_OLD = rd("submission_evidence/02_statistical_tests/detectable_effect_summary.csv")
ABL_OLD = rd("submission_evidence/05_seed_variability/ablation_seed_variability.csv")
CLAIMS = rd("submission_evidence/11_audit/REPORT_CLAIM_EVIDENCE_MATRIX.csv")
SEL = jd("outputs/07_feature_selection/selection_manifest.json")
MI = jd(f"{RUN}/select/sets/model_inputs.json")
FEAT = jd("outputs/06_features/features_manifest.json")
CLEAN = jd("outputs/05_cleaning/cleaning_manifest.json")
GFS = jd("outputs/06b_gfs/gfs_manifest.json")
CFG = __import__("yaml").safe_load((RUN_PATH / "config.yaml").read_text(encoding="utf-8"))
DER = __import__("yaml").safe_load((REPO_ROOT / "configs/derived_constants.yaml").read_text(encoding="utf-8"))
ACCESS = pd.read_csv(REPO_ROOT / "outputs/07_evaluation/test_access_log.csv")

# seed-mean benchmark, kW and target unit
def seedmean(df: pd.DataFrame, col: str = "nRMSE_kW_pct") -> pd.Series:
    return df.groupby(["model", "target", "horizon"])[col].mean()


KT, KV = seedmean(BT), seedmean(BV)
UT, UV = seedmean(BT, "nRMSE_pct"), seedmean(BV, "nRMSE_pct")
OLDK = OLD.set_index(["model", "target", "horizon"])

# claims asserted before the text is written (the notes are formatted from them, never typed)
D = DEC[DEC["model"].isin(M6)]
ME_ALL = D[(D["view"] == "all") & (D["split"] == "test")].groupby("model")["me_kw"].mean()
ME_NORM = D[(D["view"] == "normal_operation") & (D["split"] == "test")].groupby("model")["me_kw"].mean()
NEURAL5 = [m for m in M6 if m not in ("xgb_sota", "patchtst")]
NO_PT = [m for m in M6 if m != "patchtst"]
assert (ME_ALL > 0).all() and ME_NORM[NO_PT].abs().max() < 15


def route_pairs(rt: pd.DataFrame) -> pd.DataFrame:
    """direct minus indirect nRMSE in kW per model and horizon (seed mean); positive = indirect better."""
    r = rt[rt["model"].isin(M6)].groupby(["model", "horizon", "route"])["nRMSE_pct"].mean().unstack("route")
    return (r["direct"] - r["indirect_to_power"]).rename("direct_minus_indirect_pp").reset_index()


RPV, RPT = route_pairs(RT_V), route_pairs(RT_T)
DIRECT_BETTER_V, DIRECT_BETTER_T = int((RPV.iloc[:, 2] < 0).sum()), int((RPT.iloc[:, 2] < 0).sum())
assert (DIRECT_BETTER_V, DIRECT_BETTER_T) == (16, 11)  # S11-06
SIGN_DIFF = int(((RPV.iloc[:, 2] < 0) != (RPT.iloc[:, 2] < 0)).sum())
assert SIGN_DIFF == 7
CLIM_T = DMT[(DMT["model_a"] == "climatology") & (DMT["model_b"] == "xgb_sota")]
CLIM_V = DMV[(DMV["model_a"] == "climatology") & (DMV["model_b"] == "xgb_sota")]
assert int(CLIM_T["significant"].sum()) == 1 and len(CLIM_T) == 6 and CLIM_V["significant"].all()
PT_X, PT_C = PTG[PTG["model_b"] == "xgb_sota"], PTG[PTG["model_b"] == "climatology"]
assert PT_X["significant"].all() and (PT_C[PT_C["significant"]]["diff_a_minus_b_pp"] > 0).all()
assert not ENC[ENC["split"] == "val"]["significant"].any()


def cells_lowest(k: pd.Series) -> int:
    """cells (target x horizon) in which XGB-Opt has the lowest seed-mean nRMSE of the six models."""
    return sum(k.xs((t, h), level=("target", "horizon"))[M6].idxmin() == "xgb_sota"
               for t in ("indirect", "direct") for h in (1, 2, 3))


LOW_T, LOW_V = cells_lowest(KT), cells_lowest(KV)


def share_range(df: pd.DataFrame, group: str) -> tuple[float, float]:
    g = df[df["group"] == group].groupby(["target", "horizon"])["share_pct"].mean()
    return float(g.min()), float(g.max())


def sig_neural(dm: pd.DataFrame, m: str) -> int:
    d = dm[(dm["model_a"] == m) & (dm["model_b"] == "xgb_sota")]
    assert len(d) == 6 and (d[d["significant"]]["mean_diff"] > 0).all()  # XGB better where significant
    return int(d["significant"].sum())


for _dm in (DMT, DMV):  # XGB-Opt against clear sky x k and persistence: significant in all six cells
    for _r in ("clearsky_k", "persistence"):
        _d = _dm[(_dm["model_a"] == _r) & (_dm["model_b"] == "xgb_sota")]
        assert len(_d) == 6 and _d["significant"].all() and (_d["mean_diff"] > 0).all()
for _sh in (SHV, SHT):  # the weather-forecast group is the largest in every cell
    _g = _sh.groupby(["target", "horizon", "group"])["share_pct"].mean().unstack("group")
    assert (_g.idxmax(axis=1) == "nwp_gfs").all()
N6 = F.n("cells per comparison (target x horizon)", len(CLIM_T), f"{RUN}/report/report_dm_kw_test.csv", "d")


# ── text ─────────────────────────────────────────────────────────────────────
CAP = SEL["capacity_kw"]
ts, vs, ss = SPLITS.loc["train"], SPLITS.loc["val"], SPLITS.loc["test"]
al = SPLITS.loc["all"]
test_xgb_i1 = KT[("xgb_sota", "indirect", 1)]
POLLED = int(ss["daytime_polled"]) + int(ss["daytime_mixed"])
hours_gpu = sum(STEPS.loc[s, "duration_s"] for s in ("bilstm", "tft96", "tft168", "tftbase", "patchtst")) / 3600
test_rows_scored = int(ACCESS["command"].str.contains("step test: score the test split", regex=False).sum())
assert test_rows_scored == 1  # the test split was scored once (D-32)
RHO_V = float(RANKV.groupby(["target", "compare"])["spearman"].mean().min())  # as S11-07: mean over seeds
RHO_T = float(RANKT.groupby(["target", "compare"])["spearman"].mean().min())

SP = "docs/report_v7.md"
PT_ = "outputs/runs/20261114T083930Z_final"
sec = []
w = sec.append

w("# Report v7: the sections written in stage 11, part B\n")
w("""Scope of this file: the plant description (S-01), the milestones (S-05), the discussion (S-42), the
conclusions (S-43) and the disclosures (S-44) of the submitted report, rewritten from the frozen final run
`20261114T083930Z_final` (tag `g1-frozen`). The tables and figures of chapters 7 to 12 are in
`outputs/11_analyses/`, the report tables of the run and the figure sheets (S09 to S11). This file is
generated by `scripts/make_report_v7.py`; every number below is registered in `docs/traceability.csv`
with its source file and script, and the old values are in the table "Old vs new results" of
`docs/errata.md`.

Rules of reading. Every number of the test split is placed beside the validation number of the same view.
No model is declared the winner (D-37): four of the five neural models have one seed. The validation numbers
are tuning-split numbers (D-10). All errors are nRMSE in kW, in percent of the plant capacity, over the daytime
hours of the split with a measured output, the same hours for every model.
""")

w("## Reproduction and correction note (D-02)\n")
w(f"""The submitted report (v6) was reconstructed from its evidence folder and rebuilt, not reproduced one to one
(D-01). The rebuild corrects errors that changed the numbers: local time had been read as UTC (F-01), the
indirect target was clipped and its mean was a mean of ratios (F-06), rolling features saw unknown days at
D+2 and D+3 (F-07), the weather inputs were not forecasts available at issuance (F-28), and the feature
selection read the validation and test periods (F-26). Every correction has a row in `docs/errata.md` and a
test. The old numbers are therefore points of comparison and not targets. The scored hours changed from
{F.n('old test daytime hours', OLDK.loc[('xgb_sota','indirect',1),'n'], 'submission_evidence/01_benchmark/benchmark_summary.csv', ',d')} in v6 to
{F.n('test daytime hours with a measured output', ss['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')} in the rebuild, the same hours for every
model, and the validation period was added as a second view (it had been used for tuning).
""")

# S-01
w("## 3.1 Plant description (S-01)\n")
w(f"""The plant that the data describe is a section of two inverters (inv1 and inv2) of a utility-scale park
of {o('park size, MW', '40', PM)} MW in the Southern Arava, Negev, Israel; the park has {o('park inverters', '25', PM)} inverters in all (F-12, C-30). The submitted
text described the two inverters as the whole plant, which is wrong. The target is the AC power of these two
inverters. That they form the whole metered section cannot be verified from the data (A-07), and the rating
of {F.n('AC capacity, kW', CAP, 'outputs/07_feature_selection/selection_manifest.json', ',.0f')} kW used to normalise every error is the sum of two ratings that is not verified against the
inverter model (A-08); if the rating were different, every nRMSE in percent of capacity would scale by the
same factor and no ranking, test or skill score would change. The climate is hyper-arid. Site coordinates
are not published in this document (T-45; their disclosure in the final report is Q-09).

| Parameter | Value |
|---|---|
| Location | Southern Arava, Negev, Israel (coordinates not published) |
| Section modelled | two inverters of the {o('park inverters (table)', '25', PM)} of the park |
| AC capacity of the section | {F.n('AC capacity, kW (table)', CAP, 'outputs/07_feature_selection/selection_manifest.json', ',.0f')} kW |
| Panel plane | tilt {F.n('panel tilt, deg', DER['panel_plane']['tilt'], 'configs/derived_constants.yaml', '.0f')}, azimuth {F.n('panel azimuth, deg', DER['panel_plane']['azimuth'], 'configs/derived_constants.yaml', '.0f')} degrees (measured on the corrected data, F-23) |
| Resolution | hourly, from minute data ({F.n('one-minute rows', CLEAN['rows_1min'], 'outputs/05_cleaning/cleaning_manifest.json', ',d')} rows) |
| Hourly rows | {F.n('hourly rows', CLEAN['rows_1H'], 'outputs/05_cleaning/cleaning_manifest.json', ',d')} |
| Train (hours, daytime hours with a measured output) | {F.n('train hours', ts['hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')}, {F.n('train daytime valid', ts['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')} |
| Validation (hours, daytime hours with a measured output) | {F.n('val hours', vs['hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')}, {F.n('val daytime valid', vs['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')} |
| Test (hours, daytime hours with a measured output) | {F.n('test hours', ss['hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')}, {F.n('test daytime valid (table)', ss['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')} |
| Operation-limited daytime hours (train, validation, test) | {F.n('train op-limited', ts['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')}, {F.n('val op-limited', vs['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')}, {F.n('test op-limited', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} (curtailment, stops, derate; D-22) |
| Test hours of the second data source (polled snapshots) | {F.n('test hours polled or mixed', POLLED, 'outputs/03_eda/tables/table_splits.csv', ',d')} (F-18) |

The data period is 2021-09-01 to 2025-05-31 in local dates (the split table, in UTC, starts on {ts['start_utc'][:10]} and ends on {ss['end_utc'][:10]}). The daytime hour is
defined by the sun elevation at the middle of the hour (F-02).
""")

# S-05
n_sel = {k: v["n_features"] for k, v in MI.items()}
w("## 3.5 End-to-end milestones (S-05)\n")
w(f"""The comparison of the six models rests on earlier steps; each step has an artifact in the repository and
controls a risk that the submitted report had. The numbers are read from the manifests of the steps.

| Step | Artifact | Risk controlled | Key number |
|---|---|---|---|
| Problem framing (D-16) | issuance at local midnight; D+1 to D+3; two targets | ambiguous forecast origin | three horizons, two routes |
| Ingestion (stage 2) | one-minute UTC series | local time read as UTC (F-01) | {F.n('rows 1min (milestones)', CLEAN['rows_1min'], 'outputs/05_cleaning/cleaning_manifest.json', ',d')} rows |
| Logger clocks (stage 4) | clock offset per source and day | misalignment with solar geometry (F-15) | {F.n('hourly rows (milestones)', CLEAN['rows_1H'], 'outputs/05_cleaning/cleaning_manifest.json', ',d')} hourly rows |
| Cleaning (stage 5) | explicit policy, flags not erasures | spikes, frozen values, curtailment (F-14, F-17) | {F.n('op-limited daytime hours all', al['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of {F.n('daytime hours all', al['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} daytime hours flagged |
| Target and features (stage 6) | exact-inverse clear-sky index, horizon-aware lags | clipping, leaking rolling windows (F-06, F-07) | mean daytime k {F.n('MEAN_DAYTIME_K', DER['mean_daytime_k'], 'configs/derived_constants.yaml', '.6f', old='0.6279')} |
| Weather forecasts (stage 6b) | GFS runs public before issuance | reanalysis inputs not known at issuance (F-28) | {F.n('GFS runs', GFS['runs'] if isinstance(GFS['runs'], int) else len(GFS['runs']), 'outputs/06b_gfs/gfs_manifest.json', ',d')} runs |
| Feature selection (stage 7) | allow-list, train only, one validation look | selection on all periods (F-26) | {F.span('indirect inputs per horizon', min(n_sel[k] for k in n_sel if k.startswith('indirect')), max(n_sel[k] for k in n_sel if k.startswith('indirect')), PT_ + '/select/sets/model_inputs.json', 'd')} indirect, {F.span('direct inputs per horizon', min(n_sel[k] for k in n_sel if k.startswith('direct')), max(n_sel[k] for k in n_sel if k.startswith('direct')), PT_ + '/select/sets/model_inputs.json', 'd')} direct inputs |
| Evaluation layer (stage 8) | six models, three references, paired tests | unequal rows, unpaired claims | HLN and Bonferroni, minimum detectable effect |
| Gate G-1 (stage 9) | audit, reproduction from raw data, external review, tag `g1-frozen` | changes after the test is seen | every review note answered (`docs/g1_review_log.md`) |
| Final run (stage 10) | budgets of v6, three seeds for XGB and BiLSTM | unequal tuning | {F.n('GPU hours', hours_gpu, PT_ + '/report/report_steps.csv', '.1f')} GPU hours |
| Test (stage 10, phase 2) | step `test`, run once | repeated looks at the test split | scored {F.n('test scoring events', test_rows_scored, 'outputs/07_evaluation/test_access_log.csv', 'd')} time |
""")

# S-42
w("## 13. Discussion (S-42)\n")
w("### 13.1 The value of feature engineering (replaces the submitted \"paradox\")\n")
gap_x = PT_X["diff_a_minus_b_pp"]
w(f"""The submitted report read PatchTST, which sees only the past of the target, as a model that "nearly
matches" XGB-Opt, and turned the gap of {o('v6 PatchTST gap, points', '1.2')} points into an upper bound on the value of feature engineering
(M49). In the rebuild PatchTST is worse than XGB-Opt by {F.span('PatchTST minus XGB, pp', gap_x.min(), gap_x.max(), 'outputs/11_analyses/patchtst_gap.csv')} pp, significantly in
{F.n('PatchTST vs XGB significant cells', int(PT_X['significant'].sum()), 'outputs/11_analyses/patchtst_gap.csv', 'd')} of {F.n('PatchTST vs XGB cells', len(PT_X), 'outputs/11_analyses/patchtst_gap.csv', 'd')} cells (validation and test, two routes, three horizons). Against climatology it is worse by
{F.span('PatchTST minus climatology, pp', PT_C['diff_a_minus_b_pp'].min(), PT_C['diff_a_minus_b_pp'].max(), 'outputs/11_analyses/patchtst_gap.csv')} pp, significantly in {F.n('PatchTST vs climatology significant cells', int(PT_C['significant'].sum()), 'outputs/11_analyses/patchtst_gap.csv', 'd')} of {F.n('PatchTST vs climatology cells', len(PT_C), 'outputs/11_analyses/patchtst_gap.csv', 'd')}, always in the worse direction. No bound is stated: the XGB
has weather-forecast inputs that PatchTST does not, and the comparison has one seed for PatchTST.

What the inputs are worth is measured by the ablation of XGB-Opt (S-41, three seeds, refitted without a
group). Removing the weather-forecast (GFS) group raises the error by
{F.span('ablation no_nwp_gfs validation, pp', ABV[ABV['mode']=='no_nwp_gfs']['mean_delta_pp'].min(), ABV[ABV['mode']=='no_nwp_gfs']['mean_delta_pp'].max(), PT_ + '/report/report_ablation.csv')} pp on validation and by
{F.span('ablation no_nwp_gfs test, pp', ABT[ABT['mode']=='no_nwp_gfs']['mean_delta_pp'].min(), ABT[ABT['mode']=='no_nwp_gfs']['mean_delta_pp'].max(), PT_ + '/report/report_ablation_test.csv')} pp on test. The history groups change it by little: the largest change of any other group
on validation is {F.n('ablation other groups max validation, pp', ABV[~ABV['mode'].isin(['minimal','no_nwp_gfs'])]['mean_delta_pp'].abs().max(), PT_ + '/report/report_ablation.csv')} pp, close to the seed margin of the rule (M41). The
weather-forecast group holds {F.span('GFS share of mean absolute SHAP, validation, %', *share_range(SHV, 'nwp_gfs'), PT_ + '/explain/xgb/xgb_shap_groups.csv', '.0f')} percent of the mean absolute SHAP
value on validation and {F.span('GFS share of mean absolute SHAP, test, %', *share_range(SHT, 'nwp_gfs'), PT_ + '/test/shap_test/xgb_shap_groups_test.csv', '.0f')} percent on test, and it is the largest group in every cell on both splits. The value of the engineering
therefore sits mainly in the issued weather forecast. This agrees with the submitted statement that real
forecasts "would likely re-establish a larger gap", but the comparison is not a test of that statement: PatchTST
has one seed and a different design.
""")

w("### 13.2 The route decision\n")
rp_all = pd.concat([RPV.iloc[:, 2], RPT.iloc[:, 2]])
w(f"""The submitted report called the indirect route "the most operationally actionable finding" because it
beat the direct route by {o('v6 route advantage, points', '1.66 to 2.10')} points (v6, chapter 13.2). That comparison used a truth that was
rebuilt from the clipped index for one route (C-16). In the rebuild both routes are scored in kW against the
actual AC on the same hours. The direct route has the lower error in {F.n('direct better pairs, validation', DIRECT_BETTER_V, 'evaluate/val/route_comparison.csv; test/test/route_comparison.csv', 'd')} of {F.n('model-horizon pairs', len(RPV), 'evaluate/val/route_comparison.csv', 'd')} pairs of a model and a horizon on validation and in {F.n('direct better pairs, test', DIRECT_BETTER_T, PT_ + '/test/test/route_comparison.csv', 'd')} of {F.n('model-horizon pairs (test)', len(RPT), PT_ + '/test/test/route_comparison.csv', 'd')} on test. The
differences range from {F.n('route difference min, pp', rp_all.min(), PT_ + '/test/test/route_comparison.csv', '+.2f')} to {F.n('route difference max, pp', rp_all.max(), PT_ + '/test/test/route_comparison.csv', '+.2f')} points (direct minus indirect), and the sign differs
between the two periods for {F.n('pairs with a changed sign', SIGN_DIFF, PT_ + '/test/test/route_comparison.csv', 'd')} of the {F.n('pairs (sign)', len(RPV), PT_ + '/test/test/route_comparison.csv', 'd')} pairs. The evidence for either route is within the seed and period
uncertainty; the submitted recommendation to switch to the indirect route is withdrawn. The mechanism offered
in v6 (the direct model must re-learn the solar geometry) was not tested and is not repeated.
""")

w("### 13.3 Architecture insights\n")
a_all = ATT[ATT["block"] == "all"]
dm_enc = ENC[ENC["split"] == "test"]
w(f"""The three TFT models, the BiLSTM and XGB-Opt are close. On test, XGB-Opt has the lowest seed-mean error of the
six models in {F.n('cells where XGB lowest, test', LOW_T, PT_ + '/report/report_benchmark_test.csv', 'd')} of {N6} cells and on validation in {F.n('cells where XGB lowest, validation', LOW_V, PT_ + '/report/report_benchmark_val.csv', 'd')} of {N6}. Paired in kW (HLN, Bonferroni), XGB-Opt is
significantly better than the BiLSTM in {F.n('BiLSTM significant cells, test', sig_neural(DMT, 'lstm_sota'), PT_ + '/report/report_dm_kw_test.csv', 'd')} of {N6} cells on test and than the three TFT models in
{F.n('TFT-96 significant cells, test', sig_neural(DMT, 'tft_96'), PT_ + '/report/report_dm_kw_test.csv', 'd')}, {F.n('TFT-168 significant cells, test', sig_neural(DMT, 'tft_168'), PT_ + '/report/report_dm_kw_test.csv', 'd')} and {F.n('TFT baseline significant cells, test', sig_neural(DMT, 'tft'), PT_ + '/report/report_dm_kw_test.csv', 'd')} of {N6} cells (TFT-96, TFT-168 and the TFT baseline). The models are not
separated; the nominal order is not a result.

Encoder length. The submitted report found a "consistent advantage" of the 96-hour encoder from gaps of
about one point (M48). With a paired test per cell, the TFT-168 minus TFT-96 difference lies between
{F.n('encoder diff validation min, pp', ENC[ENC['split']=='val']['diff_a_minus_b_pp'].min(), 'outputs/11_analyses/encoder_length.csv', '+.2f')} and {F.n('encoder diff validation max, pp', ENC[ENC['split']=='val']['diff_a_minus_b_pp'].max(), 'outputs/11_analyses/encoder_length.csv', '+.2f')} pp on validation, with no significant cell, and between
{F.n('encoder diff test min, pp', dm_enc['diff_a_minus_b_pp'].min(), 'outputs/11_analyses/encoder_length.csv', '+.2f')} and {F.n('encoder diff test max, pp', dm_enc['diff_a_minus_b_pp'].max(), 'outputs/11_analyses/encoder_length.csv', '+.2f')} pp on test, with {F.n('encoder significant cells', int(ENC['significant'].sum()), 'outputs/11_analyses/encoder_length.csv', 'd')} of {F.n('encoder cells', len(ENC), 'outputs/11_analyses/encoder_length.csv', 'd')} cells significant.
The sign depends on the route. Each model was tuned separately with one seed, so no preference is stated.

Attention. The attention of a TFT is a description and not evidence of use. In the six cases of target and
model, the weight on the clock hour of the forecast hour is {F.span('same-hour attention ratio', a_all['same_hour_ratio'].min(), a_all['same_hour_ratio'].max(), PT_ + '/explain/tft/tft_attention_summary.csv')} times a uniform attention, and the
weight on the oldest encoder position is {F.span('first-position attention ratio', a_all['first_position_ratio'].min(), a_all['first_position_ratio'].max(), PT_ + '/explain/tft/tft_attention_summary.csv')} times uniform. The submitted
statement that the shorter window concentrates on noise steps, and that SHAP and attention agree on lags
at {o('v6 lags', '-24, -48 and -168')} hours, is not supported: the SHAP groups are dominated by the issued forecast and the
attention follows the daily cycle.

The submitted account of the timestamp error of the BiLSTM (a difference of about {o('v6 BiLSTM alignment, points', '13')} points) is a property of
v6 and is not tested again; the rebuild scores every model on the same hours with one definition (C-35).
""")

w("### 13.4 Statistical power\n")
cm = CLIM_T["pct_of_mde"]
w(f"""The observed difference between two models is compared with the minimum detectable effect of the paired test
({F.n('statistical power of the MDE, %', 100 * 0.8, 'src/solarpv/evaluation/stats_tests.py', '.0f')} percent power). XGB-Opt is significantly better than clear sky x k and persistence in all
{N6} cells on test and validation. Against climatology it is significant in {F.n('climatology significant cells, test', int(CLIM_T['significant'].sum()), PT_ + '/report/report_dm_kw_test.csv', 'd')} of {F.n('climatology cells', len(CLIM_T), PT_ + '/report/report_dm_kw_test.csv', 'd')} cells on test and in
{F.n('climatology significant cells, validation', int(CLIM_V['significant'].sum()), PT_ + '/report/report_dm_kw_val.csv', 'd')} of {F.n('climatology cells (validation)', len(CLIM_V), PT_ + '/report/report_dm_kw_val.csv', 'd')} on validation. On test the observed effect is {F.span('climatology effect, percent of MDE', cm.min(), cm.max(), PT_ + '/report/report_dm_kw_test.csv', '.0f')} percent of the minimum
detectable effect: this is a limitation of the data (one section of a plant, {F.n('test hours (power)', ss['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')} hours, autocorrelated
errors), and it means that the sentence "XGB-Opt is better than climatology" is not made. The submitted
recommendations stand: report intervals, report the effect as a share of the minimum detectable effect, and
pool sites. In v6, {F.n('old underpowered pairs', int((MDE_OLD['conclusion']=='UNDERPOWERED').sum()), 'submission_evidence/02_statistical_tests/detectable_effect_summary.csv', 'd')} of {F.n('old pairs with an MDE', len(MDE_OLD), 'submission_evidence/02_statistical_tests/detectable_effect_summary.csv', 'd')} pairs were underpowered and the significance counted in v6 ({F.n('old significant pairs', int(DM_OLD['significant_5pct'].sum()), 'submission_evidence/02_statistical_tests/dm_tests.csv', 'd')} of {F.n('old DM pairs', len(DM_OLD), 'submission_evidence/02_statistical_tests/dm_tests.csv', 'd')}) had no HLN correction and no Bonferroni correction (F-04).
""")

w("### 13.5 Seasonal generalisability\n")
sx = SEA[(SEA["model"] == "xgb_sota") & (SEA["target"] == "indirect") & (SEA["horizon"] == 1)].set_index(["season", "split"])
cx = SEA[(SEA["model"] == "climatology") & (SEA["target"] == "indirect") & (SEA["horizon"] == 1)].set_index(["season", "split"])
lim_fall_t = sx.loc[("Fall", "test"), "op_limited_share"]
lim_fall_v = sx.loc[("Fall", "val"), "op_limited_share"]
w(f"""The test period has no summer, and a season is not the same weather in the two periods: the autumn of the
test split is {sx.loc[('Fall','test'),'months']} and the autumn of the validation split is {sx.loc[('Fall','val'),'months']}. The submitted estimate that a summer would lower all errors "by
{o('v6 summer estimate, points', '5 to 8')} points" is not supported by any computation and is withdrawn. For XGB-Opt (indirect, D+1) the error in
the autumn is {F.n('XGB Fall validation, %', sx.loc[('Fall','val'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} percent on validation and {F.n('XGB Fall test, %', sx.loc[('Fall','test'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} percent on test; for climatology it is
{F.n('climatology Fall validation, %', cx.loc[('Fall','val'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} and {F.n('climatology Fall test, %', cx.loc[('Fall','test'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} percent. The rise of the test period is concentrated in the autumn, where
{F.n('Fall op-limited share test, %', 100*lim_fall_t, 'outputs/11_analyses/seasonal_kw.csv', '.0f')} percent of the test hours are operation-limited against {F.n('Fall op-limited share validation, %', 100*lim_fall_v, 'outputs/11_analyses/seasonal_kw.csv', '.0f')} percent of the validation hours. In spring the
two periods are close ({F.n('XGB Spring validation, %', sx.loc[('Spring','val'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} and {F.n('XGB Spring test, %', sx.loc[('Spring','test'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} percent). The error of the references rises by a similar amount in the
autumn, so the rise is a property of the period and of the operation-limited hours, not of the models.
""")

w("### 13.6 Modelling coordinates\n")
w("""The submitted report used anonymised coordinates and disclosed a sensitivity analysis. The rebuild uses the
coordinates of the solar field itself in all clear-sky and solar-position calculations (D-05); they are kept in a
private configuration and are not published (T-45). The old sensitivity analysis compared two other points and
does not apply; the mask of the daytime hours is the same under either location (F-02). Whether the final report
discloses the coordinates is the decision Q-09.
""")

# S-43
tx = HR[(HR["model"] == "xgb_sota")].set_index(["target", "split"])["D3_minus_D1_pp"]
hz = HR[HR["model"].isin(M6)]
w("## 14. Conclusions (S-43)\n")
w(f"""This study compares six models at three horizons on one section of a plant, from an end-to-end pipeline
in which the errors of the submitted project were corrected. The conclusions of the submitted report are
restated below, each with what the rebuild found.

C1. Ranking. No model is declared the best (D-37). On test, the seed-mean nRMSE in kW at D+1 of the indirect
route is {F.n('XGB test indirect D1', KT[('xgb_sota','indirect',1)], PT_ + '/report/report_benchmark_test.csv', '.2f', old='29.09')} for XGB-Opt, {F.n('BiLSTM test indirect D1', KT[('lstm_sota','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} for BiLSTM-Opt, {F.n('TFT-96 test indirect D1', KT[('tft_96','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} for TFT-96, {F.n('TFT-168 test indirect D1', KT[('tft_168','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} for TFT-168,
{F.n('TFT baseline test indirect D1', KT[('tft','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} for the TFT baseline and {F.n('PatchTST test indirect D1', KT[('patchtst','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} for PatchTST; on validation the same numbers are {F.n('XGB val indirect D1', KV[('xgb_sota','indirect',1)], PT_ + '/report/report_benchmark_val.csv')},
{F.n('BiLSTM val indirect D1', KV[('lstm_sota','indirect',1)], PT_ + '/report/report_benchmark_val.csv')}, {F.n('TFT-96 val indirect D1', KV[('tft_96','indirect',1)], PT_ + '/report/report_benchmark_val.csv')}, {F.n('TFT-168 val indirect D1', KV[('tft_168','indirect',1)], PT_ + '/report/report_benchmark_val.csv')}, {F.n('TFT baseline val indirect D1', KV[('tft','indirect',1)], PT_ + '/report/report_benchmark_val.csv')} and {F.n('PatchTST val indirect D1', KV[('patchtst','indirect',1)], PT_ + '/report/report_benchmark_val.csv')}. The references
are {F.n('climatology test indirect D1', KT[('climatology','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} (climatology) and {F.n('persistence test indirect D1', KT[('persistence','indirect',1)], PT_ + '/report/report_benchmark_test.csv')} (persistence) on test, and {F.n('climatology val indirect D1', KV[('climatology','indirect',1)], PT_ + '/report/report_benchmark_val.csv')} and
{F.n('persistence val indirect D1', KV[('persistence','indirect',1)], PT_ + '/report/report_benchmark_val.csv')} on validation. Every model and reference is about {F.n('mean rise val to test, pp', float(KT.loc[M6 + REFS].mean() - KV.loc[M6 + REFS].mean()), PT_ + '/report/report_benchmark_test.csv', '.1f')} points worse on test than on validation (mean over the six models and the three references, both routes and all horizons), and the rise is shared with the references. The values
of the submitted ranking ({F.n('old XGB indirect D1 (C1)', OLDK.loc[('xgb_sota','indirect',1),'nRMSE_pct'], 'submission_evidence/01_benchmark/benchmark_summary.csv', '.2f')} for XGB-Opt and {F.n('old TFT baseline indirect D1 (C1)', OLDK.loc[('tft','indirect',1),'nRMSE_pct'], 'submission_evidence/01_benchmark/benchmark_summary.csv', '.2f')} for the TFT baseline in target units) are not comparable
with these (F-06, F-01) and are shown in the table of old and new results.

C2. Route. The submitted dominance of the indirect route is not reproduced (13.2): the direct route is nominally
better in {F.n('direct better validation (C2)', DIRECT_BETTER_V, 'evaluate/val/route_comparison.csv', 'd')} of {F.n('pairs (C2)', len(RPV), 'evaluate/val/route_comparison.csv', 'd')} pairs on validation and in {F.n('direct better test (C2)', DIRECT_BETTER_T, PT_ + '/test/test/route_comparison.csv', 'd')} of {F.n('pairs (C2 test)', len(RPT), PT_ + '/test/test/route_comparison.csv', 'd')} on test, with a sign that changes between the periods.

C3. Horizon. The submitted claim of a flat horizon profile for XGB-Opt (no degradation from D+1 to D+3) and a
degradation of PatchTST is replaced: the sign of the change from D+1 to D+3 reverses between the periods. For
the six models and two routes it is negative in {F.n('pairs with D3<D1, validation', int((hz[hz['split']=='val']['D3_minus_D1_pp']<0).sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')} of {F.n('pairs for horizons', int((hz['split']=='val').sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')} pairs on validation and positive in {F.n('pairs with D3>D1, test', int((hz[hz['split']=='test']['D3_minus_D1_pp']>0).sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')} of {F.n('pairs for horizons (test)', int((hz['split']=='test').sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')} on test; its size is at most
{F.n('largest D3 minus D1, pp', hz['D3_minus_D1_pp'].abs().max(), 'outputs/11_analyses/horizon_robustness.csv')} points. The order of the horizons is a property of the period (R6b-07, R7-05), not of a model.

C4. Autoregression. PatchTST is worse than XGB-Opt in every cell and than climatology in most (13.1). The
temporal structure of the series carries information, and climatology already holds most of it.

C5. Encoder length. A consistent advantage of the 96-hour encoder is not found (13.3).

C6. Interpretability. The weather-forecast group dominates the XGB attributions, and the rank correlation of the
feature importances between two horizons or two seeds is at least
{F.n('rank correlation minimum, validation', RHO_V, PT_ + '/explain/xgb/xgb_shap_rank_stability.csv')} on validation and {F.n('rank correlation minimum, test', RHO_T, PT_ + '/test/shap_test/xgb_shap_rank_stability_test.csv')} on test (mean over seeds, as in S11-07). The TFT attention is a daily pattern. The submitted
convergence of the two methods on three lags is not reproduced, and no agreement between the methods is claimed.

C7. Statistical power. Differences between the top models are below the minimum detectable effect; XGB-Opt
separates from climatology in one cell of six on test (13.4).

C8. Production regimes. The profile of the submitted section 12.4 (error highest in the second quintile, a
negative bias of the highest) came from binning the hours by the outcome, with edges taken from the test sample
(M45). On test, XGB-Opt (direct, D+1, seed 42) has a mean error of {F.n('quintile 1 observed, kW', QU[(QU['model']=='xgb_sota')&(QU['seed']==42)&(QU['target']=='direct')&(QU['horizon']==1)&(QU['split']=='test')&(QU['binned_by']=='observed')&(QU['bin']==1)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')} kW in the lowest bin of the
observed power and {F.n('quintile 5 observed, kW', QU[(QU['model']=='xgb_sota')&(QU['seed']==42)&(QU['target']=='direct')&(QU['horizon']==1)&(QU['split']=='test')&(QU['binned_by']=='observed')&(QU['bin']==5)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')} kW in the highest; binned by the forecast it is
{F.n('quintile 1 forecast, kW', QU[(QU['model']=='xgb_sota')&(QU['seed']==42)&(QU['target']=='direct')&(QU['horizon']==1)&(QU['split']=='test')&(QU['binned_by']=='forecast')&(QU['bin']==1)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')} and {F.n('quintile 5 forecast, kW', QU[(QU['model']=='xgb_sota')&(QU['seed']==42)&(QU['target']=='direct')&(QU['horizon']==1)&(QU['split']=='test')&(QU['binned_by']=='forecast')&(QU['bin']==5)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')} kW. A control forecast made unbiased on validation gives
{F.n('control quintile 1, kW', QC[(QC['target']=='direct')&(QC['horizon']==1)&(QC['split']=='test')&(QC['bin']==1)]['me_calibrated_kw'].iloc[0], 'outputs/11_analyses/quintile_control.csv', '+.0f')} and {F.n('control quintile 5, kW', QC[(QC['target']=='direct')&(QC['horizon']==1)&(QC['split']=='test')&(QC['bin']==5)]['me_calibrated_kw'].iloc[0], 'outputs/11_analyses/quintile_control.csv', '+.0f')} kW in the observed bins, so the pattern is made by the binning and is not stated as a
property of the plant or of the model.

### Bias and the operation-limited hours

Every model has a positive mean error on all daytime hours of the test split: {F.span('mean error all hours, kW', ME_ALL.min(), ME_ALL.max(), 'outputs/11_analyses/decomposition.csv', '+.0f')} kW across the six models. On
the hours of normal operation it is {F.span('mean error normal operation, kW', ME_NORM[NO_PT].min(), ME_NORM[NO_PT].max(), 'outputs/11_analyses/decomposition.csv', '+.0f')} kW for XGB-Opt, BiLSTM-Opt and the three TFT
models, and {F.n('PatchTST mean error normal operation, kW', ME_NORM['patchtst'], 'outputs/11_analyses/decomposition.csv', '+.0f')} kW for PatchTST (F-30). The positive bias of the neural forecasts is
made in the {F.n('test op-limited hours (bias)', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} operation-limited hours (curtailment, stops, derate; D-22), which neither the weather nor
the history describes. Bias squared is at most {F.n('bias2 share max, validation, %', 100*D[(D['view']=='all')&(D['split']=='val')&(D['model'].isin(NO_PT))]['bias2_share'].max(), 'outputs/11_analyses/decomposition.csv', '.1f')} percent of the squared error on validation and
{F.n('bias2 share max, test, %', 100*D[(D['view']=='all')&(D['split']=='test')&(D['model'].isin(NO_PT))]['bias2_share'].max(), 'outputs/11_analyses/decomposition.csv', '.1f')} percent on test for the five models other than PatchTST, and the variance between the three seeds is at most
{F.n('seed variance share max, %', 100*DSD[DSD['view']=='all']['seed_variance_share'].max(), 'outputs/11_analyses/decomposition_seeds.csv', '.1f')} percent of the squared error.

### Practical recommendations

- Report both views of the error, all daytime hours and hours of normal operation, because the operation-limited
  hours carry most of the rise from validation to test.
- Compare models with a paired test in kW and report the effect as a share of the minimum detectable effect.
- Use the weather forecast issued before the forecast time: it is the group that matters most in the ablation
  and in the attributions.
- The submitted recommendations to use the indirect route, to avoid TFT-168 and to build an ensemble of XGB and
  TFT-96 for the peak hours are withdrawn: the first two are not supported by the paired tests, and the peak
  bias that motivated the third came from the binning (C8).

### Future work

- A forecast product of higher resolution than the one used (GFS), and quarter-hour data.
- Probabilistic forecasts: the quantile outputs of the neural models are stored (D-33); their coverage is not
  analysed here.
- Several sites, to give the paired tests the power that one section does not have.
- Dust events: the project holds no aerosol record, and the dust-day half of the by-season analysis was not done
  (Q-29, decision of the user).
- Training that prevents crossing quantiles of the TFT (D-33), and a longer search for the TFT models (D-37).
""")

# S-44
bp = jd(f"{RUN}/tft96/best_params.json")
bp168 = jd(f"{RUN}/tft168/best_params.json")
hs96 = [bp["indirect_seed42"]["params"]["hidden_size"], bp["direct_seed42"]["params"]["hidden_size"]]
hs168 = [bp168["indirect_seed42"]["params"]["hidden_size"], bp168["direct_seed42"]["params"]["hidden_size"]]
gap = G1[G1["point"] == "final fit against its best trial"]["fact"].str.extract(r"\(\+([\d.]+) pp\)")[0].astype(float)
ptst = G1[G1["point"] == "best check against the step budget"]["fact"].str.extract(r"best step (\d+) of a budget of (\d+)")
cap = G1[G1["point"] == "trials stopped at their cap"]["fact"].iloc[0]
cap_n, cap_total = re.search(r"(\d+) of (\d+) trials", cap).groups()
n_edge = int((G1["point"] == "best value on the edge of the search space").sum())
sp_x = SPREAD[SPREAD["model"] == "xgb_sota"]["std_kw_pp"]
sp_l = SPREAD[SPREAD["model"] == "lstm_sota"]["std_kw_pp"]
w("## 15. Academic disclosures (S-44)\n")
w(f"""D1. Operation-limited hours (rewritten; F-01c). The submitted disclosure of "{o('v6 downtime share, %', '13.5')} percent inverter downtime" was an
artefact of the time base: all {o('v6 zero-output hours', '339', PM)} hours with zero output were before sunrise by the label. In the rebuild,
the hours of curtailment, stops and derate are flagged and not erased (D-22): {F.n('test op-limited (D1)', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of the {F.n('test daytime hours (D1)', ss['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} daytime hours of the test
split and {F.n('val op-limited (D1)', vs['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of {F.n('val daytime hours (D1)', vs['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of the validation split. They are in the primary score and a second score leaves them out. They
do not "inflate all models equally": the models do not predict them, and they account for most of the positive mean
error (the section on bias).

D2. TFT hidden size. The submitted claim that Optuna chose the smallest size, {o('v6 TFT hidden size', '16')}, as a "genuine finding" is not
reproduced: the selected sizes are {F.n('TFT-96 hidden indirect', hs96[0], PT_ + '/tft96/best_params.json', 'd')} (indirect) and {F.n('TFT-96 hidden direct', hs96[1], PT_ + '/tft96/best_params.json', 'd')} (direct) for TFT-96 and {F.n('TFT-168 hidden indirect', hs168[0], PT_ + '/tft168/best_params.json', 'd')} and {F.n('TFT-168 hidden direct', hs168[1], PT_ + '/tft168/best_params.json', 'd')} for TFT-168, and
{F.n('studies with a value on the edge', n_edge, PT_ + '/report/report_g1_points.csv', 'd')} of the {F.n('studies in the final run', len(STUDIES), PT_ + '/report/report_studies.csv', 'd')} studies of the final run have a best value on the edge of the search space (a limitation, D-37).

D3. Internal and evaluation metrics. The tuning objective of every study is the nRMSE in kW over the daytime hours of
validation (S-14); the BiLSTM stops early on its quantile loss. The final fit of the BiLSTM is worse than its best trial by {F.span('BiLSTM final fit minus best trial, pp', gap.min(), gap.max(), PT_ + '/report/report_g1_points.csv')} points, and the best step of PatchTST is {F.n('PatchTST best step direct', int(ptst[0].astype(int).min()), PT_ + '/report/report_g1_points.csv', ',d')} to {F.n('PatchTST best step indirect', int(ptst[0].astype(int).max()), PT_ + '/report/report_g1_points.csv', ',d')} of a budget of {F.n('PatchTST step budget', int(ptst[1].astype(int).iloc[0]), PT_ + '/report/report_g1_points.csv', ',d')}
steps, so it may not have converged. {F.n('XGB trials at their cap', int(cap_n), PT_ + '/report/report_g1_points.csv', 'd')} of {F.n('XGB trials', int(cap_total), PT_ + '/report/report_g1_points.csv', ',d')} XGB trials stopped at their cap. None of the budgets, seeds or rules was
changed after the test split was read (D-37).

D4. Direction of the paired test. A positive difference in the tables of `dm_tests_kw.csv` means that the model in
the first column has the larger squared error than XGB-Opt. The submitted statistics were computed without the
HLN correction and without Bonferroni (F-04); the rebuild uses both, over the six cells of a split and
comparison, with a Newey-West bandwidth of {F.n('Newey-West lags', int(DMT['nw_lags'].iloc[0]), PT_ + '/report/report_dm_kw_test.csv', 'd')} hours.

D5. Seeds. Each seed of XGB-Opt and BiLSTM-Opt is scored by the evaluation layer on the same hours. The largest
standard deviation over the three seeds is {F.n('XGB seed std max, pp', sp_x.max(), PT_ + '/report/report_seed_spread.csv', '.3f')} points for XGB-Opt and {F.n('BiLSTM seed std max, pp', sp_l.max(), PT_ + '/report/report_seed_spread.csv', '.3f')} points for BiLSTM-Opt. The
submitted seed table for the BiLSTM held training-script metrics and is not comparable (C-07). The TFT models and
PatchTST have one seed each, so no seed interval exists for them.

D6. One section of one plant. The data are two inverters of a park (A-07), the rating is not verified (A-08), the
test period has no summer, and {F.n('test hours polled or mixed (D6)', POLLED, 'outputs/03_eda/tables/table_splits.csv', ',d')} of its {F.n('test daytime hours (D6)', ss['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} daytime hours come from a second data source whose hourly means
rest on four snapshots (F-18). The validation numbers are tuning-split numbers (D-10).

Reproducibility. The code, the configurations and the frozen run are in the repository at the tag `g1-frozen`;
the raw data are commercially sensitive and are not released.
""")

TEXT = "\n".join(sec)

# ── the table of old and new results ─────────────────────────────────────────
OVN: list[dict] = []


def ovn(section, item, old, new_val, new_test, unit, comparable, why, old_src, new_src):
    OVN.append({"section": section, "item": item, "old_value": old, "new_validation": new_val,
                "new_test": new_test, "unit": unit, "comparable": comparable, "reason": why,
                "old_source": old_src, "new_source": new_src})


for m in M6:
    for t in ("indirect", "direct"):
        for h in (1, 2, 3):
            o = OLDK.loc[(m, t, h)]
            comp = ("units differ (k is rescaled and unclipped, F-06)" if t == "indirect"
                    else "same definition, other data and time base")
            ovn("benchmark", f"{NAME[m]} {t} D+{h}", f"{o['nRMSE_pct']:.2f} (n={int(o['n'])})",
                f"{UV[(m, t, h)]:.2f}", f"{UT[(m, t, h)]:.2f}", "nRMSE %, target unit", comp,
                "F-01, F-06, F-07, F-28; C-15; scored hours C-35",
                "submission_evidence/01_benchmark/benchmark_summary.csv",
                f"{RUN}/report/report_benchmark_val.csv, report_benchmark_test.csv (seed mean)")
            ovn("benchmark in kW", f"{NAME[m]} {t} D+{h}", "not reported in v6", f"{KV[(m, t, h)]:.2f}",
                f"{KT[(m, t, h)]:.2f}", "nRMSE % of capacity, kW", "new view",
                "Q-19: the primary score is in kW", "-",
                f"{RUN}/report/report_benchmark_val.csv, report_benchmark_test.csv (seed mean)")

rt_old = RT_OLD.set_index(["model", "horizon", "route"])
for m in ("xgb_sota", "lstm_sota"):
    for h in (1, 2, 3):
        adv = rt_old.loc[(m, h, "direct"), "nRMSE_kW_pct"] - rt_old.loc[(m, h, "indirect_to_power"), "nRMSE_kW_pct"]
        v = RPV[(RPV["model"] == m) & (RPV["horizon"] == h)].iloc[0, 2]
        t_ = RPT[(RPT["model"] == m) & (RPT["horizon"] == h)].iloc[0, 2]
        ovn("route", f"{NAME[m]} D+{h}, direct minus indirect", f"{adv:+.2f}", f"{v:+.2f}", f"{t_:+.2f}",
            "pp (positive: indirect better)", "same definition, not the same truth",
            "C-16 (truth rebuilt from clipped k), C-34, F-06",
            "submission_evidence/03_route_comparison/route_comparison.csv",
            f"{RUN}/evaluate/val/route_comparison.csv, test/test/route_comparison.csv")

sv_old = SEED_OLD[SEED_OLD["model"] == "xgb_sota"]["nRMSE_std"].max()
ovn("seeds", "XGB-Opt, largest std over seeds", f"{sv_old:.3f}", f"{sp_x.max():.3f}", "-", "pp",
    "other data", "C-10; every seed scored by the evaluation layer",
    "submission_evidence/05_seed_variability/seed_variability_summary.csv", f"{RUN}/report/report_seed_spread.csv")
ovn("tests", "pairs significant at 5 percent without correction (v6) / cells significant with HLN and Bonferroni, XGB-Opt against climatology (rebuild)",
    f"{int(DM_OLD['significant_5pct'].sum())} of {len(DM_OLD)} (all models, no correction)", f"{int(CLIM_V['significant'].sum())} of 6",
    f"{int(CLIM_T['significant'].sum())} of 6", "cells", "different test",
    "F-04 (HLN and Bonferroni were described but not computed), C-15",
    "submission_evidence/02_statistical_tests/dm_tests.csv", f"{RUN}/report/report_dm_kw_val.csv, report_dm_kw_test.csv")
ovn("tests", "pairs below the minimum detectable effect", f"{int((MDE_OLD['conclusion']=='UNDERPOWERED').sum())} of {len(MDE_OLD)}",
    "-", f"{int((DMT[DMT['model_b']=='xgb_sota']['pct_of_mde']<100).sum())} of {len(DMT[DMT['model_b']=='xgb_sota'])} (all comparisons with XGB-Opt)", "pairs",
    "different pairs", "Q-19", "submission_evidence/02_statistical_tests/detectable_effect_summary.csv",
    f"{RUN}/report/report_dm_kw_test.csv")
for t in ("indirect", "direct"):
    po = SHAP_OLD[(SHAP_OLD["target"] == t) & (SHAP_OLD["horizon"] == 1)].sort_values("pct", ascending=False).iloc[0]
    sv = SHV[(SHV["target"] == t) & (SHV["horizon"] == 1)].groupby("group")["share_pct"].mean().sort_values(ascending=False)
    st = SHT[(SHT["target"] == t) & (SHT["horizon"] == 1)].groupby("group")["share_pct"].mean().sort_values(ascending=False)
    ovn("SHAP", f"largest feature group, {t} D+1 (share of mean absolute SHAP)",
        f"{po['group']} {po['pct']:.1f}%", f"{sv.index[0]} {sv.iloc[0]:.1f}%", f"{st.index[0]} {st.iloc[0]:.1f}%",
        "% share", "groups differ", "C-31, M25 (76 inputs, leaking rolling windows), F-28",
        "submission_evidence/07_shap/shap_group_importance.csv",
        f"{RUN}/explain/xgb/xgb_shap_groups.csv, test/shap_test/xgb_shap_groups_test.csv")
for s_ in ("Winter", "Spring", "Fall"):
    old = SEA_OLD[(SEA_OLD["model"] == "xgb_sota") & (SEA_OLD["target"] == "indirect") & (SEA_OLD["horizon"] == 1) & (SEA_OLD["season"] == s_)]["nRMSE_pct"].iloc[0]
    v = SEA[(SEA["model"] == "xgb_sota") & (SEA["target"] == "indirect") & (SEA["horizon"] == 1) & (SEA["season"] == s_) & (SEA["split"] == "val")]["nRMSE_kW_pct"]
    t_ = SEA[(SEA["model"] == "xgb_sota") & (SEA["target"] == "indirect") & (SEA["horizon"] == 1) & (SEA["season"] == s_) & (SEA["split"] == "test")]["nRMSE_kW_pct"]
    ovn("seasons", f"XGB-Opt indirect D+1, {s_}", f"{old:.2f}", f"{v.iloc[0]:.2f}" if len(v) else "no hours",
        f"{t_.iloc[0]:.2f}" if len(t_) else "no hours", "nRMSE %", "units differ; the months of a season differ (X104)",
        "F-06 units, X104", "submission_evidence/04_seasonal/seasonal_metrics.csv", "outputs/11_analyses/seasonal_kw.csv")
cvx = CVT[(CVT["target"] == "indirect") & (CVT["horizon"] == 1)]
ovn("cross-validation", "XGB-Opt indirect D+1, 5-fold CV mean", "16.08 (v6 text; no output file, matrix row 'XGB-Opt 5-fold CV')",
    f"{cvx['validation_uncorrected'].mean():.2f}", f"{cvx['test_uncorrected'].mean():.2f} (CV mean {cvx['cv_mean'].mean():.2f})", "nRMSE % in kW",
    "no (the script was lost)", "C-76, C-77, M43", "submission_evidence/11_audit/REPORT_CLAIM_EVIDENCE_MATRIX.csv", f"{RUN}/report/report_cv_vs_test.csv")
for mode, old_mode in (("minimal", "minimal"),):
    o = ABL_OLD[(ABL_OLD["target"] == "direct") & (ABL_OLD["horizon"] == 1) & (ABL_OLD["mode"] == old_mode)]["delta_mean"].iloc[0]
    v = ABV[(ABV["target"] == "direct") & (ABV["horizon"] == 1) & (ABV["mode"] == mode)]["mean_delta_pp"].iloc[0]
    t_ = ABT[(ABT["target"] == "direct") & (ABT["horizon"] == 1) & (ABT["mode"] == mode)]["mean_delta_pp"].iloc[0]
    ovn("ablation", "direct D+1, 'minimal' set, change of nRMSE", f"{o:+.2f}", f"{v:+.2f}", f"{t_:+.2f}", "pp",
        "no (the sets differ)", "C-71, C-73, C-75: the old 'minimal' set kept features that are not time encodings",
        "submission_evidence/05_seed_variability/ablation_seed_variability.csv",
        f"{RUN}/report/report_ablation.csv, report_ablation_test.csv")
q2 = QU[(QU["model"] == "xgb_sota") & (QU["seed"] == 42) & (QU["target"] == "indirect") & (QU["horizon"] == 1) & (QU["binned_by"] == "observed") & (QU["bin"] == 2)].set_index("split")["nRMSE_kW_pct"]
ovn("quintiles", "XGB-Opt indirect D+1, second quintile of the observed power, nRMSE", "20.71 (v6 text; no output file)",
    f"{q2['val']:.2f}", f"{q2['test']:.2f}", "nRMSE %", "no (edges from the test sample in v6)", "C-78, M45",
    "submission_evidence/11_audit/REPORT_CLAIM_EVIDENCE_MATRIX.csv", "outputs/11_analyses/quintiles.csv")
ovn("data", "daytime test hours scored (XGB-Opt, D+1)", "2,580", f"{int(vs['daytime_valid_target']):,}", f"{int(ss['daytime_valid_target']):,}", "hours",
    "no (other rows scored)", "C-35: every model on every valid daytime hour", "submission_evidence/01_benchmark/benchmark_summary.csv",
    "outputs/03_eda/tables/table_splits.csv")
ovn("target", "MEAN_DAYTIME_K", "0.6279", f"{DER['mean_daytime_k']:.6f}", "-", "k", "no", "F-06 (denominator POA_cs, no clip)",
    "submission_evidence/ (v6 text)", "configs/derived_constants.yaml")
sk_ref = KT.index  # silence linters about unused name
ovn("skill", "XGB-Opt indirect D+1, skill score against persistence", "0.90 to 0.96 (v6 text)",
    f"{BV[(BV['model']=='xgb_sota')&(BV['target']=='indirect')&(BV['horizon']==1)]['skill_score_kw'].mean():.3f}",
    f"{BT[(BT['model']=='xgb_sota')&(BT['target']=='indirect')&(BT['horizon']==1)]['skill_score_kw'].mean():.3f}", "skill", "no (reference changed)",
    "C-08", "docs/errata.md C-08", f"{RUN}/report/report_benchmark_val.csv, report_benchmark_test.csv")

# ── write ────────────────────────────────────────────────────────────────────
(REPO_ROOT / SP).write_text(TEXT, encoding="utf-8", newline="\n")
pd.DataFrame(F.rows).to_csv(REPO_ROOT / "docs/traceability.csv", index=False, encoding="utf-8", lineterminator="\n",
                            quoting=csv.QUOTE_MINIMAL)
pd.DataFrame(OVN).to_csv(REPO_ROOT / "docs/old_vs_new.csv", index=False, encoding="utf-8", lineterminator="\n")

MD = ["| Section | Item | Old (v6) | New, validation | New, test | Unit | Comparable | Why | Old source | New source |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in OVN:
    MD.append("| " + " | ".join(str(r[k]).replace("|", "/") for k in (
        "section", "item", "old_value", "new_validation", "new_test", "unit", "comparable", "reason",
        "old_source", "new_source")) + " |")
BEGIN, END = "<!-- old-vs-new:begin (generated by scripts/make_report_v7.py) -->", "<!-- old-vs-new:end -->"
block = (f"{BEGIN}\n\nGenerated from the files named in the last two columns; the old numbers are read from "
         "`submission_evidence/` (read-only) and the new ones from the final run `20261114T083930Z_final` "
         "and `outputs/11_analyses/`. A test number is always beside the validation number of the same view. "
         "'Comparable: no' rows are shown for the record and are not differences of the same quantity.\n\n"
         + "\n".join(MD) + f"\n\n{END}")
er = (REPO_ROOT / "docs/errata.md").read_text(encoding="utf-8")
if BEGIN in er:
    er = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _m: block, er, flags=re.S)
else:
    # the placeholder sat between two groups of rows of the corrections table (X84 to X93 follow it):
    # take it out so the rows are contiguous again, and put the section at the end of the file
    anchor = "\n## Old vs new results\n\nTo be filled in stage 11.\n"
    assert er.count(anchor) == 1
    er = er.replace(anchor, "").rstrip("\n") + "\n\n## Old vs new results\n\n" + block + "\n"
(REPO_ROOT / "docs/errata.md").write_text(er, encoding="utf-8", newline="\n")
print(f"report: {len(TEXT.splitlines())} lines, {len(F.rows)} registered numbers, {len(OVN)} old-vs-new rows")

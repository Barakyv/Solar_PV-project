# ruff: noqa: E501  (report text with values cannot be wrapped; see make_quality_notebook)
"""Write report v7 of stage 11 from the frozen final run (no training): the whole report, with its figures.

    python scripts/make_report_v7.py

Writes `docs/report_v7.md` (the report in the chapter order of the submitted report v6, with the figures
chosen in part C), `docs/report_v7_README.md` (the correction note of D-02, data and code availability, the
appendices and the decisions still open), `docs/report_v7_figures.csv` (one row per figure of the report and
its counterpart in v6), `docs/traceability.csv` (one row per number of the report: claim, value_old,
value_new, file, script, verified_on), `docs/old_vs_new.csv` and the table "Old vs new results" of
`docs/errata.md`.

Every number is read from a file of `outputs/` or of `submission_evidence/` (read-only) and is registered
when it is formatted (`Facts`); a test (T-178) checks that no other number is in the text. The caption of a
figure is the takeaway its builder formatted from computed facts (`outputs/figures/figures_manifest.json`),
and a test (T-186) checks that the captions are current. The test split is read only through the tables the
run already wrote (`report/`, `test/`, `outputs/11_analyses/`); the step `test` is not run and no model is
trained (D-36, D-37).
"""

import csv
import json
import re

import pandas as pd

from solarpv.config.cli import no_options
from solarpv.config.constants import BOOTSTRAP_BLOCK, BOOTSTRAP_REPS
from solarpv.config.paths import REPO_ROOT

no_options("python scripts/make_report_v7.py", __doc__)  # X62: --help runs nothing

SCRIPT = "scripts/make_report_v7.py"
VERIFIED_ON = "2026-11-15"
RUN = "outputs/runs/20261114T083930Z_final"
RUN_PATH = REPO_ROOT / RUN
M6 = ["xgb_sota", "lstm_sota", "tft_96", "tft_168", "tft", "patchtst"]
REFS = ["persistence", "climatology", "clearsky_k"]
NAME = {"xgb_sota": "XGB-Opt", "lstm_sota": "BiLSTM-Opt", "tft_96": "TFT-96", "tft_168": "TFT-168",
        "tft": "TFT baseline", "patchtst": "PatchTST", "climatology": "climatology",
        "persistence": "persistence", "clearsky_k": "clear sky x k"}
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
MANIFEST = "outputs/figures/figures_manifest.json"
FIG_TABLE = "docs/report_v7_figures.csv"
BANNED = ("is the best model", "wins", "outperforms all", "physical effect of")  # T-181


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
MKW = rd("outputs/11_analyses/metrics_kw.csv")
CI = rd("outputs/11_analyses/benchmark_ci_kw.csv")
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
CMS = rd(f"{RUN}/test/test/common_mask_summary.csv")
SHAP_OLD = rd("submission_evidence/07_shap/shap_group_importance.csv")
SEA_OLD = rd("submission_evidence/04_seasonal/seasonal_metrics.csv")
SEED_OLD = rd("submission_evidence/05_seed_variability/seed_variability_summary.csv")
DM_OLD = rd("submission_evidence/02_statistical_tests/dm_tests.csv")
MDE_OLD = rd("submission_evidence/02_statistical_tests/detectable_effect_summary.csv")
ABL_OLD = rd("submission_evidence/05_seed_variability/ablation_seed_variability.csv")
SEL = jd("outputs/07_feature_selection/selection_manifest.json")
MI = jd(f"{RUN}/select/sets/model_inputs.json")
CLEAN = jd("outputs/05_cleaning/cleaning_manifest.json")
GFS = jd("outputs/06b_gfs/gfs_manifest.json")
CFG = __import__("yaml").safe_load((RUN_PATH / "config.yaml").read_text(encoding="utf-8"))
DER = __import__("yaml").safe_load((REPO_ROOT / "configs/derived_constants.yaml").read_text(encoding="utf-8"))
ACCESS = pd.read_csv(REPO_ROOT / "outputs/07_evaluation/test_access_log.csv")
FIGS = jd(MANIFEST)["figures"]


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
assert (PT_C["diff_a_minus_b_pp"] > 0).all()  # PatchTST is worse than climatology in every cell
assert (CVT["validation_minus_cv_pp"] < 0).all() and (CVT["test_minus_cv_pp"] > 0).all()  # S10-06
_ci = CI[CI["model"].isin(M6[:5]) & (CI["split"] == "test") & (CI["seed"] == 42)]
assert all(g["ci_lo"].max() < g["ci_hi"].min() for _, g in _ci.groupby(["target", "horizon"]))  # 7.1: the five overlap
assert (CMS["n"] == CMS["n"].iloc[0]).all()  # the common mask keeps every hour of every model (S-25)


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

# ── the figures of the report (part C) ───────────────────────────────────────
# Chosen by following the figures of the submitted report v6 (its figures 1 to 25; the counterpart of each,
# with `v6` the number of the v6 figure) and adding the figures that complete the account of the method
# (`v6` empty). In reading order; a figure is numbered by its place in the report.
ORDER = [
    ("S01-01", "3.2", ""), ("S05-02", "3.2", ""), ("S06-01", "3.3", ""), ("S6b-01", "3.4", ""),
    ("S07-01", "3.6", ""), ("S07-02", "3.6", "22"), ("S09-35", "3.7", "24"), ("S01-02", "3.8", "11"),
    ("S06-04", "3.9", ""), ("S11-16", "5", "12"), ("S08-01", "6.1", ""), ("S11-01", "7.1", "1"),
    ("S11-03", "7.3", "13"), ("S11-02", "7.5", "2"), ("S10-03", "7.6", ""), ("S10-04", "8.1", "5"),
    ("S11-04", "9.1", "3"), ("S11-05", "9.2", "4"), ("S11-17", "9.2", ""), ("S11-06", "10", "10"),
    ("S11-07", "11.1", "6"), ("S11-08", "11.2", "7"), ("S11-09", "11.3", ""), ("S11-14", "11.4", ""),
    ("S11-10", "11.5", "8"), ("S10-05", "12.1", "17"), ("S10-06", "12.2", "18"), ("S11-11", "12.3", "9"),
    ("S11-12", "12.4", "16"), ("S11-13", "12.5", "15"), ("S11-19", "16", ""),
]
# v6 figures with no counterpart, and why (docs/defense_notes.md section 17)
V6_NOT_REDRAWN = {
    "14": "radar of the seasonal errors: the same numbers as the heatmap (C-84)",
    "19": "three-method consensus pipeline: replaced by the selection protocol on train only (D-23, F-26)",
    "20": "SHAP backward-elimination curves: replaced by the selection protocol on train only (D-23, F-26)",
    "21": "stability-selection probabilities: replaced by the selection protocol on train only (D-23, F-26)",
    "23": "mutual information against stability: replaced by the selection protocol on train only (D-23, F-26)",
    "25": "stability-floor sensitivity: replaced by the selection protocol on train only (D-23, F-26)",
}
assert len({v for _, _, v in ORDER if v} | set(V6_NOT_REDRAWN)) == 25  # every v6 figure is accounted for
FIGNO = {fid: i + 1 for i, (fid, _, _) in enumerate(ORDER)}
PLACED: list[str] = []
REFERENCED: set[str] = set()
for fid, _, _ in ORDER:
    assert fid in FIGS and FIGS[fid]["kind"] == "process", fid
    assert not any(b in FIGS[fid]["takeaway"] for b in BANNED), fid


def fr(fid: str) -> str:
    """A reference to a figure in the text: 'Figure n'."""
    REFERENCED.add(fid)
    return "Figure " + F.n(f"number of figure {fid} in the report", FIGNO[fid], FIG_TABLE, "d")


def fig(fid: str) -> str:
    """The figure itself: the image and its caption, the takeaway of its builder."""
    assert fid not in PLACED and [f for f, _, _ in ORDER].index(fid) == len(PLACED), fid
    PLACED.append(fid)
    spec = FIGS[fid]
    no = F.n(f"number of figure {fid} in the report", FIGNO[fid], FIG_TABLE, "d")
    path = F.s(f"file of figure {fid}", "../" + spec["file"], spec["file"])
    title = F.s(f"title of figure {fid}", spec["title"], MANIFEST)
    cap = F.s(f"caption of figure {fid} (takeaway of its builder)", spec["takeaway"], MANIFEST)
    return f"\n![{fid}]({path})\n\n**Figure {no}. {title}.** {cap} *(Figure book: `{fid}`.)*\n"


# ── references (the list of v6, chapter 17; M58: one citation of v6 had no entry and is not used) ──────────
# Two entries added by the user's decision of 2026-11-15 (R11-04): the method of the SHAP values and the source of the
# weather forecasts, which v6 used without an entry. Each is checked against the code and data record (explain.py uses
# shap.TreeExplainer; stage 6b reads the bucket noaa-gfs-bdp-pds).
REFS_V6 = {
    "akiba": ("Akiba et al., 2019", "Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. KDD 2019."),
    "antonanzas": ("Antonanzas et al., 2016", "Antonanzas, J., Osorio, N., Escobar, R., Urraca, R., Martinez-de-Pison, F. J., & Antonanzas-Torres, F. (2016). Review of photovoltaic power forecasting. Solar Energy, 136, 78–111."),
    "chen": ("Chen & Guestrin, 2016", "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016."),
    "diebold": ("Diebold & Mariano, 1995", "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263."),
    "efron": ("Efron & Tibshirani, 1993", "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall."),
    "graves": ("Graves & Schmidhuber, 2005", "Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5–6), 602–610."),
    "harvey": ("Harvey et al., 1997", "Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281–291."),
    "hochreiter": ("Hochreiter & Schmidhuber, 1997", "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780."),
    "lundberg2017": ("Lundberg & Lee, 2017", "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30 (NIPS 2017), 4765–4774."),
    "lundberg2020": ("Lundberg et al., 2020", "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56–67."),
    "noaa": ("NOAA NCEP, GFS", "NOAA National Centers for Environmental Prediction (NCEP). NOAA Global Forecast System (GFS), 1.0° global forecast grids. NOAA Open Data Dissemination, Registry of Open Data on AWS, bucket noaa-gfs-bdp-pds. https://registry.opendata.aws/noaa-gfs-bdp-pds/"),
    "li": ("Li et al., 2016", "Li, J., Ward, J. K., Tong, J., Collins, L., & Platt, G. (2016). Machine learning for solar irradiance forecasting of photovoltaic system. Renewable Energy, 90, 542–553."),
    "lim": ("Lim et al., 2021", "Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748–1764."),
    "mellit": ("Mellit & Pavan, 2010", "Mellit, A., & Pavan, A. M. (2010). A 24-h forecast of solar irradiance using artificial neural network. Renewable Energy, 35(11), 2603–2616."),
    "nespoli": ("Nespoli et al., 2022", "Nespoli, A., Niccolai, A., Ogliari, E., Perego, G., Collino, E., & Ronzio, D. (2022). Machine learning techniques for solar irradiance and power forecasting. Energies, 15(21), 8129."),
    "newey": ("Newey & West, 1987", "Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703–708."),
    "nie": ("Nie et al., 2023", "Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. ICLR 2023."),
    "schenker": ("Schenker & Gentleman, 2001", "Schenker, N., & Gentleman, J. F. (2001). On judging the significance of differences by examining the overlap between confidence intervals. The American Statistician, 55(3), 182–186."),
    "sweeney": ("Sweeney et al., 2020", "Sweeney, C., Bessa, R. J., Browell, J., & Pinson, P. (2020). The future of forecasting for renewable energy. Wiley Interdisciplinary Reviews: Energy and Environment, 9(2), e365."),
    "wang": ("Wang et al., 2019", "Wang, H., Yi, H., Peng, J., Wang, G., Liu, Y., Jiang, H., & Liu, W. (2019). Deterministic and probabilistic forecasting of photovoltaic power based on deep convolutional neural network. Energy Conversion and Management, 195, 1150–1163."),
}
CITED: set[str] = set()


def c(*keys: str) -> str:
    """A citation of an entry of the reference list of v6."""
    CITED.update(keys)
    return "[" + "; ".join(o(f"citation {k}", REFS_V6[k][0]) for k in keys) + "]"


# ── values used in several chapters ──────────────────────────────────────────
CAP = SEL["capacity_kw"]
ts, vs, ss = SPLITS.loc["train"], SPLITS.loc["val"], SPLITS.loc["test"]
al = SPLITS.loc["all"]
POLLED = int(ss["daytime_polled"]) + int(ss["daytime_mixed"])
hours_gpu = sum(STEPS.loc[s, "duration_s"] for s in ("bilstm", "tft96", "tft168", "tftbase", "patchtst")) / 3600
test_rows_scored = int(ACCESS["command"].str.contains("step test: score the test split", regex=False).sum())
assert test_rows_scored == 1  # the test split was scored once (D-32)
RHO_V = float(RANKV.groupby(["target", "compare"])["spearman"].mean().min())  # as S11-07: mean over seeds
RHO_T = float(RANKT.groupby(["target", "compare"])["spearman"].mean().min())
PT_ = RUN
BTF, BVF = f"{RUN}/report/report_benchmark_test.csv", f"{RUN}/report/report_benchmark_val.csv"
MKF = "outputs/11_analyses/metrics_kw.csv"
XV = MKW[MKW["model"] == "xgb_sota"].groupby(["split", "view"])["nRMSE_kW_pct"].mean()
CV_ = MKW[MKW["model"] == "climatology"].groupby(["split", "view"])["nRMSE_kW_pct"].mean()
_mv = MKW[MKW["model"].isin(M6 + REFS)].groupby(["model", "split", "view"])["nRMSE_kW_pct"].mean().unstack(["split", "view"])
# the share of the rise from validation to test that the operation-limited hours account for (as S10-03)
RISE_SHARE = (((_mv[("test", "all")] - _mv[("val", "all")]) - (_mv[("test", "normal_operation")] - _mv[("val", "normal_operation")]))
              / (_mv[("test", "all")] - _mv[("val", "all")]))
assert ((RISE_SHARE > 0) & (RISE_SHARE < 1)).all() and RISE_SHARE["xgb_sota"] > 0.5
n_sel = {k: v["n_features"] for k, v in MI.items()}
cm = CLIM_T["pct_of_mde"]
a_all = ATT[ATT["block"] == "all"]
dm_enc = ENC[ENC["split"] == "test"]
gap_x = PT_X["diff_a_minus_b_pp"]
rp_all = pd.concat([RPV.iloc[:, 2], RPT.iloc[:, 2]])
sx = SEA[(SEA["model"] == "xgb_sota") & (SEA["target"] == "indirect") & (SEA["horizon"] == 1)].set_index(["season", "split"])
cx = SEA[(SEA["model"] == "climatology") & (SEA["target"] == "indirect") & (SEA["horizon"] == 1)].set_index(["season", "split"])
hz = HR[HR["model"].isin(M6)]
QX = QU[(QU["model"] == "xgb_sota") & (QU["seed"] == 42) & (QU["target"] == "direct") & (QU["horizon"] == 1) & (QU["split"] == "test")]
QCX = QC[(QC["target"] == "direct") & (QC["horizon"] == 1) & (QC["split"] == "test")]
bp = jd(f"{RUN}/tft96/best_params.json")
bp168 = jd(f"{RUN}/tft168/best_params.json")
hs96 = [bp["indirect_seed42"]["params"]["hidden_size"], bp["direct_seed42"]["params"]["hidden_size"]]
hs168 = [bp168["indirect_seed42"]["params"]["hidden_size"], bp168["direct_seed42"]["params"]["hidden_size"]]
gap = G1[G1["point"] == "final fit against its best trial"]["fact"].str.extract(r"\(\+([\d.]+) pp\)")[0].astype(float)
ptst = G1[G1["point"] == "best check against the step budget"]["fact"].str.extract(r"best step (\d+) of a budget of (\d+)")
cap_txt = G1[G1["point"] == "trials stopped at their cap"]["fact"].iloc[0]
cap_n, cap_total = re.search(r"(\d+) of (\d+) trials", cap_txt).groups()
n_edge = int((G1["point"] == "best value on the edge of the search space").sum())
sp_x = SPREAD[SPREAD["model"] == "xgb_sota"]["std_kw_pp"]
sp_l = SPREAD[SPREAD["model"] == "lstm_sota"]["std_kw_pp"]
BUD = CFG["budgets"]
assert (BUD["xgb"]["trials"], BUD["xgb"]["seeds"]) == (BUD["bilstm"]["trials"], BUD["bilstm"]["seeds"])
assert BUD["tft96"]["trials"] == BUD["tft168"]["trials"]
CIX = CI[(CI["model"] == "xgb_sota") & (CI["split"] == "test")]
CIW = (CI[CI["model"].isin(M6) & (CI["split"] == "test")]["ci_hi"] - CI[CI["model"].isin(M6) & (CI["split"] == "test")]["ci_lo"])

sec = []
w = sec.append

# ── title, abstract, reading rules, correction note ──────────────────────────
w("# Day-Ahead Solar Power Forecasting at One Section of a Utility-Scale PV Plant: A Comparative Study of Gradient Boosting, Recurrent and Transformer Architectures\n")
w(f"""Report v7, the rebuilt and corrected version of the submitted report v6. Author: Barak Yaakov, {o('year of the report', '2026')}.
Generated by `scripts/make_report_v7.py` from the frozen final run `20261114T083930Z_final` (tag `g1-frozen`).
The notes of the report (the correction note, the appendices and the availability of the code and the data) are in
`docs/report_v7_README.md`.
""")

w("## Abstract\n")
w(f"""Day-ahead forecasts of photovoltaic power are needed to schedule a plant against the grid. This report compares six
models, XGBoost with monotone constraints (XGB-Opt), a bidirectional LSTM (BiLSTM-Opt), the Temporal Fusion
Transformer with a 96-hour and a 168-hour encoder (TFT-96, TFT-168), a TFT baseline with fixed settings and PatchTST,
for three horizons (D+1 to D+3) and two routes (the clear-sky index converted to power, and power directly), on a
section of two inverters of a utility-scale plant in the Southern Arava, Israel, with an AC capacity of
{F.n('AC capacity, kW (abstract)', CAP, 'outputs/07_feature_selection/selection_manifest.json', ',.0f')} kW. The pipeline
runs from the one-minute SCADA export in true UTC to hourly features, uses only weather forecasts (GFS) that were public
before the issuance at local midnight, selects the features on the training period only, and scores every model and three
reference forecasts on the same daytime hours. The code was frozen and audited before the final training (gate G-1), and
the test period (2024-10..2025-05) was scored once.

On test, the nRMSE in kW, in percent of capacity, of the six models is
{F.span('six models, test, range of the seed-mean nRMSE, %', KT.loc[M6].min(), KT.loc[M6].max(), BTF)} percent
(validation: {F.span('six models, validation, range of the seed-mean nRMSE, %', KV.loc[M6].min(), KV.loc[M6].max(), BVF)}),
against {F.span('climatology, test, range, %', KT.loc['climatology'].min(), KT.loc['climatology'].max(), BTF)} for
climatology and {F.span('persistence, test, range, %', KT.loc['persistence'].min(), KT.loc['persistence'].max(), BTF)}
for persistence. XGB-Opt, BiLSTM-Opt and the three TFT models are close, and no model is declared the winner: in paired
Diebold-Mariano tests in kW on test the three TFT models are not separated from XGB-Opt in any cell. XGB-Opt is significantly
better than persistence and the clear-sky reference in every cell, and than climatology in
{F.n('climatology significant cells, test (abstract)', int(CLIM_T['significant'].sum()), PT_ + '/report/report_dm_kw_test.csv', 'd')}
of {N6} cells on test, where the observed difference is close to the minimum detectable effect of one section and one
test period. The error of every model and reference rises from validation to test; {F.n('share of the rise of XGB-Opt in the op-limited hours, % (abstract)', 100 * RISE_SHARE['xgb_sota'], MKF, '.0f')} percent of
the rise of XGB-Opt is in the
{F.n('test op-limited hours (abstract)', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')}
operation-limited hours (curtailment, stops, derate), which no input describes. The weather forecast is the input group
that matters most, in the ablation and in the attributions, and the two routes are equivalent within the uncertainty.
""")

w("## Rules of reading\n")
w("""Every number of the test split is placed beside the validation number of the same view.
No model is declared the winner (D-37): four of the five neural models have one seed. The validation numbers are
tuning-split numbers (D-10). All
errors are nRMSE in kW, in percent of the plant capacity, over the daytime hours of the split with a measured output, the
same hours for every model. Every number of this report is registered in `docs/traceability.csv` with its source file and
the script that formats it, and the caption of every figure is the takeaway that its builder computed from the outputs.
""")

w("## Reproduction and correction note (D-02)\n")
w(f"""This report rebuilds the submitted report (v6) from its evidence folder; it is not a one-to-one reproduction
(D-01). The rebuild corrects errors that changed the numbers: local time had been read as UTC (F-01), the
indirect target was clipped and its mean was a mean of ratios (F-06), rolling features saw unknown days at
D+2 and D+3 (F-07), the weather inputs were not forecasts available at issuance (F-28), and the feature
selection read the validation and test periods (F-26). Every correction has a row in `docs/errata.md` and a
test, and the table "Old vs new results" of the errata sets each submitted number beside the rebuilt one. The
scored hours changed from {F.n('old test daytime hours', OLDK.loc[('xgb_sota','indirect',1),'n'], 'submission_evidence/01_benchmark/benchmark_summary.csv', ',d')}
in v6 to {F.n('test daytime hours with a measured output', ss['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')}
in the rebuild, the same hours for every model, and the validation period was added as a second view (it had been
used for tuning).
""")

# ── 1. Introduction ──────────────────────────────────────────────────────────
w("## 1. Introduction\n")
w("""Solar photovoltaic generation is intermittent, weather-dependent and non-linear, and day-ahead scheduling commits a
plant to a production schedule before the weather of the delivery day is known. Physics gives the clear-sky output of a
plant; numerical weather prediction (NWP) gives the expected state of the atmosphere; machine learning learns the
relation between the two and the measured output. Attention-based models raise the question of how much of the skill
comes from the engineered inputs and how much from the history of the series itself.

This study compares six models on one plant section, from one pipeline, and makes the following contributions:

- **Issuance-time availability of every input.** Every input is checked against the issuance time at local midnight:
  measured lags end at the issuance, rolling windows and encoders do not reach past it, and the weather forecast is the
  GFS run that was public before it (sections 3.4 and 3.7).
- **Feature selection on the training period only**, with one look at validation (section 3.6).
- **A six-model benchmark on identical hours**, scored in kW against the measured AC power, with three reference
  forecasts, intervals, paired tests and the minimum detectable effect (chapters 6 to 8).
- **Two routes compared against the same truth.** The indirect route (the clear-sky index, converted to power) and the
  direct route (power) are both scored in kW against the measured AC power (chapter 10).
- **Interpretability and robustness.** SHAP attributions, TFT attention, encoder length, seeds, cross-validation, the
  error decomposition, production bins and a feature-group ablation (chapters 11 and 12).
- **Traceability.** Every number of the report is read from an output file and registered with its source, and the
  test period was scored once, on code frozen and audited before the final training (section 3.5, chapter 16).

Chapter 2 reviews related work. Chapter 3 describes the site, the data pipeline, the features, the feature selection, the
anti-leakage protocol and the split. Chapters 4 to 6 describe the methodology, the models and the evaluation. Chapters 7
to 12 present the results, chapter 13 discusses them and chapter 14 concludes.
""")

# ── 2. Related work (v6 text; M58) ───────────────────────────────────────────
w("## 2. Related Work\n")
w(f"""### 2.1 Statistical and Machine Learning Forecasters

Early machine-learning approaches to solar forecasting applied artificial neural networks to small, single-station
datasets {c('mellit')}. Gradient-boosted trees, in particular XGBoost {c('chen')}, introduced automatic handling of feature
interactions, built-in regularisation and monotone constraints, and are a dominant tabular method in solar forecasting
benchmarks {c('nespoli')}.

### 2.2 Deep Learning Architectures

Recurrent networks, in particular Long Short-Term Memory networks {c('hochreiter')}, are widely applied to multi-step solar
forecasting because they model sequential dependencies {c('wang')}. The bidirectional LSTM {c('graves')} processes the
sequence in both directions. The Temporal Fusion Transformer {c('lim')} combines gating, variable-selection networks and
multi-head attention to model static metadata, known future inputs and the dynamics of the target; its attention weights
describe which encoder positions it weights. PatchTST {c('nie')} divides the input series into patches that are treated as
tokens; its encoder-only, univariate design uses no exogenous variables and relies on the temporal structure of the target.

### 2.3 Two-Route Forecasting

Whether to forecast irradiance and convert it to power (the indirect route) or to forecast power directly (the direct
route) has received little systematic attention; studies that compare the routes do so on different training and test
sets {c('antonanzas', 'li')}. This study scores both routes of the same models in kW against the same measured truth.

### 2.4 Evaluation Standards

Solar forecasting studies differ in the normalisation denominator, the daytime filter and whether the metrics cover all
hours or daytime hours only {c('sweeney')}. This study uses one daytime filter (sun elevation above
{F.n('daytime threshold, degrees', 5, 'src/solarpv/evaluation/daytime.py', 'd')} degrees at the middle of the hour),
normalises by the stated capacity, and reports block-bootstrap intervals {c('efron')} and paired tests.
""")

# ── 3. Study site and data ───────────────────────────────────────────────────
w("## 3. Study Site and Data\n")
w("## 3.1 Plant description (S-01)\n")
w(f"""The plant that the data describe is a section of two inverters (inv1 and inv2) of a utility-scale park
of {o('park size, MW', '40', PM)} MW in the Southern Arava, Negev, Israel; the park has {o('park inverters', '25', PM)} inverters in all (F-12, C-30).
The target is the AC power of these two inverters. That they form the whole metered section cannot be verified from the
data (A-07), and the rating of {F.n('AC capacity, kW', CAP, 'outputs/07_feature_selection/selection_manifest.json', ',.0f')} kW
used to normalise every error is the sum of two ratings that is not verified against the inverter model (A-08); if the
rating were different, every nRMSE in percent of capacity would scale by the same factor and no ranking, test or skill
score would change. The climate is hyper-arid. Site coordinates are not published in this document (T-45).

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

w("## 3.2 Data processing\n")
w(f"""{fr('S01-01')} shows the pipeline. The one-minute export, written in local wall-clock time, is converted to true UTC
with the daylight-saving rule (F-01); the logger clocks are corrected per source and day against the solar geometry
(F-15); the minutes are averaged to hours; and an explicit cleaning policy removes impossible values and flags, without
erasing, the hours of curtailment, inverter stops and partial outages (D-22). The flagged hours stay in the primary score
and a second score leaves them out. {fr('S05-02')} shows them by month: they are rare in the first year of the data and
frequent from the second half of the data on, in the validation and test periods as well, which matters for the reading of the test results (section 7.6).
{fig('S01-01')}{fig('S05-02')}""")

w("## 3.3 Target variables\n")
w(f"""The direct target is the hourly AC power in kW. The indirect target is the clear-sky index k, the measured power
divided by the clear-sky power of the panel plane (D-11), with no clipping, so that k and kW convert exactly into each
other (F-06). The mean daytime k on train is {F.n('MEAN_DAYTIME_K (3.3)', DER['mean_daytime_k'], 'configs/derived_constants.yaml', '.6f')}.
{fr('S06-01')} shows the construction on a clear and on a cloudy day: dividing by the clear-sky power removes the daily
and seasonal shape and leaves the effect of the weather. A forecast of k is converted to kW with the clear-sky power of
the same hour, and both routes are scored in kW.
{fig('S06-01')}""")

w("## 3.4 Feature engineering\n")
w(f"""The candidate features are calendar and solar-geometry encodings, the clear-sky power, lags and rolling statistics of
the measured power at the days that each horizon may use (section 3.9), and the weather forecast. The weather forecast is
the GFS run of the day before the issuance {c('noaa')}, interpolated to the site and to the panel plane
({F.n('GFS runs', GFS['runs'] if isinstance(GFS['runs'], int) else len(GFS['runs']), 'outputs/06b_gfs/gfs_manifest.json', ',d')} runs, D-12).
{fr('S6b-01')} shows that every run used was public before the issuance: a run whose files were uploaded again after the
issuance is replaced by the earlier run of the same day (F-28).
{fig('S6b-01')}""")

w("## 3.5 End-to-end milestones (S-05)\n")
w(f"""The comparison of the six models rests on earlier steps; each step has an artifact in the repository and
controls a risk of the forecasting setting. The numbers are read from the manifests of the steps.

| Step | Artifact | Risk controlled | Key number |
|---|---|---|---|
| Problem framing (D-16) | issuance at local midnight; D+1 to D+3; two targets | ambiguous forecast origin | three horizons, two routes |
| Ingestion (stage 2) | one-minute UTC series | local time read as UTC (F-01) | {F.n('rows 1min (milestones)', CLEAN['rows_1min'], 'outputs/05_cleaning/cleaning_manifest.json', ',d')} rows |
| Logger clocks (stage 4) | clock offset per source and day | misalignment with solar geometry (F-15) | {F.n('hourly rows (milestones)', CLEAN['rows_1H'], 'outputs/05_cleaning/cleaning_manifest.json', ',d')} hourly rows |
| Cleaning (stage 5) | explicit policy, flags not erasures | spikes, frozen values, curtailment (F-14, F-17) | {F.n('op-limited daytime hours all', al['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of {F.n('daytime hours all', al['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} daytime hours flagged |
| Target and features (stage 6) | exact-inverse clear-sky index, horizon-aware lags | clipping, leaking rolling windows (F-06, F-07) | mean daytime k {F.n('MEAN_DAYTIME_K', DER['mean_daytime_k'], 'configs/derived_constants.yaml', '.6f', old='0.6279')} |
| Weather forecasts (stage 6b) | GFS runs public before issuance | inputs not known at issuance (F-28) | {F.n('GFS runs (milestones)', GFS['runs'] if isinstance(GFS['runs'], int) else len(GFS['runs']), 'outputs/06b_gfs/gfs_manifest.json', ',d')} runs |
| Feature selection (stage 7) | allow-list, train only, one validation look | selection on all periods (F-26) | {F.span('indirect inputs per horizon', min(n_sel[k] for k in n_sel if k.startswith('indirect')), max(n_sel[k] for k in n_sel if k.startswith('indirect')), PT_ + '/select/sets/model_inputs.json', 'd')} indirect, {F.span('direct inputs per horizon', min(n_sel[k] for k in n_sel if k.startswith('direct')), max(n_sel[k] for k in n_sel if k.startswith('direct')), PT_ + '/select/sets/model_inputs.json', 'd')} direct inputs |
| Evaluation layer (stage 8) | six models, three references, paired tests | unequal rows, unpaired claims | HLN and Bonferroni, minimum detectable effect |
| Gate G-1 (stage 9) | audit, reproduction from raw data, external review, tag `g1-frozen` | changes after the test is seen | every review note answered (`docs/g1_review_log.md`) |
| Final run (stage 10) | budgets of v6, three seeds for XGB and BiLSTM | unequal tuning | {F.n('GPU hours', hours_gpu, PT_ + '/report/report_steps.csv', '.1f')} GPU hours |
| Test (stage 10, phase 2) | step `test`, run once | repeated looks at the test split | scored {F.n('test scoring events', test_rows_scored, 'outputs/07_evaluation/test_access_log.csv', 'd')} time |
""")

w("## 3.6 Feature selection\n")
w(f"""The features are selected on the training period only (D-23): an allow-list of candidates, a ranking by the
importance of a fixed XGBoost inside train, and a stability floor; validation is looked at once, after the sets are fixed.
The selected set of a route is kept only if it is not worse than the full candidate pool inside train, on date folds and
on an inner hold-out ({fr('S07-01')}). The indirect route passes in every horizon and keeps its selected set; the direct
route does not pass, because its importance is concentrated in a few features, and it receives the full pool (F-29).
{fr('S07-02')} shows the resulting model inputs per route and horizon: the weather forecast is the largest group of the
indirect sets.
{fig('S07-01')}{fig('S07-02')}""")

w("## 3.7 Anti-leakage protocol\n")
w(f"""A model may see only what is known at the issuance time T0, local midnight before the first forecast day. An
allow-list, not a block-list, decides what reaches a model, and the leakage tests run on every table and model input.
{fr('S09-35')} draws, for each horizon, the window of every input relative to T0: no measured input reads an hour that
ends after T0, and the forecast days are described only by deterministic inputs (calendar, clear sky, geometry) and the
GFS run. Before the final training the code was audited end to end, reproduced from the raw data, reviewed externally and
frozen with the tag `g1-frozen` (gate G-1); the test split was then scored once (section 3.5).
{fig('S09-35')}""")

w("## 3.8 Train / Validation / Test split\n")
w(f"""The split is chronological, without shuffling ({fr('S01-02')}). Train is used to fit and select; validation to tune
and to decide; test is read once, in the final run, after the code was frozen (D-08).
{fig('S01-02')}""")

w("## 3.9 Lag exclusion rules per horizon\n")
w(f"""A forecast issued at T0 for day D+h may use only measured days up to the day before the issuance. {fr('S06-04')}
shows the lag days allowed for each horizon and the same-hour rolling windows, which always end on the last day before
the issuance, whatever the horizon (F-07).
{fig('S06-04')}""")

# ── 4. Methodology ───────────────────────────────────────────────────────────
w("## 4. Methodology\n")
w(f"""### 4.1 Hyperparameter optimisation

The tuned models are optimised with Optuna {c('akiba')} (TPE sampler). The objective of every study is the nRMSE in kW
over the daytime hours of validation (S-14). XGB-Opt and BiLSTM-Opt have
{F.n('XGB trials per study', BUD['xgb']['trials'], PT_ + '/config.yaml', 'd')} trials per study and
{F.n('XGB seeds', len(BUD['xgb']['seeds']), PT_ + '/config.yaml', 'd')} seeds; TFT-96 and TFT-168 have
{F.n('TFT trials per study', BUD['tft96']['trials'], PT_ + '/config.yaml', 'd')} trials per route and one seed; the TFT
baseline and PatchTST have fixed settings. The final run holds
{F.n('studies in the final run (4.1)', len(STUDIES), PT_ + '/report/report_studies.csv', 'd')} studies. These are the
budgets of the submitted study; none was changed after the test split was read (D-37).

### 4.2 Statistical evaluation

Intervals are {F.n('interval level, %', 95, 'src/solarpv/evaluation/metrics.py', 'd')} percent percentile intervals from a
circular block bootstrap {c('efron')} of the squared errors, with blocks of
{F.n('bootstrap block, daytime hours', BOOTSTRAP_BLOCK, 'src/solarpv/config/constants.py', 'd')} consecutive daytime hours
and {F.n('bootstrap replicates', BOOTSTRAP_REPS, 'src/solarpv/config/constants.py', ',d')} replicates. Two models are compared with the
Diebold-Mariano test {c('diebold')} on the squared errors in kW, with the small-sample correction of Harvey, Leybourne and
Newbold {c('harvey')}, a Newey-West variance {c('newey')} with a bandwidth of
{F.n('Newey-West lags (4.2)', int(DMT['nw_lags'].iloc[0]), PT_ + '/report/report_dm_kw_test.csv', 'd')} hours, and a
Bonferroni correction over the {N6} cells (target x horizon) of a split and comparison. Each observed difference is set
beside the minimum detectable effect of the paired test at
{F.n('statistical power of the MDE, %', 100 * 0.8, 'src/solarpv/evaluation/stats_tests.py', '.0f')} percent power.
""")

# ── 5. Model architectures ───────────────────────────────────────────────────
w("## 5. Model Architectures\n")
w(f"""{fr('S11-16')} compares the six models: their inputs, budgets and training time in the final run.

- **XGB-Opt** {c('chen')}: gradient-boosted trees on the selected tabular inputs, with a monotone constraint on the
  clear-sky input (D-15); one model per target and horizon; on the CPU.
- **BiLSTM-Opt** {c('graves')}: a bidirectional encoder-decoder over the hourly sequence with a quantile loss; one
  forecast origin per local day (D-25).
- **TFT-96 and TFT-168** {c('lim')}: the Temporal Fusion Transformer with an encoder of 96 or 168 hours and the known
  future inputs of the decoder (D-26), tuned per route.
- **TFT baseline**: the same architecture with fixed settings.
- **PatchTST** {c('nie')}: a patch-based, univariate transformer that sees only the past of the target.

The five neural models took {F.n('GPU hours (5)', hours_gpu, PT_ + '/report/report_steps.csv', '.1f')} hours of GPU time in
the final run.
{fig('S11-16')}""")

# ── 6. Evaluation framework ──────────────────────────────────────────────────
w("## 6. Evaluation Framework\n")
w(f"""### 6.1 Primary metric

The primary metric is the nRMSE in kW, the root mean squared error of the forecast power in percent of the AC capacity,
over the daytime hours of the split with a measured output. An hour is a daytime hour when the sun is above the threshold
at the middle of the hour ({fr('S08-01')}); every model and reference is scored on exactly these hours. The indirect route
is converted to kW before it is scored.
{fig('S08-01')}
### 6.2 Secondary metrics

The MAE, the R², the skill score against persistence and the nRMSE normalised by the mean daytime power (for comparison
with studies that normalise that way) are reported in the tables of the run. A second view of every metric leaves out the
operation-limited hours (the hours of normal operation).

### 6.3 Reference forecasts

Three references are scored on the same hours: persistence (the same hour of the last measured day, safe for daylight
saving time), climatology (the mean of the same calendar hour on train), and the clear-sky power times the mean daytime k.

### 6.4 Confidence intervals and significance

Intervals and paired tests follow section 4.2. Overlapping marginal intervals do not imply that a paired test is not
significant {c('schenker')}, so the comparisons of models rest on the paired test.
""")

# ── 7. Main results ──────────────────────────────────────────────────────────
w("## 7. Main Results\n")
w(f"""### 7.1 Benchmark overview

{fr('S11-01')} shows the benchmark in kW with {F.n('interval level, % (7.1)', 95, 'src/solarpv/evaluation/metrics.py', 'd')} percent intervals, for all six models and the three references, on
validation and on test. On test the error of the six models is
{F.span('six models, test, range (7.1)', KT.loc[M6].min(), KT.loc[M6].max(), BTF)} percent and on validation
{F.span('six models, validation, range (7.1)', KV.loc[M6].min(), KV.loc[M6].max(), BVF)} percent. The width of a test
interval is {F.span('width of the test intervals of the six models, pp', CIW.min(), CIW.max(), 'outputs/11_analyses/benchmark_ci_kw.csv')}
points, and the intervals of XGB-Opt, BiLSTM-Opt and the three TFT models overlap in every cell.
{fig('S11-01')}
### 7.2 Indirect target results

At D+1 of the indirect route the seed-mean nRMSE on test is
{F.n('XGB test indirect D1', KT[('xgb_sota','indirect',1)], BTF, '.2f', old='29.09')} for XGB-Opt,
{F.n('BiLSTM test indirect D1', KT[('lstm_sota','indirect',1)], BTF)} for BiLSTM-Opt,
{F.n('TFT-96 test indirect D1', KT[('tft_96','indirect',1)], BTF)} for TFT-96,
{F.n('TFT-168 test indirect D1', KT[('tft_168','indirect',1)], BTF)} for TFT-168,
{F.n('TFT baseline test indirect D1', KT[('tft','indirect',1)], BTF)} for the TFT baseline and
{F.n('PatchTST test indirect D1', KT[('patchtst','indirect',1)], BTF)} for PatchTST; on validation the same numbers are
{F.n('XGB val indirect D1', KV[('xgb_sota','indirect',1)], BVF)},
{F.n('BiLSTM val indirect D1', KV[('lstm_sota','indirect',1)], BVF)},
{F.n('TFT-96 val indirect D1', KV[('tft_96','indirect',1)], BVF)},
{F.n('TFT-168 val indirect D1', KV[('tft_168','indirect',1)], BVF)},
{F.n('TFT baseline val indirect D1', KV[('tft','indirect',1)], BVF)} and
{F.n('PatchTST val indirect D1', KV[('patchtst','indirect',1)], BVF)}. The references are
{F.n('climatology test indirect D1', KT[('climatology','indirect',1)], BTF)} (climatology) and
{F.n('persistence test indirect D1', KT[('persistence','indirect',1)], BTF)} (persistence) on test, and
{F.n('climatology val indirect D1', KV[('climatology','indirect',1)], BVF)} and
{F.n('persistence val indirect D1', KV[('persistence','indirect',1)], BVF)} on validation.

### 7.3 Direct target results

At D+1 of the direct route the seed-mean nRMSE on test is
{F.n('XGB test direct D1', KT[('xgb_sota','direct',1)], BTF)} for XGB-Opt,
{F.n('BiLSTM test direct D1', KT[('lstm_sota','direct',1)], BTF)} for BiLSTM-Opt,
{F.n('TFT-96 test direct D1', KT[('tft_96','direct',1)], BTF)} for TFT-96,
{F.n('TFT-168 test direct D1', KT[('tft_168','direct',1)], BTF)} for TFT-168,
{F.n('TFT baseline test direct D1', KT[('tft','direct',1)], BTF)} for the TFT baseline and
{F.n('PatchTST test direct D1', KT[('patchtst','direct',1)], BTF)} for PatchTST (validation:
{F.n('XGB val direct D1', KV[('xgb_sota','direct',1)], BVF)},
{F.n('BiLSTM val direct D1', KV[('lstm_sota','direct',1)], BVF)},
{F.n('TFT-96 val direct D1', KV[('tft_96','direct',1)], BVF)},
{F.n('TFT-168 val direct D1', KV[('tft_168','direct',1)], BVF)},
{F.n('TFT baseline val direct D1', KV[('tft','direct',1)], BVF)} and
{F.n('PatchTST val direct D1', KV[('patchtst','direct',1)], BVF)}). On test, XGB-Opt has the lowest seed-mean error of
the six models in {F.n('cells where XGB lowest, test', LOW_T, BTF, 'd')} of {N6} cells and on validation in
{F.n('cells where XGB lowest, validation', LOW_V, BVF, 'd')} of {N6}; the order of the five leading models is not a
result (chapter 8). {fr('S11-03')} shows the forecast against the observed power on the test split.
{fig('S11-03')}
### 7.4 Common-mask evaluation

Every model and reference is scored on the same
{F.n('common-mask hours, test', int(CMS['n'].iloc[0]), PT_ + '/test/test/common_mask_summary.csv', ',d')} daytime hours of
the test split; no model drops an hour, so the common-mask scores equal the primary scores (S-25).

### 7.5 Horizon robustness

{fr('S11-02')} shows the error against the horizon. The change from D+1 to D+3 is at most
{F.n('largest D3 minus D1, pp', hz['D3_minus_D1_pp'].abs().max(), 'outputs/11_analyses/horizon_robustness.csv')} points
for the six models and two routes. It is negative in
{F.n('pairs with D3<D1, validation', int((hz[hz['split']=='val']['D3_minus_D1_pp']<0).sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')}
of {F.n('pairs for horizons', int((hz['split']=='val').sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')} pairs on
validation and positive in
{F.n('pairs with D3>D1, test', int((hz[hz['split']=='test']['D3_minus_D1_pp']>0).sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')}
of {F.n('pairs for horizons (test)', int((hz['split']=='test').sum()), 'outputs/11_analyses/horizon_robustness.csv', 'd')}
on test: the three horizons are close (the weather forecast covers all three days, and the history groups add little,
section 12.5), and their order is a property of the period (R6b-07, R7-05), not of a model.
{fig('S11-02')}
### 7.6 From validation to test

Every model and reference has a larger error on test than on validation, by about
{F.n('mean rise val to test, pp', float(KT.loc[M6 + REFS].mean() - KV.loc[M6 + REFS].mean()), BTF, '.1f')} points on
average over the six models and the three references. {fr('S10-03')} separates the rise: for XGB-Opt the mean nRMSE goes
from {F.n('XGB all hours, validation', XV[('val','all')], MKF)} to {F.n('XGB all hours, test', XV[('test','all')], MKF)}
on all daytime hours and from {F.n('XGB normal operation, validation', XV[('val','normal_operation')], MKF)} to
{F.n('XGB normal operation, test', XV[('test','normal_operation')], MKF)} in the hours of normal operation; for
climatology from {F.n('climatology all hours, validation', CV_[('val','all')], MKF)} to
{F.n('climatology all hours, test', CV_[('test','all')], MKF)} and from
{F.n('climatology normal operation, validation', CV_[('val','normal_operation')], MKF)} to
{F.n('climatology normal operation, test', CV_[('test','normal_operation')], MKF)}. The
{F.n('test op-limited hours (7.6)', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')}
operation-limited hours of the test split account for
{F.n('share of the rise of XGB-Opt in the op-limited hours, %', 100 * RISE_SHARE['xgb_sota'], MKF, '.0f')} percent of the rise of
XGB-Opt, and for {F.span('share of the rise in the op-limited hours, all models and references, %', 100 * RISE_SHARE.min(), 100 * RISE_SHARE.max(), MKF, '.0f')}
percent across the models and references; the rest is a change of the period that every model and reference shares. The comparisons of models are paired on the same hours.
{fig('S10-03')}""")

# ── 8. Statistical significance ──────────────────────────────────────────────
w("## 8. Statistical Significance Analysis\n")
w(f"""### 8.1 Diebold-Mariano results

{fr('S10-04')} shows the paired tests in kW of every model and reference against XGB-Opt on test. XGB-Opt is significantly
better than clear sky x k and persistence in all {N6} cells on test and validation, and than PatchTST in every cell. It is
significantly better than the BiLSTM in
{F.n('BiLSTM significant cells, test', sig_neural(DMT, 'lstm_sota'), PT_ + '/report/report_dm_kw_test.csv', 'd')} of {N6}
cells on test, and than the three TFT models in
{F.n('TFT-96 significant cells, test', sig_neural(DMT, 'tft_96'), PT_ + '/report/report_dm_kw_test.csv', 'd')},
{F.n('TFT-168 significant cells, test', sig_neural(DMT, 'tft_168'), PT_ + '/report/report_dm_kw_test.csv', 'd')} and
{F.n('TFT baseline significant cells, test', sig_neural(DMT, 'tft'), PT_ + '/report/report_dm_kw_test.csv', 'd')} of {N6}
cells (TFT-96, TFT-168 and the TFT baseline). The leading models are not separated, and no model is declared the
winner (D-37).
{fig('S10-04')}
### 8.2 Statistical power analysis

Against climatology XGB-Opt is significant in
{F.n('climatology significant cells, test', int(CLIM_T['significant'].sum()), PT_ + '/report/report_dm_kw_test.csv', 'd')}
of {F.n('climatology cells', len(CLIM_T), PT_ + '/report/report_dm_kw_test.csv', 'd')} cells on test and in
{F.n('climatology significant cells, validation', int(CLIM_V['significant'].sum()), PT_ + '/report/report_dm_kw_val.csv', 'd')}
of {F.n('climatology cells (validation)', len(CLIM_V), PT_ + '/report/report_dm_kw_val.csv', 'd')} on validation. On test the
observed difference is {F.span('climatology effect, percent of MDE', cm.min(), cm.max(), PT_ + '/report/report_dm_kw_test.csv', '.0f')}
percent of the minimum detectable effect: with one section of a plant,
{F.n('test hours (power)', ss['daytime_valid_target'], 'outputs/03_eda/tables/table_splits.csv', ',d')} test hours and
autocorrelated errors, the test cannot separate differences of this size. The report therefore states the margin over
climatology as the observed difference and its share of the minimum detectable effect, and not as a significant result.
""")

# ── 9. Seasonal and temporal ─────────────────────────────────────────────────
w("## 9. Seasonal and Temporal Analysis\n")
lim_fall_t = sx.loc[("Fall", "test"), "op_limited_share"]
lim_fall_v = sx.loc[("Fall", "val"), "op_limited_share"]
w(f"""### 9.1 Seasonal performance

{fr('S11-04')} shows the error by season on validation and on test. The test period (October to May) has no summer, and a
season is not the same months in the two periods: the autumn of the test split is {sx.loc[('Fall','test'),'months']} and the
autumn of the validation split is {sx.loc[('Fall','val'),'months']}. For XGB-Opt (indirect, D+1) the error in the autumn is
{F.n('XGB Fall validation, %', sx.loc[('Fall','val'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} percent
on validation and {F.n('XGB Fall test, %', sx.loc[('Fall','test'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')}
percent on test; for climatology it is
{F.n('climatology Fall validation, %', cx.loc[('Fall','val'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')}
and {F.n('climatology Fall test, %', cx.loc[('Fall','test'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')}
percent. In the autumn
{F.n('Fall op-limited share test, %', 100*lim_fall_t, 'outputs/11_analyses/seasonal_kw.csv', '.0f')} percent of the test
hours are operation-limited against
{F.n('Fall op-limited share validation, %', 100*lim_fall_v, 'outputs/11_analyses/seasonal_kw.csv', '.0f')} percent of the
validation hours. In spring the two periods are close
({F.n('XGB Spring validation, %', sx.loc[('Spring','val'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} and
{F.n('XGB Spring test, %', sx.loc[('Spring','test'),'nRMSE_kW_pct'], 'outputs/11_analyses/seasonal_kw.csv', '.1f')} percent).
{fig('S11-04')}
### 9.2 Hourly error by season

{fr('S11-05')} shows the error by local hour and season on the test split for all six models and the references, with
cells of fewer than twenty hours masked. {fr('S11-17')} shows the same in the hours of normal operation: the peak of the
autumn hours falls to the level of the other cells, so it is made mostly by the operation-limited hours.
{fig('S11-05')}{fig('S11-17')}""")

# ── 10. Route ────────────────────────────────────────────────────────────────
w("## 10. Forecast Route Analysis\n")
w(f"""Both routes are scored in kW against the measured AC power on the same hours ({fr('S11-06')}). The direct route has the
lower error in {F.n('direct better pairs, validation', DIRECT_BETTER_V, 'evaluate/val/route_comparison.csv; test/test/route_comparison.csv', 'd')}
of {F.n('model-horizon pairs', len(RPV), 'evaluate/val/route_comparison.csv', 'd')} pairs of a model and a horizon on
validation and in {F.n('direct better pairs, test', DIRECT_BETTER_T, PT_ + '/test/test/route_comparison.csv', 'd')} of
{F.n('model-horizon pairs (test)', len(RPT), PT_ + '/test/test/route_comparison.csv', 'd')} on test. The differences range
from {F.n('route difference min, pp', rp_all.min(), PT_ + '/test/test/route_comparison.csv', '+.2f')} to
{F.n('route difference max, pp', rp_all.max(), PT_ + '/test/test/route_comparison.csv', '+.2f')} points (direct minus
indirect), and the sign differs between the two periods for
{F.n('pairs with a changed sign', SIGN_DIFF, PT_ + '/test/test/route_comparison.csv', 'd')} of the
{F.n('pairs (sign)', len(RPV), PT_ + '/test/test/route_comparison.csv', 'd')} pairs. The two routes are equivalent within the
seed and period uncertainty, and the choice between them can follow operational convenience.
{fig('S11-06')}""")

# ── 11. Interpretability ─────────────────────────────────────────────────────
w("## 11. Interpretability Analysis\n")
w(f"""### 11.1 XGBoost SHAP: feature-group importance

{fr('S11-07')} shows what the SHAP values {c('lundberg2017')} of XGB-Opt, computed exactly for the trees with
TreeSHAP {c('lundberg2020')}, attribute the forecast to, on validation and on test. The
weather-forecast group holds
{F.span('GFS share of mean absolute SHAP, validation, %', *share_range(SHV, 'nwp_gfs'), PT_ + '/explain/xgb/xgb_shap_groups.csv', '.0f')}
percent of the mean absolute SHAP value on validation and
{F.span('GFS share of mean absolute SHAP, test, %', *share_range(SHT, 'nwp_gfs'), PT_ + '/test/shap_test/xgb_shap_groups_test.csv', '.0f')}
percent on test, and it is the largest group in every cell on both splits. The rank correlation of the feature importances
between two horizons or two seeds is at least
{F.n('rank correlation minimum, validation', RHO_V, PT_ + '/explain/xgb/xgb_shap_rank_stability.csv')} on validation and
{F.n('rank correlation minimum, test', RHO_T, PT_ + '/test/shap_test/xgb_shap_rank_stability_test.csv')} on test (mean over
seeds): the attribution is stable.
{fig('S11-07')}
### 11.2 TFT temporal attention

{fr('S11-08')} shows how the attention of the three TFT models is spread over the encoder window. Attention is a
description of the model and not evidence of use. In the six cases of target and model, the weight on the clock hour of
the forecast hour is {F.span('same-hour attention ratio', a_all['same_hour_ratio'].min(), a_all['same_hour_ratio'].max(), PT_ + '/explain/tft/tft_attention_summary.csv')}
times a uniform attention, and the weight on the oldest encoder position is
{F.span('first-position attention ratio', a_all['first_position_ratio'].min(), a_all['first_position_ratio'].max(), PT_ + '/explain/tft/tft_attention_summary.csv')}
times uniform: the attention follows the daily cycle.
{fig('S11-08')}
### 11.3 TFT encoder length

{fr('S11-09')} compares the encoder of 168 hours with that of 96 hours in a paired test per cell. The TFT-168 minus TFT-96
difference lies between {F.n('encoder diff validation min, pp', ENC[ENC['split']=='val']['diff_a_minus_b_pp'].min(), 'outputs/11_analyses/encoder_length.csv', '+.2f')}
and {F.n('encoder diff validation max, pp', ENC[ENC['split']=='val']['diff_a_minus_b_pp'].max(), 'outputs/11_analyses/encoder_length.csv', '+.2f')}
pp on validation, with no significant cell, and between
{F.n('encoder diff test min, pp', dm_enc['diff_a_minus_b_pp'].min(), 'outputs/11_analyses/encoder_length.csv', '+.2f')} and
{F.n('encoder diff test max, pp', dm_enc['diff_a_minus_b_pp'].max(), 'outputs/11_analyses/encoder_length.csv', '+.2f')} pp on
test, with {F.n('encoder significant cells', int(ENC['significant'].sum()), 'outputs/11_analyses/encoder_length.csv', 'd')}
of {F.n('encoder cells', len(ENC), 'outputs/11_analyses/encoder_length.csv', 'd')} cells significant. No consistent
difference between the two encoder lengths is found; each model was tuned separately with one seed.
{fig('S11-09')}
### 11.4 PatchTST: the autoregressive structure

PatchTST sees only the past of the target ({fr('S11-14')}). It is worse than XGB-Opt by
{F.span('PatchTST minus XGB, pp', gap_x.min(), gap_x.max(), 'outputs/11_analyses/patchtst_gap.csv')} pp, significantly in
{F.n('PatchTST vs XGB significant cells', int(PT_X['significant'].sum()), 'outputs/11_analyses/patchtst_gap.csv', 'd')} of
{F.n('PatchTST vs XGB cells', len(PT_X), 'outputs/11_analyses/patchtst_gap.csv', 'd')} cells (validation and test, two routes,
three horizons), and worse than climatology by
{F.span('PatchTST minus climatology, pp', PT_C['diff_a_minus_b_pp'].min(), PT_C['diff_a_minus_b_pp'].max(), 'outputs/11_analyses/patchtst_gap.csv')}
pp, significantly in {F.n('PatchTST vs climatology significant cells', int(PT_C['significant'].sum()), 'outputs/11_analyses/patchtst_gap.csv', 'd')}
of {F.n('PatchTST vs climatology cells', len(PT_C), 'outputs/11_analyses/patchtst_gap.csv', 'd')}. The history of the series
carries the daily and seasonal cycle that climatology already holds; the information of the forecast day comes from the
weather forecast, which PatchTST does not see.
{fig('S11-14')}
### 11.5 Optuna convergence and hyperparameter importance (TFT-96)

{fr('S11-10')} shows how the search of TFT-96 converged and which hyperparameters mattered. The selected hidden sizes are
{F.n('TFT-96 hidden indirect', hs96[0], PT_ + '/tft96/best_params.json', 'd')} (indirect) and
{F.n('TFT-96 hidden direct', hs96[1], PT_ + '/tft96/best_params.json', 'd')} (direct) for TFT-96 and
{F.n('TFT-168 hidden indirect', hs168[0], PT_ + '/tft168/best_params.json', 'd')} and
{F.n('TFT-168 hidden direct', hs168[1], PT_ + '/tft168/best_params.json', 'd')} for TFT-168.
{fig('S11-10')}""")

# ── 12. Robustness and stability ─────────────────────────────────────────────
w("## 12. Robustness and Stability Analysis\n")
cvx = CVT
w(f"""### 12.1 Seed variability

Each seed of XGB-Opt and BiLSTM-Opt is scored by the evaluation layer on the same hours ({fr('S10-05')}). The largest
standard deviation over the three seeds is
{F.n('XGB seed std max, pp', sp_x.max(), PT_ + '/report/report_seed_spread.csv', '.3f')} points for XGB-Opt and
{F.n('BiLSTM seed std max, pp', sp_l.max(), PT_ + '/report/report_seed_spread.csv', '.3f')} points for BiLSTM-Opt. The
TFT models and PatchTST have one seed each.
{fig('S10-05')}
### 12.2 Cross-validation (five-fold expanding window)

{fr('S10-06')} compares the five-fold expanding-window cross-validation of XGB-Opt on train with validation and test. The
test error is above the cross-validation mean in every fit, by
{F.span('test minus CV mean, pp', cvx['test_minus_cv_pp'].min(), cvx['test_minus_cv_pp'].max(), PT_ + '/report/report_cv_vs_test.csv', '.1f')}
points, and validation lies below it, by
{F.span('CV mean minus validation, pp', -cvx['validation_minus_cv_pp'].max(), -cvx['validation_minus_cv_pp'].min(), PT_ + '/report/report_cv_vs_test.csv', '.1f')}
points; the folds themselves differ by a mean standard deviation of
{F.n('mean CV std, pp', cvx['cv_std'].mean(), PT_ + '/report/report_cv_vs_test.csv', '.1f')} points. The difference between
the periods is shared by the references (section 7.6).
{fig('S10-06')}
### 12.3 Error decomposition

{fr('S11-11')} splits the squared error into the squared mean error and the variance of the error (MSE = ME² + Var(e)).
Every model has a positive mean error on all daytime hours of the test split:
{F.span('mean error all hours, kW', ME_ALL.min(), ME_ALL.max(), 'outputs/11_analyses/decomposition.csv', '+.0f')} kW across
the six models. On the hours of normal operation it is
{F.span('mean error normal operation, kW', ME_NORM[NO_PT].min(), ME_NORM[NO_PT].max(), 'outputs/11_analyses/decomposition.csv', '+.0f')}
kW for XGB-Opt, BiLSTM-Opt and the three TFT models, and
{F.n('PatchTST mean error normal operation, kW', ME_NORM['patchtst'], 'outputs/11_analyses/decomposition.csv', '+.0f')} kW
for PatchTST (F-30): the positive bias is made in the
{F.n('test op-limited hours (bias)', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')}
operation-limited hours (curtailment, stops, derate; D-22), which neither the weather nor the history describes. Bias
squared is at most
{F.n('bias2 share max, validation, %', 100*D[(D['view']=='all')&(D['split']=='val')&(D['model'].isin(NO_PT))].groupby('model')['bias2_share'].mean().max(), 'outputs/11_analyses/decomposition.csv', '.1f')}
percent of the squared error on validation and
{F.n('bias2 share max, test, %', 100*D[(D['view']=='all')&(D['split']=='test')&(D['model'].isin(NO_PT))].groupby('model')['bias2_share'].mean().max(), 'outputs/11_analyses/decomposition.csv', '.1f')}
percent on test (mean over the cells) for the five models other than PatchTST, and the variance between the three seeds is at most
{F.n('seed variance share max, %', 100*DSD[DSD['view']=='all'].groupby(['model', 'split'])['seed_variance_share'].mean().max(), 'outputs/11_analyses/decomposition_seeds.csv', '.1f')}
percent of the squared error: the error is almost entirely variance.
{fig('S11-11')}
### 12.4 Production bins (XGB-Opt, direct, D+1)

{fr('S11-12')} shows the mean error of XGB-Opt by production quintile, binned in two ways, with edges fixed on validation.
Binned by the observed power, the mean error is
{F.n('quintile 1 observed, kW', QX[(QX['binned_by']=='observed')&(QX['bin']==1)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')}
kW in the lowest bin and
{F.n('quintile 5 observed, kW', QX[(QX['binned_by']=='observed')&(QX['bin']==5)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')}
kW in the highest; binned by the forecast, which is known at issuance, it is
{F.n('quintile 1 forecast, kW', QX[(QX['binned_by']=='forecast')&(QX['bin']==1)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')}
and {F.n('quintile 5 forecast, kW', QX[(QX['binned_by']=='forecast')&(QX['bin']==5)]['me_kw'].iloc[0], 'outputs/11_analyses/quintiles.csv', '+.0f')}
kW. A control forecast made unbiased on validation gives
{F.n('control quintile 1, kW', QCX[QCX['bin']==1]['me_calibrated_kw'].iloc[0], 'outputs/11_analyses/quintile_control.csv', '+.0f')}
and {F.n('control quintile 5, kW', QCX[QCX['bin']==5]['me_calibrated_kw'].iloc[0], 'outputs/11_analyses/quintile_control.csv', '+.0f')}
kW in the observed bins. Binning by the outcome puts positive errors in the low bins and negative errors in the high bins
for any forecast that regresses to the mean, so the profile of the observed bins is made by the binning and is not stated
as a property of the plant or of the model (M45).
{fig('S11-12')}
### 12.5 Feature-group ablation

{fr('S11-13')} shows the ablation of XGB-Opt: each feature group is removed and the model refitted with three seeds.
Removing the weather-forecast (GFS) group raises the error by
{F.span('ablation no_nwp_gfs validation, pp', ABV[ABV['mode']=='no_nwp_gfs']['mean_delta_pp'].min(), ABV[ABV['mode']=='no_nwp_gfs']['mean_delta_pp'].max(), PT_ + '/report/report_ablation.csv')}
pp on validation and by
{F.span('ablation no_nwp_gfs test, pp', ABT[ABT['mode']=='no_nwp_gfs']['mean_delta_pp'].min(), ABT[ABT['mode']=='no_nwp_gfs']['mean_delta_pp'].max(), PT_ + '/report/report_ablation_test.csv')}
pp on test. The largest change of any other group on validation is
{F.n('ablation other groups max validation, pp', ABV[~ABV['mode'].isin(['minimal','no_nwp_gfs'])]['mean_delta_pp'].abs().max(), PT_ + '/report/report_ablation.csv')}
pp, close to the seed margin (M41).
{fig('S11-13')}""")

# ── 13. Discussion ───────────────────────────────────────────────────────────
w("## 13. Discussion (S-42)\n")
w("""### 13.1 The value of feature engineering

The ablation (section 12.5), the SHAP attributions (section 11.1) and PatchTST (section 11.4) agree: the value of the
engineered inputs sits mainly in the issued weather forecast. Without it XGB-Opt loses about a point; PatchTST, which sees
only the history, does not reach climatology. The history groups add little once the forecast is present. The comparison
with PatchTST is not a bound on the value of feature engineering: the models differ in design as well as in inputs, and
PatchTST has one seed.

### 13.2 The route decision

The two routes are equivalent within the uncertainty (chapter 10). The indirect route has a target whose daily and
seasonal shape is removed, and the direct route lets the model learn that shape from the clear-sky input; with the issued
weather forecast as the main input both reach the same error. The choice can follow operational convenience.

### 13.3 Architecture insights

XGB-Opt, BiLSTM-Opt and the three TFT models are close (chapters 7 and 8), and the order among them changes between the
periods. A tree model on the selected tabular inputs reaches the level of the sequence models at a fraction of their
training time (section 5). The encoder length of the TFT does not matter at this site (section 11.3), and the attention
follows the daily cycle (section 11.2).

### 13.4 Statistical power

The differences between the leading models are below the minimum detectable effect of one section and one test period,
and so is the margin of XGB-Opt over climatology on test (section 8.2). Results of this kind should be reported with
intervals and with the effect as a share of the minimum detectable effect; pooling sites would give the paired tests the
power that one section does not have.

### 13.5 Seasonal generalisability

The test period has no summer, so the results describe autumn, winter and spring. The rise of the error in the autumn of
the test period is a property of the period and of its operation-limited hours, and the references rise by a similar
amount (section 9.1).

### 13.6 Modelling coordinates

All clear-sky and solar-position calculations use the coordinates of the solar field itself (D-05). They are kept in a
private configuration and are not published in this document (T-45); the daytime mask is the same under either location
(F-02).
""")

# ── 14. Conclusions ──────────────────────────────────────────────────────────
w("## 14. Conclusions (S-43)\n")
w("""This study compares six models at three horizons and two routes on one section of a plant, from an end-to-end pipeline
with issuance-time inputs, selection on train only, identical scored hours, a frozen code base and one look at the test
period.

C1. Ranking. No model is declared the best (D-37). XGB-Opt, BiLSTM-Opt and the three TFT models lie within the
intervals of each other; PatchTST is clearly behind (sections 7.1 to 7.3, chapter 8).

C2. Route. The indirect and the direct route are equivalent within the seed and period uncertainty (chapter 10).

C3. Horizon. The three horizons are close, and their order is a property of the period, not of a model (section 7.5).

C4. Autoregression. A model that sees only the history of the target does not reach climatology; the information of the
forecast day comes from the weather forecast (section 11.4).

C5. Encoder length. No consistent difference is found between the 96-hour and the 168-hour encoders of the TFT
(section 11.3).

C6. Interpretability. The weather-forecast group dominates the XGB attributions, stably across horizons and seeds; the TFT
attention follows the daily cycle and is read as a description (sections 11.1 and 11.2).

C7. Statistical power. Differences between the leading models, and the margin of XGB-Opt over climatology on test, are
close to or below the minimum detectable effect (section 8.2).

C8. Production regimes. The mean error by production bin depends on how the hours are binned; binned by the outcome it
shows a profile that a control forecast shares, so the profile is made by the binning and is not stated as a property of
the plant or of the model (section 12.4). The positive mean error of the models is made in the operation-limited hours
(section 12.3).

### Practical recommendations

- Report both views of the error, all daytime hours and hours of normal operation, because the operation-limited
  hours carry most of the rise from validation to test.
- Compare models with a paired test in kW and report the effect as a share of the minimum detectable effect.
- Use the weather forecast issued before the forecast time: it is the group that matters most in the ablation
  and in the attributions.
- Either route can be used; the choice can follow operational convenience.

### Future work

- A forecast product of higher resolution than the one used (GFS), and quarter-hour data.
- Probabilistic forecasts: the quantile outputs of the neural models are stored (D-33); their coverage is not
  analysed here.
- Several sites, to give the paired tests the power that one section does not have.
- Dust events: the project holds no aerosol record; an aerosol product would allow an analysis of dust days (Q-29).
- Training that prevents crossing quantiles of the TFT (D-33), and a longer search for the TFT models (D-37).
""")

# ── 15. Disclosures ──────────────────────────────────────────────────────────
w("## 15. Academic Disclosures (S-44)\n")
w(f"""D1. Operation-limited hours. The hours of curtailment, stops and derate are flagged and not erased (D-22):
{F.n('test op-limited (D1)', ss['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of the
{F.n('test daytime hours (D1)', ss['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} daytime hours of the
test split and {F.n('val op-limited (D1)', vs['daytime_op_limited'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of
{F.n('val daytime hours (D1)', vs['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} of the validation split.
They are in the primary score and a second score leaves them out. The models do not predict them, and they account for
most of the positive mean error (section 12.3).

D2. Search spaces. {F.n('studies with a value on the edge', n_edge, PT_ + '/report/report_g1_points.csv', 'd')} of the
{F.n('studies in the final run', len(STUDIES), PT_ + '/report/report_studies.csv', 'd')} studies of the final run have a best
value on the edge of the search space; the search spaces are those of the submitted study and were not widened after the
test split was read (D-37).

D3. Internal and evaluation metrics. The tuning objective of every study is the nRMSE in kW over the daytime hours of
validation (S-14); the BiLSTM stops early on its quantile loss. The final fit of the BiLSTM is worse than its best trial
by {F.span('BiLSTM final fit minus best trial, pp', gap.min(), gap.max(), PT_ + '/report/report_g1_points.csv')} points,
and the best step of PatchTST is
{F.n('PatchTST best step direct', int(ptst[0].astype(int).min()), PT_ + '/report/report_g1_points.csv', ',d')} to
{F.n('PatchTST best step indirect', int(ptst[0].astype(int).max()), PT_ + '/report/report_g1_points.csv', ',d')} of a budget of
{F.n('PatchTST step budget', int(ptst[1].astype(int).iloc[0]), PT_ + '/report/report_g1_points.csv', ',d')} steps.
{F.n('XGB trials at their cap', int(cap_n), PT_ + '/report/report_g1_points.csv', 'd')} of
{F.n('XGB trials', int(cap_total), PT_ + '/report/report_g1_points.csv', ',d')} XGB trials stopped at their cap. None of the
budgets, seeds or rules was changed after the test split was read (D-37).

D4. Direction of the paired test. A positive difference in the tables of `dm_tests_kw.csv` means that the model in
the first column has the larger squared error than XGB-Opt. The tests use the HLN correction and Bonferroni over the
{N6} cells of a split and comparison, with a Newey-West bandwidth of
{F.n('Newey-West lags', int(DMT['nw_lags'].iloc[0]), PT_ + '/report/report_dm_kw_test.csv', 'd')} hours.

D5. Seeds. XGB-Opt and BiLSTM-Opt have three seeds, each scored by the evaluation layer on the same hours; the TFT
models and PatchTST have one seed each, so no seed interval exists for them (section 12.1).

D6. One section of one plant. The data are two inverters of a park (A-07), the rating is not verified (A-08), the
test period has no summer, and {F.n('test hours polled or mixed (D6)', POLLED, 'outputs/03_eda/tables/table_splits.csv', ',d')}
of its {F.n('test daytime hours (D6)', ss['daytime_hours'], 'outputs/03_eda/tables/table_splits.csv', ',d')} daytime hours
come from a second data source whose hourly means rest on four snapshots (F-18). The validation numbers are tuning-split
numbers (D-10).
""")

# ── 16. Data and code availability (S-45) ────────────────────────────────────
w("## 16. Data and Code Availability\n")
w(f"""The raw SCADA data are commercially sensitive and are not released; the site coordinates are kept in a private
configuration (T-45). The code, the configurations, the split definitions and the frozen final run are in the repository
at the tag `g1-frozen`. Two scripts rebuild the results: `scripts/reproduce_paper_results.py` runs the pipeline from the
raw data, and `scripts/make_tables_from_frozen.py` rebuilds
the tables, the figures and this report from the frozen forecasts without training and without scoring the test split
again ({fr('S11-19')}). At gate G-1 the reproduction from the raw data gave the same forecasts as the reference runs for
every model, and every file that differed had a documented reason (X99).
{fig('S11-19')}""")

# ── 17. References ───────────────────────────────────────────────────────────
w("## 17. References\n")
TEXT_BEFORE_REFS = "\n".join(sec)
assert CITED == set(REFS_V6), f"entries of the list that are not cited: {sorted(set(REFS_V6) - CITED)}"
w("\n".join(f"- {o(f'reference {k}', REFS_V6[k][1])}" for k in sorted(REFS_V6)) + "\n")

w("## Appendices\n")
w("""- **A. Errata** (`docs/errata.md`): every correction of the submitted project, its regression test and the table
  "Old vs new results".
- **B. Traceability** (`docs/traceability.csv`): every number of this report with its source file and script.
- **C. Data processing record** (`docs/data_processing_record.md`): every operation on the data, why, how, with what
  parameters and result.
- **D. Figure book** (`outputs/figures/gallery.html`, `docs/figure_sheets/`): every figure of the project, and the map of
  the figures of this report to those of the submitted report (`docs/report_v7_figures.csv`).
""")

TEXT = "\n".join(sec)
assert PLACED == [f for f, _, _ in ORDER], "every chosen figure is placed once, in order"
assert REFERENCED == set(PLACED), f"figures not referenced in the text: {sorted(set(PLACED) - REFERENCED)}"
assert not any(b in TEXT for b in BANNED)

# ── the README of the report ─────────────────────────────────────────────────
FIGROWS = []
for fid, section, v6 in ORDER:
    FIGROWS.append({"number": FIGNO[fid], "figure_id": fid, "section": section, "title": FIGS[fid]["title"],
                    "v6_figure": v6, "relation": "counterpart of the v6 figure" if v6 else "added",
                    "file": FIGS[fid]["file"]})
for v6, why in sorted(V6_NOT_REDRAWN.items(), key=lambda kv: int(kv[0])):
    FIGROWS.append({"number": "", "figure_id": "", "section": "", "title": "", "v6_figure": v6,
                    "relation": "not redrawn: " + why, "file": ""})
n_counter = sum(1 for _, _, v in ORDER if v)
README = f"""# Report v7: README

Generated by `scripts/make_report_v7.py` with the report (`docs/report_v7.md`). The report is the rebuilt version of the
submitted report v6, in its chapter order, from the frozen final run `20261114T083930Z_final` (tag `g1-frozen`).

## Correction note (D-02)

The submitted report was rebuilt from its evidence folder, not reproduced one to one (D-01). The rebuild corrects errors
that changed the numbers: local time read as UTC (F-01), a clipped indirect target with a mean of ratios (F-06), rolling
features that saw unknown days at D+2 and D+3 (F-07), weather inputs that were not forecasts available at issuance (F-28),
and a feature selection that read the validation and test periods (F-26). Every correction has a row and a regression test
in `docs/errata.md`, and the table "Old vs new results" of the errata sets every submitted number beside the rebuilt one.
The old numbers are points of comparison, not targets.

## Data and code availability (S-45, chapter 16 of v6)

- The raw SCADA data are commercially sensitive and are not released; the site coordinates are kept in
  `configs/site.private.yaml` and never leave it (T-45).
- The code, the configurations, the split definitions and the frozen final run are in the repository at the tag
  `g1-frozen`.
- `scripts/reproduce_paper_results.py` runs the pipeline from the raw data (gate G-1).
- `scripts/make_tables_from_frozen.py` rebuilds the tables, the figures and the report from the frozen forecasts, without
  training and without scoring the test split again.

## Appendices

- A. Errata: `docs/errata.md`.
- B. Traceability: `docs/traceability.csv`.
- C. Data processing record: `docs/data_processing_record.md`.
- D. Figure book: `outputs/figures/gallery.html` and `docs/figure_sheets/`.

## The figures of the report

The report has {len(ORDER)} figures. They follow the figures of the submitted report: {n_counter} are the counterparts of v6
figures, and {len(ORDER) - n_counter} are added where they complete the account of the method. {len(V6_NOT_REDRAWN)} figures
of v6 are not redrawn: five drew the feature-selection protocol of v6, which read the validation and test periods and was
replaced (F-26, D-23), and one repeated the seasonal numbers as a radar (C-84). The map is `docs/report_v7_figures.csv`
and the figure `S11-20`.

## Decisions still open (the user's)

- **`--xgb-only` (Q-30).** A training run, left for the user's decision.
- **Backup of the repository and the archive outside the VDI.** Not done yet.

## Decisions taken by the user (2026-11-15)

- **Q-09: the site coordinates are not disclosed.** The report describes the plant without its coordinates, which stay in
  the private configuration (T-45).
- **References for SHAP and for GFS are added** to the reference list of v6: Lundberg & Lee (2017), Lundberg et al.
  (2020) and the NOAA GFS archive that stage 6b read (R11-04).
"""
assert not any(b in README for b in BANNED)

# ── the table of old and new results ─────────────────────────────────────────
OVN: list[dict] = []


def ovn(section, item, old, new_val, new_test, unit, comparable, why, old_src, new_src):
    OVN.append({"section": section, "item": item, "old_value": old, "new_validation": new_val,
                "new_test": new_test, "unit": unit, "comparable": comparable, "reason": why,
                "old_source": old_src, "new_source": new_src})


for m in M6:
    for t in ("indirect", "direct"):
        for h in (1, 2, 3):
            ob = OLDK.loc[(m, t, h)]
            comp = ("units differ (k is rescaled and unclipped, F-06)" if t == "indirect"
                    else "same definition, other data and time base")
            ovn("benchmark", f"{NAME[m]} {t} D+{h}", f"{ob['nRMSE_pct']:.2f} (n={int(ob['n'])})",
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
cvi = CVT[(CVT["target"] == "indirect") & (CVT["horizon"] == 1)]
ovn("cross-validation", "XGB-Opt indirect D+1, 5-fold CV mean", "16.08 (v6 text; no output file, matrix row 'XGB-Opt 5-fold CV')",
    f"{cvi['validation_uncorrected'].mean():.2f}", f"{cvi['test_uncorrected'].mean():.2f} (CV mean {cvi['cv_mean'].mean():.2f})", "nRMSE % in kW",
    "no (the script was lost)", "C-76, C-77, M43", "submission_evidence/11_audit/REPORT_CLAIM_EVIDENCE_MATRIX.csv", f"{RUN}/report/report_cv_vs_test.csv")
for mode, old_mode in (("minimal", "minimal"),):
    ob = ABL_OLD[(ABL_OLD["target"] == "direct") & (ABL_OLD["horizon"] == 1) & (ABL_OLD["mode"] == old_mode)]["delta_mean"].iloc[0]
    v = ABV[(ABV["target"] == "direct") & (ABV["horizon"] == 1) & (ABV["mode"] == mode)]["mean_delta_pp"].iloc[0]
    t_ = ABT[(ABT["target"] == "direct") & (ABT["horizon"] == 1) & (ABT["mode"] == mode)]["mean_delta_pp"].iloc[0]
    ovn("ablation", "direct D+1, 'minimal' set, change of nRMSE", f"{ob:+.2f}", f"{v:+.2f}", f"{t_:+.2f}", "pp",
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
ovn("skill", "XGB-Opt indirect D+1, skill score against persistence", "0.90 to 0.96 (v6 text)",
    f"{BV[(BV['model']=='xgb_sota')&(BV['target']=='indirect')&(BV['horizon']==1)]['skill_score_kw'].mean():.3f}",
    f"{BT[(BT['model']=='xgb_sota')&(BT['target']=='indirect')&(BT['horizon']==1)]['skill_score_kw'].mean():.3f}", "skill", "no (reference changed)",
    "C-08", "docs/errata.md C-08", f"{RUN}/report/report_benchmark_val.csv, report_benchmark_test.csv")

# ── write ────────────────────────────────────────────────────────────────────
(REPO_ROOT / "docs/report_v7.md").write_text(TEXT, encoding="utf-8", newline="\n")
(REPO_ROOT / "docs/report_v7_README.md").write_text(README, encoding="utf-8", newline="\n")
pd.DataFrame(FIGROWS).to_csv(REPO_ROOT / FIG_TABLE, index=False, encoding="utf-8", lineterminator="\n")
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
assert er.count(BEGIN) == 1
er = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _m: block, er, flags=re.S)
(REPO_ROOT / "docs/errata.md").write_text(er, encoding="utf-8", newline="\n")
print(f"report: {len(TEXT.splitlines())} lines, {len(ORDER)} figures, {len(F.rows)} registered numbers, "
      f"{len(OVN)} old-vs-new rows")

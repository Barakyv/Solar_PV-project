"""Write notebooks/04c_nwp_gfs.ipynb (stage 6b: real GFS forecasts) and execute it.

    python -m solarpv.nwp.download
    python -m solarpv.nwp.gfs
    python -m solarpv.nwp.check
    python scripts/make_nwp_gfs_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-6b
manifest, sanity record, value check, decision file and audits, and from the download records,
never typed. Statistics on train and validation only.
"""

import json

import pandas as pd
import yaml
from _notebook import code, md, write_notebook

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR, REPO_ROOT
from solarpv.nwp import download

NB = REPO_ROOT / "notebooks" / "04c_nwp_gfs.ipynb"
MANIFEST = OUTPUTS_DIR / "06b_gfs" / "gfs_manifest.json"
SANITY = OUTPUTS_DIR / "06b_gfs" / "sanity.json"
VALUE = OUTPUTS_DIR / "06b_gfs" / "value_check.csv"
DICTIONARY = OUTPUTS_DIR / "06b_gfs" / "gfs_feature_dictionary.csv"
DECISION = CONFIGS_DIR / "nwp_decision.yaml"
SAMPLE = AUDIT_DIR / "gfs_sample_2026-10-25" / "sample_summary.json"
DIAG = AUDIT_DIR / "gfs_value_2026-10-25" / "diagnose_sample.csv"
BOOT = AUDIT_DIR / "gfs_value_2026-10-25" / "diagnose_sample_bootstrap.csv"

M = json.loads(MANIFEST.read_text(encoding="utf-8"))
S = json.loads(SANITY.read_text(encoding="utf-8"))
D = yaml.safe_load(DECISION.read_text(encoding="utf-8"))
SK = pd.DataFrame(S["horizon_skill"]).set_index("horizon")
V = pd.read_csv(VALUE).pivot_table(index=["target", "horizon"], columns="features",
                                   values="nrmse_kw_mean")
G = pd.DataFrame(D["results"]).T
DG = pd.read_csv(DIAG).set_index(["target", "horizon", "sample"])["nrmse_kw"]
BT = pd.read_csv(BOOT).set_index("target")
DICT = pd.read_csv(DICTIONARY)

records = [json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(download.OUT_DIR.glob("*/*.json"))]
margin = pd.Series({f"{r['day']} {r['run']:02d}Z": (
    download.issuance_of_run_day(r["day"])
    - max(pd.Timestamp(f["uploaded_utc"]) for f in r["files"])).total_seconds() / 3600
    for r in records if r.get("status") == "complete"})
rejected = [r["day"] for r in records if r.get("status") == "rejected_late_upload"]

cells = [
    md("""# 04c — Real weather forecasts: GFS runs published before issuance (stage 6b)

**Purpose.** Give the models a weather forecast that was really available at the issuance time
T0 (00:00 local of the first forecast day), then test on validation whether it helps (D-12).

**Why.** The original pipeline used an Open-Meteo archive series valued at the target hour: not
a forecast that existed at T0, so it was leakage, and it was removed from the final models (C-06,
F-11). The backup of the original project shows the exact request (F-28,
`00b_original_backup_review.ipynb`). The report's first Future Work item asks for genuine NWP.

**Source.** The NOAA GFS archive on AWS, 1°, the 12Z run of each day, leads 9–84 h, eleven
fields. **Outputs.** `data/processed/06b_gfs/gfs_features_D{h}.parquet`,
`outputs/06b_gfs/`, `configs/nwp_decision.yaml`.

**Rules.** A run may serve T0 only if it started at least 6 h before T0 **and** every one of its
files was uploaded before T0 (T-17, checked row by row). Statistics on train and validation; the
test split is not read. **Documented in** OP-42 to OP-44, run journal R6b-01 to R6b-08,
`docs/stage6b_plan.md`."""),
    md("## 1. Was every run public before T0?"),
    code("""import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

from solarpv.nwp import download

records = [json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(download.OUT_DIR.glob("*/*.json"))]
rows = []
for r in records:
    t0 = download.issuance_of_run_day(r["day"])
    last = max(pd.Timestamp(f["uploaded_utc"]) for f in r["files"])
    rows.append({"day": r["day"], "run": r["run"], "status": r.get("status"),
                 "hours_last_upload_before_t0": (t0 - last).total_seconds() / 3600})
runs = pd.DataFrame(rows)
display(runs.groupby("status")["hours_last_upload_before_t0"].describe().round(2))
runs.loc[runs["status"] == "complete", "hours_last_upload_before_t0"].hist(bins=40, figsize=(8, 3))
plt.xlabel("hours between the last file upload and T0 (used runs)");"""),
    md(f"""**What it shows.** For every run, the time between the upload of its last file and the T0
it serves.
- **Used runs:** {len(margin):,} ({M['fallback_runs']} of them from the 06Z run of the day). The
  smallest margin is {margin.min():.1f} h: every file of every used run was public before T0.
- **Rejected runs:** {len(rejected)} ({', '.join(rejected)}). A file was re-uploaded after T0.
  The archive keeps only the latest version, so the version public at T0 cannot be shown; the
  earlier run of the same day is used instead.

**In the original** the series had no issue time at all."""),
    md("## 2. From 3-hour averages to hourly values"),
    code("""display(Image("../00_data_audit/gfs_sample_2026-10-25/sample_days.png", width=950))"""),
    md("""**What it shows.** Three-day windows from ten periods, checked before the full download:
the clear-sky irradiance on the panel plane (grey), the GFS D+1 forecast on the panel plane after
the reconstruction (blue) and the measured in-plane irradiance (dashed). The test windows show
no measurement. The forecast follows clear days closely and catches the overcast day of
19 January 2022; on broken-cloud days (March 2022) it gets the level of the day but not the
timing of single clouds.

**How it is built (OP-43).** GFS stores irradiance as averages from the start of a 6-hour window.
They are turned back into 3-hour means, then spread over the hours in proportion to the clear-sky
irradiance, so the 3-hour mean is kept and dark hours get zero. The panel-plane irradiance, cell
temperature and physical PV power follow from a physical chain (Erbs, Hay-Davies, Faiman)."""),
    code("""dictionary = pd.read_csv("../outputs/06b_gfs/gfs_feature_dictionary.csv")
dictionary[["feature", "unit", "idea"]]"""),
    md(f"""**What it shows.** The {len(DICT)} candidates per horizon, with their literature idea
(L-01 to L-11). One of them, `irradiance_clear_index_prevday`, is not from GFS: yesterday's
measured clear-sky index (L-11a). Each horizon has its own leads: D+1 uses hours 10–34 after the
run start, D+3 hours 58–82."""),
    md("## 3. Sanity: does the forecast behave like a forecast?"),
    code("""sanity = json.loads(open("../outputs/06b_gfs/sanity.json", encoding="utf-8").read())
display(pd.DataFrame(sanity["horizon_skill"]).round(3))
pd.Series(sanity["lag_correlations_train"], name="r by lag (h)").to_frame().T"""),
    md(f"""**What it shows.** The GFS clear-sky index on the panel plane against the measured one,
on the same {int(SK.loc[1, 'days']):,} train and validation days for every horizon; and, on
train, the correlation at hourly shifts of −3 to +3 h.

**What it means.**
- **Skill falls with the horizon**, as it must: daily r {SK.loc[1, 'corr_daily']:.2f} /
  {SK.loc[2, 'corr_daily']:.2f} / {SK.loc[3, 'corr_daily']:.2f}.
- **The best lag is {S['best_lag_h']} h:** the time alignment is right.
- **Neither red flag of the plan is raised:** "too good at D+3"
  ({S['red_flag_too_good_at_D3']}) and "skill rising with the horizon"
  ({S['red_flag_skill_rises_with_horizon']}).
- **The seasonal bias is small:** between {min(S['bias_kt_poa_by_season'].values()):+.3f} and
  {max(S['bias_kt_poa_by_season'].values()):+.3f}."""),
    md("## 4. Does it help? The value check (D-12)"),
    code("""display(Image("../outputs/06b_gfs/value_check.png", width=850))
pd.read_csv("../outputs/06b_gfs/value_check.csv").pivot_table(
    index=["target", "horizon"], columns="features", values="nrmse_kw_mean").round(2)"""),
    md(f"""**What it shows.** The fixed XGBoost of D-11 (three seeds), trained on train and scored
on validation, with the stage-6 contract alone and with the GFS candidates added. nRMSE in % of the
3,449 kW capacity.

**What it means.** GFS lowers the error at every horizon and for both targets, by
{G['gain_pp'].min():.2f}–{G['gain_pp'].max():.2f} points, against margins of at most
{G['margin_pp'].max():.2f} points (rule fixed before the run). **Decision (D-12): admitted**
(`configs/nwp_decision.yaml`). Stage 7 selects among the candidates."""),
    md("## 5. A red flag examined: D+3 better than D+1 on validation"),
    code("""diag = pd.read_csv("../00_data_audit/gfs_value_2026-10-25/diagnose_sample.csv")
display(diag.pivot_table(index=["target", "sample"], columns="horizon", values="nrmse_kw").round(2))
pd.read_csv("../00_data_audit/gfs_value_2026-10-25/diagnose_sample_bootstrap.csv").round(3)"""),
    md(f"""**What it shows.** With GFS, D+3 scored better than D+1 on validation
({DG[('indirect', 3, 'validation')]:.2f} against {DG[('indirect', 1, 'validation')]:.2f} for the
indirect target). Forecasts lose skill with the horizon, so this was checked before going on.

**The checks.**
- **Horizon mapping:** correct; the leads are as designed.
- **Inner period of train** (fit to April 2023, scored May 2023 to January 2024): the expected
  order D+1 < D+2 < D+3, {DG[('indirect', 1, 'inner (train, after 2023-04)')]:.2f} /
  {DG[('indirect', 2, 'inner (train, after 2023-04)')]:.2f} /
  {DG[('indirect', 3, 'inner (train, after 2023-04)')]:.2f}.
- **Paired bootstrap on validation:** the 95 % interval of RMSE(D+3) − RMSE(D+1) includes zero:
  indirect {BT.loc['indirect', 'ci_lo']:+.2f} to {BT.loc['indirect', 'ci_hi']:+.2f} points.

**Conclusion.** A property of the validation sample (GFS at D+3 happened to be as good as at D+1
there), not a leak.

**Next stage.** Stage 7, feature selection, with the admitted candidates.

**Reproduce.** `python -m solarpv.nwp.download` (about 26 GB of transfer, resumable),
`python -m solarpv.nwp.gfs`, `python -m solarpv.nwp.check`, then
`python scripts/make_nwp_gfs_notebook.py`."""),
]

write_notebook(NB, cells, [MANIFEST, SANITY, VALUE, DICTIONARY, DECISION, SAMPLE, DIAG, BOOT])

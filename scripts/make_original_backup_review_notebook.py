"""Write notebooks/00b_original_backup_review.ipynb (audit of the original backup) and execute it.

    python 00_data_audit/original_backup_2026-11-01/inventory.py
    python 00_data_audit/original_backup_2026-11-01/selection_evidence.py
    python 00_data_audit/original_backup_2026-11-01/weather_source.py
    python scripts/make_original_backup_review_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the audit
outputs, never typed. No coordinate is shown (T-45).
"""

import json

from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import AUDIT_DIR, REPO_ROOT

no_options("python scripts/make_original_backup_review_notebook.py", __doc__)  # X62

NB = REPO_ROOT / "notebooks" / "00b_original_backup_review.ipynb"
HERE = AUDIT_DIR / "original_backup_2026-11-01"
INV = HERE / "inventory_summary.json"
SEL = HERE / "selection_evidence.json"
SHAP = HERE / "selection_shap_direct_top.csv"
WX = HERE / "weather_source.json"
LAG = HERE / "weather_lag_scan.csv"

INVENTORY = json.loads(INV.read_text(encoding="utf-8"))
S = json.loads(SEL.read_text(encoding="utf-8"))
W = json.loads(WX.read_text(encoding="utf-8"))
SK = W["skill"]
base = S["base_notebooks"]
early = S["earlier_notebook_05_dec2025"]
shap = S["direct_shap"]
req = next(iter(W["requests"].values()))[0]
dates = sorted({c["date_utc"] for c in INVENTORY["git"]["commits_in_reflog"]})

cells = [
    md("""# 00b — Review of the original project's backup (audit, 2026-11-01)

**What this is.** An older backup of the original project (`old files/solar_ds_capstone/`),
from about January 2026. **It is a step on the way, not the final project:** the project went
through several more iterations before the v6 submission. It is used as evidence of how the
original pipeline was built; wherever it disagrees with the submission, the submission prevails.

**How it was read.** Read-only. The working copy was listed, and the files that survive only in
its partial git history were recovered by decompressing the git objects; nothing from the backup
was executed. Coordinates found in it are private and are not shown.

**Why it matters.** It answers two questions the submission alone could not: which stage fed the
submitted feature selection, and where the original weather "forecast" came from.

**Documented in** `00_data_audit/original_backup_2026-11-01/README.md`, OP-45, findings F-26 to
F-28, test T-45, run journal RA-01 to RA-03."""),
    md("## 1. What the backup holds"),
    code("""import json

import pandas as pd
from IPython.display import display

from solarpv.config.paths import AUDIT_DIR

HERE = AUDIT_DIR / "original_backup_2026-11-01"
inv = json.loads((HERE / "inventory_summary.json").read_text(encoding="utf-8"))
print("files:", inv["files"])
print("commits in the reflog:", inv["git"]["commits_in_reflog"])
display(pd.Series(inv["by_group"], name="files").to_frame())
pd.Series(inv["final_code_present"], name="present in the backup").to_frame()"""),
    md(f"""**What it shows.** {INVENTORY['files']['working copy']} files in the working copy and
{INVENTORY['files']['git history only']} more that survive only in the git history. The reflog has
commits from {dates[0]} to {dates[-1]}. Most files are earlier versions of data ingest, EDA and
preprocessing, already rebuilt in stages 2–5.

**What is missing.** The final code is not there: not the tuned XGBoost (`xgb_sota.py`), not the
final PyTorch BiLSTM, not the TFT or PatchTST, and not the code of the 05c / 05e / 05f selection
steps. The reconstruction of those parts stays as planned (R-16)."""),
    md("## 2. The feature selection that fed the submission (F-26, F-27)"),
    code("""sel = json.loads((HERE / "selection_evidence.json").read_text(encoding="utf-8"))
params = {t: {k: (len(v) if isinstance(v, list) else v) for k, v in nb.items()
              if k in ("mi_threshold", "variance_threshold", "target_n_features", "cv_splits",
                       "core_features", "per_inverter_dc_forbidden",
                       "row_filter_by_split_or_date")}
          for t, nb in sel["base_notebooks"].items()}
early = sel["earlier_notebook_05_dec2025"]
params["notebook 05 (Dec 2025)"] = {"mi_threshold": early["mi_threshold"],
                                    "variance_threshold": early["variance_threshold"],
                                    "core_features": len(early["core_features"])}
display(pd.DataFrame(params))
pd.DataFrame(sel["fed_submitted_05e_D1"]).T"""),
    md(f"""**What it shows.** The configuration of the two base-selection notebooks of January 2026
(direct and indirect target), next to the earlier December notebook that survived in
`old files/`; and the comparison of their output with the submitted consensus files.

**What it means.**
- **These notebooks fed the submission.** Their
  {S['fed_submitted_05e_D1']['direct']['n_base_output']} features are exactly the starting
  features of the submitted D+1 consensus files, for both targets.
- **Their parameters differ from the December notebook:** MI threshold
  {base['direct']['mi_threshold']} instead of {early['mi_threshold']}, and a protected core of
  {len(base['direct']['core_features'])} features instead of {len(early['core_features'])}. Stage 7
  takes its base parameters from them.
- **No filter by split or date:** the selection saw the validation and test periods (F-26)."""),
    code("""starts = pd.DataFrame(sel["submitted_05e_start_sets"]).T
starts["dropped_from_D1"] = starts["dropped_from_D1"].apply(", ".join)
starts"""),
    md("""**What it shows.** The starting sets of the six submitted consensus files.

**What it means.** D+2 and D+3 start from the D+1 list minus the lags that are not known at that
horizon. The base selection ran once, on the D+1 table, and was not repeated per horizon."""),
    code("""top = pd.read_csv(HERE / "selection_shap_direct_top.csv")
ax = top.iloc[::-1].plot.barh(x="feature", y="mean_abs_shap", legend=False, figsize=(7, 3))
ax.set_xlabel("mean |SHAP| (direct target)")
top"""),
    md(f"""**What it shows.** The five most important features of the direct base selection.

**What it means.** The direct notebook forbade the site's total DC power but not the DC power of
each inverter, measured in the same hour that is being forecast: information that does not exist
at the time of the forecast. The two DC columns take {shap['dc_share_of_total']:.0%} of the total
importance; the first is {shap['first_over_third']:.0f} times the third feature (F-27). The
submitted files removed them afterwards, but by then they had already decided the ranking and
the stopping point of the selection. The indirect notebook forbade them correctly.

**What stage 7 does instead.** Selection on the training period only, from a list of allowed
features (the corrected contract and the real GFS candidates), so a leaking column cannot
enter."""),
    md("## 3. Where the original weather forecast came from (F-28)"),
    code("""wx = json.loads((HERE / "weather_source.json").read_text(encoding="utf-8"))
request = next(iter(wx["requests"].values()))[0]
print("endpoint:", request["endpoint"])
print("parameters:", request["parameters"])
print("variables:", request["hourly_variables"])
print("file:", wx["file"])"""),
    md(f"""**What it shows.** The request stored in the cache of the original feature notebook:
`{req['endpoint']}` with `models={req['parameters']['models']}`, one request from
{req['parameters']['start_date']} to {req['parameters']['end_date']}.

**What it means.** One continuous series with no issue time. Every value is the one for the
target hour, so it could not have been known at the forecast time (midnight before the forecast
day). That is the leak; it was the reason the weather features were removed from the final
models."""),
    code("""lag = pd.read_csv(HERE / "weather_lag_scan.csv")
ax = lag.plot.bar(x="shift_h", y="hourly_r", legend=False, figsize=(6, 3))
ax.set_xlabel("shift of the original series (hours)")
ax.set_ylabel("hourly r with the measurement")
pd.DataFrame(wx["skill"]["daily_r"]).T"""),
    md(f"""**What it shows.** On train and validation hours, the correlation of the original series'
clear-sky index with the measured one, for shifts of −3 to +3 hours; and the daily correlation of
the original series and of the real D+1 GFS forecast of stage 6b.

**What it means.**
- **Not a perfect observation.** Its daily correlation with the measurement
  ({SK['daily_r']['train_and_val']['original']}) is about that of a real day-ahead forecast
  ({SK['daily_r']['train_and_val']['gfs_d1']}). The problem was timing, not accuracy.
- **It needs a one-hour shift** to line up (hourly r {SK['hourly_r']['original']}), because the
  service labels each hour at its end. Joined 2–3 hours off the true clock, as in the original,
  the hourly r is {SK['hourly_r_by_shift'][str(SK['original_join_shifts_h'][0])]} and
  {SK['hourly_r_by_shift'][str(SK['original_join_shifts_h'][1])]}.
- **A likely reason the original weather features added little:** misaligned, they lost most of
  their hourly signal. The real GFS forecasts, aligned and available at issuance, lower the
  validation error by 0.9–1.7 points (D-12, `04c_nwp_gfs.ipynb`)."""),
    md("""## 4. What this changes

- **Stage 7:** the base parameters come from the January notebooks that fed the submission; the
  selection runs on train only and from an allow-list (F-26, F-27).
- **The report:** the description of the original weather source is corrected (reanalysis →
  an archive series with no issue time, of day-ahead quality, misaligned in time; F-28).
- **Unchanged:** the final model code is not in the backup; stage 9 reconstructs it from the
  report and the surviving material, as planned.

**Reproduce.** The three scripts of `00_data_audit/original_backup_2026-11-01/`, then
`python scripts/make_original_backup_review_notebook.py`."""),
]

write_notebook(NB, cells, [INV, SEL, SHAP, WX, LAG])

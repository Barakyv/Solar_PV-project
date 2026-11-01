"""Write notebooks/05_feature_selection.ipynb (stage 7) and execute it.

    python -m solarpv.selection.run
    python scripts/make_feature_selection_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-7
outputs and from the submitted selection files, never typed.
"""

import json

import pandas as pd
import yaml
from _notebook import code, md, write_notebook

from solarpv.config.paths import CONFIGS_DIR, EVIDENCE_DIR, OUTPUTS_DIR, REPO_ROOT

NB = REPO_ROOT / "notebooks" / "05_feature_selection.ipynb"
OUT = OUTPUTS_DIR / "07_feature_selection"
OLD = EVIDENCE_DIR / "09_feature_selection"
COMBOS = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]
MANIFEST = OUT / "selection_manifest.json"
DECISION = CONFIGS_DIR / "k_history_decision.yaml"
VAL = OUT / "validation_check.csv"
VOTES = OUT / "selection_votes.csv"
BASE = OUT / "base_stage.csv"

M = json.loads(MANIFEST.read_text(encoding="utf-8"))
K = yaml.safe_load(DECISION.read_text(encoding="utf-8"))
E = {c: json.loads((OUT / f"final_features_{c[0]}_D{c[1]}.json").read_text(encoding="utf-8"))
     for c in COMBOS}
F = {c: json.loads((OUT / f"refined_features_{c[0]}_D{c[1]}_05f.json").read_text(
    encoding="utf-8")) for c in COMBOS}
OLD_E = {c: json.loads((OLD / f"final_features_{c[0]}_D{c[1]}.json").read_text(encoding="utf-8"))
         for c in COMBOS}
V = pd.read_csv(VAL)
B = pd.read_csv(BASE)
CNT = pd.DataFrame(M["counts"]).T
SOURCES = [MANIFEST, DECISION, VAL, VOTES, BASE] + \
    [OUT / f"final_features_{t}_D{h}.json" for t, h in COMBOS] + \
    [OUT / f"refined_features_{t}_D{h}_05f.json" for t, h in COMBOS]


def name(c) -> str:
    return f"{c[0]} D+{c[1]}"


def n_list(values) -> str:
    return " / ".join(str(v) for v in values)


ind = [c for c in COMBOS if c[0] == "indirect"]
dirc = [c for c in COMBOS if c[0] == "direct"]
gfs_share = {c: sum(f.startswith("fcst_gfs_") for f in E[c]["recommended"]["features"])
             / max(1, E[c]["recommended"]["n_features"]) for c in COMBOS}
lags_left = {c: [f for f in E[c]["recommended"]["features"] if "_lag_" in f] for c in COMBOS}
top3 = {c: E[c]["methods"]["shap_elbow"]["shap_rank"][:3] for c in COMBOS}
red = {c: E[c]["red_flags"] for c in COMBOS}
val_flags = M["validation_red_flags"]
vv = V.pivot_table(index=["target", "horizon"], columns="features", values="nrmse_kw_mean")
dup_groups = B[B["outcome"].str.startswith("duplicate")]

cells = [
    md(f"""# 05 — Feature selection: 05c, 05e and 05f on the corrected data (stage 7)

**Purpose.** Choose, for each target and horizon, the inputs of the tuned XGBoost (XGB-Opt), with
the method of the submitted report (§3.6): a base stage, SHAP backward elimination (05c), a
consensus of three methods (05e) and a robustness audit (05f). The 05e sets are the model input
(D-13); 05f only audits them.

**Candidates.** The corrected contract of stage 6, the 22 candidates admitted in stage 6b (D-12)
and, for the indirect target, the history of k if it is admitted in section 1:
{n_list(CNT['pool'].astype(int))} candidates (indirect D+1 to D+3, then direct).

**Rules (D-23, approved by the user).**
- The selection reads **train rows only**; validation is opened once, at the end; the test split
  is never read.
- No quotas per category; the core of the original base stage (mapped to the new names) is kept
  through the base stage, as in the original.
- One score for every decision: nRMSE in kW on held-out daytime hours of date folds inside train.
- The base stage runs separately for every target and horizon.

**In the original** (backup review, `00b_original_backup_review.ipynb`): the selection read the
whole table, validation and test included (F-26), and the direct selection kept the DC power of
the same hour (F-27). **Documented in** `docs/stage7_plan.md`, OP-46 to OP-48, run journal R7."""),
    md("## 1. Does the history of k help the indirect target? (D-24)"),
    code("""import json

import pandas as pd
import yaml
from IPython.display import Image, display

from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR

OUT = OUTPUTS_DIR / "07_feature_selection"
display(pd.read_csv(OUT / "k_history_check.csv").round(3))
yaml.safe_load((CONFIGS_DIR / "k_history_decision.yaml").read_text(encoding="utf-8"))"""),
    md("PLACEHOLDER_K"),
    md("## 2. The pipeline"),
    code("""display(Image(str(OUT / "figures" / "fig19_pipeline.png")))
manifest = json.loads((OUT / "selection_manifest.json").read_text(encoding="utf-8"))
pd.DataFrame(manifest["counts"]).T"""),
    md(f"""**What it shows.** The number of features after each step, per target and horizon.

**What it means.**
- **Pool:** {n_list(CNT['pool'].astype(int))} candidates (indirect D+1–D+3, then direct).
- **Filters and duplicates:** down to {n_list(CNT['after duplicates'].astype(int))}.
  {len(dup_groups)} features were removed as near-duplicates (|r| > 0.98) of another candidate.
- **RFE:** each set to {n_list(CNT['after RFE (05c start)'].astype(int))}, as in the original.
- **05e consensus:** {n_list(CNT['05e consensus'].astype(int))} features. In the submission:
  {n_list(OLD_E[c]['recommended']['n_features'] for c in COMBOS)}."""),
    md("## 3. 05c: SHAP backward elimination"),
    code("""display(Image(str(OUT / "figures" / "fig20_shap_elimination.png")))
pd.DataFrame({f"{t} D+{h}": json.loads((OUT / f"final_features_{t}_D{h}.json").read_text())
              ["methods"]["shap_elbow"]["elbows"]
              for t in ("indirect", "direct") for h in (1, 2, 3)})"""),
    md(f"""**What it shows.** Starting from the 42 features of the base stage, the least important
feature (mean |SHAP| on held-out hours) is removed and the model refitted, down to one feature.
The curve is the nRMSE in kW on held-out train hours; the dashed line is the 1 % elbow, the
smallest set within 1 % of the score of all 42.

**What it means.**
- **Elbow sizes (1 %):** {n_list(E[c]['methods']['shap_elbow']['n_features'] for c in COMBOS)}.
- **Score of the 42 features (inner train):** indirect
  {n_list(round(E[c]['methods']['shap_elbow']['baseline_nrmse'], 2) for c in ind)} %, direct
  {n_list(round(E[c]['methods']['shap_elbow']['baseline_nrmse'], 2) for c in dirc)} %."""),
    md("## 4. 05e: three votes and the consensus"),
    code("""display(Image(str(OUT / "figures" / "fig21_stability.png")))
votes = pd.read_csv(OUT / "selection_votes.csv")
votes.groupby(["target", "horizon"])[["vote_shap", "vote_stability", "vote_mi", "in_05e"]].sum()"""),
    md(f"""**What it shows.** The selection probability of each feature over 30 subsamples of 70 % of
the train days, and the number of votes from each method.

**The three votes.**
- **SHAP:** the 05c elbow set.
- **Stability:** at least 1 % of the importance on the held-out days, in at least 60 % of the
  subsamples.
- **MI:** mutual information above the 95th percentile of shuffled copies.

**Consensus (2 of 3):** {n_list(E[c]['consensus']['n_features'] for c in COMBOS)} features."""),
    code("""sets = {f"{t} D+{h}": json.loads((OUT / f"final_features_{t}_D{h}.json").read_text())
        ["recommended"]["features"] for t in ("indirect", "direct") for h in (1, 2, 3)}
pd.DataFrame({k: pd.Series(v) for k, v in sets.items()}).fillna("")"""),
    md("PLACEHOLDER_SETS"),
    md("## 5. 05f: the robustness audit"),
    code("""display(Image(str(OUT / "figures" / "fig23_mi_stability.png")))
display(Image(str(OUT / "figures" / "fig25_floor_sensitivity.png")))
pd.DataFrame({f"{t} D+{h}": {tier: v["n"] for tier, v in json.loads(
    (OUT / f"refined_features_{t}_D{h}_05f.json").read_text())["tiers"].items()}
    for t in ("indirect", "direct") for h in (1, 2, 3)})"""),
    md(f"""**What it shows.** Each feature of the 42 placed by its stability and its mutual
information, in the four tiers of the submitted audit, and the number of recommended features
for stability floors from 0.25 to 0.40.

**What it means.** 05f keeps {n_list(F[c]['recommended_05f']['n'] for c in COMBOS)} features
(CORE, SUPPORTED and CONDITIONAL). As in the original, the audit does not change the model input;
it shows which of the 05e features are robust."""),
    md("## 6. Operational availability"),
    code("""display(Image(str(OUT / "figures" / "fig24_availability.png")))
pd.read_csv(OUT / "feature_availability_classification.csv")"""),
    md("""**What it shows.** The classes of the report (§3.6.6) for every feature that appears in a
05e set: FORECAST_SAFE (geometry, clear sky, time, rolling statistics), HORIZON_DEPENDENT (lags
and other histories) and PIPELINE_DEPENDENT (the GFS forecasts, which need the weather pipeline
running every day).

**What it means.** The columns show in which of the six sets each feature is used."""),
    md("## 7. One look at validation"),
    code("""pd.read_csv(OUT / "validation_check.csv").round(3)"""),
    md("PLACEHOLDER_VAL"),
    md("## 8. Against the submitted selection"),
    code("""from solarpv.config.paths import EVIDENCE_DIR

rows = []
for t in ("indirect", "direct"):
    for h in (1, 2, 3):
        old = json.loads((EVIDENCE_DIR / "09_feature_selection" / f"final_features_{t}_D{h}.json")
                         .read_text())["recommended"]["features"]
        new = sets[f"{t} D+{h}"]
        rows.append({"set": f"{t} D+{h}", "submitted": len(old), "rebuild": len(new),
                     "in both": len(set(old) & set(new))})
pd.DataFrame(rows)"""),
    md("PLACEHOLDER_OLD"),
]

write_notebook(NB, cells, SOURCES)

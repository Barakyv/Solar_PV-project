"""Write notebooks/05_feature_selection.ipynb (stage 7) and execute it.

    python -m solarpv.selection.run --skip-validation
    python -m solarpv.selection.run --validation-only
    python scripts/make_feature_selection_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-7
outputs, the k-history decision, the run log of the validation look and the submitted selection
files (read-only), never typed. Each sentence that interprets a result is guarded by an assert,
so a rerun with another outcome stops the builder instead of leaving a stale note.
"""

import json

import pandas as pd
import yaml
from _notebook import code, md, write_notebook

from solarpv.config.paths import CONFIGS_DIR, EVIDENCE_DIR, OUTPUTS_DIR, REPO_ROOT
from solarpv.selection.common import availability
from solarpv.selection.run import INNER_TRAIN_END

NB = REPO_ROOT / "notebooks" / "05_feature_selection.ipynb"
OUT = OUTPUTS_DIR / "07_feature_selection"
OLD = EVIDENCE_DIR / "09_feature_selection"
OLD_SHAP = EVIDENCE_DIR / "07_shap" / "shap_rank_stability.csv"
COMBOS = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]
SAMPLES = ("date folds", "inner hold-out")
CLASSES = ["FORECAST_SAFE", "HORIZON_DEPENDENT", "PIPELINE_DEPENDENT"]
MANIFEST = OUT / "selection_manifest.json"
DECISION = CONFIGS_DIR / "k_history_decision.yaml"
K_CHECK = OUT / "k_history_check.csv"
INNER = OUT / "inner_train_check.csv"
VAL = OUT / "validation_check.csv"
INPUTS = OUT / "model_inputs.json"
VOTES = OUT / "selection_votes.csv"
BASE = OUT / "base_stage.csv"
PAIRS = OUT / "correlated_pairs.csv"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


M = load(MANIFEST)
RUN_LOG = REPO_ROOT / M["run_log"]
VAL_LOG = REPO_ROOT / M["validation_run_log"]
K = yaml.safe_load(DECISION.read_text(encoding="utf-8"))
KC = pd.read_csv(K_CHECK).set_index(["horizon", "features"])
E = {c: load(OUT / f"final_features_{c[0]}_D{c[1]}.json") for c in COMBOS}
F = {c: load(OUT / f"refined_features_{c[0]}_D{c[1]}_05f.json") for c in COMBOS}
OLD_E = {c: load(OLD / f"final_features_{c[0]}_D{c[1]}.json") for c in COMBOS}
INP = load(INPUTS)
IN = pd.read_csv(INNER).set_index(["target", "horizon", "sample"])
V = pd.read_csv(VAL).set_index(["target", "horizon", "features"])
VO = pd.read_csv(VOTES)
B = pd.read_csv(BASE)
P = pd.read_csv(PAIRS)
CNT = pd.DataFrame(M["counts"]).T


def log_field(path, field: str) -> str:
    return next(line.split(field)[1].strip()
                for line in path.read_text(encoding="utf-8").splitlines() if field in line)


VAL_COMMIT = log_field(VAL_LOG, "git commit:")
RUN_MINUTES = float(log_field(RUN_LOG, "status: ok; duration").removesuffix("s")) / 60
SOURCES = [MANIFEST, DECISION, K_CHECK, INNER, VAL, INPUTS, VOTES, BASE, PAIRS, RUN_LOG, VAL_LOG,
           OLD_SHAP] + \
    [OUT / f"final_features_{t}_D{h}.json" for t, h in COMBOS] + \
    [OUT / f"refined_features_{t}_D{h}_05f.json" for t, h in COMBOS] + \
    [OLD / f"final_features_{t}_D{h}.json" for t, h in COMBOS]


def key(c) -> str:
    return f"{c[0]}_D{c[1]}"


def name(c) -> str:
    return f"{c[0]} D+{c[1]}"


def n_list(values) -> str:
    return " / ".join(str(v) for v in values)


def f_list(values, fmt=".3f") -> str:
    return " / ".join(format(v, fmt) for v in values)


def pct(share) -> str:
    return f"{100 * share:g} %"


def n_features(n) -> str:
    return f"{n} feature" + ("" if n == 1 else "s")


ind = [c for c in COMBOS if c[0] == "indirect"]
dirc = [c for c in COMBOS if c[0] == "direct"]
pool = [int(CNT.loc[key(c), "pool"]) for c in COMBOS]
n_start = E[COMBOS[0]]["n_starting_features"]
assert all(E[c]["n_starting_features"] == n_start for c in COMBOS)
model = E[COMBOS[0]]["selection_model"]
assert all(E[c]["selection_model"] == model for c in ind)
score_rows = {M["rows"][key(c)]["score"] for c in COMBOS}
assert len(score_rows) == 1  # both routes scored on the same daytime hours
score_rows = score_rows.pop()

# ── section 1: k history (D-24) ──────────────────────────────────────────────
kr = {h: K["results"][f"indirect_D{h}"] for h in (1, 2, 3)}
k_added = [int(KC.loc[(h, "pool+k_history"), "n_features"] - KC.loc[(h, "pool"), "n_features"])
           for h in (1, 2, 3)]
assert not K["admitted"] and not kr[1]["passes"]
k_miss = kr[1]["margin_pp"] - kr[1]["gain_pp"]

# ── section 2: the pipeline ──────────────────────────────────────────────────
removed = B[B["outcome"].isin(["variance", "mi"])].groupby("outcome").size()
n_var, n_mi = int(removed.get("variance", 0)), int(removed.get("mi", 0))
by_filter = "all by the MI filter" if n_var == 0 else f"{n_var} by variance and {n_mi} by MI"
n_admitted = B[B["group"].isin(["nwp_gfs", "history_other"])].groupby(["target", "horizon"]).size()
assert n_admitted.nunique() == 1
pairs = P.groupby(["target", "horizon"]).size()

# ── sections 3 and 4: 05c and 05e ────────────────────────────────────────────
el = {c: E[c]["methods"]["shap_elbow"] for c in COMBOS}
thr = float(el[COMBOS[0]]["threshold_used"].removesuffix("pct"))
c_min = min(COMBOS, key=lambda c: el[c]["n_features"])
margin_min = IN.loc[(*c_min, "date folds"), "margin_pp"]
assert margin_min > 0.05  # the margin is 2 × the pooled seed std, not the 0.05 pp floor
elbow_tol = el[c_min]["baseline_nrmse"] * thr / 100
assert 0.75 < elbow_tol / (margin_min / 2) < 1.25  # "about one pooled seed std"
st = E[COMBOS[0]]["methods"]["stability_selection"]
mi_vote = E[COMBOS[0]]["methods"]["mutual_information"]
assert VO["vote_mi"].all()  # R7-02: the MI vote endorses every starting feature
assert (VO["in_05e"] == (VO["vote_shap"] | VO["vote_stability"])).all()
gfs_share = {c: sum(f.startswith("fcst_gfs_") for f in E[c]["recommended"]["features"])
             / E[c]["recommended"]["n_features"] for c in COMBOS}
lags_left = {c: sum("_lag_" in f for f in E[c]["recommended"]["features"]) for c in COMBOS}
top3 = {c: el[c]["shap_rank"][:3] for c in COMBOS}
gfs_in_top3 = min(sum(f.startswith("fcst_gfs_") for f in top3[c]) for c in COMBOS)
assert gfs_in_top3 >= 2  # "GFS features lead every ranking"
assert not any(any(E[c]["red_flags"].values()) for c in COMBOS)
tiers = F[COMBOS[0]]["stability_axis"]
floors = [s["floor"] for s in F[COMBOS[0]]["floor_sensitivity"]]

# ── section 7: inside train (R7-04) ──────────────────────────────────────────
inner_pass = pd.read_csv(INNER).groupby(["target", "horizon"])["passes"].all()
assert all(inner_pass[c] for c in ind)
assert not any(IN.loc[(*c, s), "passes"] for c in dirc for s in SAMPLES)
direct_gaps = [IN.loc[(*c, s), "05e_minus_full_pp"] for c in dirc for s in SAMPLES]
full_inner = {(t, s): [IN.loc[(t, h, s), "nrmse_full"] for h in (1, 2, 3)]
              for t in ("indirect", "direct") for s in SAMPLES}
assert all(v == sorted(v) for v in full_inner.values())  # error rises with the horizon


def inner_line(c) -> str:
    g = [IN.loc[(*c, s)] for s in SAMPLES]
    return (f"  - {name(c)}: {g[0]['05e_minus_full_pp']:+.3f} / {g[1]['05e_minus_full_pp']:+.3f}"
            f" pp, margins {g[0]['margin_pp']:.3f} / {g[1]['margin_pp']:.3f}")


# ── section 8: F-29 ──────────────────────────────────────────────────────────
VO["share"] = VO["shap_mean_abs"] / VO.groupby(["target", "horizon"])["shap_mean_abs"].transform(
    "sum")
CONC = {}
for (t, h), g in VO.groupby(["target", "horizon"], sort=False):
    top = g.sort_values("share", ascending=False)
    CONC[(t, h)] = {"top3": 100 * top["share"].head(3).sum(),
                    "top3_names": list(top["feature"].head(3)),
                    "ge1": int((g["share"] >= st["share"]).sum()),
                    "stab": int(g["vote_stability"].sum()),
                    "out": int((~g["in_05e"]).sum()),
                    "share_out": 100 * g.loc[~g["in_05e"], "share"].sum()}
assert min(CONC[c]["top3"] for c in dirc) > max(CONC[c]["top3"] for c in ind)
assert max(CONC[c]["ge1"] for c in dirc) < min(CONC[c]["ge1"] for c in ind)
ge1_dir, ge1_ind = (n_list(CONC[c]["ge1"] for c in cs) for cs in (dirc, ind))

# ── section 9: the third validation look (R7-05) ─────────────────────────────
vgap = {c: V.loc[(*c, "05e"), "05e_minus_full_pp"] for c in COMBOS}
vmar = {c: V.loc[(*c, "05e"), "margin_pp"] for c in COMBOS}
vflag = M["validation_red_flags"]
assert all(bool(V.loc[(*c, "05e"), "red_flag"]) == vflag[key(c)] for c in COMBOS)
assert not any(vflag[key(c)] for c in ind)
flagged = [f"D+{h}" for t, h in dirc if vflag[f"{t}_D{h}"]]
d1 = ("direct", 1)
assert vgap[d1] < -vmar[d1] and all(IN.loc[(*d1, s), "05e_minus_full_pp"] > 0 for s in SAMPLES)
full_val = {t: [V.loc[(t, h, "full pool"), "nrmse_kw_mean"] for h in (1, 2, 3)]
            for t in ("indirect", "direct")}
assert all(v[2] < v[0] for v in full_val.values())  # the R6b-07 sample effect

# ── section 10: the input of stage 9 ─────────────────────────────────────────
for c in COMBOS:
    d = INP[key(c)]
    if c in ind:
        assert d["source"] == "05e" and d["features"] == E[c]["recommended"]["features"]
    else:
        assert d["source"] == "full pool" and d["n_features"] == int(CNT.loc[key(c), "pool"])
assert "test" not in M["splits_read_for_selection"]
cls = {c: pd.Series([availability(f) for f in INP[key(c)]["features"]]).value_counts()
       .reindex(CLASSES, fill_value=0) for c in COMBOS}
n_old_model = len(pd.read_csv(OLD_SHAP))

# ── section 11: against the submission ───────────────────────────────────────
pool_of = {c: set(B.loc[(B["target"] == c[0]) & (B["horizon"] == c[1]), "feature"])
           for c in COMBOS}
assert all(len(pool_of[c]) == int(CNT.loc[key(c), "pool"]) for c in COMBOS)
old_f = {c: OLD_E[c]["recommended"]["features"] for c in COMBOS}
missing = sorted(set().union(*(set(old_f[c]) - pool_of[c] for c in COMBOS)))
weather = [f for f in missing
           if (f.startswith("fcst_") and not f.startswith("fcst_gfs_")) or f.endswith("_forecast")]
others = [f for f in missing if f not in weather]
dc_removed = sorted(set().union(*(OLD_E[c]["recommended"].get("_leakage_fix_removed", [])
                                  for c in dirc)))
in_both = n_list(len(set(old_f[c]) & set(E[c]["recommended"]["features"])) for c in COMBOS)
in_pool = n_list(len(set(old_f[c]) & pool_of[c]) for c in COMBOS)


def codes(names) -> str:
    return ", ".join(f"`{n}`" for n in names)


cells = [
    md(f"""# 05 — Feature selection: 05c, 05e and 05f on the corrected data (stage 7)

**Purpose.** Choose, for each target and horizon, the inputs of the tuned XGBoost (XGB-Opt), with
the method of the submitted report (§3.6): a base stage, SHAP backward elimination (05c), a
consensus of three methods (05e) and a robustness audit (05f). The report makes the 05e sets the
model input (D-13); 05f only audits them. The rebuild adds a check the original never made: each
05e set against the full pool, inside train and then once on validation (R7-04). Its outcome,
the input of stage 9, is `model_inputs.json` (section 10).

**Candidates.** The corrected contract of stage 6, the {n_admitted.iloc[0]} candidates admitted
in stage 6b (D-12) and, for the indirect target, the history of k if it is admitted in section 1:
{n_list(pool)} candidates (indirect D+1 to D+3, then direct).

**Rules (D-23, approved by the user).**
- The selection reads **train rows only** (manifest: {", ".join(M["splits_read_for_selection"])});
  validation is read at the end only ({M["validation_read"]}); the test split is never read.
- No quotas per category; the core of the original base stage (mapped to the new names) is kept
  through the base stage, as in the original.
- One score for every decision: nRMSE in kW (% of {M["capacity_kw"]:,.0f} kW) on held-out daytime
  hours of date folds inside train, {score_rows:,} hours in both routes.
- The base stage runs separately for every target and horizon.
- **Selection model (R7-04, Q-17):** the fixed XGBoost of D-11 in both routes,
  {model["n_estimators"]} trees, depth {model["max_depth"]}, learning rate
  {model["learning_rate"]}, {model["n_jobs"]} threads. Where two sets are compared: three seeds
  and the margin of the D-12 rule, max(0.05 pp, 2 × pooled seed std).

**In the original** (backup review, `00b_original_backup_review.ipynb`): the selection read the
whole table, validation and test included (F-26), and the direct selection kept the DC power of
the same hour (F-27). **Documented in** `docs/stage7_plan.md`, OP-46 to OP-48, run journal R7-01
to R7-05, errata F-29."""),
    md("## 1. Does the history of k help the indirect target? (D-24)"),
    code("""import json

import pandas as pd
import yaml
from IPython.display import Image, display

from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR

OUT = OUTPUTS_DIR / "07_feature_selection"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


display(pd.read_csv(OUT / "k_history_check.csv").round(3))
yaml.safe_load((CONFIGS_DIR / "k_history_decision.yaml").read_text(encoding="utf-8"))"""),
    md(f"""**What it shows.** The fixed XGBoost of D-11, three seeds, on the date folds inside
train: the pool alone and the pool with the history of k ({n_list(k_added)} candidates at D+1 /
D+2 / D+3: lags of k known at issuance and same-hour rolling means).

**What it means.**
- **Gain of the k history:** {f_list(kr[h]["gain_pp"] for h in (1, 2, 3))} pp at D+1 / D+2 /
  D+3, against margins of {f_list(kr[h]["margin_pp"] for h in (1, 2, 3))} pp.
- **The rule (D-12, fixed before the run):** the gain at D+1 must exceed its margin, and D+2 and
  D+3 must not be worse by more than theirs. D+1 misses its margin by {k_miss:.3f} pp.
- **Decision ({K["decision"]}): not admitted.** The rule is applied as registered, even for a
  miss this small. The pools below contain no k history.

**In the original** the report shows `target_k_lag_1d` (§3.4), which is in no submitted set and
not in the XGB input (C-32): a text correction in v7."""),
    md("## 2. The pipeline"),
    code("""display(Image(str(OUT / "figures" / "fig19_pipeline.png")))
manifest = load("selection_manifest.json")
display(pd.DataFrame(manifest["counts"]).T)
pd.read_csv(OUT / "correlated_pairs.csv").groupby(["target", "horizon"]).size().rename(
    "pairs with |r| > 0.98").to_frame().T"""),
    md(f"""**What it shows.** The number of features after each step, per target and horizon, and
the number of strongly correlated pairs.

**What it means.**
- **Pool:** {n_list(pool)} candidates (indirect D+1–D+3, then direct).
- **Filters (variance, MI; the core exempt):** down to
  {n_list(CNT["after filters"].astype(int))}, {by_filter}.
- **Correlated pairs:** {n_list(int(pairs[c]) for c in COMBOS)} pairs with |r| > 0.98 are
  reported, not removed (R7-01: removing chained groups threw away real information). The RFE,
  one feature per step with a refit, removes redundant copies one at a time.
- **RFE:** each set to {n_list(CNT["after RFE (05c start)"].astype(int))}, as in the original.
- **05e consensus:** {n_list(CNT["05e consensus"].astype(int))} features. In the submission:
  {n_list(OLD_E[c]["recommended"]["n_features"] for c in COMBOS)}."""),
    md("## 3. 05c: SHAP backward elimination"),
    code("""display(Image(str(OUT / "figures" / "fig20_shap_elimination.png")))
pd.DataFrame({f"{t} D+{h}": load(f"final_features_{t}_D{h}.json")["methods"]["shap_elbow"]
              ["elbows"] for t in ("indirect", "direct") for h in (1, 2, 3)})"""),
    md(f"""**What it shows.** Starting from the {n_start} features of the base stage, the least
important feature (mean |SHAP| on held-out hours) is removed and the model refitted, down to one
feature. The curve is the nRMSE in kW on held-out train hours, one seed; the dashed line is the
{thr:g} % elbow, the smallest set within {thr:g} % of the score of all {n_start}.

**What it means.**
- **Elbow sizes ({thr:g} %):** {n_list(el[c]["n_features"] for c in COMBOS)}.
- **Score of the {n_start} features (inner train):** indirect
  {f_list((el[c]["baseline_nrmse"] for c in ind), ".2f")} %, direct
  {f_list((el[c]["baseline_nrmse"] for c in dirc), ".2f")} %.
- **The smallest elbow, {name(c_min)}:** it keeps {n_features(el[c_min]["n_features"])},
  scoring {el[c_min]["nrmse_at_elbow"]:.3f} % against {el[c_min]["baseline_nrmse"]:.3f} % for
  all {n_start}. There {thr:g} % of the score is {elbow_tol:.3f} pp, about one pooled seed std on
  the same folds ({margin_min / 2:.3f} pp, half the margin of section 7). A one-seed curve cannot
  resolve that, so this elbow lies within the noise (F-29, section 8)."""),
    md("## 4. 05e: three votes and the consensus"),
    code("""display(Image(str(OUT / "figures" / "fig21_stability.png")))
display(Image(str(OUT / "figures" / "fig22_counts.png")))
votes = pd.read_csv(OUT / "selection_votes.csv")
cols = ["vote_shap", "vote_stability", "vote_mi", "in_05e"]
votes.groupby(["target", "horizon"], sort=False)[cols].sum()"""),
    md(f"""**What it shows.** The selection probability of each feature over
{st["n_subsamples"]} subsamples of {pct(st["subsample_frac"])} of the train days, and the number
of votes from each method.

**The three votes.**
- **SHAP:** the 05c elbow set.
- **Stability:** at least {pct(st["share"])} of the mean |SHAP| on the held-out days, in at
  least {pct(st["threshold"])} of the subsamples.
- **MI:** {mi_vote["description"]}.

**What it means.**
- **Stability votes:** {n_list(CNT["stability votes"].astype(int))}.
- **Consensus (2 of 3):** {n_list(E[c]["consensus"]["n_features"] for c in COMBOS)} features.
- **The MI vote, a limitation (R7-02):** it endorsed all {n_start} starting features in every
  set, because the MI filter of the base stage had already removed the features near the
  shuffled null. The consensus is therefore the union of the SHAP and stability votes. Kept as
  registered (D-23)."""),
    code("""sets = {f"{t} D+{h}": load(f"final_features_{t}_D{h}.json")["recommended"]["features"]
        for t in ("indirect", "direct") for h in (1, 2, 3)}
pd.DataFrame({k: pd.Series(v) for k, v in sets.items()}).fillna("")"""),
    md(f"""**What it shows.** The six 05e sets.

**What it means.**
- **GFS features:** {f_list((100 * gfs_share[c] for c in COMBOS), ".0f")} % of each set. They
  lead every SHAP ranking (05c): at least {gfs_in_top3} of the top three in every set.
""" + "\n".join(f"  - {name(c)}: {codes(top3[c])}" for c in COMBOS) + f"""
- **Lag features left:** {n_list(lags_left[c] for c in COMBOS)} per set.
- **Red flags of the selection (plan §6):** no feature dominates the ranking, and every set
  keeps a GFS irradiance feature. The other two red flags of the plan, the order of the horizons
  and a set worse than the full pool, are examined in sections 7 and 9.
- **Use:** the indirect sets are the input of stage 9; the direct sets fail the check of
  section 7, and the direct route takes the full pool (section 10)."""),
    md("## 5. 05f: the robustness audit"),
    code("""display(Image(str(OUT / "figures" / "fig23_mi_stability.png")))
display(Image(str(OUT / "figures" / "fig25_floor_sensitivity.png")))
pd.DataFrame({f"{t} D+{h}": {tier: v["n"] for tier, v in load(
    f"refined_features_{t}_D{h}_05f.json")["tiers"].items()}
    for t in ("indirect", "direct") for h in (1, 2, 3)})"""),
    md(f"""**What it shows.** Each of the {n_start} starting features placed by its stability and
its mutual information, in the four tiers of the submitted audit (CORE ≥
{tiers["stable_core"]}, SUPPORTED ≥ {tiers["moderate"]}, CONDITIONAL ≥
{tiers["unstable_threshold"]} with MI above its 75th percentile, otherwise EXCLUDED), and the
number of recommended features for stability floors from {min(floors)} to {max(floors)}.

**What it means.** 05f keeps {n_list(F[c]["recommended_05f"]["n"] for c in COMBOS)} features
(CORE, SUPPORTED and CONDITIONAL). As in the original, the audit does not change the model input;
it shows which of the 05e features are robust."""),
    md("## 6. Operational availability"),
    code("""display(Image(str(OUT / "figures" / "fig24_availability.png")))
pd.read_csv(OUT / "feature_availability_classification.csv")"""),
    md("""**What it shows.** The classes of the report (§3.6.6) for every feature that appears in a
05e set: FORECAST_SAFE (geometry, clear sky, time, rolling statistics), HORIZON_DEPENDENT (lags
and other histories) and PIPELINE_DEPENDENT (the GFS forecasts, which need the weather pipeline
running every day).

**What it means.** The columns show in which of the six sets each feature is used. The classes
of the stage-9 inputs, including the full pools of the direct route, are counted in section
10."""),
    md("## 7. Each 05e set against the full pool, inside train (R7-04)"),
    code("""inner = pd.read_csv(OUT / "inner_train_check.csv")
inner.round(3)"""),
    md(f"""**What it shows.** For every target and horizon, the fixed XGBoost of D-11 with three
seeds, trained once with the 05e set and once with the full pool, on two samples inside train:
the date folds, and an inner hold-out (fit to {INNER_TRAIN_END:%Y-%m-%d}, scored on the later
train days). The difference is 05e minus full pool, in pp of nRMSE. A set passes when on both
samples it is worse by at most the margin of the D-12 rule. The rule was written before the run
(`docs/stage7_plan.md` §7).

**What it means.**
- **Indirect: passes at every horizon** (date folds / inner hold-out):
""" + "\n".join(inner_line(c) for c in ind) + """
- **Direct: fails at every horizon, on both samples:**
""" + "\n".join(inner_line(c) for c in dirc) + f"""
- **Sanity:** the full-pool error rises with the horizon on both samples and in both routes:
  indirect {f_list(full_inner[("indirect", SAMPLES[0])], ".2f")} % on the date folds, direct
  {f_list(full_inner[("direct", SAMPLES[0])], ".2f")} %.

**Decision (R7-04 step 3).** A target and horizon that fails takes the full pool, and validation
is not used for it: the direct route takes the full pool at every horizon. The validation look
of section 9 ran at commit `{VAL_COMMIT}` (its run log), the commit that recorded this decision
before the look.

**Limit.** Both samples lie inside the rows the selection used, so this check can refute a set
but cannot prove it; validation is the real test (section 9)."""),
    md("## 8. Why the direct sets fail: concentrated importance (F-29)"),
    code("""votes["share"] = votes["shap_mean_abs"] / votes.groupby(["target", "horizon"])[
    "shap_mean_abs"].transform("sum")
rows = []
for (t, h), g in votes.groupby(["target", "horizon"], sort=False):
    top = g.sort_values("share", ascending=False)
    rows.append({"set": f"{t} D+{h}", "top 3 share %": 100 * top["share"].head(3).sum(),
                 "share >= 1 %": int((g["share"] >= 0.01).sum()),
                 "stability votes": int(g["vote_stability"].sum()),
                 "in 05e": int(g["in_05e"].sum()), "left out": int((~g["in_05e"]).sum()),
                 "share left out %": 100 * g.loc[~g["in_05e"], "share"].sum()})
pd.DataFrame(rows).round(1)"""),
    md(f"""**What it shows.** The share of each of the {n_start} starting features in their total
mean |SHAP| (`selection_votes.csv`), summarised per set.

**What it means.**
- **Top three features:** {f_list((CONC[c]["top3"] for c in dirc), ".1f")} % of the importance
  in the direct route, against {f_list((CONC[c]["top3"] for c in ind), ".1f")} % in the indirect
  route. The direct top three:
""" + "\n".join(f"  - {name(c)}: {codes(CONC[c]['top3_names'])}" for c in dirc) + f"""
- **Features with at least {pct(st["share"])} of the importance:** {ge1_dir} of {n_start} in
  the direct route, {ge1_ind} in the indirect route. The stability vote needs that share on the
  held-out days, so it keeps
  {n_list(CONC[c]["stab"] for c in dirc)} direct features.
- **Left out of the direct 05e sets:** {n_list(CONC[c]["out"] for c in dirc)} starting features,
  which still carry {f_list((CONC[c]["share_out"] for c in dirc), ".1f")} % of the importance
  together.

**Finding F-29.** In the direct route a few features hold most of the importance, so a relative
{pct(st["share"])} share admits few features, and the one-seed 05c curve is too noisy to place
the elbow reliably (section 3). These are the registered rules (D-23), so this is a
limitation of the method, not a code bug; the larger selection model of R7-04 did not remove it.
The original pipeline never compared its 05e sets with the full pool, so it could not see it."""),
    md("## 9. The third look at validation (R7-05)"),
    code("""val = pd.read_csv(OUT / "validation_check.csv")
val.round(3)"""),
    md(f"""**What it shows.** Trained on train and scored on validation: the fixed XGBoost of D-11,
three seeds, with the 05e set and with the full pool. The third look of stage 7 (the first two
were the laptop runs R7-01, void, and R7-02, with the same data); the test split is not read.

**What it means.**
- **Indirect, decided here:** 05e minus full pool
  {f_list((vgap[c] for c in ind), "+.3f")} pp at D+1 / D+2 / D+3, against margins of
  {f_list(vmar[c] for c in ind)} pp. No red flag, so the input is the 05e set (D-13).
- **Direct, reported only** (decided in section 7): {f_list((vgap[c] for c in dirc), "+.3f")} pp
  against {f_list(vmar[c] for c in dirc)} pp. Red flags at {" and ".join(flagged)}, as inside
  train.
- **A limitation, the direct D+1 disagreement:** on validation the 05e set is
  {-vgap[d1]:.3f} pp better than the full pool (margin {vmar[d1]:.3f} pp), while inside train it
  was {f_list((IN.loc[(*d1, s), "05e_minus_full_pp"] for s in SAMPLES), "+.3f")} pp worse. The
  registered rule decides, so the full pool stays; a choice made from this number would be a
  fourth look at validation.
- **Horizons on validation:** the full-pool error is
  {f_list(full_val["indirect"], ".2f")} % (indirect) and {f_list(full_val["direct"], ".2f")} %
  (direct), lower at D+3 than at D+1. This is the validation-sample effect examined in stage 6b
  (R6b-07, notebook `04c_nwp_gfs`); inside train the order is the expected one (section 7).

**In the original** the 05e sets were never compared with the full pool (F-29), and the
selection itself had read validation and test (F-26)."""),
    md("## 10. The input of stage 9 (`model_inputs.json`)"),
    code("""from solarpv.features.leakage import assert_no_leakage
from solarpv.selection.common import availability
from solarpv.selection.run import decide_input

classes = ["FORECAST_SAFE", "HORIZON_DEPENDENT", "PIPELINE_DEPENDENT"]
display({k: manifest[k] for k in ("splits_read_for_selection", "validation_read")})
inputs = load("model_inputs.json")
passes = inner.groupby(["target", "horizon"])["passes"].all()
flags = val.groupby(["target", "horizon"])["red_flag"].any()
rows = []
for k, d in inputs.items():
    t, h = k.split("_D")
    assert_no_leakage(d["features"])
    counts = pd.Series([availability(f) for f in d["features"]]).value_counts()
    rows.append({"set": f"{t} D+{h}", "source": d["source"], "n_features": d["n_features"],
                 "the rule gives": decide_input(passes[(t, int(h))], flags[(t, int(h))])[0],
                 **counts.reindex(classes, fill_value=0).to_dict()})
pd.DataFrame(rows)"""),
    md(f"""**What it shows.** The input of stage 9 for every target and horizon, the decision rule
of R7-04 recomputed from the two checks above, a leakage check of every input and its
availability classes.

**What it means.**
- **Indirect:** the 05e sets, {n_list(INP[key(c)]["n_features"] for c in ind)} features.
- **Direct:** the full pool, {n_list(INP[key(c)]["n_features"] for c in dirc)} features.
- **The rule recomputed** gives the recorded source for all six; `assert_no_leakage` passes on
  every input.
- **Availability of the direct input:** {n_list(int(cls[c]["PIPELINE_DEPENDENT"]) for c in dirc)}
  GFS features (PIPELINE_DEPENDENT), {n_list(int(cls[c]["HORIZON_DEPENDENT"]) for c in dirc)}
  histories (HORIZON_DEPENDENT) and {n_list(int(cls[c]["FORECAST_SAFE"]) for c in dirc)}
  FORECAST_SAFE features at D+1 / D+2 / D+3.
- **For the direct route this departs from D-13** (the 05e sets as the model input). The
  original XGBoost was trained on all its columns anyway, not on the 05e sets: its SHAP file has
  {n_old_model} rows (C-31).

Stage 9 reads this file, not the 05e files."""),
    md("## 11. Against the submitted selection"),
    code("""from solarpv.config.paths import EVIDENCE_DIR

base = pd.read_csv(OUT / "base_stage.csv")
rows = []
for t in ("indirect", "direct"):
    for h in (1, 2, 3):
        path = EVIDENCE_DIR / "09_feature_selection" / f"final_features_{t}_D{h}.json"
        old = json.loads(path.read_text(encoding="utf-8"))["recommended"]["features"]
        new = sets[f"{t} D+{h}"]
        pool = set(base.loc[(base["target"] == t) & (base["horizon"] == h), "feature"])
        rows.append({"set": f"{t} D+{h}", "submitted": len(old), "rebuild 05e": len(new),
                     "in both": len(set(old) & set(new)),
                     "submitted, in the rebuild pool": len(set(old) & pool),
                     "stage-9 input": inputs[f"{t}_D{h}"]["n_features"]})
pd.DataFrame(rows)"""),
    md(f"""**What it shows.** The submitted 05e sets (`submission_evidence/09_feature_selection/`,
read-only) against the rebuild.

**What it means.**
- **Sizes:** submitted {n_list(len(old_f[c]) for c in ind)} (indirect) and
  {n_list(len(old_f[c]) for c in dirc)} (direct); rebuild 05e
  {n_list(E[c]["recommended"]["n_features"] for c in ind)} and
  {n_list(E[c]["recommended"]["n_features"] for c in dirc)}.
- **Few features in common:** {in_both}. Of the submitted features, {in_pool} exist in the
  rebuild pool at all.
- **The missing names:** {len(weather)} are the Open-Meteo weather columns and the forecasts
  derived from them ({codes(weather)}), values at the target hour rather than forecasts (F-11,
  F-28, C-09), replaced by GFS runs public before issuance. The others ({codes(others)}) are not
  in the stage-6 contract.
- **The submitted direct sets** had the per-inverter DC power ({codes(dc_removed)}) removed only
  after the selection (F-27): their consensus had
  {n_list(OLD_E[c]["consensus"]["n_features"] for c in dirc)} features.

**Conclusion.** The rebuild is not meant to reproduce the submitted sets (D-01): true UTC (F-01),
GFS forecasts available at issuance, and a selection on train only (F-26). The method of the
report (05c, 05e, 05f) is kept; the inside-train check is added.

**Next stage.** Stage 9 (`docs/stage9_plan.md`), with the inputs of section 10.

**Reproduce.** `python -m solarpv.selection.run --skip-validation` (about {RUN_MINUTES:.0f}
minutes on the college machine), then `python -m solarpv.selection.run --validation-only`, then
`python scripts/make_feature_selection_notebook.py`."""),
]

write_notebook(NB, cells, SOURCES)

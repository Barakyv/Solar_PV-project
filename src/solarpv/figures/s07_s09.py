"""Figures of stages 7, 8 and 9: feature selection, the evaluation layer and the light models."""

import json

import numpy as np
import pandas as pd
import yaml

from solarpv.config.constants import (
    DEV_OPTUNA_TRIALS,
    DEV_SEED,
    FINAL_OPTUNA_TRIALS,
    RANDOM_SEEDS,
)
from solarpv.config.paths import REPO_ROOT
from solarpv.evaluation.daytime import sun_elevation_mid_hour
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, arrow, blank, box, subplots

INNER = "outputs/07_feature_selection/inner_train_check.csv"
MODEL_INPUTS = "outputs/07_feature_selection/model_inputs.json"
DICTIONARY = "outputs/06_features/feature_dictionary.csv"
FIXTURE = "outputs/07_evaluation_fixture/benchmark_summary.csv"
SUBMITTED = "submission_evidence/01_benchmark/benchmark_summary.csv"
DEV_BENCH = f"{D.DEV_RUN}/evaluate/val/benchmark_summary.csv"
DEV_DM_KW = f"{D.DEV_RUN}/evaluate/val/dm_tests_kw.csv"
D15 = "outputs/09_xgb_checks/d15_check.csv"
D15_DECISION = "configs/monotone_decision.yaml"
L09 = "outputs/09_xgb_checks/l09_check.csv"
L09_DECISION = "configs/mos_decision.yaml"
SEQ_CHECKS = "outputs/09_sequence_data/checks_dev.csv"
REF_NAMES = {"persistence": "persistence (same hour, h days earlier)",
             "climatology": "climatology", "clearsky_k": "clear sky x mean k",
             "xgb_sota": "XGB, development run"}
REF_STYLE = {"persistence": (C["grey"], "^:"), "climatology": (C["green"], "s--"),
             "clearsky_k": (C["orange"], "D-."), "xgb_sota": (C["blue"], "o-")}
KEYS = [f"{t}_D{h}" for t in ("indirect", "direct") for h in (1, 2, 3)]


@register(
    "S07-01", "7", "inner_train_check",
    "Selected sets against the full candidate pool, inside train",
    "Does the selected feature set predict as well as the full candidate pool, before "
    "validation is opened?",
    "Error of the selected set minus error of the full pool (points; negative is better) on two "
    "train-only samples, with the acceptance margin of each case (short bars).",
    "Indirect route: the selected set passes in {ind_pass} of {ind_n} checks (difference "
    "{ind_min:+.2f} to {ind_max:+.2f} points). Direct route: it fails in {dir_fail} of {dir_n} "
    "(the selected set is worse by {dir_min:.2f} to {dir_max:.2f} points), because its "
    "importance is concentrated in a few features and the stability floor kept too few.",
    "The indirect route takes its selected sets, the direct route takes the full pool (F-29); "
    "the decision was committed before validation was looked at.",
    (INNER,))
def s07_inner():
    c = D.csv(INNER)
    fig, ax = subplots(height=2.8)
    xs = {(t, h): i + (0 if t == "indirect" else 3.6) for t in ("indirect", "direct")
          for i, h in enumerate((1, 2, 3))}
    for sample, mk, off in (("date folds", "o", -0.12), ("inner hold-out", "^", 0.12)):
        r = c[c["sample"] == sample]
        x = [xs[(t, h)] + off for t, h in zip(r["target"], r["horizon"], strict=False)]
        ax.scatter(x, r["05e_minus_full_pp"], marker=mk, s=28,
                   c=np.where(r["passes"], C["blue"], C["red"]), label=sample, zorder=3)
    for (t, h), x in xs.items():
        m = c[(c["target"] == t) & (c["horizon"] == h)]["margin_pp"].max()
        ax.plot([x - 0.3, x + 0.3], [m, m], color="black", lw=1.2)
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_xticks(list(xs.values()), [f"{t}\nD+{h}" for t, h in xs])
    ax.set_ylabel("selected minus full pool (points)")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, "blue: passes   red: fails   bar: margin", transform=ax.transAxes,
            ha="right", fontsize=7.5)
    ind, dr = c[c["target"] == "indirect"], c[c["target"] == "direct"]
    facts = {"ind_pass": int(ind["passes"].sum()), "ind_n": int(len(ind)),
             "ind_min": float(ind["05e_minus_full_pp"].min()),
             "ind_max": float(ind["05e_minus_full_pp"].max()),
             "dir_fail": int((~dr["passes"].astype(bool)).sum()), "dir_n": int(len(dr)),
             "dir_min": float(dr["05e_minus_full_pp"].min()),
             "dir_max": float(dr["05e_minus_full_pp"].max())}
    return fig, facts


def _group_of():
    d = D.csv(DICTIONARY).set_index("feature")["group"].to_dict()
    return lambda f: "weather forecast" if f.startswith("fcst_gfs") else d.get(f, "other")


@register(
    "S07-02", "7", "model_inputs", "Model inputs per route and horizon",
    "What does each model finally receive, per route and horizon?",
    "Stacked bars: number of input features of each group for every route and horizon, as "
    "written in the model-input file read by stage 9.",
    "Indirect route: {i1} / {i2} / {i3} inputs at D+1 / D+2 / D+3, of which {i_gfs} are "
    "weather-forecast variables at D+1. Direct route: {d1} / {d2} / {d3} (the full pool, "
    "{d_gfs} forecast variables at D+1).",
    "These lists are the input of every model of stage 9 and stage 10; the runner checks them "
    "against the tables they were selected on.",
    (MODEL_INPUTS, DICTIONARY))
def s07_inputs():
    mi = D.jload(MODEL_INPUTS)
    group = _group_of()
    rows = {k: pd.Series([group(f) for f in mi[k]["features"]]).value_counts() for k in KEYS}
    tab = pd.DataFrame(rows).fillna(0).T
    order = ["weather forecast", "clearsky", "lag_power", "rolling_stats", "time_encodings",
             "other"]
    order = [o for o in order if o in tab.columns]
    fig, ax = subplots(height=2.8)
    bottom = np.zeros(len(tab))
    colours = [C["blue"], C["orange"], C["green"], C["purple"], C["sky"], C["grey"]]
    hatches = [None, "///", "...", "xx", "\\\\", None]
    for i, g in enumerate(order):
        ax.bar(range(len(tab)), tab[g], bottom=bottom, color=colours[i], hatch=hatches[i],
               edgecolor="white", label=g.replace("_", " "))
        bottom += tab[g].to_numpy()
    for i, v in enumerate(bottom):
        ax.text(i, v + 1, str(int(v)), ha="center", fontsize=8)
    ax.set_xticks(range(len(tab)), [k.replace("_D", "\nD+") for k in tab.index])
    ax.set_ylabel("input features")
    ax.legend(loc="upper left", ncols=2)
    n = {k: int(mi[k]["n_features"]) for k in KEYS}
    facts = {"i1": n["indirect_D1"], "i2": n["indirect_D2"], "i3": n["indirect_D3"],
             "d1": n["direct_D1"], "d2": n["direct_D2"], "d3": n["direct_D3"],
             "i_gfs": int(tab.loc["indirect_D1"].get("weather forecast", 0)),
             "d_gfs": int(tab.loc["direct_D1"].get("weather forecast", 0))}
    assert [int(v) for v in bottom] == [n[k] for k in tab.index]
    return fig, facts


@register(
    "S08-01", "8", "evaluation_hours", "Which hours are scored",
    "Which hours count in the evaluation, and how many are there on validation?",
    "Sun elevation at the middle of each UTC hour on a winter and a summer validation day; "
    "hours above the 5-degree line (shaded) are scored.",
    "An hour is scored when the sun is above 5 degrees at its middle: {winter_n} hours on "
    "{winter_day} and {summer_n} on {summer_day}. Validation has {val_n:,} scored hours with a "
    "measured truth; every model and reference is scored on exactly these hours.",
    "One daytime rule from geometry only, the same for both routes and all models, so the "
    "errors are comparable (T-39).",
    (D.EVAL_VAL,))
def s08_hours():
    fig, ax = subplots(height=2.6)
    facts = {}
    for day, name, col, style in (("2024-02-15", "winter", C["blue"], "o-"),
                                  ("2024-06-21", "summer", C["orange"], "s--")):
        idx = pd.date_range(f"{day} 00:00", f"{day} 23:00", freq="h", tz="UTC")
        el = sun_elevation_mid_hour(idx)
        ax.plot(idx.hour, el, style, ms=3.5, color=col, label=f"{name} day, {day}")
        on = el > 5
        ax.fill_between(idx.hour, 0, el, where=on, color=col, alpha=0.12, step=None)
        facts[f"{name}_n"] = int(on.sum())
        facts[f"{name}_day"] = day
    ax.axhline(5, color=C["red"], lw=1, ls=":")
    ax.text(0.2, 6.5, "5 degrees", color=C["red"], fontsize=8)
    ax.set_ylim(-60, 90)
    ax.set_xlabel("UTC hour (elevation at mid-hour)")
    ax.set_ylabel("sun elevation (degrees)")
    ax.legend(loc="upper right")
    b = D.csv(D.EVAL_VAL)
    facts["val_n"] = int(b["n"].iloc[0])
    assert b["n"].nunique() == 1, "all rows share the scored hours"
    return fig, facts


@register(
    "S08-02", "8", "evaluation_code_check", "Evaluation code checked on the surviving forecasts",
    "Does the new evaluation code compute the metrics the same way as the submitted study?",
    "For the one model whose forecasts survived (BiLSTM), the nRMSE computed by the new code "
    "minus the nRMSE in the submitted benchmark table, per route and horizon, with equal hour "
    "counts.",
    "On the same forecasts the new code reproduces the submitted nRMSE within {max_ind:.4f} "
    "points (indirect) and {max_dir:.4f} points (direct), on the same {n_min:,}-{n_max:,} "
    "hours: the metric definitions did not change, so later differences come from the data "
    "and the models.",
    "The evaluation layer was then run on the rebuild's own tables (validation only) with the "
    "references and statistical tests.",
    (FIXTURE, SUBMITTED))
def s08_fixture():
    new = D.csv(FIXTURE)
    new = new[new["model"] == "lstm_sota"].set_index(["target", "horizon"])
    old = D.csv(SUBMITTED)
    old = old[old["model"] == "lstm_sota"].set_index(["target", "horizon"])
    diff = (new["nRMSE_pct"] - old["nRMSE_pct"]).reindex(new.index)
    assert (new["n"] == old["n"].reindex(new.index)).all()
    fig, ax = subplots(height=2.4)
    labels = [f"{t}\nD+{h}" for t, h in diff.index]
    ax.bar(labels, diff.values, color=[C["blue"] if t == "indirect" else C["orange"]
                                       for t, _ in diff.index], width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    lim = max(0.002, float(diff.abs().max()) * 1.5)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("new minus submitted nRMSE (points)")
    ax.set_title("BiLSTM forecasts of the submission, scored by the new code")
    facts = {"max_ind": float(diff.loc["indirect"].abs().max()),
             "max_dir": float(diff.loc["direct"].abs().max()),
             "n_min": int(new["n"].min()), "n_max": int(new["n"].max())}
    return fig, facts


@register(
    "S08-03", "8", "references", "Reference forecasts on validation",
    "How hard is the forecasting problem, judged by simple references?",
    "Validation nRMSE in kW of the three references at each horizon (direct target).",
    "Climatology ({clim:.2f} %) beats persistence at every horizon ({p1:.2f} / {p2:.2f} / "
    "{p3:.2f} %), since a day says little about the next one; clear sky times the mean k gives "
    "{cs:.2f} %. A model must beat climatology to be useful. The indirect-route references, "
    "converted to kW, differ from these by at most {route_gap:.2f} points.",
    "These references are the yardstick of the skill scores and the significance tests of "
    "stage 9.",
    (D.EVAL_VAL,))
def s08_references():
    allb = D.csv(D.EVAL_VAL).set_index(["model", "target", "horizon"])["nRMSE_kW_pct"]
    gap = (allb.xs("indirect", level="target") - allb.xs("direct", level="target")).abs().max()
    b = D.csv(D.EVAL_VAL)
    b = b[b["target"] == "direct"]
    fig, ax = subplots(height=2.6)
    x = np.arange(3)
    models = ["persistence", "climatology", "clearsky_k"]
    hatches = ["///", None, "..."]
    for i, m in enumerate(models):
        r = b[b["model"] == m].sort_values("horizon")
        ax.bar(x + (i - 1) * 0.26, r["nRMSE_kW_pct"], 0.24, color=REF_STYLE[m][0],
               hatch=hatches[i], edgecolor="white", label=REF_NAMES[m])
    ax.set_xticks(x, ["D+1", "D+2", "D+3"])
    ax.set_ylabel("validation nRMSE (% of 3,449 kW)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    p = b[b["model"] == "persistence"].sort_values("horizon")["nRMSE_kW_pct"].to_numpy()
    clim = b[b["model"] == "climatology"]["nRMSE_kW_pct"]
    assert (clim.to_numpy() < p).all()
    facts = {"clim": float(clim.iloc[0]), "p1": float(p[0]), "p2": float(p[1]),
             "p3": float(p[2]),
             "cs": float(b[b["model"] == "clearsky_k"]["nRMSE_kW_pct"].iloc[0]),
             "route_gap": float(gap)}
    return fig, facts


@register(
    "S09-01", "9", "run_modes", "Run modes of the pipeline runner",
    "How is a development run kept away from the test period, and when may the final run start?",
    "Three run modes, the data each one may read, and the gate in front of the final run.",
    "Smoke: a short slice to test the plumbing. Development: train and validation, "
    "{dev_trials} tuning trials, seed {dev_seed}. Final: {final_trials} trials, seeds "
    "{final_seeds}, and the only mode that reads the test period, refused unless the code "
    "carries the frozen tag.",
    "Light models are developed in the dev mode until gate G-1; the final run of the six "
    "models happens once (stage 10).",
    ())
def s09_modes():
    fig, ax = subplots(height=2.8)
    blank(ax)
    cols = [("Smoke", "short slice\ntrain + validation\nchecks the plumbing", C["light"]),
            ("Development", f"train + validation\n{DEV_OPTUNA_TRIALS} trials, seed {DEV_SEED}\n"
             "decisions on validation", C["sky"] + "66"),
            ("Final", f"train + validation + test\n{FINAL_OPTUNA_TRIALS} trials, seeds "
             f"{', '.join(map(str, RANDOM_SEEDS))}\nonce, on frozen code", C["orange"] + "55")]
    for i, (name, text, face) in enumerate(cols):
        x = 0.03 + i * 0.33
        box(ax, x, 0.35, 0.27, 0.45, f"{name}\n\n{text}", face=face, size=8)
    box(ax, 0.69, 0.04, 0.27, 0.18, "gate G-1 passed,\ncode tagged g1-frozen", face="white",
        edge=C["red"], size=8, color=C["red"])
    arrow(ax, 0.825, 0.22, 0.825, 0.35, color=C["red"])
    arrow(ax, 0.30, 0.575, 0.36, 0.575)
    arrow(ax, 0.63, 0.575, 0.69, 0.575)
    ax.text(0.03, 0.92, "Each run writes its own folder with the inputs' hashes; a test-period "
            "row in a smoke or development run is an error.", fontsize=8)
    facts = {"dev_trials": DEV_OPTUNA_TRIALS, "dev_seed": DEV_SEED,
             "final_trials": FINAL_OPTUNA_TRIALS,
             "final_seeds": ", ".join(map(str, RANDOM_SEEDS))}
    return fig, facts


@register(
    "S09-02", "9", "xgb_dev_vs_references", "XGB development run against the references",
    "Does a lightly tuned XGBoost beat the references on validation, in both routes?",
    "Validation nRMSE in kW per horizon for the development XGBoost and the three references, "
    "for the indirect route (left, converted to kW) and the direct route (right).",
    "XGB-dev: {xi1:.2f} / {xi2:.2f} / {xi3:.2f} % (indirect) and {xd1:.2f} / {xd2:.2f} / "
    "{xd3:.2f} % (direct), below climatology ({clim:.2f} %) and persistence at every horizon. "
    "These are tuning-split numbers, optimistic by construction; the unbiased test comes once, "
    "in the final run.",
    "The pipeline runs end to end with a light model; the neural models are added next (W4) "
    "on the same datasets and scoring.",
    (DEV_BENCH,))
def s09_dev():
    b = D.csv(DEV_BENCH)
    fig, axes = subplots(1, 2, height=2.8, sharey=True)
    for ax, t in zip(axes, ("indirect", "direct"), strict=False):
        for m in ("persistence", "climatology", "clearsky_k", "xgb_sota"):
            r = b[(b["target"] == t) & (b["model"] == m)].sort_values("horizon")
            col, style = REF_STYLE[m]
            ax.plot(r["horizon"], r["nRMSE_kW_pct"], style, color=col, ms=4, label=REF_NAMES[m])
        ax.set_title(f"{t} route")
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
        ax.set_ylim(0, None)
    axes[0].set_ylabel("validation nRMSE (% of 3,449 kW)")
    axes[1].legend(loc="lower right", fontsize=7)
    x = {t: b[(b["target"] == t) & (b["model"] == "xgb_sota")].sort_values("horizon")
         ["nRMSE_kW_pct"].to_numpy() for t in ("indirect", "direct")}
    clim = b[b["model"] == "climatology"]["nRMSE_kW_pct"].min()
    assert max(x["indirect"].max(), x["direct"].max()) < clim
    facts = {"xi1": x["indirect"][0], "xi2": x["indirect"][1], "xi3": x["indirect"][2],
             "xd1": x["direct"][0], "xd2": x["direct"][1], "xd3": x["direct"][2],
             "clim": float(clim)}
    return fig, facts


@register(
    "S09-03", "9", "monotone_check", "Monotone constraint on clear-sky irradiance",
    "Should the indirect model keep the submitted study's rule that the prediction may only "
    "rise with clear-sky irradiance?",
    "Validation nRMSE in kW of a fixed XGBoost with the constraint (A) and without it (B), "
    "three seeds each (points) and their means (bars), per horizon.",
    "Dropping the constraint does not help: the gain is {g1:+.3f} / {g2:+.3f} / {g3:+.3f} "
    "points at D+1 / D+2 / D+3, against a pre-registered margin of {m1:.2f} at D+1.",
    "The constraint stays in both routes as in the submitted study (D-15); its exact feature "
    "list is now checked against the trained model.",
    (D15, D15_DECISION))
def s09_monotone():
    c = D.csv(D15)
    dec = yaml.safe_load((REPO_ROOT / D15_DECISION).read_text(encoding="utf-8"))
    fig, ax = subplots(height=2.6)
    for i, h in enumerate((1, 2, 3)):
        for j, (s, col, mk) in enumerate((("A", C["blue"], "o"), ("B", C["orange"], "^"))):
            r = c[(c["horizon"] == h) & (c["setting"] == s)]["nrmse_kw_pct"]
            x = i + (j - 0.5) * 0.3
            ax.bar(x, r.mean(), 0.26, color=col, alpha=0.35,
                   label=("A: with constraint" if s == "A" else "B: without") if i == 0 else None)
            ax.scatter([x] * len(r), r, color=col, marker=mk, s=18, zorder=3)
    ax.set_xticks(range(3), ["D+1", "D+2", "D+3"])
    lo = c["nrmse_kw_pct"].min()
    ax.set_ylim(lo - 0.6, c["nrmse_kw_pct"].max() + 0.2)
    ax.set_ylabel("validation nRMSE (% of 3,449 kW)\naxis does not start at zero")
    ax.legend(loc="upper left")
    res = dec["results"]
    facts = {"g1": res["D1"]["gain_pp"], "g2": res["D2"]["gain_pp"], "g3": res["D3"]["gain_pp"],
             "m1": res["D1"]["margin_pp"], "decision": dec["settings"]["indirect"]}
    return fig, facts


@register(
    "S09-04", "9", "linear_correction", "Linear correction of the forecast in kW",
    "Does a straight-line correction of the forecast, fitted inside train, reduce the error?",
    "Validation nRMSE in kW before (circles) and after (squares) the correction a + b x "
    "forecast, fitted on out-of-fold train forecasts; three seeds per horizon and route.",
    "Indirect route: the error drops by {i1:.2f} / {i2:.2f} / {i3:.2f} points, above every "
    "margin, so the correction is adopted. Direct route: it rises by {d1:.2f} / {d2:.2f} / "
    "{d3:.2f} points, so it is not used.",
    "The correction runs inside the XGB step for the indirect route, refitted on each run "
    "(L-09); the uncorrected forecast is kept beside it.",
    (L09, L09_DECISION))
def s09_l09():
    c = D.csv(L09)
    dec = yaml.safe_load((REPO_ROOT / L09_DECISION).read_text(encoding="utf-8"))["results"]
    fig, axes = subplots(1, 2, height=2.6, sharey=True)
    for ax, t in zip(axes, ("indirect", "direct"), strict=False):
        r = c[c["target"] == t]
        for h in (1, 2, 3):
            q = r[r["horizon"] == h]
            ax.plot([h - 0.1] * len(q), q["uncorrected_pct"], "o", color=C["grey"], ms=4,
                    label="before" if h == 1 else None)
            ax.plot([h + 0.1] * len(q), q["corrected_pct"], "s", ms=4,
                    color=C["blue"] if dec[t]["adopted"] else C["red"],
                    label="after" if h == 1 else None)
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
        ax.set_title(f"{t} route: " + ("adopted" if dec[t]["adopted"] else "not adopted"))
    axes[0].set_ylabel("validation nRMSE (% of 3,449 kW)")
    axes[0].legend(loc="upper left")
    g = {t: [dec[t]["by_horizon"][f"D{h}"]["gain_pp"] for h in (1, 2, 3)]
         for t in ("indirect", "direct")}
    assert all(v["passes"] for v in dec["indirect"]["by_horizon"].values())
    assert dec["indirect"]["adopted"] and not dec["direct"]["adopted"]
    assert all(x < 0 for x in g["direct"])
    facts = {"i1": g["indirect"][0], "i2": g["indirect"][1], "i3": g["indirect"][2],
             "d1": -g["direct"][0], "d2": -g["direct"][1], "d3": -g["direct"][2]}
    return fig, facts


@register(
    "S09-05", "9", "significance_kw", "Are the differences significant? Diebold-Mariano in kW",
    "Is XGB-dev better than each reference beyond what chance in the validation hours allows?",
    "Grid of the Bonferroni-corrected Diebold-Mariano tests (squared error in kW, HLN "
    "correction): each cell is XGB-dev against one reference at one horizon; the text is the "
    "p-value and the observed effect as a share of the minimum detectable effect.",
    "XGB-dev is significantly better in {n_sig} of {n_all} comparisons. Against climatology "
    "the effect is {mde_min:.0%} to {mde_max:.0%} of the minimum detectable effect: real, but "
    "modest.",
    "The kW table is the primary acceptance test of the final run (Q-19); the k-unit table is "
    "kept as secondary.",
    (DEV_DM_KW,))
def s09_dm():
    d = D.csv(DEV_DM_KW)
    d = d[d["model_b"] == "xgb_sota"]
    rows = [(t, m) for t in ("indirect", "direct") for m in ("persistence", "climatology",
                                                             "clearsky_k")]
    fig, ax = subplots(height=3.0)
    for i, (t, m) in enumerate(rows):
        for h in (1, 2, 3):
            r = d[(d["target"] == t) & (d["model_a"] == m) & (d["horizon"] == h)].iloc[0]
            sig = bool(r["significant"])
            ax.add_patch(_rect(h - 0.48, i - 0.45, 0.96, 0.9,
                                C["green"] + "55" if sig else C["red"] + "55"))
            p = r["p_value_hln"]
            ptxt = "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"
            ax.text(h, i, f"{ptxt}\n{r['pct_of_mde']:.0f} % of MDE",
                    ha="center", va="center", fontsize=7.5)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
    ax.set_yticks(range(len(rows)), [f"{t}: vs {REF_NAMES[m].split(' (')[0]}" for t, m in rows])
    ax.grid(False)
    ax.set_title("green: significant at the Bonferroni level, red: not")
    clim = d[d["model_a"] == "climatology"]["pct_of_mde"] / 100
    facts = {"n_sig": int(d["significant"].astype(bool).sum()), "n_all": int(len(d)),
             "mde_min": float(clim.min()), "mde_max": float(clim.max())}
    return fig, facts


def _rect(x, y, w, h, face):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=face, ec="white", lw=1)


@register(
    "S09-06", "9", "sequence_datasets", "Sequence datasets: one forecast origin per local day",
    "How are the sequence models fed so that they forecast the same hours as XGB?",
    "Left: one forecast origin at local midnight, with the encoder history of each model and "
    "the decoder covering the next three local days. Right: decoder block lengths of the "
    "train and validation origins (23 and 25 hours on daylight-saving days).",
    "{n_train} train and {n_val} validation origins, one per local day. Blocks are {n24:,} "
    "days of 24 hours, {n23} of 23 and {n25} of 25; every check passed (each hour forecast once "
    "per horizon, the encoder ends at issuance, no step reaches the test period).",
    "BiLSTM, the three TFTs and PatchTST read these datasets in W4, with the loss on exactly the "
    "rows XGB is trained on.",
    (SEQ_CHECKS,))
def s09_sequence():
    c = D.csv(SEQ_CHECKS)
    checks = [col for col in c.columns if c[col].dtype == bool]
    assert c[checks].all().all(), "a sequence-data check failed"
    first = c[(c["target"] == "indirect")].drop_duplicates("scope")
    lengths = {s: json.loads(r)
               for s, r in zip(first["scope"], first["block_lengths"], strict=True)}
    fig, (a1, a2) = subplots(1, 2, height=2.6, gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (name, enc) in enumerate((("BiLSTM", 72), ("TFT-96", 96), ("TFT-168", 168))):
        a1.barh(i, enc, left=-enc, color=C["sky"], height=0.5)
        a1.text(-enc + 2, i, f"{name}: {enc} h", va="center", fontsize=7.5)
    for j, h in enumerate((1, 2, 3)):
        a1.barh(3.2, 24, left=j * 24, color=[C["orange"], C["yellow"], C["green"]][j],
                height=0.5, edgecolor="white")
        a1.text(j * 24 + 12, 3.2, f"D+{h}", ha="center", va="center", fontsize=8)
    a1.axvline(0, color=C["red"], ls="--", lw=1)
    a1.text(2, 2.6, "issuance (00:00 local)", color=C["red"], fontsize=7.5)
    a1.set_xlim(-175, 76)
    a1.set_yticks([])
    a1.set_xlabel("hours from issuance")
    a1.set_title("encoder (blue) and decoder")
    counts = pd.DataFrame(lengths).fillna(0).astype(int)
    counts.index = counts.index.astype(int)
    counts = counts.sort_index()
    xx = np.arange(len(counts))
    a2.bar(xx - 0.18, counts.get("train", 0), 0.34, color=C["blue"], label="train")
    a2.bar(xx + 0.18, counts.get("val", counts.get("validation", 0)), 0.34, color=C["orange"],
           hatch="///", edgecolor="white", label="validation")
    a2.set_yscale("log")
    a2.set_xticks(xx, [f"{n} h" for n in counts.index])
    a2.set_ylabel("decoder day blocks (log)")
    a2.legend(loc="upper left")
    origins = c.drop_duplicates("scope").set_index("scope")["n_origins"]
    tot = counts.sum(axis=1)
    facts = {"n_train": int(origins.get("train")),
             "n_val": int(origins.get("val", origins.get("validation", 0))),
             "n23": int(tot.get(23, 0)), "n24": int(tot.get(24, 0)), "n25": int(tot.get(25, 0))}
    return fig, facts

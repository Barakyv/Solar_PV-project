"""Figures of stage 9, W5: the final protocol, the explanations and the run report
(docs/stage9_plan.md §19). Process figures read the two development runs that W5 explained: the
XGB dev run of W5 and the neural dev run of W4. The correction figure C-10 compares the
submitted SHAP groups with the rebuild's."""

import numpy as np
from matplotlib.patches import Patch

from solarpv.config.constants import DEV_SEED
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, arrow, blank, box, subplots

XDEV = D.W5_XGB_DEV_RUN
NDEV = D.NEURAL_DEV_RUN
SHAP_FEATURES = f"{XDEV}/explain/xgb/xgb_shap_features.csv"
SHAP_GROUPS = f"{XDEV}/explain/xgb/xgb_shap_groups.csv"
SHAP_CHECKS = f"{XDEV}/explain/xgb/xgb_shap_checks.csv"
ATTENTION = f"{NDEV}/explain/tft/tft_attention_summary.csv"
HP_XGB = f"{XDEV}/explain/hparam_importance.csv"
HP_NEURAL = f"{NDEV}/explain/hparam_importance.csv"
WARN = (f"{XDEV}/report/report_warnings.csv", f"{NDEV}/report/report_warnings.csv")
V6_GROUPS = "submission_evidence/07_shap/shap_group_importance.csv"
GROUP_STYLE = {  # colour and hatch (style rule: never colour alone)
    "nwp_gfs": (C["blue"], ""), "clearsky": (C["orange"], "//"), "other": (C["grey"], ".."),
    "time_encodings": (C["green"], "\\\\"), "lag_power": (C["red"], "xx"),
    "rolling_stats": (C["purple"], "--"), "history_other": (C["yellow"], "oo"),
    "k_history": (C["sky"], "++")}
GROUP_NAMES = {"nwp_gfs": "GFS forecasts", "clearsky": "clear sky", "other": "geometry, other",
               "time_encodings": "time encodings", "lag_power": "lags",
               "rolling_stats": "rolling statistics", "history_other": "previous-day index",
               "k_history": "k history"}
TFT = (("tft96", "TFT-96"), ("tft168", "TFT-168"), ("tftbase", "TFT baseline"))
DEV_STEP_FILES = tuple(f"{run}/{step}/step.json" for run, step in (
    (XDEV, "xgb"), *((NDEV, s) for s in ("bilstm", "tft96", "tft168", "tftbase", "patchtst"))))


@register(
    "S09-13", "9", "final_protocol", "The two phases of the final run",
    "How does the final run keep the test split away from every decision?",
    "The steps of a final run in order, the gate checked before every step, the stage-10 "
    "acceptance written between the phases, and the step that alone scores the test split.",
    "Phase 1 runs {n_phase1} steps; each model forecasts validation and test, but only "
    "validation is scored. The gate (tag {tag}, clean tree, the preflight's commit) is checked "
    "before every step. Phase 2 is the step `test`, refused without --allow-test and without "
    "the acceptance file `{acceptance}`; every read of a test row is logged.",
    "W6 runs this protocol once, after G-1, by docs/runbook_final.md.",
    ())
def s09_final_protocol():
    from solarpv.pipeline.config import FROZEN_TAG, STEP_ORDER, TEST_STEP
    from solarpv.pipeline.runner import select_steps
    from solarpv.pipeline.steps import ACCEPTANCE_FILE

    phase1 = select_steps(None, None, "report")
    assert TEST_STEP not in phase1 and STEP_ORDER[-1] == TEST_STEP  # D-32
    fig, ax = subplots(height=3.0)
    blank(ax)
    w = 0.86 / len(phase1)
    for i, step in enumerate(phase1):
        x = 0.02 + i * w
        box(ax, x, 0.58, w * 0.9, 0.16, step, face=C["sky"] + "55", size=6.5)
        ax.plot(x + w * 0.45, 0.80, "v", color=C["red"], ms=4)
    ax.text(0.02, 0.86, "phase 1 (stage 10): train, forecast validation and test, score "
            "validation only", fontsize=8)
    ax.text(0.02, 0.50, "▼ gate before every step: HEAD carries the tag, clean tree, the "
            "preflight's commit (X38)", fontsize=7, color=C["red"])
    box(ax, 0.10, 0.12, 0.30, 0.22, f"stage-10 acceptance\nwritten to {ACCEPTANCE_FILE}\n"
        "(untracked, X51)", face="white", edge=C["grey"], size=7)
    box(ax, 0.55, 0.12, 0.30, 0.22, "phase 2: step test\n--allow-test; scores the test split,\n"
        "XGB SHAP on test; logged", face=C["orange"] + "55", size=7)
    arrow(ax, 0.40, 0.23, 0.55, 0.23)
    arrow(ax, 0.25, 0.58, 0.25, 0.34)
    facts = {"n_phase1": len(phase1), "tag": FROZEN_TAG, "acceptance": ACCEPTANCE_FILE}
    return fig, facts


@register(
    "S09-14", "9", "xgb_shap_top", "What the development XGBoost relies on",
    "Which inputs carry the largest attributions of the XGB-dev forecasts, in kW?",
    "Mean |SHAP| in kW of the eight largest features per target and horizon (seed 42), on the "
    "scored validation daytime hours; colour and hatch by feature group. The indirect route's "
    "attributions are the booster's k values times theo (and the L-09 slope).",
    "The largest attribution comes from a GFS forecast at every target and horizon: "
    "{top_ind} (indirect) and {top_dir} (direct) at D+1. SHAP adds up to the booster forecast "
    "within {add:.1e} of the output scale on {n_rows:,} hours per horizon, and XGBoost's own "
    "contributions agree. A description of the fitted model, not a causal effect.",
    "The final run's SHAP (validation, and test in phase 2) uses the same code; the report "
    "flags a dominant feature (the stage-7 rule).",
    (SHAP_FEATURES, SHAP_CHECKS))
def s09_xgb_shap_top():
    f = D.csv(SHAP_FEATURES)
    c = D.csv(SHAP_CHECKS)
    f = f[f["seed"] == DEV_SEED]
    fig, axes = subplots(2, 3, height=5.0)
    tops = {}
    for i, t in enumerate(("indirect", "direct")):
        for j, h in enumerate((1, 2, 3)):
            ax = axes[i][j]
            g = f[(f["target"] == t) & (f["horizon"] == h)].sort_values("rank").head(8)
            assert g["group"].iloc[0] == "nwp_gfs", f"{t} D+{h}: the top feature is not GFS"
            for k, r in enumerate(g.itertuples()):
                col, hatch = GROUP_STYLE.get(r.group, (C["grey"], ""))
                ax.barh(-k, r.mean_abs_kw, color=col, hatch=hatch, edgecolor="white")
                ax.text(0, -k, f" {r.feature}", va="center", fontsize=5.5)
            ax.set_yticks([])
            ax.set_title(f"{t}, D+{h}", fontsize=8)
            ax.set_xlabel("mean |SHAP| (kW)", fontsize=7)
            if h == 1:
                tops[t] = g["feature"].iloc[0]
    handles = [Patch(facecolor=col, hatch=hatch, edgecolor="white", label=GROUP_NAMES[g])
               for g, (col, hatch) in GROUP_STYLE.items() if g in set(f["group"])]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5)
    assert c["n_rows"].nunique() == 1
    facts = {"top_ind": tops["indirect"], "top_dir": tops["direct"],
             "add": float(c["additivity_relative"].max()), "n_rows": int(c["n_rows"].iloc[0])}
    return fig, facts


@register(
    "S09-15", "9", "xgb_shap_groups", "Attribution of the feature groups",
    "How is the XGB-dev attribution shared between the feature groups, and does the measure "
    "matter?",
    "Share of each feature group in the group attribution, the mean over hours of |sum of "
    "SHAP over the group| (bars), per target and horizon (seed 42); the dots show the share "
    "that the sum of the members' mean |SHAP| would give on the same values.",
    "The GFS forecasts hold {gfs_min:.0f} to {gfs_max:.0f} % of the group attribution. The "
    "two measures differ by up to {diff:.1f} points ({diff_group}, {diff_key}): members of one "
    "group often push in opposite directions, so summing their separate magnitudes overstates "
    "the group.",
    "The report and stage 11 use the group attribution (M27).",
    (SHAP_GROUPS,))
def s09_xgb_shap_groups():
    g = D.csv(SHAP_GROUPS)
    g = g[g["seed"] == DEV_SEED]
    keys = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]
    fig, ax = subplots(height=3.2)
    groups = [k for k in GROUP_STYLE if k in set(g["group"])]
    bottom = np.zeros(len(keys))
    diffs = []
    for grp in groups:
        vals, v6 = [], []
        for t, h in keys:
            r = g[(g["target"] == t) & (g["horizon"] == h) & (g["group"] == grp)]
            vals.append(float(r["share_pct"].iloc[0]) if len(r) else 0.0)
            v6.append(float(r["share_pct_v6_measure"].iloc[0]) if len(r) else 0.0)
            diffs.append((abs(vals[-1] - v6[-1]), GROUP_NAMES[grp], f"{t} D+{h}"))
        col, hatch = GROUP_STYLE[grp]
        x = np.arange(len(keys))
        ax.bar(x, vals, bottom=bottom, color=col, hatch=hatch, edgecolor="white",
               label=GROUP_NAMES[grp])
        ax.plot(x, bottom + np.array(v6), "o", color="black", ms=2.5)
        bottom += np.array(vals)
    ax.set_xticks(np.arange(len(keys)), [f"{t}\nD+{h}" for t, h in keys], fontsize=7)
    ax.set_ylabel("share of the group attribution (%)")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    gfs = g[g["group"] == "nwp_gfs"]["share_pct"]
    diff, name, key = max(diffs)
    facts = {"gfs_min": float(gfs.min()), "gfs_max": float(gfs.max()), "diff": diff,
             "diff_group": name, "diff_key": key}
    return fig, facts


@register(
    "S09-16", "9", "tft_same_hour", "TFT attention on the same hour of earlier days",
    "Do the TFT models attend to the same solar hour of earlier days, beyond what a flat "
    "attention would give?",
    "Ratio of the encoder attention on positions a whole number of days before each step's "
    "valid hour to the share of such positions (1 = no daily preference), per model, target "
    "and horizon block, on the validation forecasts of the development run.",
    "Every model puts more weight on the same hour of earlier days than a flat attention: the "
    "ratio is {rmin:.2f} to {rmax:.2f}. The first position of the window is {edge_min:.2f} to "
    "{edge_max:.2f} times the same hour of the later days (the window's edge, M22).",
    "Stage 11 reads the final models' attention the same way; agreement with SHAP is not "
    "claimed as validation (M29).",
    (ATTENTION,))
def s09_tft_same_hour():
    a = D.csv(ATTENTION)
    fig, ax = subplots(height=2.9)
    x0 = 0
    ticks, labels = [], []
    markers = {"1": "o", "2": "s", "3": "^"}
    for step, name in TFT:
        for t in ("indirect", "direct"):
            g = a[(a["step"] == step) & (a["target"] == t)]
            allb = g[g["block"] == "all"]
            if allb.empty:
                continue
            ax.bar(x0, float(allb["same_hour_ratio"].iloc[0]), 0.6, color=C["sky"],
                   edgecolor=C["blue"], hatch="//" if t == "direct" else "")
            for b, m in markers.items():
                r = g[g["block"] == b]
                if len(r):
                    ax.plot(x0 + (int(b) - 2) * 0.15, float(r["same_hour_ratio"].iloc[0]), m,
                            color="black", ms=3.5, label=f"D+{b}" if x0 == 0 else None)
            ticks.append(x0)
            labels.append(f"{name}\n{t}")
            x0 += 1
    ax.axhline(1.0, color=C["red"], ls="--", lw=1, label="flat attention")
    ax.set_xticks(ticks, labels, fontsize=6.5)
    ax.set_ylabel("same-hour ratio")
    ax.legend(fontsize=6.5)
    allb = a[a["block"] == "all"]
    assert (allb["same_hour_ratio"] > 1).all(), "a TFT without a daily preference"
    facts = {"rmin": float(allb["same_hour_ratio"].min()),
             "rmax": float(allb["same_hour_ratio"].max()),
             "edge_min": float(allb["first_position_ratio"].min()),
             "edge_max": float(allb["first_position_ratio"].max())}
    return fig, facts


@register(
    "S09-17", "9", "hparam_importance", "Hyper-parameter importance of the development studies",
    "Which hyper-parameters mattered in the development searches, and how stable is that "
    "reading?",
    "fANOVA importance (mean and std over ten evaluator seeds) per searched parameter: XGB "
    "averaged over its six studies (left); BiLSTM and the two TFTs per study (right), hatched "
    "where every trial lies inside TPE's random start-up.",
    "For XGB the most important parameter is {xgb_top} ({xgb_imp:.2f} on average). {n_startup} "
    "of {n_neural} neural studies lie entirely inside the random start-up, so their importances "
    "describe a handful of random trials only (M30).",
    "The final run (60 and 20 trials) gives the importances of the report; the development "
    "values decide nothing.",
    (HP_XGB, HP_NEURAL))
def s09_hparam_importance():
    hx = D.csv(HP_XGB).dropna(subset=["importance_mean"])
    hn = D.csv(HP_NEURAL).dropna(subset=["importance_mean"])
    hx = hx[hx["step"] == "xgb"]
    fig, axes = subplots(1, 2, height=3.2, width=7.6)
    m = hx.groupby("param")["importance_mean"].mean().sort_values()
    s = hx.groupby("param")["importance_std"].mean().reindex(m.index)
    axes[0].barh(m.index, m.values, xerr=s.values, color=C["blue"], ecolor="black")
    axes[0].set_xlabel("importance (mean of 6 studies)")
    axes[0].set_title("XGB-dev", fontsize=8)
    studies = hn.drop_duplicates(["step", "target", "seed"])
    y = 0
    yt, yl = [], []
    for r in studies.itertuples():
        g = hn[(hn["step"] == r.step) & (hn["target"] == r.target)].sort_values(
            "importance_mean", ascending=False).head(3)
        for q in g.itertuples():
            axes[1].barh(y, q.importance_mean, xerr=q.importance_std, color=C["orange"],
                         hatch="//" if r.all_in_startup else "", edgecolor="white",
                         ecolor="black")
            yt.append(y)
            yl.append(f"{r.step} {r.target[:3]}: {q.param}")
            y -= 1
        y -= 0.5
    axes[1].set_yticks(yt, yl, fontsize=5.5)
    axes[1].set_xlabel("importance (top three per study)")
    axes[1].set_title("neural studies", fontsize=8)
    facts = {"xgb_top": m.index[-1], "xgb_imp": float(m.iloc[-1]),
             "n_startup": int(studies["all_in_startup"].sum()), "n_neural": int(len(studies))}
    return fig, facts


@register(
    "S09-18", "9", "final_time", "Expected duration of the final run",
    "How long will the final run take, step by step, and what is its worst case?",
    "Duration of each step of a final run, scaled from its development run by the number of "
    "fits of the two budgets (bars), and the upper bound if every neural fit ran to its epoch "
    "cap and every XGB trial took as long as the longest development trial (whiskers).",
    "About {total:.1f} h expected and up to about {upper:.1f} h; the longest step is {longest} "
    "(about {longest_h:.1f} h, up to {longest_up:.1f} h), which cannot be split.",
    "docs/runbook_final.md runs the final in one sequence or in segments, by the answer of the "
    "college IT (Q-15).",
    DEV_STEP_FILES)
def s09_final_time():
    from solarpv.config.paths import REPO_ROOT
    from solarpv.pipeline.runner import scale_dev_to_final

    rows = []
    for rel in DEV_STEP_FILES:
        folder = (REPO_ROOT / rel).parent
        rec = D.jload(rel)
        ratio, upper = scale_dev_to_final(folder.name, folder)
        rows.append((folder.name, rec["duration_s"] * ratio / 3600,
                     rec["duration_s"] * upper / 3600))
    fig, ax = subplots(height=2.8)
    names = [r[0] for r in rows]
    exp = np.array([r[1] for r in rows])
    up = np.array([r[2] for r in rows])
    ax.barh(names, exp, xerr=[np.zeros_like(exp), up - exp], color=C["sky"], ecolor="black",
            capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("hours")
    i = int(np.argmax(exp))
    facts = {"total": float(exp.sum()), "upper": float(up.sum()), "longest": names[i],
             "longest_h": float(exp[i]), "longest_up": float(up[i])}
    return fig, facts


@register(
    "S09-19", "9", "report_warnings", "Automatic warnings of the development runs",
    "What did the automatic checks of §3.5 find in the two development runs, and do the "
    "report's recomputed warnings match what the steps logged?",
    "Warnings of the XGB and neural development runs by kind (rows) and step (bars), from the "
    "run reports; the hatched part is a warning that the steps logged and the report "
    "reproduced from the stored records.",
    "{n_total} warnings in all, {n_matched} reproduced exactly from the records; {n_step_only} "
    "are kinds the report does not recompute (test access, missing decisions), and "
    "{n_recomputed_only} appear only in the recomputation ({why}).",
    "The final run's report applies the same checks; every warning goes to the run journal "
    "before the stage-10 acceptance.",
    WARN)
def s09_report_warnings():
    import pandas as pd

    w = pd.concat([D.csv(p) for p in WARN], ignore_index=True)
    counts = w.groupby(["kind", "step"]).size().unstack(fill_value=0)
    matched = w[w["status"] == "matched"].groupby(["kind", "step"]).size().unstack(
        fill_value=0).reindex_like(counts).fillna(0)
    fig, ax = subplots(height=3.0)
    left = np.zeros(len(counts))
    for k, step in enumerate(counts.columns):
        col = list(C.values())[1 + k % 8]
        ax.barh(counts.index, counts[step], left=left, color=col, edgecolor="white",
                label=step)
        ax.barh(counts.index, matched[step], left=left, color="none", hatch="///",
                edgecolor="black", lw=0)
        left += counts[step].to_numpy()
    ax.set_xlabel("warnings")
    ax.legend(fontsize=6.5, ncol=3)
    only = w[w["status"] == "recomputed only"]
    why = ("the effective hidden_continuous_size, X45" if len(only) and (
        only["kind"] == "edge_space").all() else "see the reports") if len(only) else "none"
    facts = {"n_total": int(len(w)), "n_matched": int((w["status"] == "matched").sum()),
             "n_step_only": int((w["status"] == "step log only").sum()),
             "n_recomputed_only": int(len(only)), "why": why}
    return fig, facts


@register(
    "C-10", "correction", "shap_groups_v6", "SHAP groups: the submission against the rebuild",
    "Which feature groups led the submitted SHAP analysis, and which lead the rebuild's?",
    "Share of each feature group at D+1: the submitted table (sum of the members' mean |SHAP| "
    "of the 76-feature model) against the rebuild's group attribution of XGB-dev.",
    "In the submission rolling statistics led with {v6_roll_ind:.0f} % (indirect) and "
    "{v6_roll_dir:.0f} % (direct); those features leak at D+2 and D+3 (F-07) and the model had "
    "76 inputs (C-31). In the rebuild the GFS forecasts lead with {rb_gfs_ind:.0f} % and "
    "{rb_gfs_dir:.0f} %, and rolling statistics hold {rb_roll_max:.0f} %.",
    "docs/defense_notes.md §8; the v7 text of §11.1 is written from the final run (C-59, C-60).",
    (V6_GROUPS, SHAP_GROUPS), kind="correction")
def c10_shap_groups():
    v6 = D.csv(V6_GROUPS)
    rb = D.csv(SHAP_GROUPS)
    rb = rb[(rb["seed"] == DEV_SEED) & (rb["horizon"] == 1)]
    v6 = v6[v6["horizon"] == 1]
    fig, axes = subplots(1, 2, height=3.0, sharey=True)
    groups = list(GROUP_STYLE)
    for ax, t in zip(axes, ("indirect", "direct"), strict=True):
        a = v6[v6["target"] == t].set_index("group")["pct"]
        b = rb[rb["target"] == t].set_index("group")["share_pct"]
        y = np.arange(len(groups))
        ax.barh(y + 0.2, [a.get(g, 0.0) for g in groups], 0.38, color=C["grey"],
                hatch="//", edgecolor="white", label="submission (v6)")
        ax.barh(y - 0.2, [b.get(g, 0.0) for g in groups], 0.38, color=C["blue"],
                label="rebuild (XGB-dev)")
        ax.set_yticks(y, [GROUP_NAMES[g] for g in groups], fontsize=7)
        ax.set_title(f"{t}, D+1", fontsize=8)
        ax.set_xlabel("share (%)")
    axes[1].legend(fontsize=7)
    roll = rb[rb["group"] == "rolling_stats"]["share_pct"]
    facts = {"v6_roll_ind": float(v6[(v6["target"] == "indirect")
                                     & (v6["group"] == "rolling_stats")]["pct"].iloc[0]),
             "v6_roll_dir": float(v6[(v6["target"] == "direct")
                                     & (v6["group"] == "rolling_stats")]["pct"].iloc[0]),
             "rb_gfs_ind": float(rb[(rb["target"] == "indirect")
                                    & (rb["group"] == "nwp_gfs")]["share_pct"].iloc[0]),
             "rb_gfs_dir": float(rb[(rb["target"] == "direct")
                                    & (rb["group"] == "nwp_gfs")]["share_pct"].iloc[0]),
             "rb_roll_max": float(roll.max()) if len(roll) else 0.0}
    assert facts["rb_gfs_ind"] > facts["rb_roll_max"]
    return fig, facts

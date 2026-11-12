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
SELECT = (f"{XDEV}/select/select.json", f"{NDEV}/select/select.json")
RANKS = f"{XDEV}/explain/xgb/xgb_shap_rank_stability.csv"
VSN = f"{NDEV}/explain/tft/tft_variable_importance.csv"
G1 = (f"{XDEV}/report/report_g1_points.csv", f"{NDEV}/report/report_g1_points.csv")
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
    fig, ax = subplots(height=3.0, width=8.0)
    blank(ax)
    w = 0.96 / len(phase1)
    for i, step in enumerate(phase1):
        x = 0.02 + i * w
        box(ax, x + 0.006, 0.58, w - 0.022, 0.16, step, face=C["sky"] + "55", size=6)
        ax.plot(x + w / 2 - 0.004, 0.80, "v", color=C["red"], ms=4)
    ax.text(0.02, 0.88, "phase 1 (stage 10): train, forecast validation and test, score "
            "validation only", fontsize=8)
    ax.text(0.02, 0.48, "▼ gate before every step: HEAD carries the tag, clean tree, the "
            "preflight's commit (X38)", fontsize=7, color=C["red"])
    box(ax, 0.36, 0.08, 0.27, 0.24, f"stage-10 acceptance\nwritten to {ACCEPTANCE_FILE}\n"
        "(untracked, X51)", face="white", edge=C["grey"], size=7)
    box(ax, 0.70, 0.08, 0.28, 0.24, "phase 2: step test\n--allow-test; scores the test split,\n"
        "XGB SHAP on test; logged", face=C["orange"] + "55", size=7)
    last = 0.02 + (len(phase1) - 0.5) * w  # the centre of the report step
    arrow(ax, last, 0.58, 0.495, 0.32)
    arrow(ax, 0.63, 0.20, 0.70, 0.20)
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
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.5)
    assert c["n_rows"].nunique() == 1
    facts = {"top_ind": tops["indirect"], "top_dir": tops["direct"],
             "add": float(c["additivity_relative"].max()), "n_rows": int(c["n_rows"].iloc[0])}
    return fig, facts


@register(
    "S09-15", "9", "xgb_shap_groups", "Attribution of the feature groups",
    "How is the XGB-dev attribution shared between the feature groups, and does the measure "
    "matter?",
    "Share of each feature group in the group attribution, the mean over hours of |sum of "
    "SHAP over the group| (wide bars), per target and horizon (seed 42); the narrow bars show "
    "the shares that the sum of the members' mean |SHAP| would give on the same values.",
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
    assert (g["sum_member_mean_abs_kw"] >= g["mean_abs_group_kw"] - 1e-9).all()  # |sum| <= sum
    bottom, bottom6 = np.zeros(len(keys)), np.zeros(len(keys))
    diffs = []
    x = np.arange(len(keys))
    for grp in groups:
        vals, v6 = [], []
        for t, h in keys:
            r = g[(g["target"] == t) & (g["horizon"] == h) & (g["group"] == grp)]
            vals.append(float(r["share_pct"].iloc[0]) if len(r) else 0.0)
            v6.append(float(r["share_pct_v6_measure"].iloc[0]) if len(r) else 0.0)
            diffs.append((abs(vals[-1] - v6[-1]), GROUP_NAMES[grp], f"{t} D+{h}"))
        col, hatch = GROUP_STYLE[grp]
        ax.bar(x - 0.12, vals, 0.55, bottom=bottom, color=col, hatch=hatch, edgecolor="white",
               label=GROUP_NAMES[grp])
        ax.bar(x + 0.3, v6, 0.14, bottom=bottom6, color=col, hatch=hatch, edgecolor="white",
               alpha=0.6)
        bottom += np.array(vals)
        bottom6 += np.array(v6)
    ax.set_xticks(x, [f"{t}\nD+{h}" for t, h in keys], fontsize=7)
    ax.set_ylabel("share of the group attribution (%)")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    gfs = g[g["group"] == "nwp_gfs"]["share_pct"]
    diff, name, key = max(diffs)
    facts = {"gfs_min": float(gfs.min()), "gfs_max": float(gfs.max()), "diff": diff,
             "diff_group": name, "diff_key": key}
    return fig, facts


@register(
    "S09-16", "9", "tft_same_hour", "TFT attention: same hour or fixed clock hours?",
    "Do the TFT models attend to the same solar hour of earlier days as the hour they forecast, "
    "or to fixed hours of the encoder?",
    "Left: ratio of the encoder attention on positions a whole number of days before each "
    "step's valid hour to the share of such positions (1 = no preference), per model, target "
    "and horizon block. Right: the mean attention by the local clock hour of the encoder "
    "position, direct route. Validation forecasts of the development run.",
    "No model prefers the hour it forecasts: the same-hour ratio is {rmin:.2f} to {rmax:.2f} "
    "({n_below} of {n_pairs} model-target pairs below 1). The daily rhythm of the attention "
    "(S09-11) is a preference for fixed clock hours of earlier days: in the direct route the "
    "profile peaks at {peak96}:00 (TFT-96), {peak168}:00 (TFT-168) and {peakbase}:00 (TFT "
    "baseline) local time, whatever hour is forecast.",
    "Stage 11 reads the final models' attention the same way; agreement with SHAP is not "
    "claimed as validation (M29, C-62).",
    (ATTENTION, f"{NDEV}/explain/tft/tft_attention_lag.csv"))
def s09_tft_same_hour():
    a = D.csv(ATTENTION)
    lag = D.csv(f"{NDEV}/explain/tft/tft_attention_lag.csv")
    fig, (ax, ax2) = subplots(1, 2, height=2.9, width=7.8)
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
    ax.axhline(1.0, color=C["red"], ls="--", lw=1, label="no preference")
    ax.set_xticks(ticks, labels, fontsize=6)
    ax.set_ylabel("same-hour ratio")
    ax.legend(fontsize=6.5)
    allb = a[a["block"] == "all"]
    peaks = {}
    styles = {"tft96": (C["blue"], "-"), "tft168": (C["sky"], "--"), "tftbase": (C["orange"], ":")}
    for step, name in TFT:
        g = lag[(lag["step"] == step) & (lag["target"] == "direct") & (lag["block"] == "all")]
        hour = (g["lag_h"] % 24).astype(int)  # T0 is local midnight: lag -k is clock hour -k mod 24
        prof = g.groupby(hour)["attention"].mean()
        col, ls = styles[step]
        ax2.plot(prof.index, prof.to_numpy() / prof.mean(), ls, color=col, label=name)
        peaks[step] = int(prof.idxmax())
    ax2.axhline(1.0, color="0.6", lw=0.8)
    ax2.set_xticks(range(0, 24, 6))
    ax2.set_xlabel("local clock hour of the encoder position")
    ax2.set_ylabel("attention / its mean")
    ax2.legend(fontsize=6.5)
    assert allb["same_hour_ratio"].max() < 1.2, "a TFT with a clear same-hour preference"
    facts = {"rmin": float(allb["same_hour_ratio"].min()),
             "rmax": float(allb["same_hour_ratio"].max()),
             "n_below": int((allb["same_hour_ratio"] < 1).sum()), "n_pairs": int(len(allb)),
             "peak96": peaks["tft96"], "peak168": peaks["tft168"],
             "peakbase": peaks["tftbase"]}
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
    "{n_total} warnings in all, {n_matched} reproduced exactly from the records; step log "
    "only (kinds the report does not recompute): {n_step_only}; in the recomputation only: "
    "{n_recomputed_only} ({why}).",
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
    ax.barh([], [], color="none", hatch="///", edgecolor="black", lw=0,
            label="reproduced from the records")
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
    "S09-20", "9", "select_checks", "What the select step verifies",
    "Does the runner use the inputs that stage 7 selected, without selecting again?",
    "Left: the number of model inputs per route and horizon that the select step recorded, "
    "hatched where the route uses the full pool rather than a selected set. Right: the checks "
    "of the step, as the number passed out of the number made, in the XGB development run.",
    "The step read {rows_read} data rows and reran nothing. {tracked_ok} of {tracked} stage-7 "
    "files are tracked and unmodified, {tables_ok} of {tables} tables are the ones the "
    "selection ran on, and the selection read the {splits} split only. The indirect route "
    "uses the 05e sets ({ind_lo} to {ind_hi} inputs), the direct route the full pool "
    "({dir_lo} to {dir_hi}); both development runs recorded the same inputs. Selection source "
    "files changed after the selection ran: {n_code}, listed in the record.",
    "The model steps compare the inputs file with the hash in the select record before they "
    "train; a final run refuses to train without that record (D-30).",
    SELECT)
def s09_select_checks():
    a, b = D.jload(SELECT[0]), D.jload(SELECT[1])
    assert a["inputs"] == b["inputs"] and a["model_inputs_sha256"] == b["model_inputs_sha256"]
    assert a["rows_read"] == 0 and all(r["matches"] for r in a["inputs"])
    splits = a["manifest"]["splits_read_for_selection"]
    assert splits == ["train"]
    inputs = a["inputs"]
    ind = [r["n_features"] for r in inputs if r["key"].startswith("indirect")]
    dirn = [r["n_features"] for r in inputs if r["key"].startswith("direct")]
    assert {r["source"] for r in inputs if r["key"].startswith("indirect")} == {"05e"}
    assert {r["source"] for r in inputs if r["key"].startswith("direct")} == {"full pool"}
    fig, (ax, ax2) = subplots(1, 2, height=2.9, width=7.8)
    for i, r in enumerate(inputs):
        pool = r["source"] == "full pool"
        ax.bar(i, r["n_features"], 0.6, color=C["orange"] if pool else C["blue"],
               hatch="//" if pool else "", edgecolor="white")
        ax.text(i, r["n_features"], "✓" if r["matches"] else "✗", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(range(len(inputs)), [r["key"].replace("_", "\n") for r in inputs], fontsize=7)
    ax.set_ylabel("model inputs")
    ax.legend(handles=[Patch(color=C["blue"], label="05e set"),
                       Patch(facecolor=C["orange"], hatch="//", edgecolor="white",
                             label="full pool")], fontsize=7, loc="upper left")
    checks = [("stage-7 files tracked, unmodified", sum(r["ok"] for r in a["tracked"]),
               len(a["tracked"])),
              ("tables of the selection", sum(r["ok"] for r in a["tables"]), len(a["tables"])),
              ("inputs equal the recorded sets", sum(r["matches"] for r in inputs), len(inputs))]
    for k, (_name, ok, n) in enumerate(checks):
        ax2.barh(-k, n, color=C["light"], edgecolor=C["grey"])
        ax2.barh(-k, ok, 0.5, color=C["green"])
        ax2.text(n, -k, f"  {ok} of {n}", va="center", fontsize=7)
    ax2.set_yticks([-k for k in range(len(checks))], [c[0] for c in checks], fontsize=7)
    ax2.set_xlabel("checks passed")
    ax2.set_xlim(0, max(c[2] for c in checks) * 1.35)
    ax2.set_title(f"data rows read: {a['rows_read']}; selection read: {', '.join(splits)}",
                  fontsize=8)
    facts = {"rows_read": a["rows_read"], "tracked_ok": checks[0][1], "tracked": checks[0][2],
             "tables_ok": checks[1][1], "tables": checks[1][2], "splits": ", ".join(splits),
             "ind_lo": min(ind), "ind_hi": max(ind), "dir_lo": min(dirn), "dir_hi": max(dirn),
             "n_code": len(a["code_changed_since_selection"]),
             "code": a["code_changed_since_selection"]}
    return fig, facts


def _key(r) -> str:
    return f"{r.target}\nD+{r.horizon}"


@register(
    "S09-21", "9", "shap_booster_rounds", "SHAP explains the booster of the forecast",
    "Does SHAP add up to the forecast only when the explainer gets the model that made it?",
    "Largest additivity error of TreeSHAP per target and horizon (seed 42, validation) as a "
    "fraction of the output scale (1 for k, the capacity for kW), log axis: the saved booster "
    "explained whole (hatched) and cut at the rounds the forecast uses (solid). The labels give "
    "the rounds used and the rounds saved.",
    "In {n_extra} of {n_models} models the saved booster holds {extra} rounds more than the "
    "forecast uses (the rounds after the best one, kept by early stopping). Explained whole, it "
    "misses the forecast by up to {unsliced_kw:.0f} kW in the direct route and {unsliced_k:.2f} "
    "in k in the indirect route. Cut at the forecast's rounds, SHAP adds up within "
    "{sliced:.1e} of the scale, XGBoost's own contributions agree exactly, and the booster "
    "reproduces the written forecast.",
    "explain cuts the booster before TreeSHAP and checks additivity against the uncorrected "
    "forecast (X46, T-87).",
    (SHAP_CHECKS,))
def s09_shap_booster_rounds():
    from solarpv.pipeline.explain import SCALE

    c = D.csv(SHAP_CHECKS)
    c = c[c["seed"] == DEV_SEED].reset_index(drop=True)
    assert (c["contribs_max_diff"] == 0).all() and (c["written_forecast_max_diff"] == 0).all()
    scale = c["unit"].map(SCALE)
    unsliced = c["unsliced_additivity_max"] / scale
    extra = c["rounds_in_booster"] - c["n_rounds"]
    with_extra = extra > 0
    assert (unsliced[with_extra] > 100 * c["additivity_relative"][with_extra]).all()
    assert np.allclose(unsliced[~with_extra], c["additivity_relative"][~with_extra])
    assert extra[with_extra].nunique() == 1
    fig, ax = subplots(height=3.0)
    x = np.arange(len(c))
    ax.bar(x - 0.2, unsliced, 0.38, color=C["red"], hatch="//", edgecolor="white",
           label="saved booster, whole")
    ax.bar(x + 0.2, c["additivity_relative"], 0.38, color=C["blue"],
           label="cut at the forecast's rounds")
    for i, r in enumerate(c.itertuples()):
        ax.text(i, unsliced[i] * 1.6, f"{r.n_rounds} / {r.rounds_in_booster}", ha="center",
                fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(c["additivity_relative"].min() / 5, unsliced.max() * 8)
    ax.set_xticks(x, [_key(r) for r in c.itertuples()], fontsize=7)
    ax.set_ylabel("largest |SHAP sum − forecast| / scale")
    fig.legend(loc="outside upper center", ncol=2, fontsize=7)
    direct, indirect = c["target"] == "direct", c["target"] == "indirect"
    facts = {"n_extra": int(with_extra.sum()), "n_models": int(len(c)),
             "extra": int(extra[with_extra].iloc[0]),
             "unsliced_kw": float(c.loc[direct, "unsliced_additivity_max"].max()),
             "unsliced_k": float(c.loc[indirect, "unsliced_additivity_max"].max()),
             "sliced": float(c["additivity_relative"].max())}
    return fig, facts


@register(
    "S09-22", "9", "shap_robustness", "Checks of the XGB explanation",
    "Does one feature dominate the development XGBoost, and do the horizons rank their common "
    "inputs alike?",
    "Left: the first feature's mean |SHAP| over the third's, per target and horizon (seed 42), "
    "against the stage-7 red flag (dashed). Right: Spearman correlation of the mean |SHAP| "
    "ranking over the features that two horizons share, per target.",
    "No feature dominates: the first is at most {dom_max:.1f} times the third ({dom_key}), "
    "below the red flag of {flag:.0f}. The horizons rank their common inputs alike (Spearman "
    "{rho_min:.2f} to {rho_max:.2f}). The development run has one seed; the final run adds the "
    "comparison across its three seeds.",
    "The report flags a dominant feature in the final run (the stage-7 rule, M28).",
    (SHAP_CHECKS, RANKS))
def s09_shap_robustness():
    from solarpv.pipeline.explain import DOMINANCE_FACTOR

    c = D.csv(SHAP_CHECKS)
    c = c[c["seed"] == DEV_SEED].reset_index(drop=True)
    r = D.csv(RANKS)
    r = r[r["compare"].str.startswith("D+")]
    assert c["dominance_first_over_third"].max() < DOMINANCE_FACTOR
    fig, (ax, ax2) = subplots(1, 2, height=2.9, width=7.8)
    x = np.arange(len(c))
    ax.bar(x, c["dominance_first_over_third"], 0.6, color=C["sky"], edgecolor=C["blue"])
    ax.axhline(DOMINANCE_FACTOR, color=C["red"], ls="--", lw=1, label="stage-7 red flag")
    ax.set_xticks(x, [_key(q) for q in c.itertuples()], fontsize=7)
    ax.set_ylabel("first / third feature")
    ax.set_ylim(0, DOMINANCE_FACTOR * 1.15)
    ax.legend(fontsize=7, loc="upper left")
    styles = {"indirect": (C["blue"], "o", "-"), "direct": (C["orange"], "s", "--")}
    pairs = sorted(r["compare"].unique())
    for t, (col, m, ls) in styles.items():
        g = r[r["target"] == t].set_index("compare").reindex(pairs)
        ax2.plot(range(len(pairs)), g["spearman"], ls, marker=m, color=col, label=t)
    ax2.set_xticks(range(len(pairs)), pairs, fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Spearman of the rankings")
    ax2.legend(fontsize=7, loc="lower left")
    i = int(c["dominance_first_over_third"].idxmax())
    facts = {"dom_max": float(c["dominance_first_over_third"].max()),
             "dom_key": f"{c.loc[i, 'target']} D+{c.loc[i, 'horizon']}",
             "flag": float(DOMINANCE_FACTOR), "rho_min": float(r["spearman"].min()),
             "rho_max": float(r["spearman"].max())}
    return fig, facts


def _vsn_group(variable: str) -> str:
    from solarpv.models.sequence_data import K_SERIES, SERIES
    from solarpv.pipeline.explain import feature_group

    if variable in SERIES or variable == K_SERIES:
        return "measured"  # the measured series of the encoder (W3), not a stage-6 feature
    return feature_group(variable)


@register(
    "S09-23", "9", "tft_variable_selection", "What the TFT models select",
    "Which input groups do the variable-selection networks of the three TFTs weight, and do the "
    "models agree?",
    "Share of the variable-selection weight per input group (a measured series and its fill "
    "flag counted as one variable), per model and target, in the encoder (past days) and the "
    "decoder (forecast days), averaged over the validation windows of the development run.",
    "In the decoder the GFS forecasts take {dec_gfs_min:.0f} to {dec_gfs_max:.0f} % of the "
    "weight; in the encoder the measured series take {enc_meas_min:.0f} to {enc_meas_max:.0f} %. "
    "The three models do not agree on their top input: in the direct decoder they pick "
    "{top_direct}. The largest single weight, {wmax:.2f}, goes to {wvar} ({wmodel}). A selection "
    "weight gates an input inside the network; it is not the input's effect on the forecast, "
    "and the development fits have one seed.",
    "Stage 11 reads the final models' weights with their seed spread; no input decision is "
    "taken from them (M29).",
    (VSN,))
def s09_tft_variable_selection():
    v = D.csv(VSN).drop_duplicates(["step", "target", "part", "variable"])
    v["group"] = v["variable"].map(_vsn_group)
    totals = v.groupby(["step", "target", "part"])["variable_weight"].sum()
    assert np.allclose(totals, 1.0, atol=1e-6), "weights per part do not add up to one"
    share = (v.groupby(["step", "target", "part", "group"])["variable_weight"].sum() * 100)
    styles = {**GROUP_STYLE, "measured": (C["black"], "||")}
    names = {**GROUP_NAMES, "measured": "measured series"}
    groups = [g for g in styles if g in set(v["group"])]
    keys = [(s, n, t) for s, n in TFT for t in ("indirect", "direct")]
    fig, axes = subplots(1, 2, height=3.3, width=8.0, sharey=True)
    for ax, part in zip(axes, ("encoder", "decoder"), strict=True):
        bottom = np.zeros(len(keys))
        for grp in groups:
            vals = np.array([share.get((s, t, part, grp), 0.0) for s, _, t in keys])
            col, hatch = styles[grp]
            ax.bar(range(len(keys)), vals, 0.65, bottom=bottom, color=col, hatch=hatch,
                   edgecolor="white", label=names[grp] if part == "encoder" else None)
            bottom += vals
        ax.set_xticks(range(len(keys)), [f"{n}\n{t}" for _, n, t in keys], fontsize=6)
        ax.set_title(part, fontsize=8)
    axes[0].set_ylabel("share of the selection weight (%)")
    fig.legend(loc="outside lower center", ncol=5, fontsize=6.5)
    dec = share.xs("decoder", level="part")
    enc = share.xs("encoder", level="part")
    dec_gfs = [dec.get((s, t, "nwp_gfs"), 0.0) for s, _, t in keys]
    enc_meas = [enc.get((s, t, "measured"), 0.0) for s, _, t in keys]
    tops = {}
    for s, n in TFT:
        g = v[(v["step"] == s) & (v["target"] == "direct") & (v["part"] == "decoder")]
        tops[n] = g.loc[g["variable_weight"].idxmax(), "variable"]
    assert len(set(tops.values())) > 1, "the three TFTs agree on their top input"
    big = v.loc[v["variable_weight"].idxmax()]
    model = dict(TFT)[big["step"]]
    facts = {"dec_gfs_min": float(min(dec_gfs)), "dec_gfs_max": float(max(dec_gfs)),
             "enc_meas_min": float(min(enc_meas)), "enc_meas_max": float(max(enc_meas)),
             "top_direct": ", ".join(f"{var} ({n})" for n, var in tops.items()),
             "wmax": float(big["variable_weight"]), "wvar": big["variable"],
             "wmodel": f"{model}, {big['target']} {big['part']}"}
    return fig, facts


@register(
    "S09-24", "9", "g1_points", "Points the run reports raise for G-1",
    "What do the automatic run reports put on the list for the G-1 audit?",
    "Number of points per kind in the report of each development run (XGB and neural), as the "
    "report step lists them; a point is a fact to audit, not a decision.",
    "The two reports list {n_total} points of {n_kinds} kinds ({n_xgb} XGB, {n_neural} "
    "neural); the most frequent kind is {top_kind} ({top_n}). The list stays a list until the "
    "G-1 audit; no validation decision changes because of it without the user.",
    "W6 audits every point at G-1 before the tag (PROJECT_MASTER.md §1).",
    G1)
def s09_g1_points():
    import pandas as pd

    x, n = D.csv(G1[0]), D.csv(G1[1])
    both = pd.concat([x.assign(run="XGB dev"), n.assign(run="neural dev")], ignore_index=True)
    counts = both.groupby(["point", "run"]).size().unstack(fill_value=0)
    counts = counts.reindex(counts.sum(axis=1).sort_values().index)
    fig, ax = subplots(height=3.0)
    left = np.zeros(len(counts))
    for run, col, hatch in (("XGB dev", C["blue"], ""), ("neural dev", C["orange"], "//")):
        vals = counts[run].to_numpy() if run in counts else np.zeros(len(counts))
        ax.barh(counts.index, vals, left=left, color=col, hatch=hatch, edgecolor="white",
                label=run)
        left += vals
    ax.set_xlabel("points")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    tot = counts.sum(axis=1)
    facts = {"n_total": int(len(both)), "n_kinds": int(len(counts)), "n_xgb": int(len(x)),
             "n_neural": int(len(n)), "top_kind": str(tot.idxmax()), "top_n": int(tot.max())}
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
    "docs/defense_notes.md §10; the v7 text of §11.1 is written from the final run (C-59, "
    "C-60).",
    (V6_GROUPS, SHAP_GROUPS), kind="correction")
def c10_shap_groups():
    v6 = D.csv(V6_GROUPS)
    rb = D.csv(SHAP_GROUPS)
    rb = rb[(rb["seed"] == DEV_SEED) & (rb["horizon"] == 1)]
    v6 = v6[v6["horizon"] == 1]
    fig, axes = subplots(1, 2, height=3.0, sharey=True)
    groups = [g for g in GROUP_STYLE if g in set(v6["group"]) | set(rb["group"])]
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

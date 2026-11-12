"""Figures that put the six models side by side (rule 9 of docs/figures_plan.md), and the
figures that the look-back of 2026-11-12 found for earlier tasks (rule 8).

Every model-level figure here covers XGB and the five neural models in one layout and one style
(`MODELS`, `MODEL_STYLE`), on the validation hours that every model is scored on. A model that
has no such output is named in the figure. The XGB numbers come from the W5 XGB dev run, which
reproduces the W2 dev run byte for byte (R9-21); the neural numbers from the W4 dev run.
"""

import re

import numpy as np
import pandas as pd

from solarpv.config.constants import DEV_SEED, FINAL_OPTUNA_TRIALS
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.s07_s09 import NEURAL, NEURAL_STYLE, REF_NAMES, REF_STYLE
from solarpv.figures.style import C, subplots

XDEV = D.W5_XGB_DEV_RUN
NDEV = D.NEURAL_DEV_RUN
MODELS = (("xgb", "xgb_sota", "XGB"), *NEURAL)  # (step, model, display name)
MODEL_STYLE = {"xgb_sota": REF_STYLE["xgb_sota"], **NEURAL_STYLE}  # one style per model
NAMES = {m: n for _, m, n in MODELS}
RUNS = {"xgb_sota": XDEV, **{m: NDEV for _, m, _ in NEURAL}}
QUANTILE_STEPS = (("bilstm", "lstm_sota"), ("tft96", "tft_96"), ("tft168", "tft_168"),
                  ("tftbase", "tft"))
SEASONS = ("Winter", "Spring", "Summer", "Fall")  # the validation months, February to September


def _eval(run: str, name: str) -> str:
    return f"{run}/evaluate/val/{name}"


def _pred(model: str) -> str:
    step = next(s for s, m, _ in MODELS if m == model)
    return f"{RUNS[model]}/{step}/predictions_{model}_seed{DEV_SEED}.csv"


def _config(step: str, model: str) -> str:
    return f"{NDEV}/{step}/{model}_config.json"


# ── W4: the six models side by side ─────────────────────────────────────────

CARD_INPUTS = (f"{XDEV}/select/select.json", f"{XDEV}/xgb/xgb_config.json",
               *(_config(s, m) for s, m, _ in NEURAL))


@register(
    "S09-25", "9", "model_cards", "The six models side by side",
    "What does each model see, what does it output, and how was it tuned in the development "
    "run?",
    "One row per model: models per route, inputs per route (XGB: selected features per "
    "horizon; sequence models: encoder and decoder channels), the history window, the output "
    "(every sequence model's point forecast is its median), and the development search and "
    "training budgets, read from the run configurations.",
    "Every model is tuned on one objective, the daytime nRMSE in kW of its point forecast, and "
    "scored on the same hours. They differ in what they see: XGB {xgb_ind} features "
    "(indirect) and {xgb_dir} (direct) per horizon, the sequence models {enc_ind} and {enc_dir} "
    "encoder channels over {enc_min} to {enc_max} hours, and PatchTST only its own past series. "
    "XGB fits one model per horizon; each sequence model forecasts the three days at once.",
    "The final run (stage 10) keeps these inputs and raises the budgets to those of the "
    "submitted study; the report's model table is drawn from the final configurations.",
    CARD_INPUTS)
def s09_model_cards():
    sel = D.jload(CARD_INPUTS[0])
    xcfg = D.jload(CARD_INPUTS[1])
    n_in = {r["key"]: r["n_features"] for r in sel["inputs"]}
    ind = [n_in[f"indirect_D{h}"] for h in (1, 2, 3)]
    dirn = [n_in[f"direct_D{h}"] for h in (1, 2, 3)]
    assert "nRMSE in kW" in xcfg["objective"]
    stop = re.search(r"(\d+) rounds", xcfg["early_stopping"]).group(1)
    rows = [["XGB", "3 (one per horizon)", f"{min(ind)}–{max(ind)} features",
             f"{min(dirn)}–{max(dirn)} features", "lags in the features", "point",
             f"{xcfg['budget']['trials']} trials per study", f"stops after {stop} rounds"]]
    enc = {}
    for step, model, name in NEURAL:
        c = D.jload(_config(step, model))
        assert "nRMSE in kW" in c["objective"], f"{step}: another objective"
        di, dd = c["datasets"]["indirect/train"], c["datasets"]["direct/train"]
        enc[step] = (c["encoder_hours"], di["encoder_channels"], dd["encoder_channels"],
                     di["decoder_channels"], dd["decoder_channels"])
        q = c["quantiles"]
        assert c["point_forecast"] == "quantile 0.5", f"{step}: the point is not the median"
        out = "point" if q == [0.5] else f"{len(q)} quantiles"
        b = c["budget"]
        search = (f"{b['trials']} trials" if b.get("trials") else "fixed settings")
        train = (f"up to {b['max_steps']} steps" if "max_steps" in b
                 else f"up to {b.get('epochs_final', b.get('epochs'))} epochs")
        rows.append([name, "1 (three days at once)",
                     f"{di['encoder_channels']} enc / {di['decoder_channels']} dec",
                     f"{dd['encoder_channels']} enc / {dd['decoder_channels']} dec",
                     f"{c['encoder_hours']} h", out, search, train])
    assert enc["patchtst"][1:] == (1, 1, 0, 0), "PatchTST is no longer univariate"
    seq = {k: v for k, v in enc.items() if k != "patchtst"}
    assert len({v[1] for v in seq.values()}) == 1 and len({v[2] for v in seq.values()}) == 1
    fig, ax = subplots(height=2.6, width=8.6)
    ax.set_axis_off()
    cols = ["model", "models per route", "inputs, indirect", "inputs, direct", "history",
            "output", "dev search", "dev training"]
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(6.5)
    t.scale(1, 1.55)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_facecolor(C["light"])
    v = next(iter(seq.values()))
    facts = {"xgb_ind": f"{min(ind)}–{max(ind)}", "xgb_dir": f"{min(dirn)}–{max(dirn)}",
             "enc_ind": v[1], "enc_dir": v[2],
             "enc_min": min(e[0] for e in enc.values()),
             "enc_max": max(e[0] for e in enc.values())}
    return fig, facts


SIG_INPUTS = (_eval(XDEV, "dm_tests_kw.csv"), _eval(NDEV, "dm_tests_kw.csv"))


@register(
    "S09-26", "9", "significance_vs_persistence",
    "Every model against persistence: Diebold-Mariano in kW",
    "Is each model better than persistence beyond what chance in the validation hours allows?",
    "Grid of Diebold-Mariano tests on the squared error in kW (HLN correction): one row per "
    "model and reference, one column per route and horizon; green where the model is "
    "significantly better than persistence at the Bonferroni level of its run, red where "
    "significantly worse, grey where not significant; the text is the effect as a share of the "
    "minimum detectable effect.",
    "{n_all} of {n_models} models are significantly better than persistence in all six route "
    "and horizon cases; the fewest is {weakest} ({weakest_n} of 6). No model is significantly "
    "worse. The two development runs used different references, because the evaluate step "
    "takes XGB when it is in the run and persistence otherwise: the references against XGB "
    "(Bonferroni level {alpha_x:.4f}) and every neural model against persistence "
    "({alpha_n:.4f}). This grid puts both on persistence.",
    "The final run has every model in one run, so its kW tests compare every model with XGB in "
    "one family per route and horizon; they are the primary acceptance test (Q-19).",
    SIG_INPUTS)
def s09_significance():
    x, n = D.csv(SIG_INPUTS[0]), D.csv(SIG_INPUTS[1])
    rows = []
    # mean_diff is the loss of model_a minus the loss of model_b
    xr = x[x["model_a"] == "persistence"].assign(model=x["model_b"], sign=1.0)
    nr = n[n["model_b"] == "persistence"].assign(model=n["model_a"], sign=-1.0)
    both = pd.concat([xr, nr], ignore_index=True)
    order = [m for _, m, _ in MODELS] + ["climatology", "clearsky_k"]
    cols = [(t, h) for t in ("indirect", "direct") for h in (1, 2, 3)]
    fig, ax = subplots(height=3.4, width=7.6)
    count = {}
    worse = 0
    for i, m in enumerate(order):
        good = 0
        for j, (t, h) in enumerate(cols):
            r = both[(both["model"] == m) & (both["target"] == t) & (both["horizon"] == h)]
            assert len(r) == 1, (m, t, h)
            r = r.iloc[0]
            better = r["sign"] * r["mean_diff"] > 0
            sig = bool(r["significant"])
            face = (C["green"] if better else C["red"]) + "66" if sig else C["light"]
            good += int(sig and better)
            worse += int(sig and not better)
            ax.add_patch(_cell(j, i, face))
            ax.text(j, i, f"{r['pct_of_mde']:.0f} %", ha="center", va="center", fontsize=6.5)
        count[m] = good
        rows.append(NAMES.get(m, REF_NAMES.get(m, m).split(" (")[0]))
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), [f"{t}\nD+{h}" for t, h in cols], fontsize=7)
    ax.set_yticks(range(len(order)), rows, fontsize=7)
    ax.grid(False)
    ax.set_title("share of the minimum detectable effect; green: significantly better",
                 fontsize=8)
    assert worse == 0, "a model is significantly worse than persistence"
    models = [m for _, m, _ in MODELS]
    weakest = min(models, key=lambda m: count[m])
    facts = {"n_all": sum(count[m] == len(cols) for m in models), "n_models": len(models),
             "weakest": NAMES[weakest], "weakest_n": count[weakest],
             "alpha_x": float(x["alpha_bonferroni"].iloc[0]),
             "alpha_n": float(n["alpha_bonferroni"].iloc[0])}
    return fig, facts


def _cell(x, y, face):
    from matplotlib.patches import Rectangle
    return Rectangle((x - 0.48, y - 0.45), 0.96, 0.9, fc=face, ec="white", lw=1)


ROUTE_INPUTS = (_eval(XDEV, "route_comparison.csv"), _eval(NDEV, "route_comparison.csv"))


@register(
    "S09-27", "9", "route_per_model", "The two routes, model by model",
    "For each model, is the indirect route (k, converted to kW) or the direct route (kW) "
    "better on the same hours?",
    "Validation nRMSE in kW per horizon of the indirect route converted to power (circles) and "
    "the direct route (squares), paired on the same hours with the actual AC power as truth, "
    "one panel per model.",
    "At D+1 the direct route is better for {n_direct} of {n_models} models and the indirect "
    "route for {n_indirect}; the largest gap is {gap:.2f} points ({gap_model}, D+{gap_h}). "
    "Development budgets and one seed: the route of the final models is compared again in "
    "stage 10.",
    "Both routes stay in the final run, as in the submitted study (§6.11); the paired kW "
    "comparison is part of the report.",
    ROUTE_INPUTS)
def s09_routes():
    rc = pd.concat([D.csv(p) for p in ROUTE_INPUTS], ignore_index=True)
    fig, axes = subplots(1, len(MODELS), height=2.6, width=8.4, sharey=True)
    n_dir = n_ind = 0
    gaps = []
    for ax, (_, m, name) in zip(axes, MODELS, strict=True):
        g = rc[rc["model"] == m]
        col, _ = MODEL_STYLE[m]
        vals = {}
        for route, mk, ls in (("indirect_to_power", "o", ":"), ("direct", "s", "-")):
            r = g[g["route"] == route].sort_values("horizon")
            assert r["n"].nunique() == 1
            ax.plot(r["horizon"], r["nRMSE_pct"], ls, marker=mk, color=col, ms=4, lw=1,
                    label="indirect, in kW" if route != "direct" else "direct")
            vals[route] = r.set_index("horizon")["nRMSE_pct"]
        d1 = vals["indirect_to_power"][1] - vals["direct"][1]
        n_dir += d1 > 0
        n_ind += d1 < 0
        for h in (1, 2, 3):
            gaps.append((abs(vals["indirect_to_power"][h] - vals["direct"][h]), name, h))
        ax.set_title(name, fontsize=8)
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"], fontsize=7)
    axes[0].set_ylabel("validation nRMSE (% of 3,449 kW)\naxis does not start at zero")
    axes[0].legend(fontsize=6)
    gap, gmodel, gh = max(gaps)
    facts = {"n_direct": int(n_dir), "n_indirect": int(n_ind), "n_models": len(MODELS),
             "gap": float(gap), "gap_model": gmodel, "gap_h": int(gh)}
    return fig, facts


DAY_INPUTS = tuple(_pred(m) for _, m, _ in MODELS) + (
    f"{NDEV}/baselines/predictions_persistence.csv",)


@register(
    "S09-28", "9", "example_days", "Forecasts on three validation days",
    "How do the six models' D+1 forecasts follow the measured power on a clear, a typical and "
    "an overcast day?",
    "Direct route, D+1: measured AC power (shaded) and the forecasts of every model and of "
    "persistence (grey, dashed) over the local hours of three validation days. The days are "
    "chosen by a rule, not by eye: the highest, the median and the lowest daily mean clear-sky "
    "index among the days with every daytime hour scored.",
    "On the clear day ({d_clear}, mean k {k_clear:.2f}) the daily RMSE of the models is "
    "{c_lo:.0f} to {c_hi:.0f} kW; on the median day ({d_mid}) {m_lo:.0f} to {m_hi:.0f} kW; on "
    "the overcast day ({d_cloud}, mean k {k_cloud:.2f}) {o_lo:.0f} to {o_hi:.0f} kW, where "
    "persistence misses by {o_pers:.0f} kW. Three days illustrate; the scores are in S09-09.",
    "The final report shows the same three-day panel for the final models on the test split, "
    "with the days chosen by the same rule.",
    DAY_INPUTS)
def s09_example_days():
    from solarpv.evaluation.truth import load_truth

    tz = "Asia/Jerusalem"
    truth = load_truth(exclude_splits=("test",))
    t = truth[(truth["split"] == "val") & truth["daytime"].astype(bool)]
    preds = {}
    for _, m, _ in MODELS:
        p = D.csv(_pred(m), parse_dates=["timestamp"])
        p = p[(p["target"] == "direct") & (p["horizon"] == 1)].set_index("timestamp")
        preds[m] = p
    pers = D.csv(DAY_INPUTS[-1], parse_dates=["timestamp"])
    pers = pers[(pers["target"] == "direct") & (pers["horizon"] == 1)].set_index("timestamp")
    scored = preds["xgb_sota"]["y_true"].dropna().index.intersection(t.index)

    def day_of(idx):
        return pd.Index(idx.tz_convert(tz).date)

    full = pd.Series(1, index=t.index).groupby(day_of(t.index)).sum()
    got = pd.Series(1, index=scored).groupby(day_of(scored)).sum()
    complete = got[got == full.reindex(got.index)].index
    k = pd.Series(t.loc[scored, "indirect"].to_numpy(), index=scored)
    mean_k = k.groupby(day_of(scored)).mean().loc[complete].dropna().sort_values()
    pick = {"cloud": mean_k.index[0], "mid": mean_k.index[len(mean_k) // 2],
            "clear": mean_k.index[-1]}
    fig, axes = subplots(1, 3, height=2.8, width=8.4, sharey=True)
    facts = {}
    for ax, key in zip(axes, ("clear", "mid", "cloud"), strict=True):
        day = pick[key]
        hrs = scored[day_of(scored) == day]
        x = hrs.tz_convert(tz).hour
        y = preds["xgb_sota"].loc[hrs, "y_true"].to_numpy()
        ax.fill_between(x, 0, y, color=C["light"], label="measured")
        ax.plot(x, y, "-", color=C["grey"], lw=0.8)
        rmse = []
        for _, m, name in MODELS:
            col, style = MODEL_STYLE[m]
            f = preds[m].loc[hrs, "y_pred"].to_numpy()
            ax.plot(x, f, style, color=col, ms=2.5, lw=1, label=name)
            rmse.append(float(np.sqrt(np.mean((f - y) ** 2))))
        pf = pers.loc[hrs, "y_pred"].to_numpy()
        ax.plot(x, pf, "--", color=C["grey"], lw=1, label="persistence")
        ax.set_title(f"{day} (mean k {mean_k[day]:.2f})", fontsize=8)
        ax.set_xlabel("local hour")
        facts[f"{key}_rmse"] = (min(rmse), max(rmse))
        facts[f"{key}_pers"] = float(np.sqrt(np.mean((pf - y) ** 2)))
    axes[0].set_ylabel("AC power (kW)")
    axes[2].legend(fontsize=5.5, loc="upper right")
    out = {"d_clear": str(pick["clear"]), "k_clear": float(mean_k[pick["clear"]]),
           "d_mid": str(pick["mid"]), "d_cloud": str(pick["cloud"]),
           "k_cloud": float(mean_k[pick["cloud"]]),
           "c_lo": facts["clear_rmse"][0], "c_hi": facts["clear_rmse"][1],
           "m_lo": facts["mid_rmse"][0], "m_hi": facts["mid_rmse"][1],
           "o_lo": facts["cloud_rmse"][0], "o_hi": facts["cloud_rmse"][1],
           "o_pers": facts["cloud_pers"]}
    return fig, out


QUANT_INPUTS = tuple(f"{NDEV}/{s}/quantiles_{m}_seed{DEV_SEED}.csv" for s, m in QUANTILE_STEPS) \
    + tuple(f"{NDEV}/{s}/predictions_{m}_seed{DEV_SEED}.csv" for s, m in QUANTILE_STEPS)


def quantile_checks(step: str, model: str) -> pd.DataFrame:
    """Per route, on validation daytime hours with truth: the scored hours, the share with
    quantiles out of order, and the coverage of the central 80 % interval (q0.1 to q0.9) and the
    mean quantile loss, for the raw quantiles and for the quantiles ordered per row (D-33)."""
    from solarpv.evaluation import quantiles as Q
    from solarpv.evaluation.truth import load_truth

    truth = load_truth(exclude_splits=("test",))
    day = truth[(truth["split"] == "val") & truth["daytime"].astype(bool)].index
    q = D.csv(f"{NDEV}/{step}/quantiles_{model}_seed{DEV_SEED}.csv", parse_dates=["timestamp"])
    p = D.csv(f"{NDEV}/{step}/predictions_{model}_seed{DEV_SEED}.csv",
              parse_dates=["timestamp"])
    qs = Q.quantile_columns(q)
    m = q.merge(p[["timestamp", "target", "horizon", "y_true", "y_pred"]],
                on=["timestamp", "target", "horizon"], validate="one_to_one")
    m = m[m["timestamp"].isin(day) & m["y_true"].notna()]
    assert np.allclose(m["q0.5"], m["y_pred"]), f"{step}: the point is not the median head"
    rows = []
    for t, g in m.groupby("target"):
        y = g["y_true"].to_numpy()
        o = Q.ordered(g)
        assert not Q.crossed(o).any() and (o["y_pred"] == g["y_pred"]).all()
        rows.append({"step": step, "target": t, "n": len(g), "n_quantiles": len(qs),
                     "crossed_pct": float(Q.crossed(g).mean() * 100),
                     "coverage80_pct": 100 * Q.coverage(y, g),
                     "coverage80_ordered_pct": 100 * Q.coverage(y, o),
                     "pinball_raw": Q.pinball(y, g), "pinball_ordered": Q.pinball(y, o)})
    return pd.DataFrame(rows)


@register(
    "S09-29", "9", "quantile_forecasts", "The quantile forecasts of the sequence models",
    "Are the quantiles of the BiLSTM and the TFTs in order, and what does ordering them per "
    "hour do to their central 80 % interval?",
    "Left: share of the scored validation hours (all horizons) on which the forecast "
    "quantiles are out of order, per model and route. Right: the share of those hours whose "
    "truth lies between the 0.1 and 0.9 quantiles, raw (solid) and with the quantiles ordered "
    "per hour (light, hatched), against the nominal 80 % (dashed). XGB and PatchTST forecast "
    "a point only.",
    "The quantiles are out of order on up to {cross_max:.0f} % of the hours ({cross_name}); "
    "{n_cross} of {n_pairs} model-route pairs cross on more than 1 % of the hours, and the "
    "BiLSTM never. Ordering them per hour moves the coverage of the central interval from "
    "{cov_raw_min:.0f}–{cov_raw_max:.0f} % to {cov_ord_min:.0f}–{cov_ord_max:.0f} % (it rises "
    "only where the quantiles cross) and lowers the quantile loss by up to {pin_drop:.1f} % "
    "({pin_name}); the point forecast, the 0.5 head, is unchanged.",
    "Intervals and the potential skill of L-14 read the ordered quantiles (D-33, X53); the "
    "point forecast and every score stay as the model wrote them.",
    QUANT_INPUTS)
def s09_quantiles():
    tab = pd.concat([quantile_checks(s, m) for s, m in QUANTILE_STEPS], ignore_index=True)
    names = {s: NAMES[m] for s, m in QUANTILE_STEPS}
    fig, (a1, a2) = subplots(1, 2, height=2.8, width=7.8)
    y = np.arange(len(QUANTILE_STEPS))
    for k, (t, hatch) in enumerate((("indirect", ""), ("direct", "//"))):
        g = tab[tab["target"] == t].set_index("step").reindex([s for s, _ in QUANTILE_STEPS])
        cols = [MODEL_STYLE[m][0] for _, m in QUANTILE_STEPS]
        a1.barh(y + (k - 0.5) * 0.38, g["crossed_pct"], 0.36, color=cols, hatch=hatch,
                edgecolor="white", label=t)
        a2.barh(y + (k - 0.5) * 0.38, g["coverage80_pct"], 0.36, color=cols, hatch=hatch,
                edgecolor="white")
        a2.plot(g["coverage80_ordered_pct"], y + (k - 0.5) * 0.38, "D", color="black", ms=3.5)
    for a in (a1, a2):
        a.set_yticks(y, [names[s] for s, _ in QUANTILE_STEPS], fontsize=7)
        a.invert_yaxis()
    a1.set_xlabel("hours with quantiles out of order (%)")
    a2.axvline(80, color=C["red"], ls="--", lw=1)
    a2.set_xlim(0, 100)
    a2.set_xlabel("truth inside the 0.1–0.9 interval (%)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="0.6", label="indirect"),
        Patch(facecolor="0.6", hatch="//", edgecolor="white", label="direct"),
        Line2D([], [], color="black", marker="D", ls="", ms=3.5, label="ordered per hour"),
        Line2D([], [], color=C["red"], ls="--", lw=1, label="nominal 80 %")],
        fontsize=6.5, ncol=4, loc="outside lower center")
    a1.text(0.5, 0, "0 % in both routes", va="center", fontsize=6.5)  # BiLSTM, asserted below
    top = tab.loc[tab["crossed_pct"].idxmax()]
    assert tab[tab["step"] == "bilstm"]["crossed_pct"].max() == 0
    assert (tab["pinball_ordered"] <= tab["pinball_raw"] + 1e-9).all()  # the rearrangement bound
    drop = 100 * (1 - tab["pinball_ordered"] / tab["pinball_raw"])
    j = int(drop.idxmax())
    facts = {"cross_max": float(top["crossed_pct"]),
             "cross_name": f"{names[top['step']]}, {top['target']}",
             "n_cross": int((tab["crossed_pct"] > 1).sum()), "n_pairs": int(len(tab)),
             "cov_raw_min": float(tab["coverage80_pct"].min()),
             "cov_raw_max": float(tab["coverage80_pct"].max()),
             "cov_ord_min": float(tab["coverage80_ordered_pct"].min()),
             "cov_ord_max": float(tab["coverage80_ordered_pct"].max()),
             "pin_drop": float(drop.max()),
             "pin_name": f"{names[tab.loc[j, 'step']]}, {tab.loc[j, 'target']}"}
    return fig, facts


BREAK_INPUTS = (_eval(XDEV, "seasonal_metrics.csv"), _eval(NDEV, "seasonal_metrics.csv"),
                _eval(XDEV, "benchmark_views.csv"), _eval(NDEV, "benchmark_views.csv"))


@register(
    "S09-30", "9", "error_breakdown", "Where the error comes from, model by model",
    "In which season do the models err most, and how much of the error falls on operationally "
    "limited hours?",
    "Left: validation nRMSE by season, direct route (kW), D+1, every model and persistence. "
    "Right: nRMSE in kW at D+1 on all scored hours (bars) and without the operationally "
    "limited hours (dots), both routes.",
    "Every model errs most in {worst} and least in {best} (direct route, D+1; validation spans "
    "February to September, so winter is February only, {n_winter} hours, and autumn September "
    "only, {n_fall} hours). Leaving out the "
    "limited hours lowers the error by {drop_min:.1f} to {drop_max:.1f} points for every model "
    "and route.",
    "The stage-11 report gives the same breakdown for the final models; the seasonal view of "
    "the test split covers October to May.",
    BREAK_INPUTS)
def s09_breakdown():
    s = pd.concat([D.csv(BREAK_INPUTS[0]), D.csv(BREAK_INPUTS[1])], ignore_index=True)
    v = pd.concat([D.csv(BREAK_INPUTS[2]), D.csv(BREAK_INPUTS[3])], ignore_index=True)
    s = s[(s["target"] == "direct") & (s["horizon"] == 1)].drop_duplicates(["model", "season"])
    v = v[v["horizon"] == 1].drop_duplicates(["model", "target", "view"])
    fig, (a1, a2) = subplots(1, 2, height=3.0, width=8.2,
                             gridspec_kw={"width_ratios": [1, 1.3]})
    worst, best = set(), set()
    for _, m, name in [*MODELS, ("", "persistence", "persistence")]:
        g = s[s["model"] == m].set_index("season").reindex(list(SEASONS))
        col, style = MODEL_STYLE.get(m, (C["grey"], "^:"))
        a1.plot(range(len(SEASONS)), g["nRMSE_pct"], style, color=col, ms=3.5, lw=1,
                label=name)
        if m != "persistence":
            worst.add(g["nRMSE_pct"].idxmax())
            best.add(g["nRMSE_pct"].idxmin())
    a1.set_xticks(range(len(SEASONS)), SEASONS, fontsize=7)
    a1.set_ylabel("nRMSE, direct, D+1 (% of 3,449 kW)")
    a1.legend(fontsize=5.5, ncol=2)
    assert len(worst) == 1 and len(best) == 1, "the models disagree on the worst season"
    drops = []
    x = 0
    ticks, labels = [], []
    for _, m, name in MODELS:
        for t, hatch in (("indirect", ""), ("direct", "//")):
            g = v[(v["model"] == m) & (v["target"] == t)].set_index("view")["nRMSE_kW_pct"]
            col, _ = MODEL_STYLE[m]
            a2.bar(x, g["all"], 0.8, color=col, hatch=hatch, edgecolor="white", alpha=0.75)
            a2.plot(x, g["normal_operation"], "o", color="black", ms=3)
            drops.append(g["all"] - g["normal_operation"])
            x += 1
        ticks.append(x - 1.5)
        labels.append(name)
        x += 0.6
    a2.set_xticks(ticks, labels, fontsize=6.5, rotation=20)
    a2.set_ylabel("nRMSE in kW, D+1 (%)")
    a2.set_title("bars: all hours (hatched: direct); dots: normal operation", fontsize=7)
    assert min(drops) > 0
    n_fall = int(s[(s["season"] == "Fall")]["n"].iloc[0])
    n_winter = int(s[(s["season"] == "Winter")]["n"].iloc[0])
    facts = {"worst": worst.pop(), "best": best.pop(), "n_fall": n_fall, "n_winter": n_winter,
             "drop_min": float(min(drops)), "drop_max": float(max(drops))}
    return fig, facts


# ── Look-back (rule 8): figures for earlier tasks, found on 2026-11-12 ──────

XGB_CURVES = (f"{XDEV}/xgb/learning_curves.csv", f"{XDEV}/xgb/best_params.json")


@register(
    "S09-31", "9", "xgb_learning_curves", "Learning curves of the XGB final fits",
    "Did the final XGB fits of the development run converge, and where did early stopping "
    "act?",
    "Validation RMSE (solid) and training RMSE (dotted) per boosting round of each final fit "
    "(indirect route in k, top; direct route in kW, bottom; one fit per horizon); the dot marks "
    "the best round, after which early stopping waited 50 more rounds. Same layout as the "
    "neural curves (S09-08).",
    "The best rounds lie between {best_min} and {best_max}. Early stopping ended {n_stopped} "
    "of {n_fits} fits after 50 rounds without a better validation error; in {n_capped} the best "
    "round was the last one allowed. Where the fit went on, the training error kept falling "
    "after the best round.",
    "The final run keeps early stopping with 50 rounds; its curves are compared with these in "
    "the review of stage 10.",
    XGB_CURVES)
def s09_xgb_curves():
    c = D.csv(XGB_CURVES[0])
    best = D.jload(XGB_CURVES[1])
    fig, axes = subplots(2, 3, height=4.0)
    bests, capped, stopped = [], 0, 0
    for i, t in enumerate(("indirect", "direct")):
        for j, h in enumerate((1, 2, 3)):
            ax = axes[i][j]
            g = c[(c["target"] == t) & (c["horizon"] == h) & (c["seed"] == DEV_SEED)]
            rec = best[f"{t}_D{h}_seed{DEV_SEED}"]
            ax.plot(g["round"], g["val_rmse"], "-", color=C["blue"], lw=1.1, label="validation")
            ax.plot(g["round"], g["train_rmse"], ":", color=C["grey"], lw=1.1, label="training")
            b = g[g["round"] == rec["best_iteration"]]
            ax.plot(b["round"], b["val_rmse"], "o", color=C["red"], ms=4)
            ax.set_title(f"XGB, {t} D+{h}", fontsize=8)
            ax.set_xlabel("round", fontsize=7)
            bests.append(rec["best_iteration"])
            went_on = rec["rounds_trained"] > rec["n_rounds"]
            capped += not went_on
            stopped += went_on
            if went_on:
                assert g["train_rmse"].iloc[-1] < b["train_rmse"].iloc[0]
                assert rec["rounds_trained"] - rec["n_rounds"] == rec["early_stopping_rounds"]
    axes[0][0].set_ylabel("RMSE (k)")
    axes[1][0].set_ylabel("RMSE (kW)")
    axes[0][0].legend(fontsize=6)
    facts = {"best_min": int(min(bests)), "best_max": int(max(bests)), "n_capped": int(capped),
             "n_stopped": int(stopped), "n_fits": len(bests)}
    return fig, facts


XGB_TRIALS = (f"{XDEV}/xgb/trials.csv",)


@register(
    "S09-32", "9", "xgb_optimization", "Optuna searches of XGB",
    "How did the XGB searches progress within their development budget?",
    "Validation nRMSE in kW of every trial (points) and the best so far (line), per route and "
    "horizon; one study per route and horizon with seed 42. Same layout as the neural searches "
    "(S09-10).",
    "{n_trials} trials in all; the best trial improved on the first by {gain_min:.2f} to "
    "{gain_max:.2f} points, and {n_late} of {n_studies} studies found their best value in the "
    "last quarter of their trials.",
    f"The final run searches {FINAL_OPTUNA_TRIALS} trials per study in the same space (the v6 "
    "budget).",
    XGB_TRIALS)
def s09_xgb_search():
    tr = D.csv(XGB_TRIALS[0])
    fig, axes = subplots(2, 3, height=4.0, sharey="row")
    total, gains, late, studies = 0, [], 0, 0
    for i, t in enumerate(("indirect", "direct")):
        for j, h in enumerate((1, 2, 3)):
            g = tr[(tr["target"] == t) & (tr["horizon"] == h)].sort_values("trial")
            ax = axes[i][j]
            ax.plot(g["trial"], g["value_kw"], "o", color=C["blue"], ms=3)
            ax.plot(g["trial"], g["value_kw"].cummin(), "-", color=C["red"], lw=1.2)
            ax.set_title(f"XGB, {t} D+{h}", fontsize=8)
            ax.set_xlabel("trial", fontsize=7)
            total += len(g)
            gains.append(float(g["value_kw"].iloc[0] - g["value_kw"].min()))
            late += int(g["value_kw"].to_numpy().argmin()) >= 0.75 * len(g)
            studies += 1
    axes[0][0].set_ylabel("nRMSE (% of capacity)")
    axes[1][0].set_ylabel("nRMSE (% of capacity)")
    facts = {"n_trials": total, "gain_min": min(gains), "gain_max": max(gains),
             "n_late": late, "n_studies": studies}
    return fig, facts


RUN_FILES = tuple(f"{r}/run.json" for r in D.STAGE9_RUNS)
STEP_KIND = {"baselines": ("references", C["grey"]), "xgb": ("XGB", C["blue"]),
             "evaluate": ("evaluation, explanation, report", C["green"])}


@register(
    "S09-33", "9", "stage9_runs", "The runs of stage 9 so far",
    "Which runs did stage 9 make, in which mode, and where did the time go?",
    "Every run folder of stage 9 in order: the duration of each step (hours), coloured by kind "
    "(references, XGB, neural models, and evaluation with explanation and report); the label "
    "gives the mode.",
    "{n_runs} runs: {n_smoke} smoke and {n_dev} development, no final run. {hours:.1f} h of "
    "steps in all, {neural_pct:.0f} % of them in the neural models; the longest run is "
    "{longest} ({longest_h:.1f} h). {n_failed} steps failed, in {failed_runs}, and each passed "
    "in the next run that ran it. Each run keeps its configuration hash, and every completed "
    "step its commit and fingerprint.",
    "The final run adds one folder; its time is estimated in S09-18.",
    RUN_FILES)
def s09_runs():
    fig, ax = subplots(height=3.4)
    rows, total, neural, modes, failed = [], 0.0, 0.0, [], []
    for k, rel in enumerate(RUN_FILES):
        r = D.jload(rel)
        assert r.get("config_sha256")
        assert all(s.get("commit") for s in r["steps"].values() if s["status"] == "ok")
        bad = [s for s, rec in r["steps"].items() if rec["status"] != "ok"]
        if bad:
            failed.append((k, r["run_id"], bad))
            ax.text(0.02, k, "  failed: " + ", ".join(bad), va="center", fontsize=5.5,
                    color=C["red"])
        left = 0.0
        for step, rec in r["steps"].items():
            h = float(rec.get("duration_s", 0) or 0) / 3600
            if step in {s for s, _, _ in NEURAL}:
                kind, col = "neural models", C["orange"]
            else:
                kind, col = STEP_KIND.get(step, STEP_KIND["evaluate"])
            ax.barh(k, h, left=left, color=col, edgecolor="white", lw=0.4)
            left += h
            neural += h if kind == "neural models" else 0
        total += left
        if not bad:
            label = (f"{left * 3600:.0f} s" if left < 1 / 60 else f"{left * 60:.0f} min"
                     if left < 1 else f"{left:.1f} h")
            ax.text(left, k, "  " + label, va="center", fontsize=5.5)
        rows.append((r["run_id"], left))
        modes.append(r["mode"])
    ax.set_yticks(range(len(rows)), [f"{rid} ({mode})" for (rid, _), mode in zip(rows, modes,
                                                                               strict=True)],
                  fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("hours of steps")
    from matplotlib.patches import Patch
    seen = {"references": C["grey"], "XGB": C["blue"], "neural models": C["orange"],
            "evaluation, explanation, report": C["green"]}
    ax.legend(handles=[Patch(color=c, label=n) for n, c in seen.items()], fontsize=6.5,
              loc="lower right")
    assert "final" not in modes
    for k, _, bad in failed:  # a failed step passed in the next run that ran it
        later = [D.jload(rel)["steps"] for rel in RUN_FILES[k + 1:]]
        assert all(next(s[b]["status"] for s in later if b in s) == "ok" for b in bad)
    longest = max(rows, key=lambda r: r[1])
    facts = {"n_runs": len(rows), "n_smoke": modes.count("smoke"), "n_dev": modes.count("dev"),
             "hours": total, "neural_pct": 100 * neural / total, "longest": longest[0],
             "longest_h": longest[1], "n_failed": sum(len(b) for _, _, b in failed),
             "failed_runs": ", ".join(r for _, r, _ in failed) or "none"}
    return fig, facts


VIEWS = "outputs/07_evaluation/val/benchmark_views.csv"


@register(
    "S08-04", "8", "evaluation_views", "The secondary views of the evaluation",
    "How do the secondary views (normal operation, data source) change the references' "
    "errors on validation?",
    "Validation nRMSE in kW per horizon of the three references on all scored hours, without "
    "the operationally limited hours (D-22), and on the hours of each data source (D-21), "
    "both routes.",
    "Leaving out the operationally limited hours ({n_limited} per horizon) lowers the "
    "references' error by {drop_min:.2f} to {drop_max:.2f} points. The data-source views equal "
    "all hours on validation, because no validation hour comes from the second source.",
    "Every model is reported in the same views (S09-30); the main score stays all scored hours "
    "(C-35).",
    (VIEWS,))
def s08_views():
    v = D.csv(VIEWS)
    refs = ("persistence", "climatology", "clearsky_k")
    fig, axes = subplots(1, 2, height=2.8, sharey=True)
    drops = []
    for ax, t in zip(axes, ("indirect", "direct"), strict=True):
        for m in refs:
            g = v[(v["model"] == m) & (v["target"] == t)]
            col, _ = REF_STYLE[m]
            for view, ls, mk in (("all", "-", "o"), ("normal_operation", ":", "s")):
                r = g[g["view"] == view].sort_values("horizon")
                ax.plot(r["horizon"], r["nRMSE_kW_pct"], ls, marker=mk, color=col, ms=3.5,
                        lw=1, label=(f"{REF_NAMES[m].split(' (')[0]}, {view.replace('_', ' ')}"
                                     if t == "indirect" else None))
            a = g[g["view"] == "all"].set_index("horizon")["nRMSE_kW_pct"]
            n = g[g["view"] == "normal_operation"].set_index("horizon")["nRMSE_kW_pct"]
            drops += list(a - n)
            for view in ("source_minute", "no_polled"):
                o = g[g["view"] == view].set_index("horizon")["nRMSE_kW_pct"]
                assert np.allclose(o, a), f"{m} {t}: the {view} view differs from all hours"
        ax.set_title(f"{t} route")
        ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
    axes[0].set_ylabel("validation nRMSE (% of 3,449 kW)\naxis does not start at zero")
    fig.legend(loc="outside lower center", ncol=3, fontsize=6.5)
    lim =v[(v["view"] == "all")]["n"].iloc[0] - v[(v["view"] == "normal_operation")]["n"].iloc[0]
    facts = {"n_limited": int(lim), "drop_min": float(min(drops)), "drop_max": float(max(drops))}
    return fig, facts

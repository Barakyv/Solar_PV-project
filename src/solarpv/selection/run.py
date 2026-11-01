"""Stage 7: run the selection for every target × horizon (docs/stage7_plan.md).

    python -m solarpv.selection.run

S0 decides the k-history candidates inside train (D-24); S1–S7 select per target × horizon on
train rows only; S8 compares each 05e set with the full pool once on validation (D-13). Outputs in
outputs/07_feature_selection/ and configs/k_history_decision.yaml.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from solarpv.config.constants import AC_CAPACITY_KW, HORIZONS, RANDOM_SEEDS  # noqa: E402
from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR  # noqa: E402
from solarpv.config.provenance import git_commit  # noqa: E402
from solarpv.config.runlog import RunLog  # noqa: E402
from solarpv.features.decide_k import MIN_GAIN_PP  # noqa: E402
from solarpv.features.decide_k import XGB_PARAMS as FIXED_XGB  # noqa: E402
from solarpv.nwp import check as nwp_check  # noqa: E402
from solarpv.selection import common as C  # noqa: E402
from solarpv.selection import steps as S  # noqa: E402

OUT = OUTPUTS_DIR / "07_feature_selection"
DECISION_FILE = CONFIGS_DIR / "k_history_decision.yaml"
TARGET_COLUMN = {"indirect": "target_k", "direct": "site_total_AC_kW"}
IRRADIANCE_GFS = ("fcst_gfs_ghi", "fcst_gfs_kt", "fcst_gfs_poa", "fcst_gfs_kt_poa",
                  "fcst_gfs_kt_poa_ma3", "fcst_gfs_kt_poa_daymean", "fcst_gfs_pv_kw",
                  "fcst_gfs_k_phys")
DOMINANCE_FACTOR = 10.0  # red flag: first SHAP feature > 10 × the third (the DC case was 17)


def margin_rule(without: list[float], with_: list[float]) -> tuple[float, float]:
    """(gain, margin) of the D-12 rule: gain = mean(without) − mean(with); margin = max(0.05,
    2 × pooled seed std)."""
    gain = float(np.mean(without) - np.mean(with_))
    pooled = float(np.sqrt((np.var(without, ddof=1) + np.var(with_, ddof=1)) / 2))
    return gain, max(MIN_GAIN_PP, 2 * pooled)


# ── S0: k history (D-24) ──────────────────────────────────────────────────


def k_history_check(tables: dict[int, pd.DataFrame], log) -> tuple[pd.DataFrame, dict]:
    rows, decision = [], {}
    for h, t in tables.items():
        k_cols = [c for c in t.columns if c.startswith(C.K_PREFIX)]
        base = C.pool(t, h)
        p = C.problem(t, "indirect")
        scores = {name: [C.cv_score(p, feats, FIXED_XGB, seed, with_shap=False)[0]
                         for seed in RANDOM_SEEDS]
                  for name, feats in (("pool", base), ("pool+k_history", base + k_cols))}
        gain, margin = margin_rule(scores["pool"], scores["pool+k_history"])
        passes = gain > margin if h == 1 else gain > -margin
        decision[f"indirect_D{h}"] = {"gain_pp": round(gain, 4), "margin_pp": round(margin, 4),
                                      "passes": bool(passes)}
        for name, values in scores.items():
            rows.append({"horizon": h, "features": name, "n_features": len(base) + (
                len(k_cols) if name != "pool" else 0), "nrmse_kw_mean": float(np.mean(values)),
                "nrmse_kw_std": float(np.std(values, ddof=1)),
                **{f"seed_{s}": v for s, v in zip(RANDOM_SEEDS, values, strict=True)}})
        log.note(f"S0 D+{h}: gain {gain:+.3f} pp, margin {margin:.3f} pp, passes {passes}")
    admitted = all(d["passes"] for d in decision.values())
    return pd.DataFrame(rows), {"decision": "D-24", "admitted": admitted, "results": decision}


# ── S1–S7 for one target × horizon ────────────────────────────────────────


def select(table: pd.DataFrame, target: str, horizon: int, extra: tuple[str, ...], log) -> dict:
    C.check_train_only(table)
    feats = C.pool(table, horizon, extra)
    p = C.problem(table, target)
    core = set(C.CORE) & set(feats)
    fate = {f: "kept" for f in feats}

    var_out = S.low_variance(p.X, feats, core)
    for f in var_out:
        fate[f] = "variance"
    after_var = [f for f in feats if f not in var_out]
    mi_all = S.mutual_information(p.X, p.y, after_var, p.fit)
    mi_out = S.low_mi(mi_all, core)
    for f in mi_out:
        fate[f] = "mi"
    after_mi = [f for f in after_var if f not in mi_out]
    groups = S.duplicate_groups(p.X, after_mi, p.fit)
    dup = S.deduplicate(groups, mi_all, core)
    for f, rep in dup.items():
        fate[f] = f"duplicate of {rep}"
    after_dup = [f for f in after_mi if f not in dup]
    log.note(f"{target} D+{horizon}: pool {len(feats)}, variance -{len(var_out)}, "
             f"MI -{len(mi_out)}, duplicates -{len(dup)} -> {len(after_dup)}")

    start, rfe_hist = S.rfe(p, after_dup, core, log=log)
    for f in after_dup:
        if f not in start:
            fate[f] = "rfe"
    curve = S.elimination(p, start)
    elb = S.elbows(curve)
    shap_set = S.elbow_set(curve)
    full_row = curve.loc[curve["n_features"] == len(start)].iloc[0]
    _, shap_start = C.cv_score(p, start)
    shap_rank = list(shap_start.sort_values(ascending=False).index)
    prob = S.stability(p, start)
    mi_start = mi_all[start]
    mi_votes, mi_threshold = S.mi_vote(mi_start, p.X, p.y, p.fit)
    votes = pd.DataFrame({"shap_elbow": pd.Series(start, index=start).isin(shap_set),
                          "stability": prob >= S.STABILITY_VOTE, "mutual_information": mi_votes})
    chosen = S.consensus(votes)
    n_votes = votes.sum(axis=1)
    audit = S.tiers(prob, mi_start, n_votes)
    sens = S.floor_sensitivity(prob, mi_start, n_votes)
    top = shap_start.sort_values(ascending=False)
    flags = {"dominant_feature": bool(top.iloc[0] > DOMINANCE_FACTOR * top.iloc[2]),
             "no_gfs_irradiance_in_05e": not any(f in IRRADIANCE_GFS for f in chosen)}
    log.note(f"{target} D+{horizon}: RFE {len(start)}, 05c elbow {len(shap_set)} "
             f"(score {full_row['score']:.3f} at {len(start)}), stability "
             f"{int((prob >= S.STABILITY_VOTE).sum())}, MI {int(mi_votes.sum())}, consensus "
             f"{len(chosen)}; flags {flags}")
    return {"target": target, "horizon": horizon, "pool": feats, "fate": fate, "groups": groups,
            "mi": mi_all, "rfe": rfe_hist, "start": start, "curve": curve, "elbows": elb,
            "shap_set": shap_set, "shap_rank": shap_rank, "shap_start": shap_start,
            "baseline": float(full_row["score"]), "prob": prob, "mi_votes": mi_votes,
            "mi_threshold": mi_threshold, "votes": votes, "chosen": chosen, "audit": audit,
            "sensitivity": sens, "flags": flags, "n_fit_rows": int(p.fit.sum()),
            "n_score_rows": int(p.score.sum())}


def json_05e(r: dict) -> dict:
    curve = r["curve"].set_index("n_features")["score"]
    n_elbow = len(r["shap_set"])
    return {
        "target": TARGET_COLUMN[r["target"]], "target_label": r["target"], "horizon": r["horizon"],
        "feature_selection_method": "consensus_3_methods",
        "rows": "train only (stage 7, P1); score: nRMSE in kW (% of 3,449 kW) on held-out "
                "daytime hours of 5 expanding date folds with a 3-day gap",
        "n_pool": len(r["pool"]), "n_starting_features": len(r["start"]),
        "methods": {
            "shap_elbow": {
                "description": "SHAP-guided backward elimination (05c); elbow = smallest set "
                               "within the threshold (relative) of the starting set's score",
                "threshold_used": "1.0pct", "n_features": n_elbow, "elbows": r["elbows"],
                "baseline_nrmse": round(r["baseline"], 4),
                "nrmse_at_elbow": round(float(curve[n_elbow]), 4),
                "features": sorted(r["shap_set"]), "shap_rank": r["shap_rank"]},
            "stability_selection": {
                "description": "share of total mean |SHAP| >= 1 % on the held-out 30 % of days, "
                               "over 30 subsamples of 70 % of train days",
                "n_subsamples": S.STABILITY_SUBSAMPLES, "subsample_frac": S.STABILITY_FRACTION,
                "share": S.STABILITY_SHARE, "threshold": S.STABILITY_VOTE,
                "n_features": int((r["prob"] >= S.STABILITY_VOTE).sum()),
                "selection_probabilities": {f: round(float(v), 3) for f, v in
                                            r["prob"].sort_values(ascending=False).items()}},
            "mutual_information": {
                "description": "MI (sklearn, 5 neighbours) above the 95th percentile of the MI "
                               "of 3 shuffled copies of every starting feature",
                "null_threshold": round(r["mi_threshold"], 4),
                "n_features": int(r["mi_votes"].sum()),
                "scores": {f: round(float(v), 4) for f, v in
                           r["mi"][r["start"]].sort_values(ascending=False).items()}}},
        "consensus": {"min_votes": 2, "n_features": len(r["chosen"]), "features": r["chosen"],
                      "votes": {f: int(v) for f, v in r["votes"].sum(axis=1).items()}},
        "recommended": {"method": "consensus_2of3",
                        "description": "Features endorsed by at least 2 of 3 methods (SHAP "
                                       "elbow, stability, MI); the XGB-Opt input (D-13)",
                        "features": r["chosen"], "n_features": len(r["chosen"])},
        "red_flags": r["flags"]}


def json_05f(r: dict) -> dict:
    a = r["audit"]
    tiers = {}
    for tier in ("CORE", "SUPPORTED", "CONDITIONAL", "EXCLUDED"):
        sub = a[a["tier"] == tier]
        tiers[tier] = {"n": len(sub), "features": list(sub["feature"])}
        if tier == "EXCLUDED":
            tiers[tier]["reasons"] = dict(zip(sub["feature"], sub["reason"], strict=True))
    kept = list(a.loc[a["tier"] != "EXCLUDED", "feature"])
    return {
        "source": "05f — post-hoc audit of 05e", "target": TARGET_COLUMN[r["target"]],
        "target_label": r["target"], "horizon": r["horizon"],
        "note": "05f is an audit only; the modelling input is the 05e set (D-13)",
        "05e_recommended_n": len(r["chosen"]),
        "stability_axis": {"stable_core": S.TIER_CORE, "moderate": S.TIER_SUPPORTED,
                           "unstable_threshold": S.TIER_FLOOR},
        "mi_75th_percentile": round(float(a["mi_75th"].iloc[0]), 4), "tiers": tiers,
        "recommended_05f": {"features": kept, "n": len(kept),
                            "method": "tiered_CORE_SUPPORTED_CONDITIONAL"},
        "floor_sensitivity": r["sensitivity"].to_dict("records")}


# ── S8: one look at validation ────────────────────────────────────────────


def validation_check(results: list[dict], k_admitted: bool, log) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        t = nwp_check.load(h)  # train and validation only; asserts no test row
        if k_admitted:
            t = t.join(C.k_history(t, h))
        for r in (r for r in results if r["horizon"] == h):
            target = r["target"]
            p = C.problem(t, target)
            train = (t["split"] == "train").to_numpy()
            val = pd.Series(t["split"].to_numpy() == "val", index=t.index)
            folds = [(pd.Series(train, index=t.index), val)]
            sets = {"05e": r["chosen"], "full pool": r["pool"]}
            scores = {name: [C.cv_score(p, feats, FIXED_XGB, seed, with_shap=False,
                                        folds=folds)[0] for seed in RANDOM_SEEDS]
                      for name, feats in sets.items()}
            gain, margin = margin_rule(scores["05e"], scores["full pool"])
            for name, values in scores.items():
                rows.append({"target": target, "horizon": h, "features": name,
                             "n_features": len(sets[name]),
                             "nrmse_kw_mean": float(np.mean(values)),
                             "nrmse_kw_std": float(np.std(values, ddof=1)),
                             "05e_minus_full_pp": -gain, "margin_pp": margin,
                             "red_flag": bool(-gain > margin)})
            log.note(f"S8 {target} D+{h}: 05e {np.mean(scores['05e']):.3f} vs full "
                     f"{np.mean(scores['full pool']):.3f} (margin {margin:.3f})")
    return pd.DataFrame(rows)


# ── figures 19–25 ─────────────────────────────────────────────────────────


def counts(r: dict) -> dict[str, int]:
    fate = pd.Series(r["fate"])
    return {"pool": len(r["pool"]),
            "after filters": int((~fate.isin(["variance", "mi"])).sum()),
            "after duplicates": int(fate.isin(["kept", "rfe"]).sum()),
            "after RFE (05c start)": len(r["start"]), "05c elbow": len(r["shap_set"]),
            "stability votes": int((r["prob"] >= S.STABILITY_VOTE).sum()),
            "MI votes": int(r["mi_votes"].sum()), "05e consensus": len(r["chosen"]),
            "05f recommended": int((r["audit"]["tier"] != "EXCLUDED").sum())}


def figures(results: list[dict]) -> list:
    paths = []
    key = [(r["target"], r["horizon"]) for r in results]
    table = pd.DataFrame([counts(r) for r in results],
                         index=[f"{t} D+{h}" for t, h in key])

    fig, ax = plt.subplots(figsize=(11, 4))
    steps = list(table.columns)
    for i, step in enumerate(steps):
        ax.text(i, 0.5, step.replace(" (", "\n("), ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round", "fc": "#eef2f7", "ec": "#557"})
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.62, 0.5), xytext=(i + 0.38, 0.5),
                        arrowprops={"arrowstyle": "->"})
        ax.text(i, 0.05, "\n".join(f"{v}" for v in table[step]), ha="center", va="bottom",
                fontsize=7)
    ax.text(-0.9, 0.05, "\n".join(table.index), ha="right", va="bottom", fontsize=7)
    ax.set_xlim(-2.2, len(steps) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Figure 19 — selection pipeline (counts per target × horizon)")
    paths.append(_save(fig, "fig19_pipeline.png"))

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, r in zip(axes.flat, results, strict=True):
        c = r["curve"].sort_values("n_features")
        ax.plot(c["n_features"], c["score"], marker=".", lw=1)
        ax.axvline(len(r["shap_set"]), color="C3", ls="--", lw=1, label="1 % elbow")
        ax.set_title(f"{r['target']} D+{r['horizon']}", fontsize=9)
        ax.set_ylabel("nRMSE kW (%)")
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("number of features")
    fig.suptitle("Figure 20 — SHAP backward elimination (train, held-out folds)")
    paths.append(_save(fig, "fig20_shap_elimination.png"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, r in zip(axes.flat, results, strict=True):
        pr = r["prob"].sort_values().tail(20)
        ax.barh(pr.index, pr.to_numpy(), color=["C2" if v >= S.STABILITY_VOTE else "C7"
                                                for v in pr])
        ax.axvline(S.STABILITY_VOTE, color="C3", ls="--", lw=1)
        ax.set_title(f"{r['target']} D+{r['horizon']}", fontsize=9)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Figure 21 — stability selection probabilities (top 20)")
    paths.append(_save(fig, "fig21_stability.png"))

    fig, ax = plt.subplots(figsize=(11, 4))
    table.T.plot(ax=ax, marker="o")
    ax.set_ylabel("features")
    ax.set_xticks(range(len(steps)), steps, rotation=25, ha="right", fontsize=8)
    ax.set_title("Figure 22 — feature count through the selection")
    paths.append(_save(fig, "fig22_counts.png"))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, r in zip(axes.flat, results, strict=True):
        a = r["audit"]
        for tier, colour in (("CORE", "C2"), ("SUPPORTED", "C0"), ("CONDITIONAL", "C1"),
                             ("EXCLUDED", "C7")):
            s = a[a["tier"] == tier]
            ax.scatter(s["stability"], s["mi"], s=14, color=colour, label=tier)
        ax.axhline(a["mi_75th"].iloc[0], color="C3", ls="--", lw=1)
        ax.axvline(S.TIER_FLOOR, color="grey", ls=":", lw=1)
        ax.set_title(f"{r['target']} D+{r['horizon']}", fontsize=9)
        ax.set_xlabel("stability")
        ax.set_ylabel("MI")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Figure 23 — MI against stability; 05f tiers (dashed: MI 75th percentile)")
    paths.append(_save(fig, "fig23_mi_stability.png"))

    avail = pd.DataFrame({f"{r['target']} D+{r['horizon']}":
                          pd.Series([C.availability(f) for f in r["chosen"]]).value_counts()
                          for r in results}).fillna(0).T
    fig, ax = plt.subplots(figsize=(9, 4))
    avail.plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("features in the 05e set")
    ax.set_title("Figure 24 — operational availability of the 05e features")
    paths.append(_save(fig, "fig24_availability.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for r in results:
        s = r["sensitivity"]
        ax.plot(s["floor"], s["recommended"], marker="o", label=f"{r['target']} D+{r['horizon']}")
    ax.set_xlabel("stability floor")
    ax.set_ylabel("05f recommended features")
    ax.legend(fontsize=7)
    ax.set_title("Figure 25 — sensitivity of 05f to the stability floor")
    paths.append(_save(fig, "fig25_floor_sensitivity.png"))
    return paths


def _save(fig, name: str):
    path = OUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ── outputs ───────────────────────────────────────────────────────────────


def write_outputs(results: list[dict], k_table: pd.DataFrame, k_decision: dict,
                  val: pd.DataFrame, log) -> list:
    written = []

    def dump(name: str, obj) -> None:
        path = OUT / name
        path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
        written.append(path)

    def csv(name: str, frame: pd.DataFrame) -> None:
        path = OUT / name
        frame.to_csv(path, index=False, lineterminator="\n")
        written.append(path)

    for r in results:
        dump(f"final_features_{r['target']}_D{r['horizon']}.json", json_05e(r))
        dump(f"refined_features_{r['target']}_D{r['horizon']}_05f.json", json_05f(r))
    csv("base_stage.csv", pd.concat([pd.DataFrame({
        "target": r["target"], "horizon": r["horizon"], "feature": list(r["fate"]),
        "group": [C.group(f) for f in r["fate"]], "outcome": list(r["fate"].values()),
        "mi": [round(float(r["mi"].get(f, np.nan)), 4) for f in r["fate"]]})
        for r in results]))
    csv("rfe_history.csv", pd.concat([r["rfe"].assign(target=r["target"], horizon=r["horizon"])
                                      for r in results]))
    csv("shap_elimination_curves.csv", pd.concat([
        r["curve"].drop(columns="features").assign(target=r["target"], horizon=r["horizon"])
        for r in results]))
    csv("selection_votes.csv", pd.concat([pd.DataFrame({
        "target": r["target"], "horizon": r["horizon"], "feature": r["start"],
        "shap_mean_abs": [round(float(r["shap_start"][f]), 6) for f in r["start"]],
        "stability": [round(float(r["prob"][f]), 3) for f in r["start"]],
        "mi": [round(float(r["mi"][f]), 4) for f in r["start"]],
        "vote_shap": r["votes"]["shap_elbow"].to_numpy(),
        "vote_stability": r["votes"]["stability"].to_numpy(),
        "vote_mi": r["votes"]["mutual_information"].to_numpy(),
        "in_05e": [f in r["chosen"] for f in r["start"]],
        "tier_05f": r["audit"].set_index("feature").loc[r["start"], "tier"].to_numpy()})
        for r in results]))
    union = sorted({f for r in results for f in r["chosen"]})
    csv("feature_availability_classification.csv", pd.DataFrame({
        "feature": union, "group": [C.group(f) for f in union],
        "availability": [C.availability(f) for f in union],
        **{f"{r['target']}_D{r['horizon']}": [f in r["chosen"] for f in union] for r in results}}))
    csv("k_history_check.csv", k_table)
    csv("validation_check.csv", val)
    written += figures(results)
    manifest = {
        "stage": "07_feature_selection", "git_commit": git_commit(), "run_log": str(
            log.path.relative_to(OUT.parent.parent)).replace("\\", "/"),
        "rows": {f"{r['target']}_D{r['horizon']}": {"fit": r["n_fit_rows"],
                                                     "score": r["n_score_rows"]}
                 for r in results},
        "splits_read_for_selection": ["train"], "validation_read": "S8 only",
        "counts": {f"{r['target']}_D{r['horizon']}": counts(r) for r in results},
        "k_history": k_decision,
        "red_flags": {f"{r['target']}_D{r['horizon']}": r["flags"] for r in results},
        "validation_red_flags": {f"{t}_D{h}": bool(g["red_flag"].any()) for (t, h), g in
                                 val.groupby(["target", "horizon"])},
        "capacity_kw": AC_CAPACITY_KW}
    dump("selection_manifest.json", manifest)
    return written


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunLog("07_feature_selection", "python -m solarpv.selection.run", out_dir=OUT) as log:
        log.inputs([PROCESSED_DIR / "06_features" / f"features_1H_D{h}.parquet" for h in HORIZONS]
                   + [PROCESSED_DIR / "06b_gfs" / f"gfs_features_D{h}.parquet" for h in HORIZONS])
        with log.step("tables (train only) and k history"):
            tables = {h: C.train_table(h) for h in HORIZONS}
            tables = {h: t.join(C.k_history(t, h)) for h, t in tables.items()}
        with log.step("S0 k history (D-24, inside train)"):
            k_table, k_decision = k_history_check(tables, log)
            DECISION_FILE.write_text(
                "# D-24 k-history candidates; written by python -m solarpv.selection.run\n"
                + yaml.safe_dump({**k_decision, "git_commit": git_commit()}, sort_keys=False),
                encoding="utf-8", newline="\n")
            log.value("k_history_admitted", k_decision["admitted"])
        results = []
        for target in C.TARGETS:
            for h in HORIZONS:
                t = tables[h]
                extra = tuple(c for c in t.columns if c.startswith(C.K_PREFIX)) \
                    if (target == "indirect" and k_decision["admitted"]) else ()
                with log.step(f"S1-S7 {target} D+{h}"):
                    results.append(select(t, target, h, extra, log))
        with log.step("S8 validation (once)"):
            val = validation_check(results, k_decision["admitted"], log)
        with log.step("outputs"):
            written = write_outputs(results, k_table, k_decision, val, log)
        log.outputs(written + [DECISION_FILE])
    return {"results": results, "k": k_decision}


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

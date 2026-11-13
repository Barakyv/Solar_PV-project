"""The `report` step (docs/stage9_plan.md §19.6): one summary of a run.

Every number is read from a file of the run, checked against the SHA-256 in its step record,
and every table names its source files and their hashes (M31). Nothing is typed. The report:
- the steps: status, commit, fingerprint, duration, budget and seeds;
- the metrics of `evaluate` (and of `test`, when the second phase of a final run has run);
- the training diagnostics and the studies of every model step;
- the automatic warnings of §3.5, recomputed from the stored records with the steps' own
  functions and thresholds and matched with the texts the steps logged (X49);
- the seed spread in kW from the evaluation's benchmark (X42; the target-unit spread is shown
  as secondary), checked against the steps' own values;
- the MBE files (L-15) and quantile files (L-14) that exist;
- the points of the G-1 list that the run informs, as facts without a verdict (§19.1).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.provenance import sha256
from solarpv.pipeline.config import MODEL_STEPS
from solarpv.pipeline.steps import Context, Result, RunnerError, record_file

SEED_SPREAD_WARN_PP = 0.15  # the acceptance rule of stage 10, in kW (Q-19, X42)
SPREAD_TOL = 1e-6
NEURAL = ("bilstm", "tft96", "tft168", "tftbase", "patchtst")
STUDIES = ("xgb", "bilstm", "tft96", "tft168")


class Sources:
    """The files the report reads, each checked against its step record."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.used: dict[str, str] = {}

    def path(self, step: str, name: str) -> Path:
        p = record_file(self.ctx, step, name)
        self.used[f"{self.ctx.completed[step]['folder']}/{name}"] = sha256(p)
        return p

    def has(self, step: str, name: str) -> bool:
        rec = self.ctx.completed.get(step)
        return rec is not None and name in rec["outputs"]

    def csv(self, step: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(step, name))

    def json(self, step: str, name: str) -> dict:
        return json.loads(self.path(step, name).read_text(encoding="utf-8"))


def report_inputs(ctx: Context) -> list[Path]:
    if not ctx.completed:
        raise RunnerError("report needs at least one completed step in this run")
    paths = []
    for step, rec in ctx.completed.items():
        for name in rec["outputs"]:
            if name.endswith((".csv", ".json")) and not name.startswith(("predictions_",
                                                                         "uncorrected_",
                                                                         "quantiles_",
                                                                         "ablation_forecasts_")):
                paths.append(record_file(ctx, step, name))
    return paths


# ── tables ──────────────────────────────────────────────────────────────────


def steps_table(ctx: Context) -> pd.DataFrame:
    rows = []
    for step, rec in ctx.completed.items():
        budget = (rec.get("config") or {}).get("budget")
        rows.append({"step": step, "status": rec["status"], "commit": rec["commit"],
                     "fingerprint": rec["fingerprint"][:12], "duration_s": rec["duration_s"],
                     "budget": json.dumps(budget, sort_keys=True) if budget else "",
                     "n_outputs": len(rec["outputs"]), "n_warnings": len(rec.get("warnings")
                                                                          or [])})
    for step in ctx.failed:
        if step not in ctx.completed:
            rows.append({"step": step, "status": "failed", "commit": "", "fingerprint": "",
                         "duration_s": np.nan, "budget": "", "n_outputs": 0, "n_warnings": 0})
    return pd.DataFrame(rows)


def metrics_tables(src: Sources) -> dict[str, pd.DataFrame]:
    out = {}
    for step, split in (("evaluate", "val"), ("test", "test")):
        name = f"{split}/benchmark_summary.csv"
        if src.has(step, name):
            out[f"benchmark_{split}"] = src.csv(step, name).assign(split=split)
        name = f"{split}/dm_tests_kw.csv"
        if src.has(step, name):
            out[f"dm_kw_{split}"] = src.csv(step, name).assign(split=split)
    return out


def seed_spread(bench: pd.DataFrame) -> pd.DataFrame:
    """X42: the std over seeds of the evaluated nRMSE in kW (and in the target unit)."""
    from solarpv.evaluation.evaluate import NO_SEED, REFERENCES

    models = bench[~bench["model"].isin(REFERENCES) & (bench["seed"] != NO_SEED)]
    rows = []
    for (model, target, h), g in models.groupby(["model", "target", "horizon"]):
        n = len(g)
        rows.append({"model": model, "target": target, "horizon": int(h), "n_seeds": n,
                     "std_kw_pp": float(g["nRMSE_kW_pct"].std(ddof=1)) if n > 1 else np.nan,
                     "std_target_unit_pp": float(g["nRMSE_pct"].std(ddof=1)) if n > 1
                     else np.nan,
                     "above_rule": bool(n > 1 and g["nRMSE_kW_pct"].std(ddof=1)
                                        > SEED_SPREAD_WARN_PP)})
    return pd.DataFrame(rows, columns=["model", "target", "horizon", "n_seeds", "std_kw_pp",
                                       "std_target_unit_pp", "above_rule"])


def training_tables(src: Sources) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Diagnostics of every final fit, the studies, and the recomputed warnings (X49)."""
    from solarpv.models import neural_job as J
    from solarpv.models import xgb_sota as X
    from solarpv.pipeline import neural
    from solarpv.pipeline.steps import xgb_warning_list

    ctx = src.ctx
    fits, studies, warnings = [], [], []
    if "xgb" in ctx.completed:
        best = src.json("xgb", "best_params.json")
        curves = src.csv("xgb", "learning_curves.csv")
        trials = src.csv("xgb", "trials.csv")
        for name, b in best.items():
            c = curves[(curves["target"] == b["target"]) & (curves["horizon"] == b["horizon"])
                       & (curves["seed"] == b["seed"])]
            finite = bool(np.isfinite(c[["train_rmse", "val_rmse"]].to_numpy()).all())
            n_est = int(b["xgb_params"]["n_estimators"])
            warnings += [{**w, "step": "xgb"} for w in xgb_warning_list(
                name, int(b["best_iteration"]), n_est, finite, float(b["best_val_rmse"]),
                float(b["best_train_rmse"]), b["searched"])]
            fits.append({"step": "xgb", "subject": name, "target": b["target"],
                         "horizon": b["horizon"], "seed": b["seed"],
                         "best": int(b["best_iteration"]), "cap": n_est,
                         "run": int(b["rounds_trained"]), "unit_of_best": "round",
                         "gap_ratio": float(b["best_val_rmse"] / b["best_train_rmse"]),
                         "value_kw": float(b["value_kw_forecast"]),
                         "best_trial_value_kw": float(b["best_trial_value_kw"]),
                         "final_minus_best_trial_pp": float(b["value_kw"]
                                                            - b["best_trial_value_kw"])})
            t = trials[(trials["target"] == b["target"]) & (trials["horizon"] == b["horizon"])
                       & (trials["seed"] == b["seed"])]
            studies.append({"step": "xgb", "subject": name, "n_trials": len(t),
                            "n_complete": int((t["state"] == "COMPLETE").sum()),
                            "n_at_cap": int(t["cap_hit"].sum()), "best_trial": int(b["trial"]),
                            "best_value_kw": float(b["best_trial_value_kw"]),
                            "edges": ", ".join(X.edge_params(b["searched"]))})
    for step in NEURAL:
        if step not in ctx.completed:
            continue
        family = J.STEP_FAMILY[step]
        best = src.json(step, "best_params.json")
        budget = ctx.completed[step]["config"]["budget"]
        records: dict[str, dict] = {}
        for rec in best.values():
            records.setdefault(rec["target"], {})[str(rec["seed"])] = rec
        warnings += [{**w, "step": step} for w in neural.warning_list(step, records, budget)]
        for rec in best.values():
            f = rec["final"]
            name = f"{rec['target']} seed {rec['seed']}"
            patch = family == "patchtst"
            fits.append({"step": step, "subject": name, "target": rec["target"], "horizon": "all",
                         "seed": rec["seed"],
                         "best": f.get("best_step") if patch else f.get("best_epoch"),
                         "cap": budget.get("max_steps") if patch else
                         budget.get("epochs_final") or budget.get("epochs"),
                         "run": f.get("steps_run") if patch else f.get("epochs_run"),
                         "unit_of_best": "step" if patch else "epoch",
                         "gap_ratio": (f["best_val_loss"] / f["train_loss_at_best"]
                                       if not patch and f.get("train_loss_at_best") else np.nan),
                         "value_kw": float(f["value_kw"]),
                         "best_trial_value_kw": float(rec["study"]["best_value_kw"])
                         if rec.get("study") else np.nan,
                         "final_minus_best_trial_pp": float(f["value_kw"]
                                                            - rec["study"]["best_value_kw"])
                         if rec.get("study") else np.nan})
            if rec.get("study"):
                space = J.SEARCH_SPACES.get(family) or {}
                params, eff = J.effective(rec["params"], space)
                studies.append({"step": step, "subject": name,
                                "n_trials": int(rec["study"]["n_trials"]),
                                "n_complete": int(rec["study"]["n_trials"]),
                                "n_at_cap": np.nan, "best_trial": int(rec["study"]["best_trial"]),
                                "best_value_kw": float(rec["study"]["best_value_kw"]),
                                "edges": ", ".join(J.edge_params(params, eff))})
    return pd.DataFrame(fits), pd.DataFrame(studies), warnings


KIND_WORDS = (("TEST ROWS READ", "test_access"), ("SHAP dominance", "dominance"),
              ("seed spread", "seed_spread"), ("did not complete", "models_missing"),
              ("trials completed", "trials"), ("differs from its best trial", "reproduce"))


def match_warnings(ctx: Context, recomputed: list[dict], spread: pd.DataFrame) -> pd.DataFrame:
    """X49: every step warning against the recomputed ones. A text in both: 'matched'; only in
    the step log: 'step log only' (a kind the report does not recompute, or a rule changed since
    the step ran, e.g. X45); only recomputed: 'recomputed only'."""
    rows, seen = [], set()
    by_text = {(w["step"], w["text"]): w for w in recomputed}
    for step, rec in ctx.completed.items():
        for text in rec.get("warnings") or []:
            w = by_text.get((step, text))
            if w is not None:
                seen.add((step, text))
                rows.append({"step": step, "kind": w["kind"], "subject": w["subject"],
                             "text": text, "status": "matched"})
                continue
            kind = next((k for word, k in KIND_WORDS if word in text), "other")
            rows.append({"step": step, "kind": kind, "subject": text.split(": ")[0],
                         "text": text, "status": "step log only"})
    for w in recomputed:
        if (w["step"], w["text"]) not in seen:
            rows.append({"step": w["step"], "kind": w["kind"], "subject": w["subject"],
                         "text": w["text"], "status": "recomputed only"})
    above = spread[spread["above_rule"].astype(bool)] if len(spread) else spread.iloc[0:0]
    for r in above.itertuples():
        rows.append({"step": "evaluate", "kind": "seed_spread", "subject":
                     f"{r.model} {r.target} D+{r.horizon}",
                     "text": f"seed spread {r.std_kw_pp:.3f} pp in kW above "
                             f"{SEED_SPREAD_WARN_PP} pp (stage-10 rule)",
                     "status": "from the benchmark"})
    return pd.DataFrame(rows, columns=["step", "kind", "subject", "text", "status"])


def spread_check(ctx: Context, spread: pd.DataFrame) -> list[str]:
    """X42: the steps' own spread (kW, from their objective) against the benchmark's."""
    from solarpv.models import neural_job as J

    problems = []
    model_of = {"xgb": "xgb_sota", **{s: J.STEP_MODEL[s] for s in NEURAL}}
    for step, model in model_of.items():
        notes = (ctx.completed.get(step) or {}).get("notes", {}).get("seed_spread_pp") or {}
        for key, value in notes.items():
            target, h = key.split("_D")
            row = spread[(spread["model"] == model) & (spread["target"] == target)
                         & (spread["horizon"] == int(h))]
            if row.empty or abs(float(row["std_kw_pp"].iloc[0]) - value) > SPREAD_TOL:
                problems.append(f"{step} {key}: the step's spread {value:.6f} differs from the "
                                "benchmark's")
    return problems


def g1_points(src: Sources, fits: pd.DataFrame, studies: pd.DataFrame) -> pd.DataFrame:
    """The points of the G-1 list that the run informs (§1, R9-06, R9-11, R9-17): facts only."""
    rows = []

    def add(point: str, subject: str, fact: str) -> None:
        rows.append({"point": point, "subject": subject, "fact": fact})

    if not fits.empty:
        for r in fits[fits["final_minus_best_trial_pp"].abs() > 1e-6].itertuples():
            add("final fit against its best trial", f"{r.step} {r.subject}",
                f"final {r.value_kw:.3f} %, best trial {r.best_trial_value_kw:.3f} % "
                f"({r.final_minus_best_trial_pp:+.3f} pp)")
        patch = fits[fits["step"] == "patchtst"]
        for r in patch.itertuples():
            if r.best and r.cap:
                add("best check against the step budget", f"patchtst {r.subject}",
                    f"best step {int(r.best)} of a budget of {int(r.cap)}")
    if not studies.empty:
        for r in studies[studies["edges"] != ""].itertuples():
            add("best value on the edge of the search space", f"{r.step} {r.subject}", r.edges)
        for r in studies[studies["n_trials"] <= 10].itertuples():
            add("study inside TPE's random start-up", f"{r.step} {r.subject}",
                f"{r.n_trials} trials (start-up 10)")
        xgb = studies[studies["step"] == "xgb"]
        if not xgb.empty and xgb["n_at_cap"].sum():
            add("trials stopped at their cap", "xgb",
                f"{int(xgb['n_at_cap'].sum())} of {int(xgb['n_trials'].sum())} trials")
    for step in NEURAL:
        if src.has(step, "mbe.csv"):
            m = src.csv(step, "mbe.csv")
            m = m[(m["scope"] == "val") & (m["horizon"].astype(str) == "all")]
            for r in m.itertuples():
                add("mean bias on validation", f"{step} {r.target} seed {r.seed}",
                    f"{r.mbe_pct:+.2f} % of capacity ({r.mbe_kw:+.1f} kW)")
    if src.has("evaluate", "val/dm_tests_kw.csv"):
        dm = src.csv("evaluate", "val/dm_tests_kw.csv")
        clim = dm[(dm["model_a"] == "climatology")]
        for r in clim.itertuples():
            add("margin against climatology (kW DM)", f"{r.model_b} {r.target} D+{r.horizon}",
                f"{r.pct_of_mde:.0f} % of the minimum detectable effect, HLN p "
                f"{r.p_value_hln:.4f}")
    return pd.DataFrame(rows, columns=["point", "subject", "fact"])


def analysis_tables(src: Sources) -> dict[str, pd.DataFrame]:
    """The summaries of the steps `cv` (S-38) and `ablation` (S-41), and their test values from
    the step `test`, each read from its file with its hash (X73, M31)."""
    out = {}
    for key, step, name in (("cv", "cv", "cv_summary.csv"),
                            ("ablation", "ablation", "ablation_summary.csv"),
                            ("cv_vs_test", "test", "analyses/cv_vs_test.csv"),
                            ("ablation_test", "test", "analyses/ablation_test_summary.csv")):
        if src.has(step, name):
            out[key] = src.csv(step, name)
    return out


def explain_tables(src: Sources) -> dict[str, pd.DataFrame]:
    """The highlights of `explain`: the top three SHAP features and the group shares per
    booster, the TFT same-hour attention ratio, the most important hyper-parameter per study."""
    out = {}
    if src.has("explain", "xgb/xgb_shap_features.csv"):
        f = src.csv("explain", "xgb/xgb_shap_features.csv")
        out["shap_top"] = f[f["rank"] <= 3][["target", "horizon", "seed", "rank", "feature",
                                             "group", "mean_abs_kw"]]
        g = src.csv("explain", "xgb/xgb_shap_groups.csv")
        out["shap_groups"] = g[["target", "horizon", "seed", "group", "n_members",
                                "share_pct", "share_pct_v6_measure"]]
    if src.has("explain", "tft/tft_attention_summary.csv"):
        a = src.csv("explain", "tft/tft_attention_summary.csv")
        out["tft_attention"] = a[["step", "target", "seed", "block", "same_hour_ratio",
                                  "argmax_lag_h", "first_position_ratio"]]
    if src.has("explain", "hparam_importance.csv"):
        h = src.csv("explain", "hparam_importance.csv").dropna(subset=["importance_mean"])
        if not h.empty:
            top = h.sort_values("importance_mean", ascending=False).groupby(
                ["step", "target", "horizon", "seed"], as_index=False).head(1)
            out["hparam_top"] = top[["step", "target", "horizon", "seed", "param",
                                     "importance_mean", "importance_std", "n_complete",
                                     "all_in_startup"]]
    return out


# ── the Markdown summary ────────────────────────────────────────────────────


def _md_table(frame: pd.DataFrame, columns: list[str], floats: int = 3) -> list[str]:
    if frame.empty:
        return ["(none)"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame[columns].itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.{floats}f}")
            else:
                cells.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def markdown(ctx: Context, tables: dict[str, pd.DataFrame], sources: dict[str, str],
             problems: list[str]) -> str:
    lines = [f"# Run report: {ctx.run_id}", "",
             f"Mode `{ctx.mode}`. Every number below is read from a file of this run; the files "
             "and their SHA-256 are listed at the end (M31). Validation numbers are tuning-split "
             "numbers (D-10): validation chose the hyper-parameters and the stopping round.", "",
             "## Steps", ""]
    lines += _md_table(tables["steps"], ["step", "status", "commit", "duration_s",
                                         "n_warnings"], 1)
    for split in ("val", "test"):
        key = f"benchmark_{split}"
        if key in tables:
            b = tables[key].sort_values(["target", "horizon", "model", "seed"])
            lines += ["", f"## Benchmark on {split} (nRMSE in kW, % of 3,449 kW)", ""]
            lines += _md_table(b, ["target", "horizon", "model", "seed", "n", "nRMSE_kW_pct",
                                   "skill_score_kw"], 3)
    if "seed_spread" in tables and not tables["seed_spread"].empty:
        lines += ["", "## Seed spread (stage-10 rule: below 0.15 pp in kW)", ""]
        lines += _md_table(tables["seed_spread"], ["model", "target", "horizon", "n_seeds",
                                                   "std_kw_pp", "std_target_unit_pp"], 3)
    lines += ["", "## Training diagnostics", ""]
    lines += _md_table(tables["training"], ["step", "subject", "unit_of_best", "best", "cap",
                                            "run", "gap_ratio", "value_kw"], 3)
    lines += ["", "## Studies", ""]
    lines += _md_table(tables["studies"], ["step", "subject", "n_trials", "best_trial",
                                           "best_value_kw", "edges"], 3)
    w = tables["warnings"]
    lines += ["", f"## Automatic warnings ({len(w)})", ""]
    lines += _md_table(w, ["step", "kind", "status", "text"])
    if "shap_top" in tables:
        lines += ["", "## Explanation: the three largest mean |SHAP| (kW) per booster", ""]
        lines += _md_table(tables["shap_top"], ["target", "horizon", "seed", "rank", "feature",
                                                "group", "mean_abs_kw"], 2)
    if "tft_attention" in tables:
        lines += ["", "## Explanation: TFT attention on the same hour of earlier days", ""]
        lines += _md_table(tables["tft_attention"], ["step", "target", "block",
                                                     "same_hour_ratio", "argmax_lag_h",
                                                     "first_position_ratio"], 2)
    if "hparam_top" in tables:
        lines += ["", "## Explanation: the most important hyper-parameter per study (fANOVA, "
                  "ten seeds)", ""]
        lines += _md_table(tables["hparam_top"], ["step", "target", "horizon", "param",
                                                  "importance_mean", "importance_std",
                                                  "n_complete", "all_in_startup"], 2)
    if "cv" in tables:
        lines += ["", "## Cross-validation of the xgb models on the train rows (S-38, nRMSE in "
                  "kW, mean and std over five folds)", ""]
        lines += _md_table(tables["cv"], ["target", "horizon", "seed", "cv_mean", "cv_std",
                                          "validation_uncorrected", "validation_minus_cv_pp"], 2)
    if "ablation" in tables:
        lines += ["", "## Feature-group ablation of the xgb models (S-41, delta of nRMSE in kW "
                  "against the full fit, pp)", ""]
        lines += ["With one seed (`n_seeds` 1) the margin is only the 0.05 pp floor and a "
                  "verdict is provisional: the seed spread of XGB-Opt is of that size (M41).", ""]
        lines += _md_table(tables["ablation"], ["target", "horizon", "mode", "n_seeds",
                                                "mean_delta_pp", "std_delta_pp", "margin_pp",
                                                "verdict"], 3)
    lines += ["", "## Points for the G-1 list (facts, no verdict)", ""]
    lines += _md_table(tables["g1_points"], ["point", "subject", "fact"])
    if problems:
        lines += ["", "## Consistency problems", ""] + [f"- {p}" for p in problems]
    lines += ["", "## Sources", ""] + [f"- `{k}` sha256 `{v[:16]}`" for k, v in
                                       sorted(sources.items())]
    return "\n".join(lines) + "\n"


def run_report(ctx: Context) -> Result:
    src = Sources(ctx)
    tables: dict[str, pd.DataFrame] = {"steps": steps_table(ctx)}
    tables.update(metrics_tables(src))
    spread = seed_spread(tables["benchmark_val"]) if "benchmark_val" in tables else \
        seed_spread(pd.DataFrame(columns=["model", "target", "horizon", "seed", "nRMSE_kW_pct",
                                          "nRMSE_pct"]))
    tables["seed_spread"] = spread
    fits, studies, recomputed = training_tables(src)
    tables["training"], tables["studies"] = fits, studies
    tables["warnings"] = match_warnings(ctx, recomputed, spread)
    tables["g1_points"] = g1_points(src, fits, studies)
    tables.update(explain_tables(src))
    tables.update(analysis_tables(src))
    problems = spread_check(ctx, spread)
    files = [{"step": s, "file": n} for s in NEURAL for n in
             ctx.completed.get(s, {}).get("outputs", {}) if n.startswith(("quantiles_",
                                                                          "mbe.csv"))]
    tables["l14_l15_files"] = pd.DataFrame(files, columns=["step", "file"])
    out = ctx.out
    outputs = []
    for key, frame in tables.items():
        path = out / f"report_{key}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)
    record = {"run_id": ctx.run_id, "mode": ctx.mode, "sources": src.used,
              "problems": problems, "steps_missing": [s for s in MODEL_STEPS
                                                      if s in ctx.failed],
              "tables": {k: f"report_{k}.csv" for k in tables}}
    (out / "report.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8",
                                     newline="\n")
    (out / "report.md").write_text(markdown(ctx, tables, src.used, problems), encoding="utf-8",
                                   newline="\n")
    outputs += [out / "report.json", out / "report.md"]
    w = tables["warnings"]
    for p in problems:
        ctx.log.warn(f"report: {p}")
    counts = w.groupby("status").size().to_dict() if not w.empty else {}
    summary = [f"report: {len(tables['steps'])} steps, {len(w)} warnings {counts}, "
               f"{len(tables['g1_points'])} G-1 facts, {len(src.used)} source files"]
    return Result(outputs, values={"warnings": counts, "problems": problems,
                                   "sources": len(src.used)}, summary=summary)

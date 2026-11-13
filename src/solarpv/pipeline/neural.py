"""The neural model steps of the runner: `bilstm`, `tft96`, `tft168`, `tftbase`, `patchtst`
(docs/stage9_plan.md §18, W4). This side runs in `solar`; the training runs in `solar-gpu`.

One step (X25):
  1. builds the W3 datasets of every target (training scope and every evaluation window), runs
     `check_dataset` on each (X33) and the check of the missing known inputs (X30);
  2. writes them with `neural_job` into `<step>/job/` (compressed arrays with their SHA-256, and
     `job.json`: settings, budget, seeds, channels); no timestamp is written (M13);
  3. starts `python -m solarpv.models.neural_worker` in `solar-gpu` and streams its progress
     into the step log;
  4. checks every worker output against `worker.json`, writes the prediction files through
     `sequence_data.predictions_frame` (valid hours and horizons from the dataset only: M14,
     M18), the quantiles (not scored), the MBE (M24), the trials, the learning curves, the
     configuration, the figures and the automatic warnings of §3.5.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, DATA_END_UTC, DATA_START_UTC, TEST_START
from solarpv.config.paths import REPO_ROOT
from solarpv.evaluation.evaluate import withhold_test_truth
from solarpv.evaluation.units import to_kw
from solarpv.models import neural_job as J
from solarpv.models import sequence_data as SD
from solarpv.pipeline import steps
from solarpv.pipeline.steps import Context, Result, RunnerError
from solarpv.selection.common import PREVDAY

GPU_ENV_LOCK = REPO_ROOT / "requirements-gpu.lock.txt"
GPU_PYTHON_VAR = "SOLARPV_GPU_PYTHON"
GAP_RATIO_WARN = 2.0  # validation loss more than twice the train loss at the best epoch
SEED_SPREAD_WARN_PP = 0.15  # the acceptance rule of stage 10
REPRODUCE_TOL_PP = 1e-6
NIGHT_ONLY = "airmass"  # X30: pvlib leaves the air mass undefined below the horizon
TRAIN_ONLY = PREVDAY  # X30: 24 train hours without the previous day's clear-sky index (W3)
CENTRED_MEAN = "_ma"  # X35: centred GFS means are undefined at the two edge hours of the data


# ── the GPU environment ─────────────────────────────────────────────────────


def gpu_python() -> Path:
    """`solar-gpu`'s python, next to `solar` under the same conda root (or $SOLARPV_GPU_PYTHON)."""
    if os.environ.get(GPU_PYTHON_VAR):
        return Path(os.environ[GPU_PYTHON_VAR])
    prefix = Path(sys.prefix)
    envs = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    return envs / "solar-gpu" / ("python.exe" if os.name == "nt" else "bin/python")


def gpu_report(timeout: int = 600) -> dict:
    """The report of `python -m solarpv.models.neural_env` in `solar-gpu` (no probe)."""
    exe = gpu_python()
    if not exe.exists():
        return {"errors": [f"no GPU environment at {exe} (D-29, environment-gpu.yml)"]}
    proc = subprocess.run([str(exe), "-m", "solarpv.models.neural_env", "--no-probe"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"errors": [f"the GPU environment report failed: {proc.stderr[-500:]}"]}


def code_root() -> Path:
    """The `src` folder the runner's own `solarpv` comes from."""
    return (REPO_ROOT / "src").resolve()


def worker_env() -> dict:
    """X55: `solarpv` is an editable install in both environments, pointing at one checkout; the
    runner's `src` goes first on `PYTHONPATH`, so the worker imports the code of the runner even
    when the runner runs from another checkout."""
    env = os.environ.copy()
    env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8",
                "PYTHONHASHSEED": "0"})
    env["PYTHONPATH"] = os.pathsep.join(
        [str(code_root())] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def call_worker(ctx: Context, job_dir: Path, out_dir: Path) -> None:
    """Run the worker; its progress lines go to the step log, everything to worker_stdout.log."""
    cmd = [str(gpu_python()), "-m", "solarpv.models.neural_worker", str(job_dir), str(out_dir)]
    ctx.log.note("worker: " + " ".join(cmd))
    lines = []
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=worker_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if line.startswith("[worker"):
            ctx.log.note(line)
    code = proc.wait()
    (ctx.out / "worker_stdout.log").write_text("\n".join(lines) + "\n", encoding="utf-8",
                                               newline="\n")
    if code != 0:
        tail = "\n".join(lines[-15:])
        raise RunnerError(f"the training worker failed (exit code {code}):\n{tail}")


# ── inputs ──────────────────────────────────────────────────────────────────


def neural_inputs(ctx: Context, step: str) -> list[Path]:
    from solarpv.models import xgb_sota as X

    steps.check_selected_inputs(ctx)  # M25
    return [*SD.default_paths().files(), X.MODEL_INPUTS_FILE, steps.DERIVED_CONSTANTS_FILE,
            GPU_ENV_LOCK]


def read_hours(datasets: dict) -> np.ndarray:
    """The hours (int64 ns) that the windows of a target read: every encoder hour and every
    existing decoder step (X35)."""
    parts = []
    for ds in datasets.values():
        parts += [ds.enc_time.ravel(), ds.dec_time[ds.dec_exists]]
    return np.unique(np.concatenate(parts))


def check_missing_known(sources: SD.Sources, target: str, inputs: dict,
                        hours: np.ndarray | None = None) -> dict:
    """X30: a missing known input on an hour that a window reads is the night air mass, a train
    hour of the previous day's clear-sky index, or (X35) a centred GFS mean at one of the two edge
    hours of the data at night; anything else is an error. Counts per input and split, over the
    hours read (all loaded hours when `hours` is None)."""
    f = sources.frame
    det, gfs = SD.known_inputs(target, inputs)
    night = (f["sun_elevation"].to_numpy(float) <= 0 if "sun_elevation" in f
             else np.zeros(len(f), dtype=bool))
    train = (f["split"] == "train").to_numpy()
    read = (np.ones(len(f), dtype=bool) if hours is None
            else np.isin(SD._ns(f.index), hours))
    edges = SD._ns(pd.DatetimeIndex([pd.Timestamp(DATA_START_UTC, tz="UTC"),
                                     pd.Timestamp(DATA_END_UTC, tz="UTC").floor("h")]))
    edge = np.isin(SD._ns(f.index), edges) & night
    counts, bad = {}, []
    columns = [(c, f[c]) for c in det] + [(f"{c}@D{h}", f[f"{c}@D{h}"]) for c in gfs
                                         for h in (1, 2, 3)]
    for name, col in columns:
        missing = col.isna().to_numpy() & read
        if not missing.any():
            continue
        base = name.split("@")[0]
        counts[name] = {s: int((missing & (f["split"] == s).to_numpy()).sum())
                        for s in sorted(f["split"].unique())}
        if base == NIGHT_ONLY:
            allowed = night
        elif base == TRAIN_ONLY:
            allowed = train
        elif base in gfs and CENTRED_MEAN in base:
            allowed = edge
        else:
            allowed = np.zeros(len(f), dtype=bool)
        n_bad = int((missing & ~allowed).sum())
        if n_bad:
            bad.append(f"{name}: {n_bad} hours")
    if bad:
        raise RunnerError(f"{target}: missing known inputs outside the allowed hours (X30): "
                          f"{bad}")
    return counts


# ── the job ─────────────────────────────────────────────────────────────────


def _positions(ds: SD.SequenceDataset, sources: SD.Sources) -> np.ndarray:
    first = SD._ns(sources.frame.index)[0]
    return np.where(ds.dec_exists, (ds.dec_time - first) // SD.HOUR_NS, 0)


def pack(ds: SD.SequenceDataset, key: str, sources: SD.Sources) -> dict:
    """The arrays of one scope; no timestamp (M13). The scoring arrays of §14.1 for every scope:
    daytime hours of the scope with AC truth."""
    f = sources.frame
    pos = _positions(ds, sources)
    exists = ds.dec_exists
    truth = np.where(exists, f["site_total_AC_kW"].to_numpy(float)[pos], np.nan)
    day = exists & f["daytime"].astype(bool).to_numpy()[pos]
    arrays = {
        f"{key}_enc_x": ds.enc_x, f"{key}_dec_x": ds.dec_x, f"{key}_y": ds.y,
        f"{key}_loss_mask": ds.loss_mask, f"{key}_dec_exists": exists,
        f"{key}_dec_keep": ds.dec_keep, f"{key}_dec_horizon": ds.dec_horizon,
        f"{key}_score_mask": ds.dec_keep & day & np.isfinite(truth),
        f"{key}_theo_kw": np.where(exists, f["theo_kw"].to_numpy(float)[pos], 0.0),
        f"{key}_truth_kw": truth,
        f"{key}_k_true": np.where(exists, f["target_k"].to_numpy(float)[pos], np.nan),
    }
    if key == "test":  # D-32, R-G1-04: phase 1 forecasts the test window and holds no test truth
        arrays.update({f"{key}_y": np.zeros_like(ds.y),
                       f"{key}_loss_mask": np.zeros_like(ds.loss_mask),
                       f"{key}_score_mask": np.zeros_like(arrays[f"{key}_score_mask"]),
                       f"{key}_truth_kw": np.full_like(truth, np.nan),
                       f"{key}_k_true": np.full_like(arrays[f"{key}_k_true"], np.nan)})
    return arrays


def patchtst_series(sources: SD.Sources, target: str, train: SD.SequenceDataset,
                    tune: SD.SequenceDataset) -> dict:
    """PatchTST's training series (model units, filled as its W3 windows) from the first loaded
    hour to the end of the tuning window; the mask is the loss mask of the training scope and of
    the tuning window (X31); the tuning window is the validation part."""
    from solarpv.models.xgb_sota import fit_mask

    f = sources.frame
    own = SD.TARGET_COLUMN["direct"] if target == "direct" else SD.K_SERIES
    scale = SD.TARGET_SCALE[target]
    series = np.nan_to_num(sources.filled[own].to_numpy(float) / scale, nan=0.0)
    idx = f.index
    start = int(idx.get_loc(tune.scope.hours.min()))
    end = int(idx.get_loc(tune.scope.hours.max())) + 1
    truth = f[SD.TARGET_COLUMN[target]].to_numpy(float)
    fit = fit_mask(f, target).to_numpy() & np.isfinite(truth)
    mask = fit & (idx.isin(train.scope.hours) | idx.isin(tune.scope.hours))
    if not np.allclose(series[mask] * scale, truth[mask], rtol=0, atol=1e-6):
        raise RunnerError(f"{target}: the PatchTST series differs from the target on a loss hour")
    return {"series_y": series[:end].astype(np.float32), "series_mask": mask[:end],
            "series_val_size": np.int64(end - start)}


def build_job(ctx: Context, step: str, job_dir: Path, seeds: list[int], budget: dict,
              mean_k: float) -> tuple[dict, dict, dict]:
    """Datasets of every target and scope, checked, written as the job. Returns the job, the
    datasets (target -> scope key -> dataset) and a report."""
    from solarpv.models import xgb_sota as X

    family = J.STEP_FAMILY[step]
    with_test = any(w.split == "test" for w in ctx.windows)
    inputs = X.load_model_inputs(X.MODEL_INPUTS_FILE)
    with ctx.log.step("load the inputs" + ("" if with_test else " before TEST_START")):
        sources = SD.load_sources(allow_test=with_test)
        if not with_test and sources.frame.index.max() >= pd.Timestamp(TEST_START, tz="UTC"):
            raise RunnerError("test rows reached a neural step in a development run")
        if with_test:  # final: the test window is forecast, never scored here (D-32, X40)
            n_test = int((sources.frame.index >= pd.Timestamp(TEST_START, tz="UTC")).sum())
            report_access = steps.log_test_access(
                ctx, "forecast the test window (the encoder series and the known inputs)",
                list(SD.default_paths().files()), n_test)
        scaler = SD.Scaler.fit(sources)
    tune = [w for w in ctx.windows if w.split == "val"]
    if len(tune) != 1:
        raise RunnerError("a neural step needs exactly one validation window to tune on")
    hours = SD.MODELS[step]
    job_dir.mkdir()
    datasets, report, meta, files = {}, {"datasets": {}, "missing_known": {}}, {}, {}
    for target in ctx.config["targets"]:
        with ctx.log.step(f"{target}: build and check the windows"):
            scopes = {"train": SD.training_scope(sources, ctx.mode)}
            scopes.update({w.split: SD.evaluation_scope(sources, w) for w in ctx.windows})
            dsets = {k: SD.build(sources, scaler, step, target, s, inputs=inputs,
                                 encoder_hours=hours) for k, s in scopes.items()}
            for k, ds in dsets.items():
                checks = SD.check_dataset(ds, sources)
                failed = [c for c, v in checks.items() if isinstance(v, bool) and not v]
                if failed:
                    raise RunnerError(f"{target} {k}: the dataset fails {failed} (X33)")
                report["datasets"][f"{target}/{k}"] = {**ds.info, "checks": checks,
                                                       "scope": ds.scope.label}
            if step != "patchtst":
                report["missing_known"][target] = check_missing_known(
                    sources, target, inputs, hours=read_hours(dsets))
            arrays = {}
            for k, ds in dsets.items():
                arrays.update(pack(ds, k, sources))
            if family == "patchtst":
                arrays.update(patchtst_series(sources, target, dsets["train"], dsets["val"]))
            path = job_dir / f"data_{target}.npz"
            files[target] = {"path": path.name, "sha256": J.write_arrays(path, arrays)}
            ref = dsets["train"]
            names = SD.encoder_series_names(target)
            meta[target] = {"enc_channels": ref.enc_channels, "dec_channels": ref.dec_channels,
                            "n_unknown": 0 if step == "patchtst" else 2 * len(names),
                            "encoder_hours": ref.encoder_hours,
                            "target_scale": float(ref.target_scale)}
            datasets[target] = dsets
            counts = ", ".join(f"{k} {ds.info['n_origins']} origins" for k, ds in dsets.items())
            ctx.log.note(f"{target}: {counts}; {len(ref.enc_channels)} encoder and "
                         f"{len(ref.dec_channels)} decoder channels; {hours} encoder hours")
    job = {"version": J.JOB_VERSION, "step": step, "model": J.STEP_MODEL[step],
           "family": family, "mode": ctx.mode, "run_id": ctx.run_id,
           "targets": list(ctx.config["targets"]), "seeds": seeds, "budget": budget,
           "fixed": J.FIXED, "search_space": J.SEARCH_SPACES.get(family),
           "quantiles": list(J.QUANTILES[family]), "scopes": ["train", *[w.split for w in
                                                                         ctx.windows]],
           "tune_scope": "val", "mean_k": float(mean_k), "meta": meta, "files": files}
    J.write_json(job_dir / J.JOB_FILE, job)
    report["scaler_rows"] = scaler.rows
    report["test_access"] = [report_access] if with_test else []
    return job, datasets, report


# ── reading the worker's outputs ────────────────────────────────────────────


def verify_worker(out_dir: Path, job: dict) -> dict:
    """Every output of `worker.json` exists with its hash; every expected file is there."""
    path = out_dir / J.WORKER_FILE
    if not path.exists():
        raise RunnerError("the worker wrote no worker.json")
    worker = J.read_json(path)
    for name, digest in worker["outputs"].items():
        if J.sha256_file(out_dir / name) != digest:
            raise RunnerError(f"worker/{name}: SHA-256 differs from worker.json")
    need = [n(t, s) for t in job["targets"] for s in job["seeds"]
            for n in (J.forecast_name, J.fit_name, J.curves_name)]
    missing = [n for n in need if n not in worker["outputs"]]
    if missing:
        raise RunnerError(f"the worker did not write {missing}")
    imported = worker.get("solarpv_file")
    if imported is None or not Path(imported).resolve().is_relative_to(code_root()):
        raise RunnerError(f"the worker imported solarpv from {imported}, not from the runner's "
                          f"{code_root()} (X55)")
    return worker


def mbe_rows(target: str, seed: int, key: str, point: np.ndarray, arrays: dict,
             scale: float) -> list[dict]:
    """M24: mean bias (forecast - actual AC) in kW and % of capacity, and the nRMSE in kW, on
    the scoring steps of a scope, per horizon and pooled."""
    sel = arrays[f"{key}_score_mask"].astype(bool)
    kw = to_kw(target, point * scale, arrays[f"{key}_theo_kw"])
    err = kw - arrays[f"{key}_truth_kw"]
    rows = []
    for h in (0, 1, 2, 3):
        s = sel & (arrays[f"{key}_dec_horizon"] == h) if h else sel
        e = err[s]
        rows.append({"target": target, "seed": seed, "scope": key, "horizon": h or "all",
                     "n": int(s.sum()), "mbe_kw": float(e.mean()),
                     "mbe_pct": float(e.mean() / AC_CAPACITY_KW * 100),
                     "nrmse_kw_pct": float(np.sqrt(np.mean(e ** 2)) / AC_CAPACITY_KW * 100)})
    return rows


def warning_list(step: str, records: dict, budget: dict) -> list[dict]:
    """The automatic warnings of §3.5 for every final fit and study, as records (kind, subject,
    text) computed from the worker's records alone; the step logs `<step>: <text>`, and the
    report recomputes them from the stored `best_params.json` (X49)."""
    out = []
    family = J.STEP_FAMILY[step]

    def add(kind: str, name: str, text: str) -> None:
        out.append({"kind": kind, "subject": name, "text": f"{step}: {name}: {text}"})

    for target, by_seed in records.items():
        for seed, rec in by_seed.items():
            name, f = f"{target} seed {seed}", rec["final"]
            if family == "patchtst":
                checks = int(f.get("val_checks") or 0)
                if f["best_step"] is not None and f["steps_run"] is not None:
                    if f["best_step"] <= int(budget["val_check_steps"]):
                        add("first_check", name, "best validation check at the first check")
                    if checks and f["best_step"] >= f["steps_run"] - 1:
                        add("last_step", name, "best validation check at the last step: it "
                                               "may not have converged")
                continue
            if f["nan_loss"]:
                add("nan", name, "NaN or infinite loss")
            if f["best_epoch"] == 1:
                add("first_epoch", name, "best epoch at the first epoch")
            if f["best_epoch"] == f["epochs_run"] and not f["stopped_early"]:
                add("last_epoch", name, f"best epoch {f['best_epoch']} at the last epoch: it "
                                        "may not have converged")
            if f["train_loss_at_best"] > 0 and (f["best_val_loss"] / f["train_loss_at_best"]
                                                > GAP_RATIO_WARN):
                add("gap", name, f"validation loss {f['best_val_loss']:.4g} is "
                                 f"{f['best_val_loss'] / f['train_loss_at_best']:.1f} x the "
                                 f"train loss {f['train_loss_at_best']:.4g} at the best epoch")
            space = J.SEARCH_SPACES.get(family)
            if rec.get("study") and space:
                params, eff_space = J.effective(rec["params"], space)  # X45
                edges = J.edge_params(params, eff_space)
                if edges:
                    add("edge_space", name, "best trial on the edge of the search space for "
                        + ", ".join(f"{p} = {params[p]}" for p in edges) + " (M23)")
            diff = rec.get("reproduce_diff_pp")
            if diff is not None and diff > REPRODUCE_TOL_PP:
                add("reproduce", name, f"the retrained best trial differs from the trial by "
                                       f"{diff:.3g} pp (X32)")
    return out


def _warnings(ctx: Context, step: str, worker: dict, job: dict) -> list[str]:
    """The automatic warnings of §3.5 for every final fit and study."""
    out = [w["text"].removeprefix(f"{step}: ")
           for w in warning_list(step, worker["records"], job["budget"])]
    for w in out:
        ctx.log.warn(f"{step}: {w}")
    return out


def _seed_spread(ctx: Context, step: str, records: dict) -> dict:
    spread = {}
    for target, by_seed in records.items():
        if len(by_seed) < 2:
            continue
        for h in (1, 2, 3):
            values = [r["final"][f"value_kw_D{h}"] for r in by_seed.values()]
            spread[f"{target}_D{h}"] = float(np.std(values, ddof=1))
            if spread[f"{target}_D{h}"] > SEED_SPREAD_WARN_PP:
                ctx.log.warn(f"{step}: {target} D+{h}: seed spread "
                             f"{spread[f'{target}_D{h}']:.3f} pp above {SEED_SPREAD_WARN_PP}")
    if not spread:
        ctx.log.note(f"{step}: seed spread: one seed per target, not computed")
    return spread


# ── figures ─────────────────────────────────────────────────────────────────


def plot_curves(curves: pd.DataFrame, step: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = curves[curves["fit"] == "final"]
    targets = list(dict.fromkeys(final["target"]))
    fig, axes = plt.subplots(1, max(len(targets), 1), squeeze=False,
                             figsize=(5.2 * max(len(targets), 1), 3.6))
    x = "step" if step == "patchtst" else "epoch"
    for a, target in zip(axes[0], targets, strict=False):
        for seed, g in final[final["target"] == target].groupby("seed"):
            for col, style in (("train_loss", "-"), ("val_loss", "-")):
                if col in g and g[col].notna().any():
                    gg = g[g[col].notna()]
                    a.plot(gg[x], gg[col], style, lw=1.2,
                           label=f"{col.replace('_', ' ')}, seed {seed}")
        a.set_title(f"{step} {target}: final fit", fontsize=9)
        a.set_xlabel(x)
        a.set_ylabel("loss (model units)")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(trials: pd.DataFrame, step: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys(trials["target"]))
    fig, axes = plt.subplots(1, len(targets), squeeze=False, figsize=(5.2 * len(targets), 3.4))
    for a, target in zip(axes[0], targets, strict=True):
        for seed, g in trials[trials["target"] == target].groupby("seed"):
            g = g.sort_values("trial")
            a.plot(g["trial"], g["value_kw"], "o", ms=3, label=f"trial, seed {seed}")
            a.plot(g["trial"], g["value_kw"].cummin(), "-", lw=1.2,
                   label=f"best so far, seed {seed}")
        a.set_title(f"{step} {target}", fontsize=9)
        a.set_xlabel("trial")
        a.set_ylabel("validation nRMSE [% of 3,449 kW]")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── the step ────────────────────────────────────────────────────────────────


def _csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def run_neural(ctx: Context, step: str) -> Result:
    family, model_name = J.STEP_FAMILY[step], J.STEP_MODEL[step]
    budget = dict(ctx.config["budgets"][step])
    seeds = [int(s) for s in (ctx.config["seeds"] or budget["seeds"])]
    mean_k = steps.mean_daytime_k()
    env = gpu_report()
    if env.get("errors"):
        raise RunnerError(f"the GPU environment fails its checks: {env['errors']}")
    ctx.log.note(f"{step} ({model_name}): budget {budget}, seeds {seeds}; GPU environment "
                 f"{env.get('versions', {}).get('torch')} on {env.get('gpu', {}).get('name')}")
    job_dir, out_dir = ctx.out / "job", ctx.out / "worker"
    job, datasets, report = build_job(ctx, step, job_dir, seeds, budget, mean_k)
    with ctx.log.step(f"{step}: the training worker in solar-gpu"):
        call_worker(ctx, job_dir, out_dir)
    worker = verify_worker(out_dir, job)
    quantiles = J.QUANTILES[family]
    iq = quantiles.index(J.POINT_QUANTILE)
    horizons = [int(h) for h in ctx.config["horizons"]]
    outputs: list[Path] = []
    mbe, curves, trials, best = [], [], [], {}
    arrays = {t: J.read_arrays(job_dir / f["path"], f["sha256"]) for t, f in job["files"].items()}
    for seed in seeds:
        frames, qframes = [], []
        for target in job["targets"]:
            name = J.forecast_name(target, seed)
            fc = J.read_arrays(out_dir / name, worker["outputs"][name])
            scale = job["meta"][target]["target_scale"]
            for key, ds in datasets[target].items():
                q = fc[f"{key}_q"]
                if q.shape != (ds.dec_keep.shape[0], J.MAX_STEPS, len(quantiles)):
                    raise RunnerError(f"{name}: {key} forecast of shape {q.shape}")
                point = q[..., iq].astype(float)
                if key != "test":  # D-32, R-G1-01: phase 1 scores train and validation only
                    mbe += mbe_rows(target, seed, key, point, arrays[target], scale)
                if key == "train":
                    continue
                frame = withhold_test_truth(  # D-32, R-G1-04
                    SD.predictions_frame(ds, point, model_name=model_name, seed=seed))
                frames.append(frame[frame["horizon"].isin(horizons)])
                if len(quantiles) > 1:
                    sel = ds.dec_keep
                    qf = pd.DataFrame({"timestamp": pd.to_datetime(ds.dec_time[sel], utc=True),
                                       "target": target,
                                       "horizon": ds.dec_horizon[sel].astype(int),
                                       "split": ds.scope.split})
                    for i, qq in enumerate(quantiles):
                        qf[f"q{qq:g}"] = q[..., i][sel] * scale
                    qframes.append(qf[qf["horizon"].isin(horizons)])
            curves.append(pd.read_csv(out_dir / J.curves_name(target, seed)))
            tname = J.trials_name(target, seed)
            if tname in worker["outputs"]:
                trials.append(pd.read_csv(out_dir / tname))
            best[f"{target}_seed{seed}"] = worker["records"][target][str(seed)]
        pred = pd.concat(frames, ignore_index=True).sort_values(["target", "horizon",
                                                                 "timestamp"])
        outputs.append(_csv(pred, ctx.out / f"predictions_{model_name}_seed{seed}.csv"))
        if qframes:
            qall = pd.concat(qframes, ignore_index=True).sort_values(["target", "horizon",
                                                                      "timestamp"])
            outputs.append(_csv(qall, ctx.out / f"quantiles_{model_name}_seed{seed}.csv"))
    outputs.append(_csv(pd.DataFrame(mbe), ctx.out / "mbe.csv"))
    curves_df = pd.concat(curves, ignore_index=True)
    outputs.append(_csv(curves_df, ctx.out / "learning_curves.csv"))
    trials_df = pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()
    if trials:
        outputs.append(_csv(trials_df, ctx.out / "trials.csv"))
    J.write_json(ctx.out / "best_params.json", best)
    warnings = _warnings(ctx, step, worker, job)
    spread = _seed_spread(ctx, step, worker["records"])
    if family == "tft":
        for target in job["targets"]:
            for seed in seeds:
                fc = J.read_arrays(out_dir / J.forecast_name(target, seed))
                att = fc["val_attention"].astype(np.float64)
                keep = datasets[target]["val"].dec_keep
                mean = att[keep].mean(axis=0)  # over every kept step of every origin
                L = att.shape[-1]
                outputs.append(_csv(pd.DataFrame({"lag_h": np.arange(-L, 0), "attention": mean}),
                                    ctx.out / f"attention_{target}_seed{seed}.csv"))
    config = {
        "step": step, "model": model_name, "family": family, "mode": ctx.mode, "budget": budget,
        "seeds": seeds, "windows": [w.as_dict() for w in ctx.windows],
        "encoder_hours": SD.MODELS[step], "quantiles": list(quantiles),
        "point_forecast": f"quantile {J.POINT_QUANTILE}",
        "search_space": J.SEARCH_SPACES.get(family), "fixed": J.FIXED,
        "objective": "daytime nRMSE in kW (% of 3,449 kW) of the point forecast on the daytime "
                     "steps of the tuning window with AC truth, op-limited hours included, "
                     "pooled over the three horizons; indirect through units.to_kw "
                     "(docs/stage9_plan.md §14.1, §18.4)",
        "early_stopping": "the training loss on the loss-mask steps of the tuning window; the "
                          "weights of the best epoch (PatchTST: validation check) restored",
        "loss": "a mean over the steps of the W3 loss mask only (X31)",
        "normalisers": "none fitted in the libraries: inputs scaled by the W3 scaler (train "
                       "hours), targets in model units; TFT identity output transformer, "
                       "PatchTST scaler_type identity with RevIN (X29)",
        "missing_known_inputs": report["missing_known"], "datasets": report["datasets"],
        "scaler_rows": report["scaler_rows"], "seed_spread_pp": spread,
        "warnings": warnings, "gpu_environment": worker["environment"],
        "determinism": worker["determinism"], "worker_duration_s": worker["duration_s"],
        "job_files": job["files"]}
    if family == "patchtst":
        config["l19_exogenous_futr"] = {t: r["exogenous_futr"] for t, by in
                                        worker["records"].items() for r in by.values()}
    J.write_json(ctx.out / f"{model_name}_config.json", config)
    figures = ctx.out / "figures"
    figures.mkdir()
    plot_curves(curves_df, step, figures / "learning_curves.png")
    if trials:
        plot_history(trials_df, step, figures / "optimization_history.png")
    outputs += [ctx.out / "best_params.json", ctx.out / f"{model_name}_config.json"]
    if (ctx.out / "worker_stdout.log").exists():
        outputs.append(ctx.out / "worker_stdout.log")
    outputs += sorted(p for p in figures.iterdir())
    outputs += sorted(p for p in job_dir.iterdir())
    outputs += sorted(p for p in out_dir.rglob("*") if p.is_file())
    summary = [f"{step} ({ctx.mode}, seeds {seeds}): nRMSE in kW on the tuning window, "
               f"{ctx.windows[0].label if ctx.windows else ''}"]
    values = {}
    for key, rec in best.items():
        f = rec["final"]
        values[key] = f["value_kw"]
        summary.append(f"  {key:<22} {f['value_kw']:6.3f} %  (D+1 {f['value_kw_D1']:.3f}, "
                       f"D+2 {f['value_kw_D2']:.3f}, D+3 {f['value_kw_D3']:.3f})")
    return Result(sorted(set(outputs)), values={"best": values,
                                                "test_access": report["test_access"]},
                  summary=summary, notes={"seed_spread_pp": spread, "warnings": len(warnings)})

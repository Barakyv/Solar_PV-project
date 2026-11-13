"""The neural training worker (docs/stage9_plan.md §18.3, X25). Runs in `solar-gpu`:

    <solar-gpu python> -m solarpv.models.neural_worker <job folder> <output folder>

Reads `job.json` and the arrays it names (every SHA-256 checked), trains the step's model per
target and seed, and writes into the output folder, which must not exist yet:
  forecast_<target>_seed<S>.npz   per scope `<scope>_q` [origins, 73, quantiles] in model units;
                                  for the TFT also the validation attention and the variable
                                  weights (M22)
  fit_<target>_seed<S>.json       the best parameters, the final fit, the reproduction check
  trials_<target>_seed<S>.csv     every Optuna trial (none for the TFT baseline and PatchTST)
  curves_<target>_seed<S>.csv     every epoch (validation check for PatchTST) of every fit
  models/<model>_<target>_seed<S>.pt, optuna.db
  worker.json                     the environment, the determinism settings, every output's hash
Progress goes to stdout, line by line, for the runner's log. Any error ends the process with a
non-zero exit code and no `worker.json`.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from solarpv.models import checkpoint as CK
from solarpv.models import neural_env
from solarpv.models.neural_job import (
    FIXED,
    JOB_FILE,
    JOB_VERSION,
    POINT_QUANTILE,
    QUANTILES,
    SEARCH_SPACES,
    STEP_FAMILY,
    STEP_MODEL,
    WORKER_FILE,
    JobError,
    arrays_digest,
    curves_name,
    fit_name,
    forecast_name,
    objective,
    read_arrays,
    read_json,
    sha256_file,
    trials_name,
    write_arrays,
    write_json,
)


def say(text: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {text}", flush=True)


def write_rows(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _attrs(res) -> dict:
    return {**res.objective_at_best, "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
            "best_val_loss": res.best_val_loss, "train_loss_at_best": res.train_loss_at_best,
            "stopped_early": res.stopped_early, "nan_loss": res.nan_loss,
            "duration_s": res.seconds, "gpu_peak_mb": round(res.gpu_peak_mb, 1),
            "n_parameters": res.n_parameters}


def unit_files(out: Path, target: str, seed: int, trials: bool, model_file: str) -> dict:
    """The files of a finished target x seed (§23): relative name -> the file in `out`."""
    names = [forecast_name(target, seed), fit_name(target, seed), curves_name(target, seed),
             f"models/{model_file}"]
    if trials:
        names.append(trials_name(target, seed))
    return {n: out / n for n in names}


def restore_unit(store, out: Path, unit: str, storage=None, study_name: str | None = None,
                 seed: int | None = None):
    """The record of a target x seed that an earlier try finished, with its files copied into
    `out` (each checked against its hash); the study is replayed into `storage` so that
    `optuna.db` holds it as in an uninterrupted worker. None when there is no such unit."""
    record = store.restore_unit(unit, out)
    if record is None:
        return None
    if study_name is not None:
        state = store.load_pickle(f"study_{unit}.pkl")
        if state is None:
            raise JobError(f"{unit}: the finished unit has no study checkpoint")
        from solarpv.models import checkpoint

        checkpoint.replayed_study(storage, study_name, seed, state)
    store.note("unit", unit=unit)
    say(f"{unit}: finished by an earlier try, restored from the checkpoint")
    return record


def run_torch(job: dict, target: str, arrays: dict, out: Path, device, storage,
              store=None) -> dict:
    """BiLSTM-Opt, TFT-96, TFT-168 and the TFT baseline (§18.4). `store` (§23) holds the
    checkpoints: after every epoch, every finished trial and every finished target x seed."""
    import torch

    from solarpv.models import checkpoint as CK
    from solarpv.models import lstm_sota, neural_torch, tft_model

    store = store if store is not None else CK.Store(None)

    step, family = job["step"], STEP_FAMILY[job["step"]]
    module = lstm_sota if family == "bilstm" else tft_model
    meta = job["meta"][target]
    quantiles = QUANTILES[family]
    point = quantiles.index(POINT_QUANTILE)
    scopes, tune = job["scopes"], job["tune_scope"]
    data = {s: neural_torch.Data.from_arrays(arrays, s, meta["n_unknown"], device) for s in scopes}

    def obj(pred):
        return objective(target, pred, arrays, tune, meta["target_scale"], job["mean_k"])

    budget = job["budget"]
    fixed = FIXED[family]
    records = {}
    for seed in job["seeds"]:
        unit = f"{target}_seed{seed}"
        study_name = f"{STEP_MODEL[step]}_{target}_seed{seed}"
        record = restore_unit(store, out, unit, storage,
                              None if step == "tftbase" else study_name, seed)
        if record is not None:  # §23: this target x seed was finished by an earlier try
            records[str(seed)] = record
            continue
        curves: list[dict] = []
        trial_rows: list[dict] = []
        fit_key, study_key = f"fit_{unit}.pt", f"study_{unit}.pkl"
        saved = store.load_pickle(study_key)  # the finished trials, the sampler, their curves
        if saved is not None:
            curves.extend(saved["extra"]["curves"])
            store.note("study", unit=unit, trials=len(saved["trials"]))

        def train_once(params, epochs, patience, label, seed=seed, curves=curves, unit=unit,
                       fit_key=fit_key):
            neural_torch.seed_everything(seed)
            model = (module.build_model(params, meta, len(quantiles)) if family == "bilstm"
                     else module.build_model(params, meta)).to(device)
            s = module.train_settings(params, fixed)
            key = {"label": label, "params": params, "seed": seed, "epochs": epochs,
                   "patience": patience}
            resume, ck = None, store.load_torch(fit_key)
            if ck is not None and ck["key"] == key:  # the fit that was running (§23)
                resume = ck["state"]
                curves.extend({"target": target, "seed": seed, "fit": label, **row}
                              for row in resume["curve"])
                store.note("fit", unit=unit, fit=label, epoch=resume["epoch"],
                           finished=resume["done"] or resume["epoch"] >= epochs)

            def on_epoch(row):
                curves.append({"target": target, "seed": seed, "fit": label, **row})

            def on_checkpoint(state):
                store.save_torch(fit_key, {"key": key, "state": state})
                CK.fault(f"epoch:{unit}:{label}:{state['epoch']}")

            res = neural_torch.fit(model, module.forward, data["train"], data[tune],
                                   epochs=epochs, patience=patience, lr=s["lr"],
                                   batch_size=s["batch_size"], clip=s["clip"],
                                   quantiles=quantiles, seed=seed, objective_fn=obj,
                                   point_index=point, on_epoch=on_epoch, resume=resume,
                                   on_checkpoint=on_checkpoint if store.enabled else None)
            say(f"{target} seed {seed} {label}: {res.objective_at_best['value_kw']:.4f} % kW, "
                f"best epoch {res.best_epoch} of {res.epochs_run}, {res.seconds} s")
            return model, res

        study_record, reproduce = None, None
        if step == "tftbase":
            params = dict(FIXED["tftbase"])
            model, res = train_once(params, budget["epochs"], budget["patience"], "final")
        else:
            space = SEARCH_SPACES[family]
            if step == "bilstm":
                epochs, patience = budget["epochs_trial"], None  # v6: fixed epochs per trial
            else:
                epochs, patience = budget["epochs"], budget["patience"]

            def trial_fn(params, number, epochs=epochs, patience=patience):
                _, r = train_once(params, epochs, patience, f"trial{number}")
                attrs = _attrs(r)
                if family == "tft":
                    attrs["hidden_continuous_size_effective"] = tft_model.effective_hcs(params)
                return r.objective_at_best["value_kw"], attrs

            callbacks = ([CK.study_callback(store, study_key, f"trial:{unit}",
                                            extra=lambda curves=curves: {"curves": list(curves)})]
                         if store.enabled else None)
            study = neural_torch.run_study(space, budget["trials"], seed, storage, study_name,
                                           trial_fn, state=saved, callbacks=callbacks)
            for t in study.trials:
                trial_rows.append({"target": target, "seed": seed, "trial": t.number,
                                   "state": t.state.name, "value_kw": t.value,
                                   **t.user_attrs,
                                   **{f"param_{k}": v for k, v in t.params.items()}})
            params = dict(study.best_params)
            study_record = {"best_trial": study.best_trial.number,
                            "best_value_kw": study.best_value, "n_trials": len(study.trials)}
            if step == "bilstm":
                model, res = train_once(params, budget["epochs_final"], budget["patience"],
                                        "final")
            else:  # the best TFT trial, trained again with its seed (X32)
                model, res = train_once(params, epochs, patience, "final")
                reproduce = abs(res.objective_at_best["value_kw"] - study.best_value)
        forecasts = {f"{s}_q": neural_torch.predict(model, module.forward, data[s])
                     for s in scopes}
        extra = {}
        if family == "tft":
            interp = tft_model.interpretation(model, data[tune])
            forecasts[f"{tune}_attention"] = interp["attention"]
            extra = {"encoder_importance": dict(zip(meta["enc_channels"],
                                                    map(float, interp["encoder_importance"]),
                                                    strict=True)),
                     "decoder_importance": dict(zip(meta["dec_channels"],
                                                    map(float, interp["decoder_importance"]),
                                                    strict=True)),
                     "hidden_continuous_size_effective": tft_model.effective_hcs(params)}
        write_arrays(out / forecast_name(target, seed), forecasts)
        model_path = out / "models" / f"{STEP_MODEL[step]}_{target}_seed{seed}.pt"
        torch.save({"state_dict": res.state, "params": params, "meta": meta}, model_path)
        write_rows(out / curves_name(target, seed), curves)
        if trial_rows:
            write_rows(out / trials_name(target, seed), trial_rows)
        record = {"target": target, "seed": seed, "params": params, "fixed": fixed,
                  "quantiles": list(quantiles), "study": study_record,
                  "final": _attrs(res), "reproduce_diff_pp": reproduce, **extra}
        write_json(out / fit_name(target, seed), record)
        records[str(seed)] = record
        store.save_unit(unit, unit_files(out, target, seed, bool(trial_rows),
                                         model_path.name), record)
        store.drop(fit_key)
        CK.fault(f"unit:{unit}")
    return records


def run_patchtst(job: dict, target: str, arrays: dict, out: Path, store=None) -> dict:
    import torch

    from solarpv.models import checkpoint as CK
    from solarpv.models import patchtst_model

    store = store if store is not None else CK.Store(None)
    meta = job["meta"][target]
    records = {}
    for seed in job["seeds"]:
        unit = f"{target}_seed{seed}"
        record = restore_unit(store, out, unit)
        if record is not None:  # §23: a fit takes minutes, the unit is the target x seed
            records[str(seed)] = record
            continue
        t0 = time.perf_counter()
        r = patchtst_model.fit_predict(arrays, job["scopes"], job["budget"], seed)
        values = objective(target, r["forecasts"][f"{job['tune_scope']}_q"][..., 0], arrays,
                           job["tune_scope"], meta["target_scale"], job["mean_k"])
        seconds = round(time.perf_counter() - t0, 2)
        say(f"{target} seed {seed} final: {values['value_kw']:.4f} % kW, best step "
            f"{r['best_step']} of {r['steps_run']}, {seconds} s")
        write_arrays(out / forecast_name(target, seed), r["forecasts"])
        torch.save({"state_dict": r["state"], "hparams": r["hparams"], "meta": meta},
                   out / "models" / f"patchtst_{target}_seed{seed}.pt")
        curves = ([{"target": target, "seed": seed, "fit": "final", "step": s, "train_loss": v}
                   for s, v in r["train_trajectory"]]
                  + [{"target": target, "seed": seed, "fit": "final", "step": s, "val_loss": v}
                     for s, v in r["val_checks"]])
        write_rows(out / curves_name(target, seed), curves)
        record = {"target": target, "seed": seed, "params": dict(FIXED["patchtst"]),
                  "hparams": r["hparams"], "quantiles": list(QUANTILES["patchtst"]),
                  "study": None,
                  "final": {**values, "best_step": r["best_step"], "steps_run": r["steps_run"],
                            "best_val_loss": r["best_val_loss"], "duration_s": seconds,
                            "n_parameters": r["n_parameters"],
                            "val_checks": len(r["val_checks"])},
                  "exogenous_futr": r["exogenous_futr"], "training_rows": r["training_rows"],
                  "val_size": r["val_size"], "reproduce_diff_pp": None}
        write_json(out / fit_name(target, seed), record)
        records[str(seed)] = record
        store.save_unit(unit, unit_files(out, target, seed, False,
                                         f"patchtst_{target}_seed{seed}.pt"), record)
        CK.fault(f"unit:{unit}")
    return records


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    job_dir, out = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    job = read_json(job_dir / JOB_FILE)
    if job.get("version") != JOB_VERSION:
        raise JobError(f"job version {job.get('version')}, the worker reads {JOB_VERSION}")
    if job["step"] not in STEP_FAMILY:
        raise JobError(f"unknown step {job['step']!r}")
    out.mkdir(parents=False, exist_ok=False)
    (out / "models").mkdir()
    started = time.time()
    env = neural_env.report(probe=False)
    if env["errors"]:
        raise JobError(f"the GPU environment fails its checks: {env['errors']}")
    import optuna
    import torch

    from solarpv.models import neural_torch

    determinism = neural_torch.setup_determinism()
    device = torch.device("cuda")
    say(f"{job['step']} ({job['mode']}): targets {job['targets']}, seeds {job['seeds']}, "
        f"budget {job['budget']}, on {env['gpu']['name']}")
    storage = optuna.storages.RDBStorage(f"sqlite:///{(out / 'optuna.db').as_posix()}")
    resume = job.get("resume")  # §23: the checkpoint folder and the identity it must carry
    store = CK.open_store(Path(resume["dir"]) if resume else None,
                          resume["identity"] if resume else None,
                          trust=bool(resume["trust"]) if resume else False, say=say)
    records = {}
    try:
        for target in job["targets"]:
            f = job["files"][target]
            arrays = read_arrays(job_dir / f["path"], f["sha256"])
            store.bind(target, arrays_digest(arrays))
            if STEP_FAMILY[job["step"]] == "patchtst":
                records[target] = run_patchtst(job, target, arrays, out, store)
            else:
                records[target] = run_torch(job, target, arrays, out, device, storage, store)
    finally:
        storage.remove_session()
        storage.engine.dispose()
    outputs = {p.relative_to(out).as_posix(): sha256_file(p)
               for p in sorted(out.rglob("*")) if p.is_file()}
    import solarpv

    write_json(out / WORKER_FILE, {
        "step": job["step"], "job_sha256": sha256_file(job_dir / JOB_FILE),
        # X55: the code this worker ran, checked by the runner against its own
        "solarpv_file": str(Path(solarpv.__file__).resolve()),
        "environment": env, "determinism": determinism,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1), "records": records,
        "resumed": store.events, "outputs": outputs, "numpy": np.__version__})
    say(f"done in {round(time.time() - started, 1)} s, {len(outputs)} outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

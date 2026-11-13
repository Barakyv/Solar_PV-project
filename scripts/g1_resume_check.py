"""Measures how far a fit continued from an epoch checkpoint is from the uninterrupted fit, in the
GPU environment `solar-gpu` (docs/stage9_plan.md §23.6, X95, figure S09-43).

For each configuration the script trains five epochs on the synthetic windows of the GPU tests,
once without interruption and once continued from the checkpoint after epoch 2, with the first
design (every random state restored, X95) and with the final one (the same training passes run
again, the CUDA generator not set). It writes `outputs/g1_audit/resume_check.json`: the largest
absolute difference of the weights and of the losses, and the seconds the replay costs against the
seconds of the epochs it replaces.

    <solar-gpu python> scripts/g1_resume_check.py [--out PATH]
"""

import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "gpu")]

import torch  # noqa: E402
from test_neural_gpu import (  # noqa: E402
    BILSTM,
    DEVICE,
    META,
    TFT,
    data_of,
    objective_fn,
    scope_arrays,
)

from solarpv.config.cli import parser  # noqa: E402
from solarpv.models import lstm_sota, neural_job, neural_torch, tft_model  # noqa: E402

CONFIGS = [
    ("BiLSTM, 1 layer, dropout 0.1", lstm_sota, {**BILSTM, "n_layers": 1}),
    ("BiLSTM, 2 layers, no dropout", lstm_sota, {**BILSTM, "dropout": 0.0}),
    ("BiLSTM, 2 layers, dropout 0.1", lstm_sota, BILSTM),
    ("BiLSTM, 3 layers, dropout 0.3", lstm_sota, {**BILSTM, "n_layers": 3, "dropout": 0.3}),
    ("TFT, 1 LSTM layer, dropout 0.1", tft_model, TFT),
    ("TFT, 2 LSTM layers, dropout 0.1", tft_model, {**TFT, "lstm_layers": 2}),
    ("TFT, 3 LSTM layers, dropout 0.3", tft_model, {**TFT, "lstm_layers": 3, "dropout": 0.3}),
]
EPOCHS, FROM_EPOCH = 5, 2
BATCH = 16  # several batches per epoch, also for the TFT (its own batch size is 64)


def roundtrip(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def build(module, params):
    """A freshly seeded model, the quantiles and the settings of its fit."""
    neural_torch.seed_everything(7)
    if module is lstm_sota:
        model = module.build_model(params, META, 3).to(DEVICE)
        return (model, neural_job.QUANTILES["bilstm"],
                module.train_settings(params, neural_job.FIXED["bilstm"]))
    model = module.build_model(params, META).to(DEVICE)
    return (model, neural_job.QUANTILES["tft"],
            module.train_settings(params, neural_job.FIXED["tft"]))


def train(module, params, arrays, resume=None):
    model, q, s = build(module, params)
    states = []
    t0 = time.perf_counter()
    res = neural_torch.fit(
        model, module.forward, data_of(arrays, "train"), data_of(arrays, "val"), epochs=EPOCHS,
        patience=None, lr=s["lr"], batch_size=min(s["batch_size"], BATCH), clip=s["clip"],
        quantiles=q, seed=7,
        objective_fn=objective_fn(arrays), point_index=q.index(0.5), resume=resume,
        on_checkpoint=lambda st: states.append(roundtrip(st)))
    return res, states, time.perf_counter() - t0


def replay_seconds(module, params, arrays) -> float:
    """The seconds the replay of FROM_EPOCH epochs takes (the cost of a resume)."""
    model, _, s = build(module, params)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    neural_torch.advance_cudnn_dropout(model, module.forward, data_of(arrays, "train"), FROM_EPOCH,
                                       min(s["batch_size"], BATCH), 7)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def difference(a, b) -> dict:
    weights = max(float((a.state[k].cpu() - b.state[k].cpu()).abs().max()) for k in a.state)
    losses = max(abs(x["train_loss"] - y["train_loss"]) for x, y in zip(a.curve, b.curve,
                                                                         strict=True))
    return {"max_abs_weight_diff": weights, "max_abs_train_loss_diff": losses,
            "equal": weights == 0.0 and losses == 0.0}


def main(argv: list[str] | None = None) -> int:
    p = parser("python scripts/g1_resume_check.py", __doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "g1_audit" / "resume_check.json",
                   help="where to write the measurements")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    neural_torch.setup_determinism()
    # half of the training origins are short, as the real windows are at the DST origins and at
    # the end of a split (R-G1-27)
    arrays = {**scope_arrays("train", 48, 0, short=24), **scope_arrays("val", 12, 1, short=True)}
    rows = []
    train(lstm_sota, BILSTM, arrays)  # warm-up: the first fit of a process pays for CUDA start-up
    for name, module, params in CONFIGS:
        full, states, _ = train(module, params, arrays)
        checkpoint = states[FROM_EPOCH - 1]
        interlayer = neural_torch.has_interlayer_dropout(build(module, params)[0])
        done_seconds = sum(r["seconds"] for r in full.curve[:FROM_EPOCH])
        row = {"name": name, "family": "bilstm" if module is lstm_sota else "tft",
               "interlayer_dropout": bool(interlayer), "seconds_of_the_epochs_done": done_seconds}
        real = neural_torch.has_interlayer_dropout
        neural_torch.has_interlayer_dropout = lambda m: False  # the first design: restore only
        try:
            first, _, _ = train(module, params, arrays, resume=checkpoint)
        finally:
            neural_torch.has_interlayer_dropout = real
        final, _, _ = train(module, params, arrays, resume=checkpoint)
        row["first_design"] = difference(full, first)
        row["final"] = difference(full, final)
        row["replay_seconds"] = replay_seconds(module, params, arrays) if interlayer else 0.0
        rows.append(row)
        print(f"{name:38s} first design {row['first_design']['max_abs_weight_diff']:.3e}  "
              f"final {row['final']['max_abs_weight_diff']:.3e}  replay "
              f"{row['replay_seconds']:.2f} s of {done_seconds:.2f} s", flush=True)
    result = {"epochs": EPOCHS, "resumed_after_epoch": FROM_EPOCH, "configs": rows,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "note": "synthetic windows of tests/gpu; the weights are float32"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0 if all(r["final"]["equal"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

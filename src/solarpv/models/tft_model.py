"""TFT-96, TFT-168 and the TFT baseline: pytorch-forecasting 1.7.0's TemporalFusionTransformer on
the W3 windows (PROJECT_MASTER.md §6.11; docs/stage9_plan.md §18.2-§18.4). Runs in the training
worker of `solar-gpu`.

The network is built from explicit variable lists and fed the W3 windows directly, one sample per
origin, so no `TimeSeriesDataSet` is involved (X29, M16, M17): the library fits no scaler, picks
no normaliser from the data, and builds no window of its own.

- Unknown reals: the measured encoder series and their fill flags (encoder only).
- Known reals: the calendar, clear-sky, geometry and GFS inputs of D-26 and D-27, on the encoder
  and decoder steps. The decoder rows of the unknown reals are zeros, which the network never
  reads (the decoder's variable selection takes the known reals only; T-78).
- No static input and no categorical input. Encoder length L for every sample; decoder lengths
  from the dataset (71, 72 or 73, fewer at the end of a split), handled by the library's packed
  LSTMs and attention masks.
- The output transformer is an identity normaliser with target_scale (0, 1): the network predicts
  the target in model units (k; AC / 3,449), exactly as the W3 datasets give it.
- Loss: the masked pinball loss of `neural_torch` with the quantiles of the library's default
  `QuantileLoss` (asserted equal).
"""

import torch

from solarpv.models.neural_job import MAX_STEPS, QUANTILES

_MAPS: dict = {}


def linear_map(n_in: int, n_out: int, device, dtype) -> torch.Tensor:
    """[n_in, n_out]: linear interpolation with align_corners=True as a matrix (X34)."""
    key = (n_in, n_out, str(device), dtype)
    if key not in _MAPS:
        if n_out == 1 or n_in == 1:
            pos = torch.zeros(n_out, dtype=torch.float64)
        else:
            pos = torch.arange(n_out, dtype=torch.float64) * (n_in - 1) / (n_out - 1)
        i0 = pos.floor().long().clamp(max=n_in - 1)
        i1 = (i0 + 1).clamp(max=n_in - 1)
        lam = pos - i0.to(torch.float64)
        w = torch.zeros(n_in, n_out, dtype=torch.float64)
        cols = torch.arange(n_out)
        w.index_put_((i0, cols), 1.0 - lam, accumulate=True)
        w.index_put_((i1, cols), lam, accumulate=True)
        _MAPS[key] = w.to(device=device, dtype=dtype)
    return _MAPS[key]


def _interpolate(self, x):
    """pytorch-forecasting's `TimeDistributedInterpolation.interpolate` with the same linear map
    as a matrix product: `F.interpolate(mode="linear")` has no deterministic CUDA backward (X34)."""
    upsampled = x @ linear_map(x.shape[-1], self.output_size, x.device, x.dtype)
    if self.trainable:
        upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
    return upsampled


def deterministic_interpolation() -> None:
    """Replace the interpolation of the TFT's resampling layers (X34), once per process."""
    from pytorch_forecasting.models.temporal_fusion_transformer import sub_modules

    sub_modules.TimeDistributedInterpolation.interpolate = _interpolate


def build_model(params: dict, meta: dict):
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.data.encoders import TorchNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    deterministic_interpolation()
    enc, dec, n_unknown = list(meta["enc_channels"]), list(meta["dec_channels"]), meta["n_unknown"]
    if enc[n_unknown:] != dec:
        raise ValueError("the known reals of the encoder and the decoder differ")
    loss = QuantileLoss()
    if tuple(loss.quantiles) != QUANTILES["tft"]:
        raise ValueError(f"QuantileLoss quantiles {loss.quantiles} != {QUANTILES['tft']}")
    hidden = int(params["hidden_size"])
    return TemporalFusionTransformer(
        hidden_size=hidden, lstm_layers=int(params["lstm_layers"]),
        dropout=float(params["dropout"]), output_size=len(loss.quantiles), loss=loss,
        attention_head_size=int(params["attention_head_size"]),
        max_encoder_length=int(meta["encoder_hours"]),
        static_categoricals=[], static_reals=[], time_varying_categoricals_encoder=[],
        time_varying_categoricals_decoder=[], categorical_groups={},
        time_varying_reals_encoder=enc, time_varying_reals_decoder=dec, x_reals=enc,
        x_categoricals=[], embedding_sizes={}, embedding_paddings=[], embedding_labels={},
        hidden_continuous_size=effective_hcs(params), hidden_continuous_sizes={},
        learning_rate=float(params["learning_rate"]), log_interval=-1,
        output_transformer=TorchNormalizer(method="identity", center=False))


def effective_hcs(params: dict) -> int:
    """§18.4: sampled in 8..64, used as min(value, hidden_size)."""
    return int(min(int(params["hidden_continuous_size"]), int(params["hidden_size"])))


def network_input(data, idx) -> dict:
    enc = data.enc_x[idx]
    b, length = enc.shape[0], enc.shape[1]
    lengths = data.lengths[idx]
    t = int(lengths.max())
    known = data.dec_x[idx, :t]
    unknown = torch.zeros(b, t, data.n_unknown, dtype=enc.dtype, device=enc.device)
    scale = torch.zeros(b, 2, dtype=enc.dtype, device=enc.device)
    scale[:, 1] = 1.0
    return {"encoder_cont": enc, "decoder_cont": torch.cat([unknown, known], dim=2),
            "encoder_cat": torch.zeros(b, length, 0, dtype=torch.long, device=enc.device),
            "decoder_cat": torch.zeros(b, t, 0, dtype=torch.long, device=enc.device),
            "encoder_lengths": torch.full((b,), length, dtype=torch.long, device=enc.device),
            "decoder_lengths": lengths, "target_scale": scale}


def _pad(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < MAX_STEPS:
        pad = torch.zeros(pred.shape[0], MAX_STEPS - pred.shape[1], pred.shape[2],
                          dtype=pred.dtype, device=pred.device)
        pred = torch.cat([pred, pad], dim=1)
    return pred


def forward(model, data, idx) -> torch.Tensor:
    return _pad(model(network_input(data, idx))["prediction"])


@torch.no_grad()
def interpretation(model, data, batch: int = 128) -> dict:
    """For the validation origins (M22): the encoder attention of every decoder step over the L
    encoder positions (mean over the heads), and the mean variable-selection weights of the
    encoder and decoder variables over the existing steps."""
    import numpy as np

    model.eval()
    att, enc_w, dec_w, dec_n = [], 0.0, 0.0, 0
    length = int(data.enc_x.shape[1])
    for start in range(0, data.n, batch):
        idx = torch.arange(start, min(start + batch, data.n), device=data.y.device)
        out = model(network_input(data, idx))
        a = out["encoder_attention"]  # [B, T, heads, L]
        if a.shape[-1] != length or a.ndim != 4:
            raise ValueError(f"encoder attention of shape {tuple(a.shape)}, not [B, T, heads, "
                             f"{length}]")
        a = a.mean(dim=2)
        full = torch.zeros(a.shape[0], MAX_STEPS, length, dtype=a.dtype, device=a.device)
        full[:, :a.shape[1]] = a
        att.append(full.float().cpu().numpy().astype(np.float16))
        enc_w = enc_w + out["encoder_variables"].squeeze(-2).sum(dim=(0, 1)).float().cpu()
        lengths = data.lengths[idx]
        steps = torch.arange(out["decoder_variables"].shape[1], device=a.device)[None, :]
        exists = (steps < lengths[:, None]).to(out["decoder_variables"].dtype)
        dec_w = dec_w + (out["decoder_variables"].squeeze(-2) * exists[..., None]
                         ).sum(dim=(0, 1)).float().cpu()
        dec_n += int(lengths.sum())
    return {"attention": np.concatenate(att, axis=0),
            "encoder_importance": (enc_w / (data.n * length)).numpy(),
            "decoder_importance": (dec_w / max(dec_n, 1)).numpy()}


def train_settings(params: dict, fixed: dict) -> dict:
    return {"lr": float(params["learning_rate"]), "batch_size": int(fixed["batch_size"]),
            "clip": float(fixed["grad_clip_norm"])}

"""BiLSTM-Opt: an encoder-decoder with quantile loss (PROJECT_MASTER.md §6.11; docs/stage9_plan.md
§18.4). Runs in the training worker of `solar-gpu` (it imports torch).

Inputs (D-28): the datasets of `solarpv.models.sequence_data` (W3): a 72-hour encoder, a decoder
over the three local days from T0 (D-25), and the part of the stage-7 inputs known at T0.
Horizons come from the dataset (the local day of each valid hour), never from step numbers; the
original's `valid_ts = window_end - (DECODER_LEN - 1 - target_step) h` is not used (M14, X19).

Architecture (§18.4, D-28): a bidirectional LSTM over the encoder channels; its last forward and
backward states (per layer, concatenated and projected) start a forward LSTM over the decoder's
known inputs; a linear head gives the quantiles 0.1, 0.5 and 0.9 per step. The same number of
layers and the same dropout in the encoder and the decoder. Search space of v6 §5.2.
"""

import torch
from torch import nn


class BiLSTMSeq2Seq(nn.Module):
    def __init__(self, n_enc: int, n_dec: int, hidden: int, n_layers: int, dropout: float,
                 n_quantiles: int):
        super().__init__()
        if n_dec < 1:
            raise ValueError("the decoder needs its known inputs (D-28)")
        self.hidden, self.n_layers = int(hidden), int(n_layers)
        between = float(dropout) if self.n_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_enc, self.hidden, self.n_layers, batch_first=True,
                               bidirectional=True, dropout=between)
        self.bridge_h = nn.Linear(2 * self.hidden, self.hidden)
        self.bridge_c = nn.Linear(2 * self.hidden, self.hidden)
        self.decoder = nn.LSTM(n_dec, self.hidden, self.n_layers, batch_first=True,
                               dropout=between)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.hidden, int(n_quantiles))

    def _bridge(self, state: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        # [layers * 2, B, H] -> [layers, B, 2H] (forward and backward of each layer)
        b = state.shape[1]
        state = state.view(self.n_layers, 2, b, self.hidden).permute(0, 2, 1, 3)
        return layer(state.reshape(self.n_layers, b, 2 * self.hidden))

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(enc_x)
        h0 = torch.tanh(self._bridge(h, self.bridge_h)).contiguous()
        c0 = self._bridge(c, self.bridge_c).contiguous()
        out, _ = self.decoder(dec_x, (h0, c0))
        return self.head(self.dropout(out))  # [B, steps, quantiles]


def build_model(params: dict, meta: dict, n_quantiles: int) -> BiLSTMSeq2Seq:
    return BiLSTMSeq2Seq(len(meta["enc_channels"]), len(meta["dec_channels"]),
                         params["hidden"], params["n_layers"], params["dropout"], n_quantiles)


def forward(model, data, idx) -> torch.Tensor:
    return model(data.enc_x[idx], data.dec_x[idx])


def train_settings(params: dict, fixed: dict) -> dict:
    """lr, batch size and clipping of one fit."""
    return {"lr": float(params["lr"]), "batch_size": int(params["batch_size"]),
            "clip": float(fixed["grad_clip_norm"])}

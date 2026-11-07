"""The datasets of the sequence models (docs/stage9_plan.md §17, W3; decisions D-25-D-28).

    python -m solarpv.models.sequence_data [--mode dev|smoke]

numpy/pandas only: W4 trains BiLSTM-Opt, TFT-96, TFT-168, the TFT baseline and PatchTST on the
arrays built here. The command builds the train and validation datasets of every model x target
and checks them against theory (§17.8); it writes reports only, never arrays.

Rules (tests T-66-T-71 and T-02):
  * Origins (D-16): one per local day; T0 = 00:00 Asia/Jerusalem of that day, in UTC.
  * Decoder (D-25): the hours of the three local days from T0 (72, or 71 / 73 when one of them
    has 23 / 25 hours), padded to MAX_STEPS = 73. The horizon of a step is the local day of its
    valid hour counted from T0's day, never its step number (M14, T-02), and
    issuance_utc(valid, h) == T0 on every step (X20).
  * Encoder: the L hours [T0 - L, T0); every encoder hour ends by T0.
  * Scope (M15): for a split, or the part of it in the run's window, the origins run from the
    local day of its first hour minus two days to the local day of its last hour. Only the steps
    whose valid hour is in the scope are kept, so every hour of the scope gets exactly one
    forecast per horizon. Decoders stop at the end of their split (training at VAL_START,
    validation at TEST_START in every mode); smoke and dev never read a row at or after
    TEST_START (X21).
  * Inputs (D-26-D-28): per target, the part of model_inputs.json known at T0, the union over the
    three horizons: calendar, clear sky and geometry from the stage-6 table, GFS (and the
    measured `irradiance_clear_index_prevday`) from the GFS table of the step's horizon, whose
    t0_utc is the origin's T0. The same for the BiLSTM and the three TFT models. Encoder series:
    the seven measured series of stage 6 with the D-22 masks of `masked_sources` (X22), plus
    target_k for the indirect route; gaps filled with the stage-6 rule and flagged. PatchTST:
    the target series only (L-19).
  * Scalers (A-14): z-scores from train hours (not warm-up) only; targets not transformed (k as
    is, AC / 3,449).
  * Loss (X23): `xgb_sota.fit_mask` on the kept steps.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from solarpv.config.constants import (
    AC_CAPACITY_KW,
    HORIZONS,
    NWP_MIN_AVAILABILITY_HOURS,
    TEST_START,
    TIMEZONE_LOCAL,
    VAL_START,
)
from solarpv.config.paths import output_dir
from solarpv.config.provenance import git_commit
from solarpv.config.runlog import RunLog
from solarpv.features.build import (
    HOURLY_FILE,
    INVERTER_FLAGS,
    MINUTE_FILE,
    POLICY_FILE,
    impute_same_hour,
    issuance_utc,
    masked_sources,
)
from solarpv.features.contract import FEATURE_GROUPS, GFS_PREFIX, LAG_SOURCES
from solarpv.features.leakage import assert_no_leakage
from solarpv.models.xgb_sota import fit_mask, load_model_inputs, local_day, train_scope
from solarpv.pipeline.config import Window, evaluation_windows
from solarpv.pipeline.data import INDEX, feature_file, gfs_file
from solarpv.selection.common import PREVDAY

# encoder hours per model (D-28: 72 h for the BiLSTM, E1; §6.11 for the TFT models). PatchTST's
# input size is a W4 choice; the checks of this module use CHECK_PATCHTST_HOURS
MODELS = {"bilstm": 72, "tft96": 96, "tft168": 168, "tftbase": 168, "patchtst": None}
CHECK_PATCHTST_HOURS = 168
TARGETS = ("indirect", "direct")
N_DAYS = 3
MAX_STEPS = 73  # three local days: 72 h, or 71 / 73 h around a DST day (D-25)
HOUR_NS = 3_600_000_000_000
NAT = np.iinfo(np.int64).min  # padding of the int64 time arrays (numpy's NaT)
TARGET_COLUMN = {"indirect": "target_k", "direct": "site_total_AC_kW"}
TARGET_SCALE = {"indirect": 1.0, "direct": AC_CAPACITY_KW}  # fixed, not fitted (A-14)
DETERMINISTIC = (FEATURE_GROUPS["clearsky"] + FEATURE_GROUPS["time_encodings"]
                 + FEATURE_GROUPS["other"])
PAST = frozenset(FEATURE_GROUPS["lag_power"] + FEATURE_GROUPS["rolling_stats"])
SERIES = LAG_SOURCES  # the seven measured series the stage-6 lags are built from
K_SERIES = "target_k"
BASE_COLUMNS = ("split", "warmup", "daytime", "op_limited_flag", "theo_kw", "target_k",
                "k_valid", "site_total_AC_kW")
FLAG_SUFFIX = "_filled"
REPORT_NAME = "09_sequence_data"


class SequenceDataError(RuntimeError):
    """A dataset breaks a rule of §17.5."""


def is_gfs(name: str) -> bool:
    return name.startswith(GFS_PREFIX) or name == PREVDAY


def _ns(index) -> np.ndarray:
    return pd.DatetimeIndex(index).as_unit("ns").asi8


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


# ── origins and the step -> horizon mapping (D-16, D-25, T-02) ───────────────────


def t0_of(days) -> pd.DatetimeIndex:
    """T0 of local days: 00:00 Asia/Jerusalem, in UTC. Local midnight is never ambiguous in
    Israel (the clock changes at 02:00), so the conversion raises if that ever changes."""
    days = pd.DatetimeIndex(days)
    if days.tz is not None:
        raise ValueError("local days must be naive dates")
    return days.normalize().tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")


def origin_days(hours: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The origins of a scope (M15): the local day of its first hour minus two days, to the local
    day of its last hour."""
    if len(hours) == 0:
        raise ValueError("an empty scope has no origins")
    days = local_day(pd.DatetimeIndex(hours))
    return pd.date_range(days.min() - pd.Timedelta(days=N_DAYS - 1), days.max(), freq="D")


def decoder_layout(days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray]:
    """For each origin day: T0 (int64 ns), the valid time of every decoder step [n, MAX_STEPS]
    (NAT beyond the third local day), its horizon (1..3 by the local day, 0 = padding) and the
    number of steps (71 / 72 / 73)."""
    days = pd.DatetimeIndex(days)
    t0 = _ns(t0_of(days))
    bounds = [_ns(t0_of(days + pd.Timedelta(days=k))) for k in range(1, N_DAYS + 1)]
    n_steps = (bounds[-1] - t0) // HOUR_NS
    if n_steps.max() > MAX_STEPS or n_steps.min() < 3 * 23:
        raise SequenceDataError(f"three local days of {n_steps.min()}-{n_steps.max()} hours")
    steps = np.arange(MAX_STEPS, dtype=np.int64)
    valid = t0[:, None] + steps[None, :] * HOUR_NS
    inside = steps[None, :] < n_steps[:, None]
    horizon = (1 + (valid >= bounds[0][:, None]).astype(np.int8)
               + (valid >= bounds[1][:, None]).astype(np.int8))
    horizon = np.where(inside, horizon, 0).astype(np.int8)
    valid = np.where(inside, valid, NAT)
    return t0, valid, horizon, n_steps


# ── inputs (D-26-D-28) ──────────────────────────────────────────────────────────


def known_inputs(target: str, inputs: dict[str, list[str]] | None = None
                 ) -> tuple[list[str], list[str]]:
    """The part of the stage-7 input of `target` known at T0, the union over the horizons in the
    recorded order: (deterministic, GFS). Lags and rolling statistics are dropped (the encoder
    sees the series they are built from); any other name is an error."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    inputs = inputs if inputs is not None else load_model_inputs()
    union: list[str] = []
    for h in HORIZONS:
        key = f"{target}_D{h}"
        if key not in inputs:
            raise KeyError(f"model_inputs.json has no entry {key}")
        union += [f for f in inputs[key] if f not in union]
    det = [f for f in union if f in DETERMINISTIC]
    gfs = [f for f in union if is_gfs(f)]
    other = [f for f in union if f not in det and f not in gfs and f not in PAST]
    if other:
        raise ValueError(f"{target}: inputs that are neither known at T0 nor past-day: {other}")
    if not det:
        raise ValueError(f"{target}: no calendar or clear-sky input")
    assert_no_leakage(det + gfs)
    return det, gfs


def encoder_series_names(target: str) -> list[str]:
    return list(SERIES) + ([K_SERIES] if target == "indirect" else [])


# ── loading (X21, X22) ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Paths:
    features: dict
    gfs: dict
    hourly: Path
    minute: Path
    policy: Path

    def files(self) -> list[Path]:
        return [*self.features.values(), *self.gfs.values(), self.hourly, self.minute,
                self.policy]


def default_paths() -> Paths:
    return Paths({h: feature_file(h) for h in HORIZONS}, {h: gfs_file(h) for h in HORIZONS},
                 HOURLY_FILE, MINUTE_FILE, POLICY_FILE)


@dataclass
class Sources:
    """Everything the windows read, on one complete hourly UTC index.

    frame: the stage-6 base columns and the deterministic features (from D+1), `t0@D<h>` of
    every feature table, and per GFS table `<column>@D<h>`, `gfs_t0@D<h>`, `gfs_init@D<h>`.
    series: the encoder series after the D-22 masks, before the fill; filled: after it."""

    frame: pd.DataFrame
    series: pd.DataFrame
    filled: pd.DataFrame
    lag1d: pd.DataFrame
    gfs_columns: list[str]
    allow_test: bool
    masked_hours: dict = field(default_factory=dict)

    @property
    def end(self) -> pd.Timestamp:
        """The first hour after the loaded data."""
        return self.frame.index[-1] + pd.Timedelta(hours=1)


def _read(path: Path, columns, limit) -> pd.DataFrame:
    filters = None if limit is None else [(INDEX, "<", limit)]
    frame = pd.read_parquet(path, columns=list(columns), filters=filters)
    if frame.index.name != INDEX or frame.index.tz is None or str(frame.index.tz) != "UTC":
        raise SequenceDataError(f"{Path(path).name}: the index must be UTC {INDEX!r}")
    if limit is not None and len(frame) and frame.index.max() >= limit:
        raise SequenceDataError(f"{Path(path).name}: a row at or after {limit} was read")
    return frame


def assemble(tables: dict, gfs: dict, hourly: pd.DataFrame, flags: pd.DataFrame, share: float,
             *, allow_test: bool = False) -> Sources:
    """Build the sources from tables already in memory (the loader reads them; the tests perturb
    them). `tables` and `gfs` map a horizon to its stage-6 / stage-6b table."""
    d1 = tables[1]
    index = d1.index
    steps = np.diff(_ns(index))
    if len(index) < 2 or (steps != HOUR_NS).any():
        raise SequenceDataError("the hourly index must be complete and increasing")
    named = [*((f"features D{h}", t) for h, t in tables.items()),
             *((f"GFS D{h}", g) for h, g in gfs.items()), ("hourly", hourly)]
    for name, frame in named:
        if len(frame) != len(index) or not np.array_equal(_ns(frame.index), _ns(index)):
            raise SequenceDataError(f"the {name} table is not on the index of the D+1 table")
    if not allow_test and index.max() >= _utc(TEST_START):
        raise SequenceDataError("rows of the test period in a development dataset")
    frame = d1[list(BASE_COLUMNS) + list(DETERMINISTIC)].copy()
    for h, table in tables.items():
        if h != 1:  # the deterministic inputs do not depend on the horizon (asserted)
            for col in DETERMINISTIC:
                a, b = d1[col].to_numpy(dtype=float), table[col].to_numpy(dtype=float)
                if not np.array_equal(a, b, equal_nan=True):
                    raise SequenceDataError(f"{col} differs between the D+1 and D+{h} tables")
        frame[f"t0@D{h}"] = _ns(table["t0_utc"])
    gfs_columns = [c for c in gfs[1].columns if is_gfs(c)]
    for h, g in gfs.items():
        if [c for c in g.columns if is_gfs(c)] != gfs_columns:
            raise SequenceDataError(f"the GFS columns of D+{h} differ from those of D+1")
        frame[f"gfs_t0@D{h}"] = _ns(g["t0_utc"])
        frame[f"gfs_init@D{h}"] = _ns(g["gfs_init"])
        for col in gfs_columns:
            frame[f"{col}@D{h}"] = g[col].to_numpy(dtype=float)
    sources, masked = masked_sources(hourly, flags, share)
    series = pd.DataFrame({c: sources[c] for c in SERIES}, index=index)
    # k of the encoder: as stored (0 where theo <= 50 kW, §6.5), NaN in op-limited hours (D-22)
    series[K_SERIES] = d1["target_k"].where(~d1["op_limited_flag"].astype(bool))
    filled, _ = impute_same_hour(series)
    lag1d = d1[[f"{c}_lag_1d" for c in SERIES if f"{c}_lag_1d" in d1.columns]].copy()
    return Sources(frame, series, filled, lag1d, gfs_columns, allow_test, masked)


def load_sources(*, allow_test: bool = False, paths: Paths | None = None) -> Sources:
    """Read the stage-5, stage-6 and stage-6b inputs. Smoke and dev: every file filtered inside
    the parquet reader to timestamp_utc < TEST_START, so no test row reaches pandas (X10, X21)."""
    paths = paths or default_paths()
    limit = None if allow_test else _utc(TEST_START)
    lag_cols = [f"{c}_lag_1d" for c in SERIES]
    tables = {}
    for h in HORIZONS:
        cols = ["t0_utc", *DETERMINISTIC] + (list(BASE_COLUMNS) + lag_cols if h == 1 else [])
        tables[h] = _read(paths.features[h], cols, limit)
    gfs = {}
    for h in HORIZONS:
        names = pq.read_schema(paths.gfs[h]).names
        gfs[h] = _read(paths.gfs[h], ["t0_utc", "gfs_init", *[c for c in names if is_gfs(c)]],
                       limit)
    hourly = _read(paths.hourly, [*SERIES, "op_limited_flag"], limit)
    flag_cols = sorted({f for v in INVERTER_FLAGS.values() for f in v})
    flags = _read(paths.minute, flag_cols, limit)
    policy = yaml.safe_load(Path(paths.policy).read_text(encoding="utf-8"))
    share = float(policy["hourly"]["op_limited_min_share"])
    return assemble(tables, gfs, hourly, flags, share, allow_test=allow_test)


# ── scaler (A-14) ───────────────────────────────────────────────────────────────


@dataclass
class Scaler:
    """z-score per channel, fitted on the train hours that are not warm-up (A-14)."""

    mean: pd.Series
    std: pd.Series
    rows: int

    @classmethod
    def fit(cls, sources: Sources) -> "Scaler":
        f = sources.frame
        rows = (f["split"] == "train").to_numpy() & ~f["warmup"].astype(bool).to_numpy()
        if not rows.any():
            raise SequenceDataError("no train hour to fit the scaler on")
        values = scaler_inputs(sources).loc[rows]
        mean = values.mean(skipna=True).fillna(0.0)
        std = values.std(ddof=0, skipna=True)
        std = std.where(std > 1e-12, 1.0).fillna(1.0)
        return cls(mean, std, int(rows.sum()))

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        """Scaled values; NaN (a gap the fill could not close, a missing input) becomes 0, the
        train mean."""
        z = (values - self.mean[values.columns]) / self.std[values.columns]
        return np.nan_to_num(z.to_numpy(dtype=np.float64), nan=0.0).astype(np.float32)

    def as_dict(self) -> dict:
        return {"rows": self.rows,
                "mean": {k: float(v) for k, v in self.mean.items()},
                "std": {k: float(v) for k, v in self.std.items()}}


def scaler_inputs(sources: Sources) -> pd.DataFrame:
    """The raw values the scaler is fitted on: the encoder series before the fill, the
    deterministic features, and each GFS column of D+1 (one scale per quantity, used for every
    horizon)."""
    f = sources.frame
    gfs = pd.DataFrame({c: f[f"{c}@D1"] for c in sources.gfs_columns}, index=f.index)
    return pd.concat([sources.series, f[list(DETERMINISTIC)].astype(float), gfs], axis=1)


# ── scopes ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scope:
    name: str
    split: str
    hours: pd.DatetimeIndex  # the hours whose forecasts are kept
    limit: pd.Timestamp  # decoders stop before it (the end of the split)
    label: str


def training_scope(sources: Sources, mode: str = "dev") -> Scope:
    """Train hours that are not warm-up (smoke: the last 56 local days of train, as XGB)."""
    f = sources.frame
    rows = train_scope(f, smoke=mode == "smoke") & ~f["warmup"].astype(bool)
    label = "the last 56 local days of train" if mode == "smoke" else "train after the warm-up"
    return Scope("train", "train", f.index[rows.to_numpy()], _utc(VAL_START), label)


def evaluation_scope(sources: Sources, window: Window) -> Scope:
    """The hours of `window` (the run's window of §10.3). Validation windows stop at TEST_START in
    every mode; the test window only exists with allow_test."""
    f = sources.frame
    if window.split == "test" and not sources.allow_test:
        raise PermissionError("test windows are built only in a final run with allow_test")
    rows = ((f["split"] == window.split).to_numpy() & (f.index >= window.start_ts)
            & (f.index < window.end_ts))
    limit = _utc(TEST_START) if window.split == "val" else sources.end
    return Scope(window.split, window.split, f.index[rows], limit, window.label)


# ── the dataset ─────────────────────────────────────────────────────────────────


@dataclass
class SequenceDataset:
    model: str
    target: str
    scope: Scope
    encoder_hours: int
    origins: pd.DatetimeIndex  # T0 of every window, UTC
    enc_time: np.ndarray  # [n, L] int64 ns
    enc_x: np.ndarray  # [n, L, C_enc] float32
    enc_channels: list
    dec_time: np.ndarray  # [n, MAX_STEPS] int64 ns, NAT beyond the third local day
    dec_horizon: np.ndarray  # [n, MAX_STEPS] int8: 1..3 by local day, 0 = padding
    dec_exists: np.ndarray  # [n, MAX_STEPS] bool: inside the three days and before the limit
    dec_keep: np.ndarray  # [n, MAX_STEPS] bool: the valid hour is in the scope
    dec_x: np.ndarray  # [n, MAX_STEPS, C_dec] float32, 0 on missing steps
    dec_channels: list
    y: np.ndarray  # [n, MAX_STEPS] float32, model units (k; AC / 3,449), NaN without a value
    y_true: np.ndarray  # [n, MAX_STEPS] float64, the target's unit (k; kW)
    loss_mask: np.ndarray  # [n, MAX_STEPS] bool
    target_scale: float
    known: list
    info: dict


def build(sources: Sources, scaler: Scaler, model: str, target: str, scope: Scope, *,
          inputs: dict | None = None, encoder_hours: int | None = None) -> SequenceDataset:
    """The windows of one model x target x scope (§17.5)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; one of {list(MODELS)}")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    hours = encoder_hours or MODELS[model]
    if not hours:
        raise ValueError(f"{model}: the encoder length must be given (W4 sets it)")
    f = sources.frame
    first, end = _ns(f.index)[0], _ns([sources.end])[0]
    limit = min(_ns([scope.limit])[0], end)
    days = origin_days(scope.hours)
    t0, valid, horizon, n_steps = decoder_layout(days)
    pos0 = (t0 - first) // HOUR_NS
    enc_pos = pos0[:, None] + np.arange(-hours, 0)[None, :]
    if (enc_pos < 0).any():
        raise SequenceDataError(f"{model} {target} {scope.name}: an encoder starts before the "
                                "loaded data")
    if (t0 >= limit).any():
        raise SequenceDataError(f"{scope.name}: an origin at or after the limit")
    exists = (horizon > 0) & (valid < limit) & (valid >= first)
    pos = np.where(exists, (valid - first) // HOUR_NS, 0)
    keep = exists & np.isin(valid, _ns(scope.hours))
    _check_issuance(f, t0, pos, horizon, exists, enc_pos)
    valid = np.where(exists, valid, NAT)  # steps beyond the limit are padding: no time, no row
    horizon = np.where(exists, horizon, 0).astype(np.int8)

    det, gfs = known_inputs(target, inputs) if model != "patchtst" else ([], [])
    det_z = scaler.transform(f[det].astype(float)) if det else np.zeros((len(f), 0), np.float32)
    gfs_raw = {h: pd.DataFrame({c: f[f"{c}@D{h}"] for c in gfs}, index=f.index)
               for h in HORIZONS}
    gfs_z = {h: (scaler.transform(gfs_raw[h]) if gfs else np.zeros((len(f), 0), np.float32))
             for h in HORIZONS}

    names = encoder_series_names(target)
    column = TARGET_COLUMN[target]
    scale = TARGET_SCALE[target]
    known_nan = {"encoder": 0, "decoder": 0}
    if model == "patchtst":  # L-19: the target series only, in model units, filled and unscaled
        own = TARGET_COLUMN["direct"] if target == "direct" else K_SERIES
        series = sources.filled[own].to_numpy(dtype=float) / scale
        enc_x = np.nan_to_num(series[enc_pos], nan=0.0)[..., None].astype(np.float32)
        enc_channels = [own]
        dec_x = np.zeros((len(t0), MAX_STEPS, 0), np.float32)
        dec_channels: list[str] = []
    else:
        series_z = scaler.transform(sources.filled[names])
        filled_flag = sources.series[names].isna().to_numpy().astype(np.float32)
        enc_known = np.concatenate([det_z, gfs_z[1]], axis=1)  # encoder hours: the D+1 table
        enc_x = np.concatenate([series_z[enc_pos], filled_flag[enc_pos], enc_known[enc_pos]],
                               axis=2)
        enc_channels = names + [n + FLAG_SUFFIX for n in names] + det + gfs
        raw_det = f[det].to_numpy(dtype=float)
        known_nan["encoder"] = int(np.isnan(raw_det[enc_pos]).sum()
                                   + np.isnan(gfs_raw[1].to_numpy()[enc_pos]).sum())
        gfs_steps = np.zeros((len(t0), MAX_STEPS, len(gfs)), np.float32)
        known_nan["decoder"] = int(np.isnan(raw_det[pos[exists]]).sum())
        for h in HORIZONS:
            sel = exists & (horizon == h)
            gfs_steps[sel] = gfs_z[h][pos[sel]]
            known_nan["decoder"] += int(np.isnan(gfs_raw[h].to_numpy()[pos[sel]]).sum())
        dec_x = np.concatenate([det_z[pos], gfs_steps], axis=2)
        dec_x[~exists] = 0.0
        dec_channels = det + gfs
    truth = f[column].to_numpy(dtype=float)
    y_true = np.where(exists, truth[pos], np.nan)
    y = (y_true / scale).astype(np.float32)
    loss = keep & fit_mask(f, target).to_numpy()[pos] & np.isfinite(y_true)
    info = {
        "n_origins": int(len(t0)),
        "first_t0": pd.Timestamp(t0[0], tz="UTC").isoformat(),
        "last_t0": pd.Timestamp(t0[-1], tz="UTC").isoformat(),
        "decoder_steps": {str(k): int(v) for k, v in
                          zip(*np.unique(n_steps, return_counts=True), strict=True)},
        "kept_steps_per_horizon": {f"D{h}": int((keep & (horizon == h)).sum()) for h in HORIZONS},
        "loss_steps_per_horizon": {f"D{h}": int((loss & (horizon == h)).sum()) for h in HORIZONS},
        "scope_hours": int(len(scope.hours)),
        "encoder_channels": len(enc_channels),
        "decoder_channels": len(dec_channels),
        "known_input_nan_values": known_nan,
    }
    return SequenceDataset(
        model=model, target=target, scope=scope, encoder_hours=hours,
        origins=pd.DatetimeIndex(t0, tz="UTC"), enc_time=_ns(f.index)[enc_pos],
        enc_x=enc_x.astype(np.float32), enc_channels=enc_channels, dec_time=valid,
        dec_horizon=horizon, dec_exists=exists, dec_keep=keep, dec_x=dec_x.astype(np.float32),
        dec_channels=dec_channels, y=y, y_true=y_true, loss_mask=loss, target_scale=scale,
        known=det + gfs, info=info)


def _check_issuance(f: pd.DataFrame, t0: np.ndarray, pos: np.ndarray, horizon: np.ndarray,
                    exists: np.ndarray, enc_pos: np.ndarray) -> None:
    """Every step of day D+h reads the D+h tables at their own T0, which is the origin's, and a
    GFS run that was public by T0 (init + 6 h <= T0, the rule of T-17). Encoder hours read the
    D+1 tables, issued at or before the origin's T0."""
    need = NWP_MIN_AVAILABILITY_HOURS * HOUR_NS
    for h in HORIZONS:
        sel = exists & (horizon == h)
        origin = np.broadcast_to(t0[:, None], sel.shape)[sel]
        for col in (f"t0@D{h}", f"gfs_t0@D{h}"):
            if not np.array_equal(f[col].to_numpy()[pos[sel]], origin):
                raise SequenceDataError(f"a D+{h} step reads a row of another issuance ({col})")
        if (f[f"gfs_init@D{h}"].to_numpy()[pos[sel]] + need > origin).any():
            raise SequenceDataError(f"a D+{h} step uses a GFS run not public by T0")
    for col in ("t0@D1", "gfs_t0@D1"):
        if (f[col].to_numpy()[enc_pos] > t0[:, None]).any():
            raise SequenceDataError(f"an encoder hour reads a row issued after T0 ({col})")
    if (f["gfs_init@D1"].to_numpy()[enc_pos] + need > t0[:, None]).any():
        raise SequenceDataError("an encoder hour uses a GFS run not public by T0")


# ── into the prediction contract (§6.10) ─────────────────────────────────────────


def predictions_frame(ds: SequenceDataset, y_pred, *, model_name: str, seed: int
                      ) -> pd.DataFrame:
    """The rows of §6.10 from the window forecasts `y_pred[origin, step]` (model units): one row
    per kept step, with its valid hour and its local-day horizon. A duplicate or a missing hour
    of the scope at some horizon is an error (M18)."""
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != ds.dec_keep.shape:
        raise ValueError(f"y_pred must have shape {ds.dec_keep.shape}, not {y_pred.shape}")
    sel = ds.dec_keep
    if not np.isfinite(y_pred[sel]).all():
        raise SequenceDataError("a kept step has no finite forecast")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(ds.dec_time[sel], utc=True),
        "y_true": ds.y_true[sel],
        "y_pred": y_pred[sel] * ds.target_scale,
        "target": ds.target,
        "horizon": ds.dec_horizon[sel].astype(int),
        "model": model_name,
        "seed": int(seed),
        "split": ds.scope.split,
    })
    dup = out.duplicated(["horizon", "timestamp"])
    if dup.any():
        raise SequenceDataError(f"{int(dup.sum())} duplicated (horizon, timestamp) rows")
    need = pd.DatetimeIndex(ds.scope.hours)
    for h in HORIZONS:
        have = pd.DatetimeIndex(out.loc[out["horizon"] == h, "timestamp"])
        missing = need.difference(have)
        if len(missing) or len(have) != len(need):
            raise SequenceDataError(f"D+{h}: {len(missing)} hours of the scope without a forecast")
    return out.sort_values(["horizon", "timestamp"], ignore_index=True)


# ── checks against theory (§17.8) ──────────────────────────────────────────────


def check_dataset(ds: SequenceDataset, sources: Sources) -> dict:
    """The checks of §17.8 on one dataset, recomputed independently of `build`: True = holds."""
    t0 = _ns(ds.origins)
    exists = ds.dec_exists
    valid = ds.dec_time
    out = {}
    out["encoder_ends_at_t0"] = bool((ds.enc_time[:, -1] + HOUR_NS == t0).all()
                                     and (np.diff(ds.enc_time, axis=1) == HOUR_NS).all())
    out["no_step_at_or_after_limit"] = bool((valid[exists] < _ns([ds.scope.limit])[0]).all())
    test_start = _ns([_utc(TEST_START)])[0]
    out["no_time_at_or_after_test_start"] = bool(
        sources.allow_test or ((ds.enc_time < test_start).all()
                               and (valid[exists] < test_start).all()))
    # the horizon by the local day of the valid hour, from the time zone itself (T-02)
    origin_day = local_day(ds.origins)
    steps_local = pd.DatetimeIndex(valid[exists], tz="UTC")
    day_index = ((local_day(steps_local) - np.repeat(origin_day.values, exists.sum(axis=1)))
                 // pd.Timedelta(days=1)) + 1
    out["horizon_is_local_day"] = bool((day_index == ds.dec_horizon[exists]).all())
    issue = np.empty(exists.sum(), dtype=np.int64)
    hz = ds.dec_horizon[exists]
    for h in HORIZONS:
        issue[hz == h] = _ns(issuance_utc(steps_local[hz == h], h))
    out["issuance_utc_is_t0"] = bool((issue == np.repeat(t0, exists.sum(axis=1))).all())
    # every D+h block that ends before the limit is its whole local day (23 / 24 / 25 steps)
    lengths, exact = {}, True
    limit = _ns([ds.scope.limit])[0]
    for i, day in enumerate(origin_day):
        for h in HORIZONS:
            d = day + pd.Timedelta(days=h - 1)
            start, stop = _ns(t0_of([d]))[0], _ns(t0_of([d + pd.Timedelta(days=1)]))[0]
            if stop > limit or start < _ns(sources.frame.index[:1])[0]:
                continue
            block = valid[i][ds.dec_horizon[i] == h]
            truth = np.arange(start, stop, HOUR_NS)
            exact &= bool(np.array_equal(block, truth))
            lengths[len(truth)] = lengths.get(len(truth), 0) + 1
    out["blocks_are_local_days"] = exact
    out["block_lengths"] = {str(k): v for k, v in sorted(lengths.items())}
    # coverage: every hour of the scope exactly once per horizon
    cover = True
    need = np.sort(_ns(ds.scope.hours))
    for h in HORIZONS:
        have = np.sort(valid[ds.dec_keep & (ds.dec_horizon == h)])
        cover &= bool(np.array_equal(have, need))
    out["every_scope_hour_once_per_horizon"] = cover
    # the loss rows at each horizon are the XGB fit rows of the scope (X23)
    f = sources.frame
    fit = fit_mask(f, ds.target).to_numpy() & np.isin(_ns(f.index), need)
    fit &= np.isfinite(f[TARGET_COLUMN[ds.target]].to_numpy(dtype=float))
    out["loss_rows_are_fit_rows"] = all(
        int((ds.loss_mask & (ds.dec_horizon == h)).sum()) == int(fit.sum()) for h in HORIZONS)
    out["loss_only_on_kept_steps"] = bool(not (ds.loss_mask & ~ds.dec_keep).any())
    return out


def lag_consistency(sources: Sources) -> dict:
    """X22: the encoder series equals the stage-6 lag it feeds (`<source>_lag_1d` at s + 24 h) on
    every hour whose source value is finite and not removed by the DST guard. Mismatches per
    series (expected 0)."""
    out = {}
    index = sources.frame.index
    guard_ok = (_ns(index) - 23 * HOUR_NS) <= sources.frame["t0@D1"].to_numpy()
    for col in sources.lag1d.columns:
        src = col.removesuffix("_lag_1d")
        shifted = sources.series[src].shift(24).to_numpy(dtype=float)
        lag = sources.lag1d[col].to_numpy(dtype=float)
        both = np.isfinite(shifted) & guard_ok
        out[src] = int((~np.isclose(lag[both], shifted[both], rtol=0, atol=1e-9)).sum())
    return out


def scaler_checks(sources: Sources, scaler: Scaler, seed: int = 0) -> dict:
    """A-14: the statistics recomputed independently from the train hours, and unchanged when
    every value outside train is replaced by noise."""
    f = sources.frame
    rows = (f["split"] == "train").to_numpy() & ~f["warmup"].astype(bool).to_numpy()
    raw = scaler_inputs(sources).to_numpy(dtype=float)
    mean = np.nanmean(raw[rows], axis=0)
    recomputed = bool(np.allclose(np.nan_to_num(mean), scaler.mean.to_numpy(), atol=1e-9))
    noisy = Sources(f.copy(), sources.series.copy(), sources.filled, sources.lag1d,
                    sources.gfs_columns, sources.allow_test)
    rng = np.random.default_rng(seed)
    outside = ~(f["split"] == "train").to_numpy()
    for frame, cols in ((noisy.series, list(noisy.series.columns)),
                        (noisy.frame, list(DETERMINISTIC)
                         + [f"{c}@D1" for c in sources.gfs_columns])):
        for col in cols:
            values = frame[col].to_numpy(dtype=float).copy()
            values[outside] = rng.normal(size=int(outside.sum())) * 1e3
            frame[col] = values
    again = Scaler.fit(noisy)
    unchanged = bool(np.array_equal(again.mean.to_numpy(), scaler.mean.to_numpy())
                     and np.array_equal(again.std.to_numpy(), scaler.std.to_numpy()))
    return {"recomputed_from_train": recomputed, "unchanged_when_other_splits_change": unchanged,
            "fit_rows": scaler.rows}


# ── the command (§17.8) ─────────────────────────────────────────────────────────


def _models_to_check() -> list[tuple[str, int]]:
    return [(m, h or CHECK_PATCHTST_HOURS) for m, h in MODELS.items()]


def run(mode: str = "dev", out_dir: Path | None = None, paths: Paths | None = None,
        inputs: dict | None = None) -> dict:
    """Build and check the train and validation datasets of every model x target (§17.8)."""
    if mode not in ("smoke", "dev"):
        raise ValueError("W3 builds development datasets only (smoke or dev); the test windows "
                         "are built in a final run with allow_test")
    out_dir = Path(out_dir) if out_dir else output_dir(REPORT_NAME)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    paths = paths or default_paths()
    with RunLog(REPORT_NAME, f"python -m solarpv.models.sequence_data --mode {mode}",
                out_dir=out_dir / "runs") as log:
        inputs_sha = log.inputs(paths.files())
        with log.step("load the inputs before TEST_START"):
            sources = load_sources(paths=paths)
            log.value("hours", len(sources.frame))
            log.value("last_hour", str(sources.frame.index[-1]))
        with log.step("fit the scaler on train hours"):
            scaler = Scaler.fit(sources)
        scopes = [training_scope(sources, mode),
                  evaluation_scope(sources, evaluation_windows(mode)[0])]
        rows, report = [], {"mode": mode, "git_commit": git_commit(), "datasets": {}}
        for scope in scopes:
            for target in TARGETS:
                for model, hours in _models_to_check():
                    with log.step(f"{scope.name} {target} {model}"):
                        ds = build(sources, scaler, model, target, scope, inputs=inputs,
                                   encoder_hours=hours)
                        checks = check_dataset(ds, sources)
                    key = f"{scope.name}/{target}/{model}"
                    report["datasets"][key] = {"scope": scope.label, "encoder_hours": hours,
                                               **ds.info, "checks": checks}
                    flat = {k: v for k, v in checks.items() if isinstance(v, bool)}
                    rows.append({"scope": scope.name, "target": target, "model": model,
                                 "encoder_hours": hours, "n_origins": ds.info["n_origins"],
                                 "first_t0": ds.info["first_t0"], "last_t0": ds.info["last_t0"],
                                 **flat, "block_lengths": json.dumps(checks["block_lengths"])})
                    if not all(flat.values()):
                        log.warn(f"{key}: failed {[k for k, v in flat.items() if not v]}")
        with log.step("scaler and lag checks"):
            report["scaler"] = {**scaler_checks(sources, scaler), **scaler.as_dict()}
            report["lag_consistency_mismatches"] = lag_consistency(sources)
        report["known_inputs"] = {t: dict(zip(("deterministic", "gfs"),
                                              known_inputs(t, inputs), strict=True))
                                  for t in TARGETS}
        report["filled_hours"] = {
            split: {c: int(v) for c, v in
                    sources.series[(sources.frame["split"] == split).to_numpy()]
                    .isna().sum().items()} for split in ("train", "val")}
        report["masked_power_hours"] = sources.masked_hours
        report["inputs_sha256"] = inputs_sha
        failed = [r for r in rows if not all(v for k, v in r.items() if isinstance(v, bool))]
        bad_lags = {k: v for k, v in report["lag_consistency_mismatches"].items() if v}
        scaler_ok = (report["scaler"]["recomputed_from_train"]
                     and report["scaler"]["unchanged_when_other_splits_change"])
        report["all_checks_pass"] = not failed and not bad_lags and scaler_ok
        csv_path = out_dir / f"checks_{mode}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, lineterminator="\n")
        json_path = out_dir / f"dataset_report_{mode}.json"
        report["run_log"] = log.relative_path
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8",
                             newline="\n")
        log.outputs([csv_path, json_path])
        log.value("all_checks_pass", report["all_checks_pass"])
        if not report["all_checks_pass"]:
            log.warn(f"checks failed: {failed[:3]} lags {bad_lags} scaler {scaler_ok}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("smoke", "dev"), default="dev")
    args = parser.parse_args(argv)
    report = run(args.mode)
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

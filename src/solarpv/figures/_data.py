"""Small readers shared by the figure modules. They only read stage outputs."""

import json

import pandas as pd
import pyarrow.dataset as ds

from solarpv.config.paths import REPO_ROOT

FEATURES_D1 = "data/processed/06_features/features_1H_D1.parquet"
CLEAN_1H = "data/processed/05_clean/scada_1H_clean.parquet"
INGEST_1MIN = "data/processed/01_ingest/scada_1min_utc.parquet"
GFS_D1 = "data/processed/06b_gfs/gfs_features_D1.parquet"
EVAL_VAL = "outputs/07_evaluation/val/benchmark_summary.csv"
DEV_RUN = "outputs/runs/20261107T135938Z_dev"
# W4 (the neural models): the passing smoke run, its repetition (X32), and the dev run (R9-16, R9-17)
NEURAL_SMOKE_RUN = "outputs/runs/20261111T155739Z_smoke"
NEURAL_SMOKE_REPEAT = "outputs/runs/20261111T160241Z_smoke"
NEURAL_DEV_RUN = "outputs/runs/20261111T160714Z_dev"


def table(rel: str, columns: list[str], split: str | None = None) -> pd.DataFrame:
    """Columns of a stage parquet, indexed by UTC time; ``split`` filters inside the reader so
    that no other split's rows are read."""
    dataset = ds.dataset(REPO_ROOT / rel)
    names = dataset.schema.names
    time_col = "timestamp_utc" if "timestamp_utc" in names else None
    cols = list(columns) + ([time_col] if time_col and time_col not in columns else [])
    flt = ds.field("split") == split if split else None
    df = dataset.to_table(columns=cols, filter=flt).to_pandas()
    if time_col and time_col in df.columns:
        df = df.set_index(time_col)
    df.index = pd.to_datetime(df.index, utc=True)
    if split:
        assert "split" not in df.columns or set(df["split"]) <= {split}
    return df.sort_index()


def csv(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / rel, **kw)


def jload(rel: str):
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def local(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("Asia/Jerusalem")


def naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("UTC").tz_localize(None)

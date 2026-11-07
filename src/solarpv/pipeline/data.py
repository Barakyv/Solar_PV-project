"""The model tables of the runner (docs/stage9_plan.md §13.3, X10).

`model_table(h, splits)` joins the stage-6 table of horizon h with its GFS table, like the stage-6b
and stage-7 loaders, but filters inside the parquet reader, so rows of an excluded split never
reach pandas:
- the feature table by its `split` column;
- the GFS table, which has no split column, by `timestamp_utc < TEST_START`.
Smoke and dev runs read train and validation only. The test rows are read only in a final run
with allow_test, and only to be predicted (D-08). The stage loaders (`nwp.check.load`,
`selection.common.train_table`) stay as they are: they are the record of their stages.
"""

import pandas as pd

from solarpv.config.constants import TEST_START
from solarpv.config.paths import PROCESSED_DIR

FEATURES_DIR = PROCESSED_DIR / "06_features"
GFS_DIR = PROCESSED_DIR / "06b_gfs"
SPLITS = ("train", "val", "test")
INDEX = "timestamp_utc"


def feature_file(h: int):
    return FEATURES_DIR / f"features_1H_D{h}.parquet"


def gfs_file(h: int):
    return GFS_DIR / f"gfs_features_D{h}.parquet"


def model_table(h: int, splits: tuple[str, ...] = ("train", "val"), *,
                allow_test: bool = False, features_path=None, gfs_path=None) -> pd.DataFrame:
    """The stage-6 table of D+h joined with its GFS table, rows of `splits` only.

    The test split needs `allow_test` (a final run that predicts the test hours). Without it, the
    function asserts that no row at or after TEST_START was read from either file."""
    splits = tuple(splits)
    unknown = set(splits) - set(SPLITS)
    if unknown or not splits:
        raise ValueError(f"unknown or empty splits {sorted(unknown)}; one of {SPLITS}")
    with_test = "test" in splits
    if with_test and not allow_test:
        raise PermissionError("the test split is read only in a final run with allow_test "
                              "(D-08, docs/stage9_plan.md §3.6)")
    first_test = pd.Timestamp(TEST_START, tz="UTC")
    base = pd.read_parquet(features_path or feature_file(h),
                           filters=[("split", "in", list(splits))])
    gfs_filters = None if with_test else [(INDEX, "<", first_test)]
    gfs = pd.read_parquet(gfs_path or gfs_file(h), filters=gfs_filters)
    for name, frame in (("feature", base), ("GFS", gfs)):
        if frame.index.name != INDEX or frame.index.tz is None or str(frame.index.tz) != "UTC":
            raise ValueError(f"the {name} table of D+{h} must have a UTC index {INDEX!r}")
    if not with_test:
        if (base.index >= first_test).any() or (gfs.index >= first_test).any():
            raise RuntimeError("rows of the test period were read in a development run")
    if not base["split"].isin(splits).all():
        raise RuntimeError(f"rows outside {list(splits)} were read")
    if gfs.index.has_duplicates or base.index.has_duplicates:
        raise ValueError(f"duplicated timestamps in the tables of D+{h}")
    table = base.join(gfs.drop(columns=["t0_utc"], errors="ignore"), how="left")
    if not table.index.is_monotonic_increasing:
        table = table.sort_index()
    return table

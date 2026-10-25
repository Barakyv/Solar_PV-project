"""The authoritative truth for evaluation (stage 8B, docs/stage8b_plan.md §3.2; Q-06, E8).

The truth comes from the stage-6 feature table, never from the prediction files: target_k for
the indirect route, site_total_AC_kW for the direct route, and theo_kw for the k -> kW
conversion. The targets are identical in the three horizon tables (T-33), so the D+1 table is
read. Its metadata gives the evaluation views: split, daytime (F-02), op_limited_flag (D-22) and
hour_source (D-21).
"""

import pandas as pd

from solarpv.config.paths import PROCESSED_DIR

FEATURES_FILE = PROCESSED_DIR / "06_features" / "features_1H_D1.parquet"
COLUMNS = {"target_k": "indirect", "site_total_AC_kW": "direct", "theo_kw": "theo_kw",
           "k_valid": "k_valid", "daytime": "daytime", "split": "split",
           "op_limited_flag": "op_limited_flag", "hour_source": "hour_source"}


def load_truth(path=FEATURES_FILE) -> pd.DataFrame:
    """Truth table on the complete hourly UTC grid, columns renamed for the evaluation."""
    table = pd.read_parquet(path, columns=list(COLUMNS)).rename(columns=COLUMNS)
    if table.index.tz is None or str(table.index.tz) != "UTC":
        raise ValueError("the feature table must have a tz-aware UTC index")
    expected = pd.date_range(table.index[0], table.index[-1], freq="h")
    if not table.index.equals(expected):
        raise ValueError("the truth table is not a complete hourly grid")
    return table

"""T-02 on the real tables: the sequence datasets of every model x target, train and validation
(docs/stage9_plan.md §17.7; PROJECT_MASTER.md §8.2).

For every window: unique valid timestamps per horizon (every hour of the scope exactly once), the
encoder ends at T0, the decoder steps mapped to D+1..D+3 by the local day of the valid hour (with
the 23- and 25-hour days), and no row at or after TEST_START. Skipped when the processed data are
not present.
"""

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import TEST_START, VAL_START
from solarpv.models import sequence_data as S
from solarpv.pipeline.config import split_window

pytestmark = pytest.mark.skipif(not all(p.exists() for p in S.default_paths().files()),
                                reason="processed data not present")
EXPECTED_ORIGINS = {"train": 872, "val": 246}


@pytest.fixture(scope="module")
def real():
    sources = S.load_sources()
    return sources, S.Scaler.fit(sources)


@pytest.mark.parametrize("model", ["bilstm", "tft96", "tft168", "patchtst"])
@pytest.mark.parametrize("target", ["indirect", "direct"])
@pytest.mark.parametrize("which", ["train", "val"])
def test_sequence_alignment(real, model, target, which):
    sources, scaler = real
    assert sources.frame.index.max() < pd.Timestamp(TEST_START, tz="UTC")
    scope = (S.training_scope(sources, "dev") if which == "train"
             else S.evaluation_scope(sources, split_window("val")))
    hours = S.MODELS[model] or S.CHECK_PATCHTST_HOURS
    ds = S.build(sources, scaler, model, target, scope, encoder_hours=hours)
    checks = S.check_dataset(ds, sources)
    failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    assert not failed, failed
    assert ds.info["n_origins"] == EXPECTED_ORIGINS[which]
    lengths = checks["block_lengths"]
    assert lengths.get("23", 0) >= 1  # 2024-03-29 in validation; 2022-03-25, 2023-03-24 in train
    if which == "train":
        assert lengths.get("25", 0) >= 1  # 2021-10-31, 2022-10-30, 2023-10-29
        limit = pd.Timestamp(VAL_START, tz="UTC")
        assert (ds.dec_time[ds.dec_exists] < S._ns([limit])[0]).all()
    for h in (1, 2, 3):
        kept = ds.dec_time[ds.dec_keep & (ds.dec_horizon == h)]
        assert len(np.unique(kept)) == len(kept) == len(scope.hours)

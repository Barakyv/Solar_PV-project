import pandas as pd
import pytest

from solarpv.config.paths import EVIDENCE_DIR, LSTM_FIXTURE, RAW_DIR
from solarpv.config.site import PRIVATE_FILE


def pytest_collection_modifyitems(config, items):
    skip_private = pytest.mark.skip(reason="configs/site.private.yaml not present")
    skip_raw = pytest.mark.skip(reason="raw SCADA files not present")
    for item in items:
        if "needs_private_site" in item.keywords and not PRIVATE_FILE.exists():
            item.add_marker(skip_private)
        if "needs_raw" in item.keywords and not any(RAW_DIR.glob("*.CSV")):
            item.add_marker(skip_raw)


@pytest.fixture(scope="session")
def evidence_dir():
    return EVIDENCE_DIR


@pytest.fixture(scope="session")
def lstm_fixture() -> pd.DataFrame:
    """Old LSTM predictions (built on the buggy pipeline). Used to test evaluation code only."""
    df = pd.read_csv(LSTM_FIXTURE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

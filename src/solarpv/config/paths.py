"""Repository paths. Every module resolves locations through here, never hard-coded strings."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = Path(os.environ.get("SOLAR_RAW_DIR", REPO_ROOT / "Raw data"))
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_DIR = REPO_ROOT / "docs"
AUDIT_DIR = REPO_ROOT / "00_data_audit"

# Read-only reference material from the original project
EVIDENCE_DIR = REPO_ROOT / "submission_evidence"
OLD_NOTEBOOKS_DIR = REPO_ROOT / "old files"

# The only surviving predictions; used as the evaluate.py fixture (test T-15)
LSTM_FIXTURE = EVIDENCE_DIR / "08_lstm_alignment" / "predictions_lstm_sota_aligned.csv"


def stage_dir(number: str, name: str) -> Path:
    """Processed-data stage directory: stage_dir("01", "ingest") -> data/processed/01_ingest."""
    path = PROCESSED_DIR / f"{number}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(name: str) -> Path:
    """Directory under outputs/, e.g. output_dir("06_xgb_sota")."""
    path = OUTPUTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path

"""
align_and_evaluate.py - Post-training: align LSTM/TFT predictions, re-run DM tests
====================================================================================
Run after LSTM (and optionally TFT) retraining to:
  1. Align LSTM predictions (day-ahead deduplication)
  2. Update predictions_lstm_sota_aligned.csv
  3. Re-run evaluate.py to get updated DM tests and benchmark
  4. Update seed_variability.csv with aligned evaluate.py metrics

Usage:
  python scripts/align_and_evaluate.py
  python scripts/align_and_evaluate.py --seed 123
"""
import argparse
import pathlib
import subprocess
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
LSTM_DIR = ROOT / "outputs" / "06_lstm_sota"
EVAL_DIR = ROOT / "outputs" / "07_evaluation"
EVAL_SCRIPT = ROOT / "production" / "03_evaluation" / "evaluate.py"

sys.path.insert(0, str(ROOT / "production" / "00_config"))
from constants import HORIZONS


def align_lstm(seed=42):
    suffix = f"_seed{seed}" if seed != 42 else ""
    src = LSTM_DIR / f"predictions_lstm_sota{suffix}.csv"
    dst_name = f"predictions_lstm_sota_aligned{suffix}.csv"
    dst = LSTM_DIR / dst_name

    if not src.exists():
        print(f"  [SKIP] {src.name} not found")
        return None

    df = pd.read_csv(src)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print(f"  Loaded {src.name}: {len(df):,} rows")

    # Deduplicate: keep one row per (timestamp, target, horizon)
    # The training script now uses day-ahead protocol (target_step = h_end-1)
    # so duplicates would only come from multiple sequences predicting same step
    before = len(df)
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp", "target", "horizon"], keep="last"))
    print(f"  Deduplicated: {before:,} -> {len(df):,} rows")

    df.to_csv(dst, index=False)
    print(f"  Saved {dst_name} ({len(df):,} rows)")
    return dst


def run_evaluate():
    print("\n  Running evaluate.py...")
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] evaluate.py failed:\n{result.stderr[-2000:]}")
    else:
        print(result.stdout[-3000:] if result.stdout else "  (no output)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_evaluate", action="store_true")
    args = parser.parse_args()

    print(f"[align] Aligning LSTM predictions (seed={args.seed})...")
    dst = align_lstm(args.seed)

    if dst is None:
        print("  Nothing to align. Exiting.")
        return

    # For seed=42: also update the canonical aligned file used by evaluate.py
    if args.seed == 42:
        canonical = LSTM_DIR / "predictions_lstm_sota_aligned.csv"
        if dst.resolve() != canonical.resolve():
            import shutil, time
            for attempt in range(3):
                try:
                    shutil.copy2(str(dst), str(canonical))
                    print(f"  Copied to canonical: predictions_lstm_sota_aligned.csv")
                    break
                except PermissionError:
                    print(f"  [retry {attempt+1}/3] File locked, waiting 5s...")
                    time.sleep(5)
        else:
            print(f"  Canonical file is already up-to-date (same path as dst).")

    if not args.skip_evaluate:
        run_evaluate()

    print("\n[align] Done.")


if __name__ == "__main__":
    main()

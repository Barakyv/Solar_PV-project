"""
reproduce_paper_results.py
==========================
End-to-end reproduction of all paper results from the canonical feature parquet.

Produces (in order):
  1. XGB SOTA training — 60 Optuna trials, 3 seeds (42/123/777)
  2. Full evaluation — DM tests (h=24 NW), route comparison, common mask, seed variability
  3. SHAP analysis — group importance, rank stability (horizon + season)
  4. Ablation study — feature group removal, 3 seeds
  5. Detectable-effect / extended DM table

All outputs land in outputs/07_evaluation/ and outputs/09_cv_ablation/.

GPU training (LSTM/TFT) is excluded by default because it requires:
  - Python 3.12 via venv_cuda/
  - NVIDIA GPU with CUDA 12.1+
  - ~25–30 h wall-clock time
Use --include-gpu to include it (will fail gracefully if GPU unavailable).

Usage:
    python scripts/reproduce_paper_results.py              # XGB-only
    python scripts/reproduce_paper_results.py --xgb-only  # explicit
    python scripts/reproduce_paper_results.py --include-gpu
    python scripts/reproduce_paper_results.py --skip-train # evaluation only (use existing outputs)
    python scripts/reproduce_paper_results.py --dry-run    # print commands, don't run
"""

import argparse
import pathlib
import subprocess
import sys
import time
import os

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Python interpreters
PY_XGB   = sys.executable          # Current Python (3.14) — for XGB, eval, SHAP
VENV_PY  = ROOT / "venv_cuda" / "Scripts" / "python.exe"
PY_GPU   = str(VENV_PY) if VENV_PY.exists() else sys.executable

# Script paths
XGB_SCRIPT    = ROOT / "production" / "02_models"     / "xgb_sota.py"
LSTM_SCRIPT   = ROOT / "production" / "02_models"     / "lstm_sota.py"
EVAL_SCRIPT   = ROOT / "production" / "03_evaluation" / "evaluate.py"
ALIGN_SCRIPT  = ROOT / "scripts" / "align_and_evaluate.py"
COMPILE_SCRIPT= ROOT / "scripts" / "compile_seed_variability.py"
ROUTE_SCRIPT  = ROOT / "scripts" / "compare_routes.py"
MASK_SCRIPT   = ROOT / "scripts" / "common_mask_analysis.py"
DE_SCRIPT     = ROOT / "scripts" / "detectable_effect.py"
SHAP_SCRIPT   = ROOT / "scripts" / "shap_rank_stability.py"
ABLATION_SCRIPT = ROOT / "scripts" / "run_ablation_study.py"
LSTM_ORCH     = ROOT / "scripts" / "run_lstm_gpu_all_seeds.py"

# Output dirs
LOG_DIR = ROOT / "outputs" / "reproduction_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 777]


def banner(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def run_step(label, cmd, logfile, dry_run=False):
    """Run subprocess, stream to console + logfile. Returns success bool."""
    banner(f"STEP: {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"  LOG: {logfile}")

    if dry_run:
        print("  [DRY RUN] — skipped")
        return True

    t0 = time.time()
    with open(logfile, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        proc.wait()

    elapsed = (time.time() - t0) / 60
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (code {proc.returncode})"
    print(f"\n  [{status}] {label} — {elapsed:.1f} min")
    return ok


def check_prerequisite(path, label):
    """Warn if a prerequisite file is missing."""
    if not pathlib.Path(path).exists():
        print(f"  ⚠️  Missing prerequisite: {label}")
        print(f"     Expected at: {path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all paper results from feature parquet."
    )
    parser.add_argument("--xgb-only",    action="store_true", default=True,
                        help="Run XGB path only (default; GPU not required)")
    parser.add_argument("--include-gpu", action="store_true",
                        help="Include LSTM/TFT GPU training (requires CUDA env)")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip model training; reuse existing predictions")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Random seeds for XGB (default: 42 123 777)")
    parser.add_argument("--n-trials", type=int, default=60,
                        help="Optuna trials per XGB model (default: 60)")
    args = parser.parse_args()

    if args.include_gpu:
        args.xgb_only = False

    dry = args.dry_run
    t_total = time.time()
    failures = []

    # ------------------------------------------------------------------
    # 0 — Prerequisites
    # ------------------------------------------------------------------
    banner("Prerequisites Check")
    feature_file = (ROOT / "04_feature_engineering" / "data" / "processed"
                    / "04_features_target_k" / "features_1H_target_k.parquet")
    if not check_prerequisite(feature_file, "Feature parquet"):
        print("\n  ERROR: Feature parquet is required. Cannot reproduce results.")
        print("  The raw SCADA + NWP data is proprietary; the pre-processed feature")
        print("  parquet must be present in the repository.")
        sys.exit(1)
    print(f"  ✅ Feature parquet found: {feature_file}")

    # ------------------------------------------------------------------
    # 1 — XGB SOTA Training (3 seeds)
    # ------------------------------------------------------------------
    if not args.skip_train:
        banner("Phase 1 — XGB SOTA Training")
        for seed in args.seeds:
            suffix = f"_seed{seed}" if seed != 42 else ""
            ok = run_step(
                f"XGB SOTA seed={seed} ({args.n_trials} trials)",
                [PY_XGB, "-u", str(XGB_SCRIPT),
                 "--n_trials", str(args.n_trials),
                 "--seed", str(seed)],
                LOG_DIR / f"xgb_train_seed{seed}.log",
                dry_run=dry,
            )
            if not ok:
                failures.append(f"XGB training seed={seed}")

    # ------------------------------------------------------------------
    # 2 — Canonical Evaluation (DM h=24, bootstrap CI)
    # ------------------------------------------------------------------
    banner("Phase 2 — Canonical Evaluation")
    ok = run_step(
        "evaluate.py — DM tests, benchmark_summary, route comparison",
        [PY_XGB, "-u", str(EVAL_SCRIPT)],
        LOG_DIR / "evaluate.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("evaluate.py")

    # ------------------------------------------------------------------
    # 3 — Route comparison (k→power vs direct at AC scale)
    # ------------------------------------------------------------------
    ok = run_step(
        "compare_routes.py — indirect k→power vs direct",
        [PY_XGB, "-u", str(ROUTE_SCRIPT)],
        LOG_DIR / "route_comparison.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("compare_routes.py")

    # ------------------------------------------------------------------
    # 4 — Common evaluation mask
    # ------------------------------------------------------------------
    ok = run_step(
        "common_mask_analysis.py — shared timestamp sensitivity",
        [PY_XGB, "-u", str(MASK_SCRIPT)],
        LOG_DIR / "common_mask.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("common_mask_analysis.py")

    # ------------------------------------------------------------------
    # 5 — Detectable-effect / extended DM
    # ------------------------------------------------------------------
    ok = run_step(
        "detectable_effect.py — N_eff, MDE, extended DM table",
        [PY_XGB, "-u", str(DE_SCRIPT)],
        LOG_DIR / "detectable_effect.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("detectable_effect.py")

    # ------------------------------------------------------------------
    # 6 — Seed variability summary
    # ------------------------------------------------------------------
    ok = run_step(
        "compile_seed_variability.py — mean ± std across seeds",
        [PY_XGB, "-u", str(COMPILE_SCRIPT)],
        LOG_DIR / "seed_variability.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("compile_seed_variability.py")

    # ------------------------------------------------------------------
    # 7 — SHAP rank stability
    # ------------------------------------------------------------------
    ok = run_step(
        "shap_rank_stability.py — horizon/season Spearman correlations",
        [PY_XGB, "-u", str(SHAP_SCRIPT)],
        LOG_DIR / "shap_stability.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("shap_rank_stability.py")

    # ------------------------------------------------------------------
    # 8 — Ablation study (3 seeds)
    # ------------------------------------------------------------------
    ok = run_step(
        "run_ablation_study.py — feature group ablation × 3 seeds",
        [PY_XGB, "-u", str(ABLATION_SCRIPT)],
        LOG_DIR / "ablation.log",
        dry_run=dry,
    )
    if not ok:
        failures.append("run_ablation_study.py")

    # ------------------------------------------------------------------
    # 9 — GPU training (optional)
    # ------------------------------------------------------------------
    if args.include_gpu:
        banner("Phase 9 — GPU Training (LSTM ×3 seeds, 60 trials)")
        if not VENV_PY.exists():
            print(f"  ⚠️  venv_cuda not found at {VENV_PY}")
            print("  Run: py -3.12 -m venv venv_cuda && venv_cuda\\Scripts\\activate")
            print("       pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
            print("       pip install -r production/requirements.txt")
            failures.append("GPU training — venv_cuda missing")
        else:
            ok = run_step(
                "run_lstm_gpu_all_seeds.py — 60 trials × seeds 42/123/777",
                [PY_GPU, "-u", str(LSTM_ORCH),
                 "--n_trials", "60", "--max_epochs", "60"],
                LOG_DIR / "lstm_gpu.log",
                dry_run=dry,
            )
            if not ok:
                failures.append("LSTM GPU training")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = (time.time() - t_total) / 60
    banner("REPRODUCTION COMPLETE")

    key_outputs = [
        ROOT / "outputs" / "07_evaluation" / "benchmark_summary.csv",
        ROOT / "outputs" / "07_evaluation" / "dm_tests.csv",
        ROOT / "outputs" / "07_evaluation" / "dm_tests_extended.csv",
        ROOT / "outputs" / "07_evaluation" / "route_comparison.csv",
        ROOT / "outputs" / "07_evaluation" / "seed_variability_summary.csv",
        ROOT / "outputs" / "07_evaluation" / "common_mask_summary.csv",
        ROOT / "outputs" / "07_evaluation" / "shap_group_importance.csv",
        ROOT / "outputs" / "09_cv_ablation" / "ablation_seed_variability.csv",
    ]

    print(f"\n  Total wall-clock: {elapsed_total:.1f} min")
    print(f"\n  Key output files:")
    for p in key_outputs:
        status = "✅" if p.exists() else "❌ MISSING"
        print(f"    {status}  {p.relative_to(ROOT)}")

    if failures:
        print(f"\n  ⚠️  {len(failures)} step(s) failed:")
        for f in failures:
            print(f"    - {f}")
        print("\n  Check logs in:", LOG_DIR)
        sys.exit(1)
    else:
        print("\n  ✅ All steps completed successfully.")
        print("\n  Expected paper results:")
        print("    XGB SOTA indirect D+1 nRMSE: ~29.09%")
        print("    XGB SOTA direct   D+1 nRMSE: ~12.06%")
        print("    LSTM SOTA indirect D+1 nRMSE: ~30.48% (aligned timestamps)")
        print("    LSTM SOTA direct   D+1 nRMSE: ~12.60% (aligned timestamps)")
        print("    DM LSTM vs XGB: p > 0.10 (not significant, underpowered)")
        print("    Route: indirect→power outperforms direct by 1.7–2.1 pp nRMSE")


if __name__ == "__main__":
    main()

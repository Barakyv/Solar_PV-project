"""The stage-9 runner (docs/stage9_plan.md §3).

    python -m solarpv.pipeline list
    python -m solarpv.pipeline check --mode smoke
    python -m solarpv.pipeline run --mode smoke --only baselines,evaluate
    python -m solarpv.pipeline run --mode dev --from baselines --to evaluate --dry-run
    python -m solarpv.pipeline run --mode smoke --only evaluate --resume <run_id> [--force]
    python -m solarpv.pipeline status [<run_id>]

Modes: smoke (does the code run? a 28-day validation window), dev (stage 9, the whole
validation split), final (stage 10, only on the commit tagged g1-frozen with a clean tree; the
test split only with --allow-test). Smoke and dev never load a test row.
"""

import argparse
import sys

from solarpv.pipeline import checks
from solarpv.pipeline.config import MODES, STEPS
from solarpv.pipeline.runner import Runner, select_steps
from solarpv.pipeline.steps import IMPLEMENTED


def _csv(text: str | None, cast=str):
    return [cast(x) for x in text.split(",") if x] if text else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m solarpv.pipeline",
                                     description="Stage-9 runner (docs/stage9_plan.md).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the steps, in order")
    p_check = sub.add_parser("check", help="environment, data and gates; nothing is run")
    p_check.add_argument("--mode", choices=MODES, default="smoke")
    p_run = sub.add_parser("run", help="run the selected steps")
    p_run.add_argument("--mode", choices=MODES, default="smoke")
    p_run.add_argument("--only", help="comma-separated steps, e.g. baselines,evaluate")
    p_run.add_argument("--from", dest="from_", help="first step of a section")
    p_run.add_argument("--to", help="last step of a section")
    p_run.add_argument("--targets", help="indirect,direct (default both)")
    p_run.add_argument("--horizons", help="1,2,3 (default all)")
    p_run.add_argument("--seeds", help="override the seeds of the budget, e.g. 42")
    p_run.add_argument("--resume", metavar="RUN_ID", help="continue this run: completed steps "
                       "with the same fingerprint are skipped")
    p_run.add_argument("--force", action="store_true", help="rerun completed steps")
    p_run.add_argument("--keep-going", action="store_true",
                       help="a failed step does not stop the sequence")
    p_run.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    p_run.add_argument("--allow-test", action="store_true",
                       help="final mode only: also evaluate the test split (stage 11)")
    p_status = sub.add_parser("status", help="the runs, or one run in detail")
    p_status.add_argument("run_id", nargs="?")
    args = parser.parse_args(argv)
    command = "python -m solarpv.pipeline " + " ".join(argv if argv is not None else sys.argv[1:])

    if args.command == "list":
        for s in STEPS:
            state = "implemented" if s.name in IMPLEMENTED else f"comes with {s.work_item}"
            print(f"{s.name:<10} stage {s.stage:<5} {s.device:<10} {state:<16} {s.purpose}; "
                  f"needs {s.needs}; produces {s.produces}")
        print("preflight of every run: env (environment report), data (inputs against their "
              "stage manifests)")
        return 0
    if args.command == "check":
        report = checks.preflight(args.mode)
        for line in checks.format_preflight(report):
            print(line)
        return 1 if report["errors"] else 0
    if args.command == "status":
        return Runner().status(args.run_id)
    try:
        steps = select_steps(_csv(args.only), args.from_, args.to)
        return Runner().run(
            args.mode, steps, resume=args.resume, force=args.force, keep_going=args.keep_going,
            dry_run=args.dry_run, targets=_csv(args.targets), horizons=_csv(args.horizons, int),
            seeds=_csv(args.seeds, int), allow_test=args.allow_test, command=command)
    except (ValueError, PermissionError) as exc:
        print(f"run refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

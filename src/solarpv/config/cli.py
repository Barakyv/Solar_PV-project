"""Argument parsing for the entry points that write data or documents (X62 of
docs/stage9_plan.md §20).

Until W6 most stage modules ran their work whatever the arguments: `python -m
solarpv.data.clean --help` cleaned the data again and rewrote the stage-5 files (it happened on
2026-11-12, run journal R9-26). Every entry point now parses its arguments first: `--help` prints
the module's docstring and exits 0 before any work, and an unknown argument is an error (exit 2).
"""

import argparse
import sys


def parser(prog: str, doc: str | None) -> argparse.ArgumentParser:
    """A parser whose help is the module's docstring. No abbreviations: with argparse's default a
    typo such as `--valid` would be read as `--validation-only` and start the validation look."""
    return argparse.ArgumentParser(prog=prog, description=(doc or "").strip(), allow_abbrev=False,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)


def no_options(prog: str, doc: str | None, argv: list[str] | None = None) -> None:
    """For an entry point without options: returns only when there is no argument; `--help`
    exits 0, anything else exits 2. Call it before any work."""
    parser(prog, doc).parse_args(sys.argv[1:] if argv is None else argv)

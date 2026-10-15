# Solar PV Forecasting Capstone — rebuild and correction

Read `PROJECT_MASTER.md` before doing anything. It is the single source of truth for the plan, the canonical spec (§6), findings (F-xx), decisions (D-xx) and status (§1).

- Work on the "next action" in §1. Do not start a stage whose blocking decisions (§3) are still open.
- Principle (D-01): do not reproduce the old results one-to-one. Every bug found gets fixed, gets a regression test, and gets an entry in `docs/errata.md`. Old numbers are for comparison only.
- Build and validate the whole pipeline end-to-end with light models first (baselines, XGB-dev). Final training of all models (stage 10) happens only after gate G-1 passes and the code is tagged `g1-frozen` (D-08). Never use the test split for any decision.
- Every assumption goes into the register in §6.13 and must be verified before G-1.
- All processing is in true UTC (F-01). The indirect target is never clipped at 1.2, and k ↔ kW must invert exactly (F-06).
- `submission_evidence/`, `old files/` and `Raw data/` are read-only.
- At the end of each session update `PROJECT_MASTER.md`: §1 status table and next action, any changed D/C/R/Q rows, and a dated line in §17.
- Edit text files (especially the Hebrew `PROJECT_MASTER.md`) only with UTF-8-aware tools, never with PowerShell `Get-Content`/`Set-Content`: a cp1252 round trip once turned the whole document into mojibake (`tests/contracts/test_docs_encoding.py` guards this).
- Before committing, check `git status --ignored`: an unanchored `.gitignore` pattern once kept `src/solarpv/data/` out of git.
- **Every operation on the data is documented as it is made** (user request, 2026-10-15): an OP entry in `docs/data_processing_record.md` in the same commit as the code (why, method, parameters, result, verification, evidence, difference from the original); after rerunning a stage, `python scripts/build_data_record.py`. `tests/contracts/test_data_record.py` fails if the numbers are stale or a stage, rule or OP is missing.

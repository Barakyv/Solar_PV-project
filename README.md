# Solar PV Forecasting Capstone — Ktora plant (rebuild)

Day-ahead (D+1 to D+3) hourly forecasting of AC power for a 3.45 MW section (2 of 25 inverters) of the
40 MW Ketura Solar park in the Southern Arava, comparing a direct kW target with a clear-sky
performance-index target.

This repository rebuilds the original project after its source code was lost. It is a corrected
rebuild, not a one-to-one reproduction: every bug found on the way is fixed and listed in
[`docs/errata.md`](docs/errata.md).

**Start here:** [`PROJECT_MASTER.md`](PROJECT_MASTER.md) — plan, specification, findings, decisions and status.

## Setup (Windows)

```powershell
conda env create -f environment.yml      # creates env "solar" and installs this package (editable)
conda activate solar
pytest
```

The raw SCADA data (`Raw data/`) and the true site coordinates (`configs/site.private.yaml`) are not in git.
Without the private site file the code falls back to anonymised coordinates and site-dependent tests are skipped.

## Layout

| Path | Content |
|---|---|
| `src/solarpv/` | Pipeline code: `config`, `data`, `features`, `nwp`, `models`, `evaluation` |
| `tests/` | `unit`, `leakage`, `contracts`, `regression` (test IDs in PROJECT_MASTER.md §8) |
| `configs/` | Split manifest, site parameters |
| `scripts/` | Command-line entry points |
| `notebooks/` | One narrative notebook per stage, importing from `src/` |
| `docs/` | Errata, deviations, leakage audit |
| `submission_evidence/`, `old files/` | Read-only material from the original project |
| `00_data_audit/` | Raw data inventory, checksums, verification scripts |

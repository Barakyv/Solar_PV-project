# Errata — differences from the submitted report (v6)

Every bug found during the rebuild is listed here with what changed and its effect on the reported numbers.
Old values come from `submission_evidence/`; new values are filled in after the final training (stage 10).
IDs refer to PROJECT_MASTER.md.

| ID | Issue in the original pipeline | Fix | Effect on results |
|---|---|---|---|
| F-01 | Local Israel time was treated as UTC (notebook 03, cell 3), so production was shifted 2–3 h against solar geometry; afternoon production was zeroed (F-01b); the "13.5 % inverter downtime" were pre-dawn hours (F-01c) | Localise to Asia/Jerusalem, convert to UTC; night rule on true geometry | pending |
| F-06 | Indirect target clipped at 1.2 (16.8 % of daytime test hours); hourly k was a mean of 15-min ratios, so k → kW was not invertible | Hourly k as ratio of means, no clipping, exact inversion | pending |
| F-07 | Rolling same-time-of-day features included T−24 h at D+2/D+3 (leakage) | Horizon-aware rolling windows | pending |
| C-15 | DM test reported as HLN + Bonferroni but computed without them | Implemented as described; uncorrected value also reported | pending |
| C-16 | Route comparison used clipped k × GHI_cs as "truth" for the indirect route | Actual AC is the truth for both routes, on shared timestamps | pending |
| C-18 | `theoretical_power` (listed as leakage) was a model input | Removed | pending |
| C-07 | Ablation baseline contained leaky features and reanalysis NWP | Rerun on the leak-free contract; no_nwp row removed | pending |
| C-08 | Skill score vs persistence reported as 90–96 % | Recomputed against horizon-matched persistence (old pipeline: ~22 %) | pending |
| F-15 | Logger clock drift not detected: two logger clocks drifting up to −16 min with separate resyncs (2023-05-09, 2024-01-21), so production and irradiance were misaligned with solar geometry and with each other | Per-day clock offsets per source and clock group, estimated on clear days (irradiance vs clear-sky POA) and by DC/irradiance cross-correlation; residual ≤ 2 min (D-19) | pending |
| F-14 | Spikes up to 40,825 kW and a 32,766 sentinel stayed in the target (notebook 04: 1-min maximum 12,444 kW); the −270 fault code of the temperature sensors was only flagged; seven communication freezes repeated the last value for hours, including 1,296 kW at night; inverter-1 DC read 0 for 7,038 minutes in March 2023 | Explicit policy `configs/cleaning_policy.yaml` (R1–R6), every change logged and reconciled (T-23) | pending |
| F-17 | Grid curtailment (setpoint < 100 %) and inverter stops were not handled; quality flags were used as model features | Flagged, never erased; excluded from the training loss and from past-day features, reported in a secondary evaluation (D-22); quality columns blocked as features (T-01) | pending |
| F-22 | Partial inverter outages (one inverter at ~75 % of its twin for hours or days, status code 5/6) were not detected, so the models learned an equipment fault as weather; a drifting curtailment setpoint (Feb 2022) and the pyranometer's intra-day faults were also mis-handled (F-20, F-21) | New flag F3, used like F1/F2 (D-22); F2 ignores non-integer drifts; R7 per hour (stage 3b) | pending |
| F-18 | From 2025-02-26 the export switches to a second source: 15-min instantaneous snapshots held for 15 min (≈ 7.5 min lag, hourly means from 4 stale samples) on its own clock, mixed hour by hour with the old 1-min source during the transition; the old pipeline used it as 1-min data | Source detected per tag-hour; polled tags rebuilt at their poll instants and interpolated; each source corrected with its own clock | pending (affects ~35 % of the test period) |

## Old vs new results

To be filled in stage 11.

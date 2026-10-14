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

## Old vs new results

To be filled in stage 11.

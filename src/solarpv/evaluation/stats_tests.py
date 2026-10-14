"""Diebold-Mariano test with Newey-West (Bartlett, 24 lags), HLN correction and Bonferroni
(stage 8, decision D-14, PROJECT_MASTER.md §6.9).

The uncorrected statistic must also be reported; it reproduces the old dm_tests.csv (finding F-04).
Acceptance: test T-03 against a hand-computed example.
"""

"""Stage 6: build the feature matrix defined in contract.py from the hourly UTC data.

Spec: PROJECT_MASTER.md §6.6. Horizon-aware rolling (D-04): for D+h,
    rolling_<stat>_<N>d_same_tod = stat(shift(h*24), ..., shift((h+N-1)*24))
Imputation uses only the past (expanding median, then forward fill; never backward fill).
Acceptance: tests T-09, T-10, T-18.
"""

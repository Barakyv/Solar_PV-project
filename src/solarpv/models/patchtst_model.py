"""PatchTST via neuralforecast 3.1.7: patch_len 16, stride 8, the target series only (L-19;
PROJECT_MASTER.md §6.11). Code in W4.

Origins and horizons come from `solarpv.models.sequence_data` (W3): one origin per local day,
from two days before the evaluated split, so every hour of the split is forecast at every horizon
(245 test origins, not the 243 of v6; M15, X19). The input size is fixed in W4. Before gate G-1
only smoke and dev runs.
"""

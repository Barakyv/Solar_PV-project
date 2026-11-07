"""BiLSTM-Opt encoder-decoder with quantile loss (PROJECT_MASTER.md §6.11). Code in W4.

Inputs (D-28): the datasets of `solarpv.models.sequence_data` (W3): a 72-hour encoder, a decoder
over the three local days from T0 (D-25), and the part of the stage-7 inputs known at T0.
Horizons come from the dataset (the local day of each valid hour), never from step numbers; the
original's `valid_ts = window_end - (DECODER_LEN - 1 - target_step) h` is not used (M14, X19).
Everything runs on the college machine (docs/stage9_plan.md §2); before gate G-1 only smoke and
dev runs.
"""

"""BiLSTM-Opt encoder-decoder with quantile loss (PROJECT_MASTER.md §6.11).

Before gate G-1: smoke test only (2 epochs on a subset). Final training on Colab in stage 10.
Alignment: valid_ts = window_end - (DECODER_LEN - 1 - target_step) h (test T-02).
"""

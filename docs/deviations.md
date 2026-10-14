# Deviations log

Record here any result outside its tolerance, any assumption (PROJECT_MASTER.md §6.13) that could not be
verified and is kept as a stated limitation, and any change made after gate G-1.

| Date | Stage | What | Why | Impact |
|---|---|---|---|---|
| 2026-10-14 | 8 | Fixture (T-15): R² is 0.0002–0.0004 above the old values; direct nRMSE up to 0.0012 pp off | Old evaluate.py probably took y_true from the feature file (Q-06); the new code will do the same from stage 6 | None on conclusions; within tolerance |
| 2026-10-14 | 8 | Fixture CIs differ from the old ones by up to 0.27 pp | Old bootstrap RNG/implementation unknown; new one is circular blocks of 24 daytime rows, 1,000 reps, seed 42 | Within the ±0.3 pp tolerance |
| 2026-10-14 | 8 | Seasonal split: one LSTM hour counted as Fall instead of Winter (571/857 vs 570/858) | New code assigns seasons by local calendar month; the old rule is unknown | ≤0.03 pp on seasonal nRMSE |

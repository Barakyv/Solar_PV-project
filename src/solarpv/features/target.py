"""Stage 6: targets (PROJECT_MASTER.md §6.5, fix for finding F-06).

    theo_t = mean clear-sky irradiance over hour t * 3449 / 1000      (GHI or POA, decision D-11)
    k_t    = AC_t / theo_t  if theo_t > 50 kW, else 0 with k_valid = False
No clipping at 1.2; k > 1.5 is flagged as a data issue. Inversion AC_t = k_t * theo_t is exact.
Also writes configs/derived_constants.yaml with mean_daytime_k (train, daytime mask).
Acceptance: tests T-11, T-13.
"""

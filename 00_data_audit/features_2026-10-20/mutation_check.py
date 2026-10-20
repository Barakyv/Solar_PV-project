"""Do the stage-6 tests catch a leaky feature construction? (docs/stage6_plan.md)

Plants one known fault at a time in solarpv.features.build and runs the matching tests
(tests/unit/test_features_build.py) against it. Every line must read "caught".
  (a) rolling that starts one day too late back (the F-07 bug of the original: for D+2 it
      would include the last day before the target day, unknown at T0)
  (b) imputation with backward fill (the next hour's value fills a gap, as v4 did for NWP)
  (b2) imputation from the same hour on later days (a future-looking median)
  (c) lags without the DST guard (a plain 24-hour shift)

    python 00_data_audit/features_2026-10-20/mutation_check.py
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "unit"))
import test_features_build as T  # noqa: E402

from solarpv.features import build as B  # noqa: E402

ORIGINAL = {name: getattr(B, name) for name in ("rolling_features", "impute_same_hour",
                                                "past_value")}


@contextmanager
def patched(name, replacement):
    setattr(B, name, replacement)
    try:
        yield
    finally:
        setattr(B, name, ORIGINAL[name])


def rolling_shift_from_one(sources, horizon, t0):
    """F-07: days 1..N back for every horizon, as in the original v4 code."""
    out = {}
    for name, col in B.ROLLING_COLUMNS.items():
        for n in B.ROLLING_WINDOWS_DAYS:
            stack = np.vstack([sources[col].shift(24 * k).to_numpy() for k in range(1, 1 + n)])
            out[f"{name}_rolling_mean_{n}d_same_tod"] = B._nanmean(stack)
            out[f"{name}_rolling_std_{n}d_same_tod"] = B._nanstd(stack)
    return out


def impute_with_bfill(frame, window_days=B.IMPUTE_WINDOW_DAYS):
    return frame.bfill(), frame.isna().mean()


def impute_from_later_days(frame, window_days=B.IMPUTE_WINDOW_DAYS):
    hour = frame.index.hour
    later = frame[::-1].groupby(hour[::-1]).transform(
        lambda s: s.shift(1).rolling(window_days, min_periods=1).median())[::-1]
    return frame.fillna(later), frame.isna().mean()


def past_value_without_guard(series, days, t0):
    return series.shift(24 * days).to_numpy(dtype=float)


def expect_fail(name, fn) -> bool:
    try:
        fn()
    except AssertionError:
        print(f"caught      {name}")
        return True
    print(f"NOT CAUGHT  {name}")
    return False


def main() -> int:
    results = []
    with patched("rolling_features", rolling_shift_from_one):
        results.append(expect_fail("(a) rolling from shift 1 day at D+2 (F-07)",
                                   lambda: T.test_nothing_after_issuance_reaches_the_forecast(
                                       "fall back", 2)))
    with patched("impute_same_hour", impute_with_bfill):
        results.append(expect_fail("(b) imputation with backward fill",
                                   lambda: T.test_nothing_after_issuance_reaches_the_forecast(
                                       "spring forward", 1)))
    with patched("impute_same_hour", impute_from_later_days):
        results.append(expect_fail("(b2) imputation from the same hour on later days",
                                   lambda: T.test_nothing_after_issuance_reaches_the_forecast(
                                       "spring forward", 3)))
    with patched("past_value", past_value_without_guard):
        results.append(expect_fail("(c) lags without the DST guard (25th hour)",
                                   T.test_the_25th_hour_of_the_fall_back_day_is_guarded))
        results.append(expect_fail("(c) lags without the DST guard (perturbation, D+1)",
                                   lambda: T.test_nothing_after_issuance_reaches_the_forecast(
                                       "fall back", 1)))
    # sanity: the unmodified code passes the same tests
    T.test_nothing_after_issuance_reaches_the_forecast("fall back", 2)
    T.test_the_25th_hour_of_the_fall_back_day_is_guarded()
    print("unmodified code passes")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

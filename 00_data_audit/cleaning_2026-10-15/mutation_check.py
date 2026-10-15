"""Do the stage-5 contract tests catch a wrong cleaning? (docs/stage5_test_plan.md)

Plants one known fault at a time in a copy of the real stage-5 output and runs the matching
contract test on it. Every line must read "caught".

    python 00_data_audit/cleaning_2026-10-15/mutation_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "contracts"))
import test_clean_output as T  # noqa: E402

from solarpv.config.site import load_site  # noqa: E402
from solarpv.data.clock import (  # noqa: E402
    Orientation,
    clear_sky_components,
    poa_from_components,
)

AC = "s01_inv1_AC_kW"


def expect_fail(name, fn) -> bool:
    try:
        fn()
    except AssertionError:
        print(f"caught      {name}")
        return True
    print(f"NOT CAUGHT  {name}")
    return False


def main() -> int:
    before = pd.read_parquet(T.BEFORE)
    after = pd.read_parquet(T.MINUTE)
    site = load_site()
    comp = clear_sky_components(after.index, site)
    sun = (comp["elev"].to_numpy(),
           poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth)))
    elev = sun[0]
    rng = np.random.default_rng(0)

    def pick(mask):
        return int(rng.choice(np.flatnonzero(np.asarray(mask))))

    def mutate(i, col, value, code=None):
        a = after.copy()
        a.iloc[i, a.columns.get_loc(col)] = value
        if code is not None:
            a.iloc[i, a.columns.get_loc(f"rule_{col}")] = code
        return a

    untouched_day = (after[f"rule_{AC}"].to_numpy() == 0) & (elev > 20)
    i = pick(untouched_day)
    silent = mutate(i, AC, before[AC].iloc[i] * 0.9)
    i = pick(untouched_day & (before[AC].to_numpy() > 100))
    zeroed = mutate(i, AC, 0.0, code=4)
    i = int(np.flatnonzero((after[f"rule_{AC}"].to_numpy() == 1)
                           & (before[AC].to_numpy() > 1897))[0])
    spike = mutate(i, AC, before[AC].iloc[i], code=0)
    idx = np.flatnonzero(after["rule_irradiance"].to_numpy() == 3)[:10]
    frozen = after.copy()
    frozen.iloc[idx, frozen.columns.get_loc("irradiance")] = before["irradiance"].iloc[idx]
    frozen.iloc[idx, frozen.columns.get_loc("rule_irradiance")] = 0
    i = pick(after["op_limited_flag"].to_numpy() & (after[f"rule_{AC}"].to_numpy() == 0)
             & (before[AC].to_numpy() > 50))
    flag_changed = mutate(i, AC, 0.0)
    unflagged = after.copy()
    unflagged.iloc[pick(after["curtail_flag"]), unflagged.columns.get_loc("curtail_flag")] = False
    manifest = json.loads((T.STAGE / "cleaning_manifest.json").read_text(encoding="utf-8"))
    manifest["rule_minutes"][next(iter(manifest["rule_minutes"]))] += 1

    checks = [
        ("value changed without a rule",
         lambda: T.test_every_change_is_explained_and_every_rule_hit_is_justified(
             AC, before, silent, sun)),
        ("daytime production zeroed, labelled as night rule",
         lambda: T.test_every_change_is_explained_and_every_rule_hit_is_justified(
             AC, before, zeroed, sun)),
        ("T-16 on the same fault", lambda: T.test_real_daytime_production_is_never_removed(
            AC, before, zeroed, sun)),
        ("spike left in place (reconciliation)",
         lambda: T.test_every_change_is_explained_and_every_rule_hit_is_justified(
             AC, before, spike, sun)),
        ("spike left in place (ranges)", lambda: T.test_values_are_inside_the_policy_ranges(spike)),
        ("frozen run partly kept",
         lambda: T.test_every_change_is_explained_and_every_rule_hit_is_justified(
             "irradiance", before, frozen, sun)),
        ("flag changed a value", lambda: T.test_flags_never_change_values(before, flag_changed)),
        ("curtailment minute not flagged", lambda: T.test_flags_match_their_definition(
            unflagged, sun)),
        ("manifest count off by one", lambda: T.test_log_and_manifest_match_the_output(
            after, manifest)),
    ]
    results = [expect_fail(name, fn) for name, fn in checks]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

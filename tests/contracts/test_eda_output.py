"""T-30: stage-3 EDA outputs (docs/stage3_eda_plan.md).

Checks that every planned figure and table exists, that the tables agree with the stage-5
manifest and with physics, and that the outputs were made from the current inputs.
Skipped when the EDA has not been run.
"""

import json

import pandas as pd
import pytest

from solarpv.config.paths import OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import sha256, text_sha256

EDA = OUTPUTS_DIR / "03_eda"
MANIFEST = EDA / "eda_manifest.json"
FIGURES = ["fig01_overview", "fig02_time_bug", "fig03_clock", "fig04_polled", "fig05_cleaning",
           "fig06_diurnal", "fig07_power_irradiance", "fig08_variability", "fig09_operations",
           "fig10_sensors"]
TABLES = ["table_splits", "table_time_bug", "table_physics", "table_variability",
          "table_sensors"]

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="stage 3 not run")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def table(name):
    return pd.read_csv(EDA / "tables" / f"{name}.csv")


def test_every_planned_output_exists(manifest):
    assert sorted(manifest["figures"]) == sorted(f"{f}.png" for f in FIGURES)
    assert sorted(manifest["tables"]) == sorted(f"{t}.csv" for t in TABLES)
    for f in FIGURES:
        assert (EDA / "figures" / f"{f}.png").stat().st_size > 20_000, f
    plan = (REPO_ROOT / "docs" / "stage3_eda_plan.md").read_text(encoding="utf-8")
    for name in FIGURES + TABLES:
        assert name in plan, f"{name} not in the stage-3 plan"


def test_outputs_were_made_from_the_current_inputs(manifest):
    for rel, digest in manifest["inputs_sha256"].items():
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.skip(f"input not present: {rel}")
        current = text_sha256(path) if path.suffix in (".csv", ".yaml") else sha256(path)
        assert current == digest, f"{rel} changed since the EDA was run"


def test_split_table_matches_stage5():
    t = table("table_splits").set_index("split")
    clean = json.loads((OUTPUTS_DIR / "05_cleaning" / "cleaning_manifest.json")
                       .read_text(encoding="utf-8"))["hours_by_split"]
    assert t.loc[["train", "val", "test"], "hours"].sum() == t.loc["all", "hours"] == 32_856
    assert t.loc["test", "daytime_hours"] == 2_580
    for s in ("train", "val", "test"):
        assert t.loc[s, "daytime_hours"] == clean[s]["daytime_hours"]
        assert t.loc[s, "daytime_op_limited"] == clean[s]["op_limited_hours"]
        assert t.loc[s, "daytime_hours"] - t.loc[s, "daytime_valid_target"] \
            == clean[s]["target_nan_hours"]
    assert t.loc["train", "daytime_polled"] == t.loc["val", "daytime_polled"] == 0


def test_time_bug_is_quantified():
    t = table("table_time_bug")
    orig = t[t["reading"].str.startswith("original")]["median_centroid_minus_solar_noon_min"]
    new = t[t["reading"].str.startswith("rebuild")]["median_centroid_minus_solar_noon_min"]
    assert orig.between(100, 200).all()  # 2 h (winter) / 3 h (summer) minus the clock drift
    assert new.abs().max() <= 5


def test_physics_is_plausible():
    coeff = table("table_physics").set_index("quantity").iloc[0]["value"]
    assert -0.6 <= coeff <= -0.2  # crystalline silicon: about -0.3 to -0.5 %/K


def test_irradiance_sensor_is_consistent_and_r7_worked():
    t = table("table_sensors")
    assert t["irradiance_over_clearsky_p90"].between(1.0, 1.3).all()
    after = t["pyranometer_over_irradiance_after_cleaning"].dropna()
    assert after.between(0.9, 1.1).all()
    assert (t["pyranometer_over_irradiance_median"] < 0.8).sum() >= 7  # the fault is visible

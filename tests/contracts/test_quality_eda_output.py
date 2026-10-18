"""T-31: stage-3b data-quality analysis (docs/stage3b_quality_eda_plan.md).

Checks that every planned analysis was produced from the current inputs, that the notebook was
built from the current manifest with a markdown note under every code cell, and that the claims
the notes make (the evidence for each cleaning threshold) still hold. Skipped when stage 3b has
not been run.
"""

import json

import pytest

from solarpv.config.paths import OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import sha256, text_sha256

OUT = OUTPUTS_DIR / "03b_quality"
MANIFEST = OUT / "quality_manifest.json"
NOTEBOOK = REPO_ROOT / "notebooks" / "02b_data_quality.ipynb"
PLAN = REPO_ROOT / "docs" / "stage3b_quality_eda_plan.md"
ANALYSES = [
    "q01_inventory", "q02_held_values", "q03_sampling_regime", "q04_distributions",
    "q05_range_limits", "q06_range_margin", "q07_night_offset", "q08_frozen_runs",
    "q09_freeze_events", "q10_freeze_resume", "q11_night", "q12_dc_consistency",
    "q13_scale_fault", "q14_downtime", "q15_curtailment", "q16_derate", "q17_hourly_thresholds",
    "q18_changes", "q19_distributions", "q20_energy", "q21_missing_after", "q22_residual",
    "q23_inverter_ceiling", "q24_temperature_sensors", "q25_correlations", "q26_lags",
    "q27_sky_conditions", "q28_weekday", "q29_long_term", "q30_unused_tags", "q31_splits",
]
TABLE_ONLY = {"q01_inventory"}
TEMPERATURES = ("ambient_temp", "site_ambient_temp", "mod_temp1", "mod_temp2")
POWER = ("s01_inv1_AC_kW", "s01_inv2_AC_kW", "s01_inv1_DC_kW", "s01_inv2_DC_kW")

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="stage 3b not run")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def f(manifest):
    return {name.split("_")[0]: entry["facts"] for name, entry in manifest["analyses"].items()}


def test_every_planned_analysis_exists(manifest):
    assert list(manifest["analyses"]) == ANALYSES
    plan = PLAN.read_text(encoding="utf-8")
    for name in ANALYSES:
        entry = manifest["analyses"][name]
        assert (OUT / "tables" / entry["table"]).exists(), name
        if name not in TABLE_ONLY:
            assert (OUT / "figures" / entry["figure"]).stat().st_size > 20_000, name
        assert f"| {name[:3].upper()} |" in plan, f"{name} not in the plan"


def test_outputs_were_made_from_the_current_inputs(manifest):
    for rel, digest in manifest["inputs_sha256"].items():
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.skip(f"input not present: {rel}")
        current = text_sha256(path) if path.suffix in (".csv", ".yaml", ".json") else sha256(path)
        assert current == digest, f"{rel} changed since stage 3b was run"


def test_notebook_is_current_and_every_code_cell_has_a_note():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert nb.metadata["solarpv"]["quality_manifest_sha256"] == text_sha256(MANIFEST), \
        "run: python scripts/make_quality_notebook.py"
    cells = nb.cells
    code = [i for i, c in enumerate(cells) if c.cell_type == "code"]
    assert len(code) == len(ANALYSES) + 1  # one setup cell
    for i in code:
        assert i + 1 < len(cells) and cells[i + 1].cell_type == "markdown", f"cell {i}"
        assert cells[i].outputs, f"cell {i} was not executed"
        assert not any(o.get("output_type") == "error" for o in cells[i].outputs), f"cell {i}"
    for name in ANALYSES:
        assert any(c.cell_type == "code" and f"Q.{name}(d)" in c.source for c in cells), name


# the claims of the notes: Part A ---------------------------------------------------------

def test_part_a_claims(f):
    assert f["q01"]["n_tags"] == 27 and f["q01"]["raw_nan_total"] == 0
    assert f["q03"]["ambiguous_share"] < 0.02


# Part B: the evidence for each threshold ---------------------------------------------------

def test_r1_limits_sit_in_empty_regions(f):
    c = f["q05"]
    assert all(v <= 0.9 for v in c["power_largest_inside_over_upper"].values())
    assert all(v >= 2 for v in c["power_smallest_above_over_upper"].values() if v is not None)
    assert all(v >= -20 + 15 for v in c["temperature_smallest_inside_not_next"].values())
    assert f["q06"]["share_beyond_normal"]["temperatures"] >= 0.2


def test_r2_and_r3_evidence(f):
    assert f["q07"]["irradiance_min_train"] >= -20
    assert f["q07"]["irradiance_negative_share_sun_above_10"] == 0
    for col, ratio in f["q08"]["ratio_15_60"].items():
        if col not in ("ambient_temp", "site_ambient_temp"):
            assert ratio <= 1.3, col
    for col in ("ambient_temp", "site_ambient_temp"):
        assert f["q08"]["ratio_30_120"][col] <= 1.3, col
    assert f["q09"]["share_multi_column_train"] >= 0.8
    assert f["q10"]["blend_share_train"] >= 0.5


def test_night_and_dc_evidence(f):
    c = f["q11"]
    assert all(v == 0 for v in c["max_not_frozen_below_minus6"].values())
    assert c["r5_share_in_or_next_to_freeze"] >= 0.99
    assert c["r4_candidates_in_twilight_share"] >= 0.9
    d = f["q12"]
    assert d["r6_top_month"]["inv1"] == "2023-03" and d["r6_share_top_month"]["inv1"] >= 0.95
    assert all(0.95 <= v <= 1.0 for v in d["efficiency_median"].values())
    assert all(v < 0.01 for v in d["share_ac_above_dc_stable"].values())


def test_r7_two_populations(f):
    c = f["q13"]
    assert len(c["train_gap_days"]) <= 1  # an intra-day switch, explained in the note
    assert abs(c["train_faulty_median"] - 0.75) <= 0.03
    assert abs(c["train_good_median"] - 1.0) <= 0.01


def test_flag_evidence(f):
    assert all(v <= 0.005 for v in f["q14"]["share_20_to_100"].values())
    c = f["q15"]
    assert c["relative_output_by_setpoint_train"]["50"] < 0.9
    assert all(v >= 0.95 for v in c["relative_output_drift_days"].values())
    assert c["train_transition_run_max"] <= 10 < min(c["train_drift_runs"])
    assert all(abs(v - 1) <= 0.05 for v in c["mode_relative_output"].values())
    d = f["q16"]
    assert d["train_valley_share"] < 0.01 and d["train_low_stable_share"] >= 0.5
    assert d["ratio_by_state"]["inv1 code 5 run >= 30 min"] < 0.85
    assert abs(d["ratio_by_state"]["inv1 code 1 all"] - 1) <= 0.05
    clean = json.loads((OUTPUTS_DIR / "05_cleaning" / "cleaning_manifest.json")
                       .read_text(encoding="utf-8"))["hours_by_split"]
    assert d["hours_by_split"] == {s: clean[s]["hours_with_derate"] for s in d["hours_by_split"]}
    h = f["q17"]
    assert h["share_partial"] <= 0.01
    assert abs(h["op_hours_at"]["0.02"] - h["op_hours_at"]["0.1"]) <= 0.1 * h["op_hours_at"]["0.1"]


# Part C: after cleaning -------------------------------------------------------------------

def test_after_cleaning_claims(f):
    assert f["q18"]["matches_cleaning_log"]
    shift = f["q19"]["max_bulk_shift_pct"]
    for col in ("s01_inv1_AC_kW", "s01_inv2_AC_kW", "irradiance", "mod_temp1", "mod_temp2",
                "ambient_temp"):
        assert shift[col] < 1, col
    assert all(v == 0 for v in f["q19"]["outside_range_after"].values())
    assert set(f["q20"]["days_changed_by_rule"]) <= {"R1_range", "R3_frozen"}
    c = f["q22"]
    assert all(v == 0 for v in c["frozen_daytime_minutes_left"].values())
    assert c["csi_above_1_2_low_sun_share"] >= 0.9
    assert c["imbalance_hours"] * 5 < c["imbalance_hours_without_f3"]


# Part D: for modelling --------------------------------------------------------------------

def test_modelling_claims(f):
    assert f["q24"]["pairs"]["ambient_temp - site_ambient_temp"]["pearson_r"] >= 0.95
    assert f["q28"]["kruskal_p"] > 0.05
    assert f["q29"]["sensor_vs_clear_sky_p"] > 0.05
    assert f["q30"]["precipitation_positive_minutes"] == 0
    assert f["q31"]["test_hours_JJA"] == 0

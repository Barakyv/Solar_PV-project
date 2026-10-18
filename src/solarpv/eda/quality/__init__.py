"""Stage 3b: data quality before and after cleaning (docs/stage3b_quality_eda_plan.md).

    python -m solarpv.eda.quality

Every analysis takes the loaded inputs (data.load()) and returns a Result: a table, a figure
and the facts its notebook note relies on. run() writes them to outputs/03b_quality/ with a
manifest; notebooks/02b_data_quality.ipynb calls the same functions cell by cell.
EDA changes no data.
"""

import json

import matplotlib.pyplot as plt

from solarpv.config.paths import OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import git_commit, sha256, text_sha256
from solarpv.eda.quality.common import Result, py
from solarpv.eda.quality.data import INPUT_FILES, TEXT_INPUTS, Data, load
from solarpv.eda.quality.part_a import (
    q01_inventory,
    q02_held_values,
    q03_sampling_regime,
    q04_distributions,
)
from solarpv.eda.quality.part_b import (
    q05_range_limits,
    q06_range_margin,
    q07_night_offset,
    q08_frozen_runs,
    q09_freeze_events,
    q10_freeze_resume,
    q11_night,
    q12_dc_consistency,
    q13_scale_fault,
    q14_downtime,
    q15_curtailment,
    q16_derate,
    q17_hourly_thresholds,
)
from solarpv.eda.quality.part_c import (
    q18_changes,
    q19_distributions,
    q20_energy,
    q21_missing_after,
    q22_residual,
)
from solarpv.eda.quality.part_d import (
    EDA_MANIFEST,
    q23_inverter_ceiling,
    q24_temperature_sensors,
    q25_correlations,
    q26_lags,
    q27_sky_conditions,
    q28_weekday,
    q29_long_term,
    q30_unused_tags,
    q31_splits,
)

OUT = OUTPUTS_DIR / "03b_quality"
FIG, TAB = OUT / "figures", OUT / "tables"
MANIFEST = OUT / "quality_manifest.json"
DPI = 130

ANALYSES = (
    q01_inventory, q02_held_values, q03_sampling_regime, q04_distributions,
    q05_range_limits, q06_range_margin, q07_night_offset, q08_frozen_runs, q09_freeze_events,
    q10_freeze_resume, q11_night, q12_dc_consistency, q13_scale_fault, q14_downtime,
    q15_curtailment, q16_derate, q17_hourly_thresholds,
    q18_changes, q19_distributions, q20_energy, q21_missing_after, q22_residual,
    q23_inverter_ceiling, q24_temperature_sensors, q25_correlations, q26_lags,
    q27_sky_conditions, q28_weekday, q29_long_term, q30_unused_tags, q31_splits,
)

__all__ = ["ANALYSES", "MANIFEST", "OUT", "Data", "Result", "load", "run", "save"]


def save(result: Result) -> dict:
    """Write the table and figure of one analysis; return its manifest entry."""
    entry = {"title": result.title, "facts": py(result.facts)}
    if result.table is not None:
        TAB.mkdir(parents=True, exist_ok=True)
        path = TAB / f"{result.name}.csv"
        result.table.to_csv(path, index=False, lineterminator="\n")
        entry["table"] = path.name
    if result.figure is not None:
        FIG.mkdir(parents=True, exist_ok=True)
        path = FIG / f"{result.name}.png"
        result.figure.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(result.figure)
        entry["figure"] = path.name
    return entry


def input_hashes() -> dict:
    files = {p: sha256(p) for p in INPUT_FILES}
    files.update({p: text_sha256(p) for p in (*TEXT_INPUTS, EDA_MANIFEST)})
    return {p.relative_to(REPO_ROOT).as_posix(): h for p, h in files.items()}


def run(data: Data | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    data = data or load()
    analyses = {}
    for fn in ANALYSES:
        result = fn(data)
        analyses[result.name] = save(result)
        print(f"{result.name}: done")
    manifest = {"stage": "03b_quality", "git_commit": git_commit(),
                "inputs_sha256": input_hashes(), "analyses": analyses}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return manifest

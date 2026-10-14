"""Stage 4: one consistent true-UTC minute grid, then hourly means (D-19, F-15, F-18).

    python -m solarpv.data.preprocess

On the 1-minute grid of stage 2 (stamped time):
  1. the source of every tag-minute, and reconstruction of polled tags (resolution.py, F-18);
  2. clock correction per source and clock group (clock.py, configs/clock_offsets.csv, D-19);
  3. site_total_AC_kW = inv1 + inv2, minute by minute (NaN if either is missing).
Hourly means (PROJECT_MASTER.md §6.4): left-labelled, valid if at least 45 of the 60 minutes are
present, else NaN. Cleaning (stage 5) works on the corrected minute data and re-aggregates with
hourly_mean(); clear-sky and geometry are averaged in stage 6 over the same minutes as the target.

Outputs in data/processed/04_preprocess/:
  scada_1min_true_utc.parquet  corrected minute data, plus the source of three reference tags
                               (src_meteo, src_inv1, src_inv2: 0 minute, 1 polled, -1 none)
  scada_1H_utc.parquet         hourly means (32,856 rows) and valid-minute counts (n_<tag>)
  preprocess_manifest.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW, HOURLY_MIN_COVERAGE, N_HOURS_EXPECTED
from solarpv.config.paths import CONFIGS_DIR, stage_dir
from solarpv.config.provenance import git_commit, sha256
from solarpv.config.site import load_site
from solarpv.data.clock import MIN_ELEV, apply_correction, clear_sky_components, load_offsets
from solarpv.data.resolution import harmonise_sources, local_day
from solarpv.evaluation.daytime import daytime_mask

OFFSETS_FILE = CONFIGS_DIR / "clock_offsets.csv"
TARGET = "site_total_AC_kW"
AC_COLUMNS = ("s01_inv1_AC_kW", "s01_inv2_AC_kW")
SOURCE_REFERENCES = {"src_meteo": "irradiance", "src_inv1": "s01_inv1_AC_kW",
                     "src_inv2": "s01_inv2_AC_kW"}
COUNTED = (TARGET, *AC_COLUMNS, "irradiance", "mod_temp1", "mod_temp2", "ambient_temp")


def hourly_mean(df: pd.DataFrame, min_valid: int = HOURLY_MIN_COVERAGE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-labelled hourly means and valid-minute counts; hours with < min_valid minutes -> NaN."""
    g = df.resample("1h", label="left", closed="left")
    counts = g.count()
    return g.mean().where(counts >= min_valid), counts


def correct_minutes(raw: pd.DataFrame, offsets: pd.DataFrame, site=None) -> tuple[
        pd.DataFrame, pd.DataFrame, object]:
    """Steps 1-3 above. Returns (corrected minute data, source codes, polled era)."""
    comp = clear_sky_components(raw.index, site or load_site())
    data, source, era = harmonise_sources(raw, comp["elev"].to_numpy() > MIN_ELEV)
    days = pd.DatetimeIndex(sorted(set(local_day(raw.index))))
    needed = ["minute_meteo", "minute_inverter"]
    if offsets[needed].reindex(days).isna().any().any():
        raise ValueError("clock offsets do not cover every local day of the data")
    if era is not None:
        polled_days = days[days >= era.first_day]
        polled = offsets[["polled_meteo", "polled_inverter"]].reindex(polled_days)
        if polled.isna().any().any():
            raise ValueError("polled-source offsets missing: rerun scripts/estimate_clock_drift.py")
    corrected = apply_correction(data, offsets, source)
    refs = list(SOURCE_REFERENCES.values())
    codes = source[refs].replace({"minute": 0.0, "polled": 1.0}).astype(float)
    moved = apply_correction(codes, offsets, source[refs])
    src = pd.DataFrame({name: moved[ref].fillna(-1).astype("int8")
                        for name, ref in SOURCE_REFERENCES.items()}, index=raw.index)
    corrected[TARGET] = corrected[list(AC_COLUMNS)].sum(axis=1, min_count=len(AC_COLUMNS))
    return corrected, src, era


def preprocess(in_dir: Path | None = None, out_dir: Path | None = None,
               offsets_file: Path = OFFSETS_FILE) -> pd.DataFrame:
    in_dir = in_dir or stage_dir("01", "ingest")
    out_dir = out_dir or stage_dir("04", "preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file = in_dir / "scada_1min_utc.parquet"
    raw = pd.read_parquet(raw_file)
    site = load_site()
    corrected, src, era = correct_minutes(raw, load_offsets(offsets_file), site)
    lost = (corrected[raw.columns].isna().sum() - raw.isna().sum()).astype(int)
    del raw

    minute_file = out_dir / "scada_1min_true_utc.parquet"
    corrected.join(src).to_parquet(minute_file)
    hourly, counts = hourly_mean(corrected)
    for col in COUNTED:
        hourly[f"n_{col}"] = counts[col].astype("int16")
    hourly.index.name = "timestamp_utc"
    if len(hourly) != N_HOURS_EXPECTED:
        raise ValueError(f"{len(hourly)} hourly rows, expected {N_HOURS_EXPECTED}")
    hourly_file = out_dir / "scada_1H_utc.parquet"
    hourly.to_parquet(hourly_file)

    day = daytime_mask(hourly.index, site)
    total = hourly[TARGET]
    manifest = {
        "stage": "04_preprocess",
        "git_commit": git_commit(),
        "input": {"file": str(raw_file.name), "sha256": sha256(raw_file)},
        "clock_offsets": {"file": offsets_file.name, "sha256": sha256(offsets_file)},
        "polled_source": None if era is None else {"first_day": str(era.first_day.date()),
                                                   "era_start": str(era.era_start.date())},
        "rows_1min": len(corrected),
        "rows_1H": len(hourly),
        "hourly_rule": f"left-labelled mean, valid if >= {HOURLY_MIN_COVERAGE}/60 minutes",
        "minutes_lost_by_clock_correction": {k: int(v) for k, v in lost.items() if v},
        "hourly_nan": {c: int(hourly[c].isna().sum()) for c in COUNTED},
        "hourly_nan_daytime": {c: int(hourly.loc[day, c].isna().sum()) for c in COUNTED},
        "site_total_AC_kW": {
            "max_hourly": round(float(total.max()), 1),
            "p99_99_hourly": round(float(total.quantile(0.9999)), 1),
            "hours_above_capacity": int((total > AC_CAPACITY_KW).sum()),
            "daytime_hours_zero": int(((total <= 0) & day).sum()),
            "daytime_hours": int(day.sum()),
        },
        "outputs": {p.name: sha256(p) for p in (minute_file, hourly_file)},
    }
    (out_dir / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                      encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("polled_source", "rows_1H", "hourly_nan_daytime",
                                               "site_total_AC_kW")}, indent=2))
    return hourly


def main() -> int:
    preprocess()
    return 0


if __name__ == "__main__":
    sys.exit(main())


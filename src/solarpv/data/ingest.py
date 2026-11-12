"""Stage 2: merge the 45 monthly raw CSVs, map tags to canonical names, convert to true UTC.

    python -m solarpv.data.ingest

Spec: PROJECT_MASTER.md §6.1 (tags) and §6.2 (time). Fix for finding F-01: the raw t_stamp is
Israel local time with DST, so it is localised to Asia/Jerusalem and converted to UTC.
October files repeat local 01:00-01:59 as two consecutive blocks with different values; they are
resolved by file order (ambiguous="infer"). March files have no local 02:00-02:59.

Values are not changed here; anomalies are only counted (cleaning rules belong to stage 5).
Outputs in data/processed/01_ingest/: scada_1min_utc.parquet, ingest_manifest.json,
ingest_qa.csv (per-column counts), tag_map.csv.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from solarpv.config.cli import no_options
from solarpv.config.constants import (
    AC_CAPACITY_KW,
    DATA_END_UTC,
    DATA_START_UTC,
    N_MINUTES_EXPECTED,
    TIMEZONE_LOCAL,
)
from solarpv.config.paths import AUDIT_DIR, RAW_DIR, stage_dir

TIME_COLUMN = "t_stamp"
RAW_FILE_PATTERN = re.compile(r"^\d{4}\.CSV$", re.IGNORECASE)
N_RAW_FILES = 45

TAG_MAP: dict[str, str] = {
    "Sites/PVS/Ktora/IEC/Mode": "iec_mode",
    "Sites/PVS/Ktora/IEC/SP_W": "iec_sp_w",
    "Sites/PVS/Ktora/IEC/SP_Q": "iec_sp_q",
    "Sites/PVS/Ktora/IEC/SP_V": "iec_sp_v",
    "Sites/PVS/Ktora/IEC/SP_PF": "iec_sp_pf",
    "Sites/PVS/Ktora/meteo/amb_temp/st02_amb_temp/eu": "ambient_temp",
    "Sites/PVS/Ktora/meteo/irrad/st01_irrad/eu": "irradiance",  # in-plane (POA) sensor, F-13
    "Sites/PVS/Ktora/meteo/mod_temp/st01_mod_temp1/eu": "mod_temp1",
    "Sites/PVS/Ktora/meteo/mod_temp/st01_mod_temp2/eu": "mod_temp2",
    "Sites/PVS/Ktora/meteo/precipitation/st05_precipitation/eu": "precipitation",
    "Sites/PVS/Ktora/meteo/soil/soil_soil/eu": "soil",
    "Sites/PVS/Ktora/meteo/soil/soiling01_soil/eu": "soiling01",
    "Sites/PVS/Ktora/meteo/soil/soiling02_soil/eu": "soiling02",
    "Sites/PVS/Ktora/meteo/wind_direction/st05_wind_direction/eu": "wind_direction",
    "Sites/PVS/Ktora/meteo/wind_speed/st05_wind_speed/eu": "wind_speed",
    "sites/pvs/ktora/stations/st1/inverters/inv1/AC_kW": "s01_inv1_AC_kW",
    "sites/pvs/ktora/stations/st1/inverters/inv1/AC_kVar": "s01_inv1_AC_kVar",
    "sites/pvs/ktora/stations/st1/inverters/inv1/DC_kW": "s01_inv1_DC_kW",
    "sites/pvs/ktora/stations/st1/inverters/inv1/PF": "s01_inv1_PF",
    "sites/pvs/ktora/stations/st1/inverters/inv2/AC_kW": "s01_inv2_AC_kW",
    "sites/pvs/ktora/stations/st1/inverters/inv2/AC_kVar": "s01_inv2_AC_kVar",
    "sites/pvs/ktora/stations/st1/inverters/inv2/DC_kW": "s01_inv2_DC_kW",
    "sites/pvs/ktora/stations/st1/inverters/inv2/PF": "s01_inv2_PF",
    "Sites/PVS/Ktora/SiteData/Pyranometer": "pyranometer",  # identical to irradiance, F-13
    "Sites/PVS/Ktora/SiteData/AmbientTemp": "site_ambient_temp",
    "Sites/PVS/Ktora/stations/st1/inverters/inv1/operativestate": "s01_inv1_state",
    "Sites/PVS/Ktora/stations/st1/inverters/inv2/operativestate": "s01_inv2_state",
}

INVERTER_AC_COLUMNS = ("s01_inv1_AC_kW", "s01_inv2_AC_kW")
INVERTER_AC_FLAG_KW = 1.1 * AC_CAPACITY_KW / 2  # above 110 % of one inverter's share
SENTINELS = (32766.0, 32767.0, -32768.0, 65535.0)


def raw_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    files = sorted(p for p in raw_dir.iterdir() if RAW_FILE_PATTERN.match(p.name))
    if len(files) != N_RAW_FILES:
        raise ValueError(f"expected {N_RAW_FILES} monthly files in {raw_dir}, found {len(files)}")
    return files


def to_utc(local_naive: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Israel local clock (with DST) -> UTC. Rows must be in recording order (for the repeated
    October hour). Raises on a timestamp that does not exist locally (spring-forward gap)."""
    idx = pd.DatetimeIndex(local_naive)
    if idx.tz is not None:
        raise ValueError("expected naive local timestamps")
    return idx.tz_localize(TIMEZONE_LOCAL, ambiguous="infer", nonexistent="raise").tz_convert("UTC")


def read_raw_month(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """One raw file -> UTC-indexed frame with canonical columns, plus non-numeric counts."""
    df = pd.read_csv(path, dtype=str)
    expected = {TIME_COLUMN, *TAG_MAP}
    if set(df.columns) != expected:
        missing, extra = expected - set(df.columns), set(df.columns) - expected
        raise ValueError(f"{path.name}: missing {sorted(missing)}, unexpected {sorted(extra)}")
    local = pd.to_datetime(df.pop(TIME_COLUMN), format="%Y-%m-%d %H:%M:%S.%f")
    df = df.rename(columns=TAG_MAP)[list(TAG_MAP.values())]
    non_numeric = {}
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        non_numeric[col] = int((values.isna() & df[col].notna()).sum())
        df[col] = values
    df.index = to_utc(local)
    df.index.name = "timestamp_utc"
    return df, non_numeric


def qa_table(df: pd.DataFrame, non_numeric: dict[str, int]) -> pd.DataFrame:
    """Per-column anomaly counts. Nothing is modified here."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "n_missing": int(s.isna().sum()),
            "n_non_numeric": non_numeric.get(col, 0),
            "n_sentinel": int(s.isin(SENTINELS).sum()),
            "min": float(s.min()),
            "p50": float(s.median()),
            "max": float(s.max()),
            "n_negative": int((s < 0).sum()),
            "n_above_inverter_flag": int((s > INVERTER_AC_FLAG_KW).sum())
            if col in INVERTER_AC_COLUMNS else 0,
        })
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _verify_against_checksums(files: list[Path]) -> int:
    """Raw files must be byte-identical to the stage-0 checksums. Returns how many were checked."""
    ref_file = AUDIT_DIR / "raw_checksums.sha256"
    ref = {}
    for line in ref_file.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        ref[name] = digest
    bad = [p.name for p in files if ref.get(p.name) != _sha256(p)]
    if bad:
        raise ValueError(f"raw files differ from {ref_file.name}: {bad}")
    return len(files)


def check_time_axis(index: pd.DatetimeIndex) -> None:
    """Contiguous, unique, 1-minute UTC grid covering the full data range."""
    expected = pd.date_range(DATA_START_UTC, DATA_END_UTC, freq="min", tz="UTC")
    if len(index) != N_MINUTES_EXPECTED or not index.equals(expected):
        dup = int(index.duplicated().sum())
        missing = len(expected.difference(index))
        raise ValueError(f"time axis not contiguous: {len(index)} rows, {dup} duplicates, "
                         f"{missing} missing minutes")


def ingest(raw_dir: Path = RAW_DIR, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or stage_dir("01", "ingest")
    files = raw_files(raw_dir)
    n_verified = _verify_against_checksums(files)

    frames, non_numeric = [], {}
    for path in files:
        df, nn = read_raw_month(path)
        frames.append(df)
        for col, n in nn.items():
            non_numeric[col] = non_numeric.get(col, 0) + n
        print(f"  {path.name}: {len(df):>6} rows  {df.index[0]} .. {df.index[-1]}", flush=True)
    data = pd.concat(frames)
    check_time_axis(data.index)

    qa = qa_table(data, non_numeric)
    parquet = out_dir / "scada_1min_utc.parquet"
    data.to_parquet(parquet, compression="snappy")
    qa.to_csv(out_dir / "ingest_qa.csv", index=False)
    pd.DataFrame({"raw_tag": list(TAG_MAP), "column": list(TAG_MAP.values())}).to_csv(
        out_dir / "tag_map.csv", index=False)

    manifest = {
        "stage": "01_ingest",
        "rows": len(data),
        "columns": list(data.columns),
        "utc_start": str(data.index[0]),
        "utc_end": str(data.index[-1]),
        "time_rule": "tz_localize('Asia/Jerusalem', ambiguous='infer', nonexistent='raise') "
                     "-> tz_convert('UTC')",
        "raw_files": [p.name for p in files],
        "raw_files_verified_against_checksums": n_verified,
        "parquet_sha256": _sha256(parquet),
        "anomalies": {
            "sentinel_values": int(qa["n_sentinel"].sum()),
            "non_numeric_values": int(qa["n_non_numeric"].sum()),
            "inverter_ac_above_flag": int(qa["n_above_inverter_flag"].sum()),
            "inverter_ac_flag_kw": INVERTER_AC_FLAG_KW,
        },
    }
    (out_dir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> int:
    no_options("python -m solarpv.data.ingest", __doc__, argv)  # X62: --help runs nothing
    data = ingest()
    print(f"\n{len(data):,} rows, {data.index[0]} .. {data.index[-1]}")
    print(f"missing values (all columns): {int(data.isna().sum().sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

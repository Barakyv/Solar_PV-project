"""Decode GFS GRIB2 messages in memory and keep only the grid points around the site
(stage 6b, docs/stage6b_plan.md S1).

On Windows the ecCodes DLL of the conda environment is found only when Library/bin is on PATH,
which is not the case when python.exe runs without an activated environment; the directory is
added here before eccodes is imported.
"""

import os
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":  # pragma: no cover - depends on the platform
    _lib = Path(sys.prefix) / "Library" / "bin"
    if _lib.is_dir():
        os.environ["PATH"] = str(_lib) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(_lib))

import eccodes  # noqa: E402

# the 3x3 neighbourhood of the 1-degree grid point nearest to the site (30N 35E); the site
# (about 29.97N 35.07E) lies in the cell 29-30N x 35-36E
NEIGHBOURHOOD_LATS = (31.0, 30.0, 29.0)
NEIGHBOURHOOD_LONS = (34.0, 35.0, 36.0)
GRID_1P00 = {"Ni": 360, "Nj": 181, "latitudeOfFirstGridPointInDegrees": 90.0,
             "longitudeOfFirstGridPointInDegrees": 0.0, "iDirectionIncrementInDegrees": 1.0,
             "jDirectionIncrementInDegrees": 1.0}


def check_grid(handle) -> None:
    """Assert the regular 1-degree grid, north to south and east from 0 (plan B5)."""
    for key, expected in GRID_1P00.items():
        value = eccodes.codes_get(handle, key)
        if not np.isclose(float(value), float(expected)):
            raise ValueError(f"unexpected GRIB grid: {key} = {value}, expected {expected}")
    if eccodes.codes_get(handle, "jScansPositively") != 0:
        raise ValueError("unexpected GRIB scanning direction (jScansPositively != 0)")


def grid_index(lat: float, lon: float) -> int:
    """Index of (lat, lon) in the values array of the 1-degree grid (rows from 90N, columns
    from 0E)."""
    return int(round((90.0 - lat) * 360 + (lon % 360.0)))


def decode_message(data: bytes) -> dict:
    """Decode one GRIB2 message: short name, level, step range, step type, and the values at
    the 3x3 neighbourhood points."""
    handle = eccodes.codes_new_from_message(data)
    try:
        check_grid(handle)
        values = eccodes.codes_get_values(handle)
        points = {(lat, lon): float(values[grid_index(lat, lon)])
                  for lat in NEIGHBOURHOOD_LATS for lon in NEIGHBOURHOOD_LONS}
        return {"shortName": eccodes.codes_get(handle, "shortName"),
                "typeOfLevel": eccodes.codes_get(handle, "typeOfLevel"),
                "stepRange": str(eccodes.codes_get(handle, "stepRange")),
                "stepType": eccodes.codes_get(handle, "stepType"),
                "dataDate": int(eccodes.codes_get(handle, "dataDate")),
                "dataTime": int(eccodes.codes_get(handle, "dataTime")),
                "points": points}
    finally:
        eccodes.codes_release(handle)

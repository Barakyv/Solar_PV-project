"""Site parameters (decision D-05).

The true coordinates are kept in configs/site.private.yaml, which is not committed to git.
configs/site.example.yaml holds anonymised placeholder values so the code runs without it,
but results computed with the example file are not valid for the study.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from solarpv.config.paths import CONFIGS_DIR

PRIVATE_FILE = CONFIGS_DIR / "site.private.yaml"
EXAMPLE_FILE = CONFIGS_DIR / "site.example.yaml"


@dataclass(frozen=True)
class Site:
    name: str
    latitude: float
    longitude: float
    altitude_m: float
    surface_tilt: float
    surface_azimuth: float
    albedo: float
    timezone: str
    is_private: bool


def load_site(path: Path | None = None) -> Site:
    """Load site parameters, preferring the private file."""
    if path is None:
        path = PRIVATE_FILE if PRIVATE_FILE.exists() else EXAMPLE_FILE
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["site"]
    return Site(
        name=cfg["name"],
        latitude=float(cfg["latitude"]),
        longitude=float(cfg["longitude"]),
        altitude_m=float(cfg["altitude_m"]),
        surface_tilt=float(cfg["surface_tilt"]),
        surface_azimuth=float(cfg["surface_azimuth"]),
        albedo=float(cfg["albedo"]),
        timezone=cfg["timezone"],
        is_private=Path(path).resolve() == PRIVATE_FILE.resolve(),
    )

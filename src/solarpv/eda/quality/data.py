"""Inputs of the stage-3b data-quality analysis (docs/stage3b_quality_eda_plan.md).

"Before cleaning" is the stage-4 output: clock-corrected, true-UTC minutes with the polled data
rebuilt, which is exactly what the cleaning rules see. "After cleaning" is the stage-5 output.
The raw export (stage 2) is loaded only when an analysis needs the export format itself.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from solarpv.config.paths import CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from solarpv.config.site import Site, load_site
from solarpv.data.clean import load_policy, run_lengths, split_of
from solarpv.data.clock import Orientation, clear_sky_components, poa_from_components

INGEST_FILE = PROCESSED_DIR / "01_ingest" / "scada_1min_utc.parquet"
STAGE4_DIR = PROCESSED_DIR / "04_preprocess"
STAGE5_DIR = PROCESSED_DIR / "05_clean"
STAGE4_MIN = STAGE4_DIR / "scada_1min_true_utc.parquet"
STAGE4_H = STAGE4_DIR / "scada_1H_utc.parquet"
STAGE5_MIN = STAGE5_DIR / "scada_1min_clean.parquet"
STAGE5_H = STAGE5_DIR / "scada_1H_clean.parquet"
CLEANING_LOG = OUTPUTS_DIR / "05_cleaning" / "cleaning_log.csv"
POLICY_FILE = CONFIGS_DIR / "cleaning_policy.yaml"
INPUT_FILES = (INGEST_FILE, STAGE4_MIN, STAGE4_H, STAGE5_MIN, STAGE5_H)
TEXT_INPUTS = (CLEANING_LOG, POLICY_FILE)

INV_AC = ("s01_inv1_AC_kW", "s01_inv2_AC_kW")
INV_DC = ("s01_inv1_DC_kW", "s01_inv2_DC_kW")
POWER = (*INV_AC, *INV_DC)
TEMPERATURES = ("mod_temp1", "mod_temp2", "ambient_temp", "site_ambient_temp")
# the columns that carry the cleaning rules and are used to recognise communication freezes
MAIN = ("irradiance", "pyranometer", *POWER, *TEMPERATURES)
SPLITS = ("train", "val", "test")


@dataclass
class Data:
    """All inputs, aligned on the 1,971,360-minute true-UTC grid."""

    s4: pd.DataFrame
    s5: pd.DataFrame
    h4: pd.DataFrame
    h5: pd.DataFrame
    elev: np.ndarray
    poa_cs: np.ndarray
    site: Site
    policy: dict
    _runs: dict = field(default_factory=dict, repr=False)

    @cached_property
    def split(self) -> np.ndarray:
        return split_of(self.s4.index)

    @cached_property
    def train(self) -> np.ndarray:
        return self.split == "train"

    @cached_property
    def daytime(self) -> np.ndarray:
        """The cleaning policy's daytime: sun above 10 deg at the minute."""
        return self.elev > self.policy["sun"]["daytime_min_elevation"]

    @cached_property
    def hour_split(self) -> np.ndarray:
        return split_of(self.h5.index)

    @cached_property
    def hour_daytime(self) -> np.ndarray:
        """Evaluation daytime of each hourly row (sun > 5 deg at mid-hour, F-02)."""
        from solarpv.evaluation.daytime import daytime_mask
        return daytime_mask(self.h5.index, self.site)

    @cached_property
    def raw(self) -> pd.DataFrame:
        """The export as delivered (stage 2): local stamps converted to UTC, logger clocks and
        held polled values not yet corrected."""
        return pd.read_parquet(INGEST_FILE)

    @cached_property
    def raw_elev(self) -> np.ndarray:
        return clear_sky_components(self.raw.index, self.site)["elev"].to_numpy()

    def runs(self, column: str, stage: str = "s4") -> tuple[np.ndarray, np.ndarray]:
        """(run id, run length) of every minute; NaN breaks runs (clean.run_lengths)."""
        key = (stage, column)
        if key not in self._runs:
            frame = self.s4 if stage == "s4" else self.s5
            self._runs[key] = run_lengths(frame[column].to_numpy(dtype=float))
        return self._runs[key]

    def frozen(self, column: str, min_minutes: int = 30, stage: str = "s4") -> np.ndarray:
        """Minutes inside a run of one value lasting at least min_minutes; for power columns
        runs of exactly 0 are inverter stops, not frozen values (as in rule R3)."""
        frame = self.s4 if stage == "s4" else self.s5
        v = frame[column].to_numpy(dtype=float)
        _, length = self.runs(column, stage)
        out = np.isfinite(v) & (length >= min_minutes)
        if column in self.policy["frozen_ignore_zero"] or column.endswith("_kVar"):
            out &= v != 0
        return out


def load() -> Data:
    site = load_site()
    s4 = pd.read_parquet(STAGE4_MIN)
    s5 = pd.read_parquet(STAGE5_MIN)
    comp = clear_sky_components(s4.index, site)
    poa_cs = poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth))
    elev = comp["elev"].to_numpy()
    sky = pd.DataFrame({"poa_clearsky": poa_cs, "elev": elev}, index=s4.index).resample("1h").mean()
    h4 = pd.read_parquet(STAGE4_H).join(sky)
    h5 = pd.read_parquet(STAGE5_H).join(sky)
    return Data(s4=s4, s5=s5, h4=h4, h5=h5, elev=elev, poa_cs=poa_cs, site=site,
                policy=load_policy(POLICY_FILE))

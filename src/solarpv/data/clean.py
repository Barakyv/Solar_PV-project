"""Stage 5: explicit, logged cleaning rules (faults, spikes, frozen, night, stops, curtailment).

    python -m solarpv.data.clean

Policy: configs/cleaning_policy.yaml (every threshold with its evidence). Test plan:
docs/stage5_test_plan.md. Works on the clock-corrected minute data of stage 4.

Value rules replace a wrong measurement; the rule that did it is kept in rule_<column>:
  R1 range        outside the physical range, plus a margin of neighbours      -> NaN
  R2 offset       small negative irradiance (sensor offset at night)          -> 0
  R3 frozen       long run of one repeated value that reaches daytime, plus a  -> NaN
                  margin of neighbours (blended or interpolated from it)
  R4 night zero   |power| <= noise with the sun below the horizon             -> 0
  R5 night stale  power > noise with the sun far below the horizon            -> NaN
  R6 dead DC      DC = 0 while the same inverter delivers AC                  -> NaN
  R7 scale fault  a redundant sensor off its reference by a constant factor   -> NaN for
                  over a local day or a UTC hour (the pyranometer read x0.75)    that period
Precedence when several apply to one minute: R1, R3, R6, R7, R5, R2, R4.
Flag rules never change a value (decision D-22):
  F1 downtime     sun up and high in-plane irradiance, inverter gives ~nothing
  F2 curtailment  park setpoint below 100 % (a commanded step, not a non-integer drift)
  F3 derate       the inverter runs at ~75 % of its twin (status code, or AC ratio)
The night rule uses true-UTC geometry, so real afternoon production is never zeroed (F-01b,
test T-16). Hourly output: hourly_mean() of stage 4 on the cleaned minutes, plus the share of
flagged minutes per hour, op_limited_flag (D-22) and hour_source (D-21).

Outputs:
  data/processed/05_clean/scada_1min_clean.parquet   cleaned minutes, rule_<column>, flags
  data/processed/05_clean/scada_1H_clean.parquet     hourly means, flag shares, hour_source
  data/processed/05_clean/cleaning_manifest.json
  outputs/05_cleaning/cleaning_log.csv, cleaning_policy_applied.yaml   (tracked in git)
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from solarpv.config.constants import TEST_START, TIMEZONE_LOCAL, TRAIN_END, VAL_END, VAL_START
from solarpv.config.paths import CONFIGS_DIR, output_dir, stage_dir
from solarpv.config.provenance import git_commit, sha256, text_sha256
from solarpv.config.site import load_site
from solarpv.data.clock import Orientation, clear_sky_components, poa_from_components
from solarpv.data.preprocess import AC_COLUMNS, TARGET, hourly_mean
from solarpv.evaluation.daytime import daytime_mask

POLICY_FILE = CONFIGS_DIR / "cleaning_policy.yaml"
RULES = {1: "R1_range", 2: "R2_offset", 3: "R3_frozen", 4: "R4_night_zero",
         5: "R5_night_stale", 6: "R6_dc_dead", 7: "R7_scale_fault"}
NAN_RULES = (1, 3, 5, 6, 7)
ZERO_RULES = (2, 4)
PRECEDENCE = (1, 3, 6, 7, 5, 2, 4)
SOURCE_POLLED = 1
DOWNTIME_FLAGS = {"s01_inv1_AC_kW": "downtime_inv1_flag", "s01_inv2_AC_kW": "downtime_inv2_flag"}
DERATE_FLAGS = {"s01_inv1_AC_kW": "derate_inv1_flag", "s01_inv2_AC_kW": "derate_inv2_flag"}
STATE_COLUMNS = {"s01_inv1_AC_kW": "s01_inv1_state", "s01_inv2_AC_kW": "s01_inv2_state"}
CURTAIL_FLAG = "curtail_flag"
OP_FLAG = "op_limited_flag"
MINUTE_FLAGS = (*DOWNTIME_FLAGS.values(), CURTAIL_FLAG, *DERATE_FLAGS.values(), OP_FLAG)
SOURCE_COLUMNS = ("src_meteo", "src_inv1", "src_inv2")
COUNTED = (TARGET, *AC_COLUMNS, "irradiance", "mod_temp1", "mod_temp2", "ambient_temp")


def load_policy(path: Path = POLICY_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cleaned_columns(policy: dict) -> list[str]:
    cols = (set(policy["range"]) | set(policy["negative_to_zero"])
            | set(policy["frozen_min_minutes"]) | set(policy["night"]["columns"])
            | set(policy["dc_dead"]["pairs"]) | set(policy["scale_fault"]["pairs"]))
    return sorted(cols)


def local_day_number(index: pd.DatetimeIndex) -> np.ndarray:
    """Days since 1970-01-01 of each minute's local calendar date."""
    local = index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    return ((local - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).to_numpy()


def source_reference(column: str) -> str:
    """Stage-4 source column of the clock group a tag belongs to."""
    if column.startswith("s01_inv2_"):
        return "src_inv2"
    if column.startswith(("s01_", "iec_")):
        return "src_inv1"
    return "src_meteo"


# ── value rules: each returns the minutes it would change (on stage-4 values) ──


def _dilate(mask: np.ndarray, minutes: int) -> np.ndarray:
    if minutes <= 0:
        return mask
    window = 2 * minutes + 1
    return (pd.Series(mask.astype(np.int8)).rolling(window, center=True, min_periods=1).max()
            .to_numpy().astype(bool))


def rule_range(v: np.ndarray, lo: float, hi: float, source: np.ndarray,
               margins: dict) -> np.ndarray:
    """R1: outside [lo, hi], plus neighbours within the margin of the minute's source."""
    finite = np.isfinite(v)
    bad = finite & ((v < lo) | (v > hi))
    return finite & _near(bad, source, margins)


def rule_offset(v: np.ndarray, lo: float) -> np.ndarray:
    """R2: lo <= v < 0 (the sensor's offset)."""
    return np.isfinite(v) & (v < 0) & (v >= lo)


def run_lengths(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run id and run length of every minute; NaNs break runs."""
    finite = np.isfinite(v)
    same = np.r_[False, (v[1:] == v[:-1]) & finite[1:] & finite[:-1]]
    rid = np.cumsum(~same) - 1
    return rid, np.bincount(rid)[rid]


def _near(core: np.ndarray, source: np.ndarray, margins: dict) -> np.ndarray:
    """`core` widened by the margin of each minute's source (minute / polled)."""
    return np.where(source == SOURCE_POLLED, _dilate(core, margins["polled"]),
                    _dilate(core, margins["minute"]))


def rule_frozen(v: np.ndarray, min_minutes: int, daytime: np.ndarray, ignore_zero: bool,
                source: np.ndarray | None = None, margins: dict | None = None) -> np.ndarray:
    """R3: runs of one value lasting at least min_minutes that reach daytime, plus the
    neighbours within the margin of the minute's source."""
    rid, length = run_lengths(v)
    reaches_day = np.bincount(rid, weights=daytime.astype(float)) > 0
    core = np.isfinite(v) & (length >= min_minutes) & reaches_day[rid]
    if ignore_zero:
        core &= v != 0
    if margins is None:
        return core
    src = np.zeros(len(v), dtype=np.int8) if source is None else source
    return np.isfinite(v) & _near(core, src, margins)


def rule_night_zero(v: np.ndarray, elev: np.ndarray, max_elev: float, noise: float) -> np.ndarray:
    """R4: sun below max_elev and 0 < |v| <= noise."""
    return np.isfinite(v) & (elev < max_elev) & (np.abs(v) <= noise) & (v != 0)


def rule_night_stale(v: np.ndarray, elev: np.ndarray, below: float, noise: float) -> np.ndarray:
    """R5: sun below `below` and v > noise, when no production is possible."""
    return np.isfinite(v) & (elev < below) & (v > noise)


def rule_dc_dead(dc: np.ndarray, ac: np.ndarray, min_ac: float) -> np.ndarray:
    """R6: DC = 0 while the same inverter delivers more than min_ac of AC."""
    return np.isfinite(dc) & (dc == 0) & np.isfinite(ac) & (ac > min_ac)


def rule_scale_fault(v: np.ndarray, ref: np.ndarray, elev: np.ndarray, day: np.ndarray,
                     p: dict, hour: np.ndarray | None = None) -> np.ndarray:
    """R7: periods (local days, and UTC hours if `hour` is given) on which the median ratio
    v / ref over the usable minutes (sun high, reference bright; at least
    p["min_minutes"][period] of them) lies outside p["ratio_range"]; the whole period of v is
    marked."""
    ok = (np.isfinite(v) & np.isfinite(ref) & (elev > p["min_elevation"])
          & (ref > p["min_reference"]))
    ratio = pd.Series(v[ok] / ref[ok])
    lo, hi = p["ratio_range"]
    out = np.zeros(len(v), dtype=bool)
    for period, key in (("day", day), ("hour", hour)):
        if key is None:
            continue
        g = ratio.groupby(key[ok]).agg(["median", "size"])
        g = g[g["size"] >= p["min_minutes"][period]]
        bad = g.index[(g["median"] < lo) | (g["median"] > hi)]
        out |= np.isin(key, bad)
    return np.isfinite(v) & out


def rule_masks(column: str, v: np.ndarray, elev: np.ndarray, source: np.ndarray,
               policy: dict, ac: np.ndarray | None = None, ref: np.ndarray | None = None,
               day: np.ndarray | None = None, hour: np.ndarray | None = None
               ) -> dict[int, np.ndarray]:
    """All value-rule masks for one column, before precedence. `ac`: the stage-4 AC of the
    same inverter, for the DC columns (R6). `ref`, `day` and `hour` (local-day number and UTC
    hour number per minute): the cleaned reference sensor, for R7."""
    none = np.zeros(len(v), dtype=bool)
    daytime = elev > policy["sun"]["daytime_min_elevation"]
    night = policy["night"]
    masks = dict.fromkeys(RULES, none)
    if column in policy["range"]:
        lo, hi = policy["range"][column]
        masks[1] = rule_range(v, lo, hi, source, policy["range_margin_minutes"])
        if column in policy["negative_to_zero"]:
            masks[2] = rule_offset(v, lo)
    if column in policy["frozen_min_minutes"]:
        masks[3] = rule_frozen(v, policy["frozen_min_minutes"][column], daytime,
                               column in policy["frozen_ignore_zero"], source,
                               policy["frozen_margin_minutes"])
    if column in night["columns"]:
        masks[4] = rule_night_zero(v, elev, night["max_elevation"], night["noise_kw"])
        masks[5] = rule_night_stale(v, elev, night["stale_below_elevation"], night["noise_kw"])
    if column in policy["dc_dead"]["pairs"] and ac is not None:
        masks[6] = rule_dc_dead(v, ac, policy["dc_dead"]["min_ac_kw"])
    if column in policy["scale_fault"]["pairs"] and ref is not None and day is not None:
        masks[7] = rule_scale_fault(v, ref, elev, day, policy["scale_fault"], hour)
    return masks


def apply_value_rules(column: str, v: np.ndarray, elev: np.ndarray, source: np.ndarray,
                      policy: dict, ac: np.ndarray | None = None, ref: np.ndarray | None = None,
                      day: np.ndarray | None = None, hour: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(cleaned values, rule code per minute, 0 = unchanged)."""
    masks = rule_masks(column, v, elev, source, policy, ac, ref, day, hour)
    code = np.zeros(len(v), dtype=np.int8)
    for r in PRECEDENCE:
        code[masks[r] & (code == 0)] = r
    out = v.astype(float).copy()
    out[np.isin(code, NAN_RULES)] = np.nan
    out[np.isin(code, ZERO_RULES)] = 0.0
    return out, code


# ── flag rules: never change a value ───────────────────────────────────────


def flag_downtime(ac: np.ndarray, irradiance: np.ndarray, poa_cs: np.ndarray, elev: np.ndarray,
                  policy: dict) -> np.ndarray:
    """F1: the sun is up and the in-plane irradiance is high, but the inverter gives ~0."""
    p = policy["downtime"]
    bright = np.where(np.isfinite(irradiance), irradiance > p["min_irradiance"],
                      poa_cs > p["fallback_min_poa_clearsky"])
    return (elev > p["min_elevation"]) & bright & np.isfinite(ac) & (ac < p["max_ac_kw"])


def true_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Length of the stretch of consecutive True values each minute belongs to (0 where False)."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    out = np.zeros(len(mask), dtype=np.int64)
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        out[a:b] = b - a
    return out


def flag_curtailment(setpoint: np.ndarray, elev: np.ndarray, policy: dict) -> np.ndarray:
    """F2: daytime minute with the park setpoint below 100 %, unless the setpoint is drifting
    through non-integer values for longer than a step change takes."""
    p = policy["curtailment"]
    limited = np.isfinite(setpoint) & (setpoint < p["below"])
    non_integer = limited & (np.abs(setpoint - np.round(setpoint)) > 1e-6)
    drift = true_run_lengths(non_integer) > p["max_noninteger_run_minutes"]
    return (elev > p["min_elevation"]) & limited & ~drift


def flag_derate(ac: np.ndarray, other_ac: np.ndarray, state: np.ndarray, elev: np.ndarray,
                policy: dict) -> np.ndarray:
    """F3: the inverter produces, but in a derated status code for a sustained period, or its AC
    stays at a stable fraction well below that of the other inverter (windowed log ratio: low
    median, small inter-quartile range)."""
    p = policy["derate"]
    code = np.round(state)
    integer = np.isfinite(state) & (np.abs(state - code) < 1e-6)
    by_state = ((elev > p["min_elevation"]) & integer & np.isin(code, p["states"])
                & np.isfinite(ac) & (ac >= p["min_ac_kw"]))
    by_state &= true_run_lengths(by_state) >= p["state_min_run_minutes"]
    both = ((elev > p["ratio_min_elevation"]) & np.isfinite(ac) & np.isfinite(other_ac)
            & (ac > p["ratio_min_ac_kw"]) & (other_ac > p["ratio_min_ac_kw"]))
    log_ratio = np.full(len(ac), np.nan)
    log_ratio[both] = np.log(ac[both] / other_ac[both])
    window = pd.Series(log_ratio).rolling(p["ratio_window_minutes"], center=True,
                                          min_periods=p["ratio_min_valid_minutes"])
    median = window.median().to_numpy()
    iqr = (window.quantile(0.75) - window.quantile(0.25)).to_numpy()
    by_ratio = both & (median < np.log(p["ratio_below"])) & (iqr < p["ratio_max_iqr"])
    return by_state | by_ratio


def utc_hour_number(index: pd.DatetimeIndex) -> np.ndarray:
    """Hours since 1970-01-01 00:00 UTC of each minute."""
    return ((index.tz_convert("UTC").tz_localize(None) - pd.Timestamp("1970-01-01"))
            // pd.Timedelta(hours=1)).to_numpy()


# ── whole table ────────────────────────────────────────────────────────────


def clean_minutes(df: pd.DataFrame, elev: np.ndarray, poa_cs: np.ndarray, policy: dict
                  ) -> pd.DataFrame:
    """All stage-4 columns (cleaned where a rule exists), rule_<column> codes, minute flags,
    and site_total recomputed from the cleaned inverters."""
    out = df.copy()
    day = local_day_number(df.index)
    hour = utc_hour_number(df.index)
    references = set(policy["scale_fault"]["pairs"].values())
    columns = sorted(cleaned_columns(policy), key=lambda c: c not in references)
    for col in columns:  # reference sensors first: R7 compares against their cleaned values
        if col not in df:
            continue
        src_col = source_reference(col)
        source = df[src_col].to_numpy() if src_col in df else np.zeros(len(df), dtype=np.int8)
        pair = policy["dc_dead"]["pairs"].get(col)
        ac = df[pair].to_numpy(dtype=float) if pair in df else None
        ref_col = policy["scale_fault"]["pairs"].get(col)
        ref = out[ref_col].to_numpy(dtype=float) if ref_col in out else None
        values, code = apply_value_rules(col, df[col].to_numpy(dtype=float), elev, source, policy,
                                         ac, ref, day, hour)
        out[col] = values
        out[f"rule_{col}"] = code
    out[TARGET] = out[list(AC_COLUMNS)].sum(axis=1, min_count=len(AC_COLUMNS))
    irr = out["irradiance"].to_numpy()
    for ac, name in DOWNTIME_FLAGS.items():
        out[name] = flag_downtime(out[ac].to_numpy(), irr, poa_cs, elev, policy)
    setpoint = df[policy["curtailment"]["setpoint_column"]].to_numpy(dtype=float)
    out[CURTAIL_FLAG] = flag_curtailment(setpoint, elev, policy)
    for (ac, name), other in zip(DERATE_FLAGS.items(), reversed(AC_COLUMNS), strict=True):
        state = (df[STATE_COLUMNS[ac]].to_numpy(dtype=float) if STATE_COLUMNS[ac] in df
                 else np.full(len(df), np.nan))
        out[name] = flag_derate(out[ac].to_numpy(), out[other].to_numpy(), state, elev, policy)
    flags = [*DOWNTIME_FLAGS.values(), CURTAIL_FLAG, *DERATE_FLAGS.values()]
    out[OP_FLAG] = out[flags].any(axis=1)
    return out


def hour_source(minutes: pd.DataFrame) -> pd.Series:
    """'minute', 'polled' or 'mixed' per hour, from the stage-4 source codes (D-21)."""
    src = minutes[list(SOURCE_COLUMNS)]
    polled = src.eq(1).any(axis=1).resample("1h").max()
    minute = src.eq(0).any(axis=1).resample("1h").max()
    out = pd.Series("minute", index=polled.index)
    out[polled & minute] = "mixed"
    out[polled & ~minute] = "polled"
    return out


def to_hourly(minutes: pd.DataFrame, value_columns: list[str], policy: dict) -> pd.DataFrame:
    hourly, counts = hourly_mean(minutes[value_columns])
    for col in COUNTED:
        if col in counts:
            hourly[f"n_{col}"] = counts[col].astype("int16")
    shares = minutes[list(MINUTE_FLAGS)].astype(float).resample("1h").mean()
    for flag in MINUTE_FLAGS:
        hourly[f"frac_{flag}"] = shares[flag]
    hourly[OP_FLAG] = shares[OP_FLAG] >= policy["hourly"]["op_limited_min_share"]
    hourly["hour_source"] = hour_source(minutes)
    hourly.index.name = "timestamp_utc"
    return hourly


def verify_reconciliation(before: pd.DataFrame, after: pd.DataFrame, policy: dict) -> None:
    """Runtime self-check (T-23): a value changed if and only if its rule code is set."""
    for col in cleaned_columns(policy):
        if col not in before:
            continue
        a, b = before[col].to_numpy(dtype=float), after[col].to_numpy(dtype=float)
        changed = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        coded = after[f"rule_{col}"].to_numpy() != 0
        if not np.array_equal(changed, coded):
            raise AssertionError(f"{col}: {int((changed != coded).sum())} unexplained changes")


def split_of(index: pd.DatetimeIndex) -> np.ndarray:
    t = index.tz_convert("UTC").tz_localize(None)
    return np.select([t <= pd.Timestamp(TRAIN_END),
                      (t >= pd.Timestamp(VAL_START)) & (t <= pd.Timestamp(VAL_END)),
                      t >= pd.Timestamp(TEST_START)], ["train", "val", "test"], "gap")


def cleaning_log(before: pd.DataFrame, after: pd.DataFrame, elev: np.ndarray,
                 policy: dict) -> pd.DataFrame:
    """One row per rule and column: minutes changed, daytime minutes, hours touched, and for
    the inverter AC columns the energy taken out (kWh)."""
    rows = []
    hour = after.index.floor("h")
    daytime = elev > policy["sun"]["daytime_min_elevation"]
    for col in cleaned_columns(policy):
        if col not in before:
            continue
        code = after[f"rule_{col}"].to_numpy()
        for r, name in RULES.items():
            m = code == r
            if not m.any():
                continue
            row = {"rule": name, "column": col, "minutes": int(m.sum()),
                   "daytime_minutes": int((m & daytime).sum()),
                   "hours_touched": int(pd.Index(hour[m]).nunique()), "kwh_changed": np.nan}
            if col in AC_COLUMNS:
                removed = before[col].to_numpy()[m] - np.nan_to_num(after[col].to_numpy()[m])
                row["kwh_changed"] = round(float(np.nansum(removed)) / 60, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def hours_by_split(hourly: pd.DataFrame, site) -> dict:
    day = daytime_mask(hourly.index, site)
    split = split_of(hourly.index)
    down = (hourly["frac_downtime_inv1_flag"] + hourly["frac_downtime_inv2_flag"]) > 0
    derate = (hourly["frac_derate_inv1_flag"] + hourly["frac_derate_inv2_flag"]) > 0
    out = {}
    for s in ("train", "val", "test"):
        m = day & (split == s)
        out[s] = {
            "daytime_hours": int(m.sum()),
            "op_limited_hours": int((hourly[OP_FLAG] & m).sum()),
            "hours_with_downtime": int((down & m).sum()),
            "hours_with_curtailment": int(((hourly["frac_curtail_flag"] > 0) & m).sum()),
            "hours_with_derate": int((derate & m).sum()),
            "target_nan_hours": int(hourly.loc[m, TARGET].isna().sum()),
            "polled_hours": int(((hourly["hour_source"] == "polled") & m).sum()),
            "mixed_hours": int(((hourly["hour_source"] == "mixed") & m).sum()),
        }
    return out


def clean(in_dir: Path | None = None, out_dir: Path | None = None,
          policy_file: Path = POLICY_FILE) -> pd.DataFrame:
    in_dir = in_dir or stage_dir("04", "preprocess")
    out_dir = out_dir or stage_dir("05", "clean")
    log_dir = output_dir("05_cleaning")
    policy = load_policy(policy_file)
    site = load_site()
    in_file = in_dir / "scada_1min_true_utc.parquet"
    before = pd.read_parquet(in_file)
    comp = clear_sky_components(before.index, site)
    elev = comp["elev"].to_numpy()
    poa_cs = poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth))

    after = clean_minutes(before, elev, poa_cs, policy)
    verify_reconciliation(before, after, policy)
    log = cleaning_log(before, after, elev, policy)
    hourly = to_hourly(after, [c for c in before.columns if not c.startswith("src_")], policy)

    minute_file = out_dir / "scada_1min_clean.parquet"
    hourly_file = out_dir / "scada_1H_clean.parquet"
    after.to_parquet(minute_file)
    hourly.to_parquet(hourly_file)
    log.to_csv(log_dir / "cleaning_log.csv", index=False, lineterminator="\n")
    shutil.copyfile(policy_file, log_dir / "cleaning_policy_applied.yaml")

    total = hourly[TARGET]
    manifest = {
        "stage": "05_clean",
        "git_commit": git_commit(),
        "input": {"file": in_file.name, "sha256": sha256(in_file)},
        "policy": {"file": policy_file.name, "sha256": text_sha256(policy_file)},
        "rows_1min": len(after),
        "rows_1H": len(hourly),
        "rule_minutes": {f"{r['rule']}:{r['column']}": r["minutes"]
                         for r in log.to_dict("records")},
        "flag_minutes": {f: int(after[f].sum()) for f in MINUTE_FLAGS},
        "hours_by_split": hours_by_split(hourly, site),
        "site_total_AC_kW": {"max_hourly": round(float(total.max()), 1),
                             "p99_99_hourly": round(float(total.quantile(0.9999)), 1)},
        "outputs": {p.name: sha256(p) for p in (minute_file, hourly_file)},
    }
    text = json.dumps(manifest, indent=2)
    for path in (out_dir / "cleaning_manifest.json", log_dir / "cleaning_manifest.json"):
        path.write_text(text, encoding="utf-8", newline="\n")
    print(log.to_string(index=False))
    print(json.dumps({k: manifest[k] for k in ("flag_minutes", "hours_by_split",
                                               "site_total_AC_kW")}, indent=2))
    return hourly


def main() -> int:
    clean()
    return 0


if __name__ == "__main__":
    sys.exit(main())

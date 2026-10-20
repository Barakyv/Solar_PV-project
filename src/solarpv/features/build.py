"""Stage 6: the modelling table, one per horizon D+1..D+3 (docs/stage6_plan.md, S2-S6).

    python -m solarpv.features.build

Input: the cleaned hourly data of stage 5 (scada_1H_clean.parquet) and, for the per-inverter
D-22 masks, the minute flags of stage 5. Output, per horizon h:
    data/processed/06_features/features_1H_D{h}.parquet
        metadata  t0_utc, split, daytime (F-02), warmup, op_limited_flag, hour_source
        targets   site_total_AC_kW; theo_{ghi,poa}_kw, k_{ghi,poa}, k_valid_{ghi,poa};
                  and, once D-11 is decided (configs/target.yaml), theo_kw / target_k /
                  k_valid / k_suspect for the chosen denominator
        features  the contract (contract.py, 74) minus the lags of LAG_DROP[h]
plus features_manifest.json, outputs/06_features/feature_dictionary.csv, and
configs/derived_constants.yaml (MEAN_DAYTIME_K, once D-11 is decided).

Rules that make the table forecast-safe (tests T-09, T-18):
  * Issuance (D-16): the forecast of local day D+h is made at T0 = 00:00 local of day D+1.
    A past-day value v(s) may feed hour t only if its hour ends by T0:  s + 1h <= T0(t).
    With LAG_DROP this always holds, except on the day the clock falls back (25 hours):
    there the last hour's lag would come from inside the forecast day. The guard turns such
    values into NaN instead of relying on the 24-hour shift.
  * Rolling per horizon (D-04): mean / std over the same hour on days h .. h+N-1 back.
  * D-22: before any lag or rolling, an inverter's AC is NaN in hours where its own stop (F1)
    or derate (F3) or the curtailment (F2) covers >= 5 % of the minutes; the site total is
    NaN in op-limited hours. Weather columns are not masked.
  * Imputation (fix of the original): a missing past-day value is filled with the median of
    the same feature at the same UTC hour over the previous 30 days (older values only). The
    original filled with the expanding median of the whole column, nights included, which pulls
    daytime gaps towards zero and mixes seasons. Never backward fill.
  * Warm-up: the first 14 days have too little history; they carry warmup = True and are
    excluded from training (§6.6).
"""

import json
import sys

import numpy as np
import pandas as pd
import yaml

from solarpv.config.constants import (
    DERIVED_CONSTANTS_FILE,
    HORIZONS,
    LAG_DROP,
    THEO_MIN_KW,
    TIMEZONE_LOCAL,
)
from solarpv.config.paths import CONFIGS_DIR, PROCESSED_DIR, output_dir, stage_dir
from solarpv.config.provenance import git_commit, text_sha256
from solarpv.config.runlog import RunLog
from solarpv.config.site import load_site
from solarpv.data.clean import split_of
from solarpv.evaluation.daytime import daytime_mask
from solarpv.features.contract import (
    FEATURE_GROUPS,
    LAG_DAYS,
    LAG_SOURCES,
    ROLLING_SOURCES,
    ROLLING_WINDOWS_DAYS,
    features_for_horizon,
    group_of,
)
from solarpv.features.geometry import (
    PANEL_FILE,
    daily_sun_times,
    hourly_geometry,
    load_panel_plane,
)
from solarpv.features.target import (
    DENOMINATORS,
    clear_sky_index,
    mean_daytime_k,
    share_above,
    suspect,
    theoretical_power,
)

HOURLY_FILE = PROCESSED_DIR / "05_clean" / "scada_1H_clean.parquet"
MINUTE_FILE = PROCESSED_DIR / "05_clean" / "scada_1min_clean.parquet"
POLICY_FILE = CONFIGS_DIR / "cleaning_policy.yaml"
TARGET_FILE = CONFIGS_DIR / "target.yaml"  # the D-11 decision (written by decide_k)
TARGET = "site_total_AC_kW"
WARMUP_DAYS = 14
IMPUTE_WINDOW_DAYS = 30
# rolling source name -> hourly column it is computed from
ROLLING_COLUMNS = {name: f"{name}_AC_kW" for name in ROLLING_SOURCES}
# per-inverter minute flags that limit that inverter (D-22): own stop, own derate, curtailment
INVERTER_FLAGS = {"s01_inv1_AC_kW": ("downtime_inv1_flag", "derate_inv1_flag", "curtail_flag"),
                  "s01_inv2_AC_kW": ("downtime_inv2_flag", "derate_inv2_flag", "curtail_flag")}
SEASONS = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
WEEKEND_DAYS = (4, 5)  # Friday and Saturday (pandas dayofweek: Monday = 0), the Israeli weekend


# ── issuance ───────────────────────────────────────────────────────────────


def issuance_utc(hours: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """T0 of each target hour for D+horizon: 00:00 local of the local day (horizon - 1) days
    before the hour's local day, in UTC. Local midnight is never ambiguous in Israel (the clock
    changes at 02:00), so the conversion is unique."""
    local_day = hours.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    issue_day = local_day - pd.Timedelta(days=horizon - 1)
    return issue_day.tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")


def past_value(series: pd.Series, days: int, t0: pd.DatetimeIndex) -> np.ndarray:
    """Value of `series` at the same UTC hour `days` days earlier, NaN unless that hour ended
    by the row's issuance time (the DST guard). The hourly index must be complete."""
    shifted = series.shift(24 * days).to_numpy(dtype=float)
    source_end = series.index - pd.Timedelta(hours=24 * days - 1)  # end of the source hour
    return np.where(source_end <= t0, shifted, np.nan)


# ── D-22 masks ─────────────────────────────────────────────────────────────


def limited_hours(minute_flags: pd.DataFrame, columns: tuple[str, ...], share: float
                  ) -> pd.Series:
    """Hours in which the union of the given minute flags covers at least `share` of the hour
    (the same rule as op_limited_flag in stage 5, D-22)."""
    union = minute_flags[list(columns)].any(axis=1).astype(float)
    return union.resample("1h").mean() >= share


def masked_sources(hourly: pd.DataFrame, minute_flags: pd.DataFrame, share: float
                   ) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """The seven lag sources, with the power columns masked by D-22, and the number of masked
    hours per power column."""
    sources, counts = {}, {}
    for col in LAG_SOURCES:
        s = hourly[col].astype(float).copy()
        if col in INVERTER_FLAGS:
            limited = limited_hours(minute_flags, INVERTER_FLAGS[col], share).reindex(s.index)
            limited = limited.fillna(False).astype(bool)
        elif col == TARGET:
            limited = hourly["op_limited_flag"].astype(bool)
        else:
            limited = None  # weather: a plant limitation does not change it
        if limited is not None:
            counts[col] = int(limited.sum())
            s[limited.to_numpy()] = np.nan
        sources[col] = s
    return sources, counts


# ── past-day features ─────────────────────────────────────────────────────


def kept_lags(horizon: int) -> list[int]:
    """Lag days known at issuance for D+horizon (LAG_DROP: lag_1d dropped at D+2, lag_1d and
    lag_2d at D+3)."""
    return [d for d in LAG_DAYS if f"lag_{d}d" not in LAG_DROP[horizon]]


def lag_features(sources: dict[str, pd.Series], horizon: int, t0: pd.DatetimeIndex
                 ) -> dict[str, np.ndarray]:
    """<source>_lag_<d>d for the lags kept at this horizon, with the DST guard."""
    return {f"{col}_lag_{d}d": past_value(sources[col], d, t0)
            for col in LAG_SOURCES for d in kept_lags(horizon)}


def rolling_features(sources: dict[str, pd.Series], horizon: int, t0: pd.DatetimeIndex
                     ) -> dict[str, np.ndarray]:
    """<name>_rolling_{mean,std}_{N}d_same_tod for D+horizon (D-04): statistics over the same
    hour on days horizon .. horizon+N-1 back. Mean needs 1 valid value, std (ddof 1) needs 2."""
    out = {}
    for name, col in ROLLING_COLUMNS.items():
        for n in ROLLING_WINDOWS_DAYS:
            stack = np.vstack([past_value(sources[col], k, t0)
                               for k in range(horizon, horizon + n)])
            out[f"{name}_rolling_mean_{n}d_same_tod"] = _nanmean(stack)  # >= 1 valid value
            out[f"{name}_rolling_std_{n}d_same_tod"] = _nanstd(stack)  # >= 2 valid values
    return out


def _nanmean(stack: np.ndarray) -> np.ndarray:
    count = np.isfinite(stack).sum(axis=0)
    total = np.nansum(stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _nanstd(stack: np.ndarray) -> np.ndarray:
    count = np.isfinite(stack).sum(axis=0)
    mean = _nanmean(stack)
    sq = np.nansum((stack - mean) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 1, np.sqrt(sq / (count - 1)), np.nan)


def impute_same_hour(frame: pd.DataFrame, window_days: int = IMPUTE_WINDOW_DAYS
                     ) -> tuple[pd.DataFrame, pd.Series]:
    """Fill each NaN with the median of the same column at the same UTC hour over the previous
    `window_days` days (shift by one day first, so only older rows are used). Rows with no
    earlier value at that hour stay NaN (they fall in the warm-up). Returns (filled frame,
    share of filled values per column)."""
    hour = frame.index.hour
    filled = frame.copy()
    shares = {}
    for col in frame.columns:
        past = (frame[col].groupby(hour)
                .transform(lambda s: s.shift(1).rolling(window_days, min_periods=1).median()))
        missing = frame[col].isna()
        filled[col] = frame[col].where(~missing, past)
        shares[col] = float(missing.mean())
    return filled, pd.Series(shares)


# ── calendar and geometry ─────────────────────────────────────────────────


def calendar_features(hours: pd.DatetimeIndex, sun_times: pd.DataFrame, max_day_length: float
                      ) -> pd.DataFrame:
    """Time encodings and solar-time features. The cyclic hour uses the UTC hour (§6.6); the
    calendar uses the local day of the target hour; solar times come from pvlib (transit,
    sunrise, sunset of the local day)."""
    local = hours.tz_convert(TIMEZONE_LOCAL)
    mid = hours + pd.Timedelta(minutes=30)
    day_length = (sun_times["sunset"] - sun_times["sunrise"]).dt.total_seconds() / 3600
    out = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hours.hour / 24),
        "hour_cos": np.cos(2 * np.pi * hours.hour / 24),
        "month_sin": np.sin(2 * np.pi * local.month / 12),
        "month_cos": np.cos(2 * np.pi * local.month / 12),
        "month": local.month,
        "season": np.array([SEASONS[m] for m in local.month]),
        "is_weekend": np.isin(local.dayofweek, WEEKEND_DAYS).astype(int),
        "hour_local": local.hour,
        "doy_sin": np.sin(2 * np.pi * local.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * local.dayofyear / 365.25),
        "day_length": day_length.to_numpy(),
        # hours from the middle of the row's hour to solar noon (positive before noon)
        "time_to_solar_noon": ((sun_times["transit"] - mid).dt.total_seconds() / 3600)
        .to_numpy(),
        "daylight_fraction": (day_length / max_day_length).to_numpy(),
        "day_of_week": local.dayofweek,
    }, index=hours)
    return out


def max_day_length(site) -> float:
    """Longest day of the year at the site [h] (pvlib, a full year around the solstices)."""
    days = pd.date_range("2022-01-01", "2022-12-31", freq="D").tz_localize(TIMEZONE_LOCAL)
    import pvlib
    t = pvlib.solarposition.sun_rise_set_transit_spa(days, site.latitude, site.longitude)
    return float(((t["sunset"] - t["sunrise"]).dt.total_seconds() / 3600).max())


# ── targets ────────────────────────────────────────────────────────────────


def load_target_decision() -> dict | None:
    """The D-11 decision (configs/target.yaml), or None before it is taken."""
    if not TARGET_FILE.exists():
        return None
    return yaml.safe_load(TARGET_FILE.read_text(encoding="utf-8"))


def target_columns(hourly: pd.DataFrame, geom: pd.DataFrame) -> pd.DataFrame:
    """theo and k for both candidate denominators (D-11)."""
    out = pd.DataFrame(index=hourly.index)
    out[TARGET] = hourly[TARGET]
    for name, cs_col in DENOMINATORS.items():
        theo = theoretical_power(geom[cs_col])
        k, valid = clear_sky_index(hourly[TARGET], theo)
        out[f"theo_{name}_kw"] = theo
        out[f"k_{name}"] = k
        out[f"k_valid_{name}"] = valid
    return out


# ── assembly ──────────────────────────────────────────────────────────────


def build_base(hourly: pd.DataFrame, site, plane) -> pd.DataFrame:
    """Everything that does not depend on the horizon: metadata, targets, geometry, calendar."""
    hours = hourly.index
    geom = hourly_geometry(hours, site, plane)
    sun_times = daily_sun_times(hours, site)
    base = target_columns(hourly, geom)
    base["split"] = split_of(hours)
    base["daytime"] = daytime_mask(hours, site)
    base["op_limited_flag"] = hourly["op_limited_flag"].astype(bool)
    base["hour_source"] = hourly["hour_source"]
    first = hours[0].tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    local_day = hours.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    base["warmup"] = local_day < first + pd.Timedelta(days=WARMUP_DAYS)
    for col in FEATURE_GROUPS["clearsky"] + ("aoi", "solar_azimuth", "solar_zenith", "airmass"):
        base[col] = geom[col]
    cal = calendar_features(hours, sun_times, max_day_length(site))
    return base.join(cal)


def past_day_features(sources: dict[str, pd.Series], horizon: int
                      ) -> tuple[pd.DataFrame, pd.Series, int, pd.DatetimeIndex]:
    """All past-day features of D+horizon (lags and rolling), guarded and imputed. This is the
    only part of the table that depends on measured data, so the leakage tests (T-09, T-18)
    perturb its inputs. Returns (features, imputed share per feature, values removed by the
    DST guard, T0 per row)."""
    hours = next(iter(sources.values())).index
    t0 = issuance_utc(hours, horizon)
    past = pd.DataFrame({**lag_features(sources, horizon, t0),
                         **rolling_features(sources, horizon, t0)}, index=hours)
    # values the DST guard removed (a plain 24-hour shift would have had a value there)
    guard = sum(int((sources[col].shift(24 * d).notna() & past[f"{col}_lag_{d}d"].isna()).sum())
                for col in LAG_SOURCES for d in kept_lags(horizon))
    filled, shares = impute_same_hour(past)
    return filled, shares, guard, t0


def build_horizon(base: pd.DataFrame, sources: dict[str, pd.Series], horizon: int
                  ) -> tuple[pd.DataFrame, pd.Series, int]:
    """The table for D+horizon: base columns, T0, past-day features (guarded, imputed).
    Returns (table, imputed share per past-day feature, values removed by the DST guard)."""
    filled, shares, guard, t0 = past_day_features(sources, horizon)
    table = base.copy()
    table.insert(0, "t0_utc", t0)
    table = table.join(filled)
    expected = features_for_horizon(horizon)
    missing = [f for f in expected if f not in table.columns]
    if missing:
        raise RuntimeError(f"D+{horizon}: features missing from the table: {missing}")
    return table, shares, guard


def add_decided_target(table: pd.DataFrame, decision: dict) -> pd.DataFrame:
    """theo_kw, target_k, k_valid, k_suspect for the denominator chosen in D-11."""
    den = decision["k_denominator"]
    table["theo_kw"] = table[f"theo_{den}_kw"]
    table["target_k"] = table[f"k_{den}"]
    table["k_valid"] = table[f"k_valid_{den}"]
    table["k_suspect"] = suspect(table["target_k"])
    return table


def target_checks(t: pd.DataFrame, ac_minutes: pd.Series) -> dict:
    """Facts about the decided target that the data-processing record quotes (OP-33, OP-34):
    the smallest theoretical power of a daytime hour (§6.5 assumes every daytime hour is above
    the 50 kW threshold), the k_suspect hours (§6.5: examined and reported, never removed), and
    the daytime hours whose AC mean rests on fewer than 60 minutes while theo covers the full
    hour (the deviation from §6.4)."""
    day = t["daytime"].to_numpy()
    suspect_rows = t["k_suspect"].to_numpy(dtype=bool)
    partial = day & t["site_total_AC_kW"].notna().to_numpy() & (ac_minutes.to_numpy() < 60)
    valid = t["k_valid"].to_numpy(dtype=bool)
    split = t["split"].to_numpy()

    def by_split(mask: np.ndarray) -> dict[str, int]:
        return {s: int((mask & (split == s)).sum()) for s in ("train", "val", "test")}

    error = (t["target_k"] * t["theo_kw"] - t["site_total_AC_kW"]).abs()[valid]
    elevation = t.loc[suspect_rows, "sun_elevation"]
    return {
        "daytime_min_theo_kw": round(float(t.loc[day, "theo_kw"].min()), 1),
        "valid_hours": int(valid.sum()),
        "max_inversion_error_kw": float(error.max()),
        "k_suspect_hours": by_split(suspect_rows),
        "k_suspect_in_daytime": int((suspect_rows & day).sum()),
        "k_suspect_op_limited": int((suspect_rows & t["op_limited_flag"].to_numpy()).sum()),
        "k_suspect_sun_elevation_deg": [round(float(elevation.min()), 1),
                                        round(float(elevation.max()), 1)],
        "k_suspect_max_theo_kw": round(float(t.loc[suspect_rows, "theo_kw"].max()), 1),
        "partial_ac_daytime_hours": by_split(partial),
        "partial_ac_min_minutes": int(ac_minutes[partial].min()),
    }


GROUP_METHODS = {  # how each group is computed, and why it is known at T0
    "clearsky": ("hourly mean of the 60 minute values (pvlib Ineichen clear sky, monthly Linke "
                 "turbidity; POA by Hay-Davies on the panel plane, albedo 0.2)",
                 "astronomy and clear-sky model: known at any time"),
    "rolling_stats": ("D-22 masks, DST guard and same-hour 30-day median for gaps, then mean / "
                      "std over days h..h+N-1 back (D-04)", "only hours that ended by T0"),
    "lag_power": ("D-22 masks on power, DST guard, same-hour 30-day median for gaps",
                  "only hours that ended by T0"),
    "time_encodings": ("calendar of the target hour", "calendar: known at any time"),
    "other": ("pvlib solar position (apparent zenith) and sunrise / sunset / transit of the "
              "local day; hourly means of minute values", "astronomy and calendar: known at any "
              "time"),
}
FIXED_FEATURES = {  # name: (definition, unit)
    "ghi_clearsky": ("clear-sky global horizontal irradiance", "W/m2"),
    "dhi_clearsky": ("clear-sky diffuse horizontal irradiance", "W/m2"),
    "dni_clearsky": ("clear-sky direct normal irradiance", "W/m2"),
    "sun_elevation": ("apparent sun elevation (refraction corrected)", "deg"),
    "poa_global_clearsky": ("clear-sky irradiance on the panel plane, total (the k denominator, "
                            "D-11)", "W/m2"),
    "poa_direct_clearsky": ("clear-sky irradiance on the panel plane, direct beam", "W/m2"),
    "poa_diffuse_clearsky": ("clear-sky irradiance on the panel plane, sky + ground diffuse",
                             "W/m2"),
    "poa_sky_diffuse_clearsky": ("clear-sky irradiance on the panel plane, sky diffuse", "W/m2"),
    "poa_ground_diffuse_clearsky": ("clear-sky irradiance on the panel plane, ground reflected",
                                    "W/m2"),
    "hour_sin": ("sin(2 pi hour / 24), hour of the UTC timestamp (§6.6)", "-"),
    "hour_cos": ("cos(2 pi hour / 24), hour of the UTC timestamp (§6.6)", "-"),
    "month_sin": ("sin(2 pi month / 12), local month", "-"),
    "month_cos": ("cos(2 pi month / 12), local month", "-"),
    "month": ("local month, 1..12", "-"),
    "season": ("0 Dec-Feb, 1 Mar-May, 2 Jun-Aug, 3 Sep-Nov (local month)", "-"),
    "is_weekend": ("1 on Friday and Saturday (the Israeli weekend), local day", "0/1"),
    "hour_local": ("local clock hour of the hour's start, 0..23", "h"),
    "aoi": ("angle of incidence of the beam on the panel plane", "deg"),
    "solar_azimuth": ("sun azimuth, circular mean of the hour (north = 0, east = 90)", "deg"),
    "solar_zenith": ("apparent sun zenith", "deg"),
    "airmass": ("relative air mass (Kasten-Young 1989); NaN while the sun is below the horizon",
                "-"),
    "doy_sin": ("sin(2 pi day-of-year / 365.25), local day", "-"),
    "doy_cos": ("cos(2 pi day-of-year / 365.25), local day", "-"),
    "day_length": ("sunset - sunrise of the local day", "h"),
    "time_to_solar_noon": ("transit - middle of the hour; positive before solar noon", "h"),
    "daylight_fraction": ("day_length / the longest day of the year at the site", "-"),
    "day_of_week": ("local day of week, Monday = 0", "-"),
}
SOURCE_UNITS = {"s01_inv1_AC_kW": ("inverter 1 AC power", "kW"),
                "s01_inv2_AC_kW": ("inverter 2 AC power", "kW"),
                "site_total_AC_kW": ("site total AC power", "kW"),
                "irradiance": ("measured irradiance on the panel plane (D-09)", "W/m2"),
                "mod_temp1": ("module temperature, sensor 1", "degC"),
                "mod_temp2": ("module temperature, sensor 2", "degC"),
                "ambient_temp": ("ambient temperature (D-09)", "degC")}


def describe_feature(name: str) -> tuple[str, str]:
    """(definition, unit) of one contract feature."""
    if name in FIXED_FEATURES:
        return FIXED_FEATURES[name]
    if "_lag_" in name:
        source, days = name.rsplit("_lag_", 1)
        what, unit = SOURCE_UNITS[source]
        return f"{what} at the same UTC hour {days[:-1]} day(s) back", unit
    if "_rolling_" in name:  # e.g. s01_inv1_rolling_std_7d_same_tod
        source, rest = name.split("_rolling_")
        stat, window = rest.split("_")[:2]
        what, unit = SOURCE_UNITS[ROLLING_COLUMNS[source]]
        word = {"mean": "mean", "std": "standard deviation (ddof 1)"}[stat]
        return (f"{word} of {what} at the same UTC hour over {window[:-1]} days, starting h days "
                "back", unit)
    raise KeyError(name)


def feature_dictionary(imputed: dict[int, pd.Series]) -> pd.DataFrame:
    rows = []
    for name in features_for_horizon(1):
        group = group_of(name)
        definition, unit = describe_feature(name)
        method, availability = GROUP_METHODS[group]
        row = {"feature": name, "group": group, "definition": definition, "unit": unit,
               "method": method, "available_at_T0": availability,
               "horizons": ",".join(str(h) for h in HORIZONS
                                    if name in features_for_horizon(h))}
        for h, shares in imputed.items():
            row[f"imputed_share_D{h}"] = round(float(shares.get(name, 0.0)), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict:
    out_dir = stage_dir("06", "features")
    report_dir = output_dir("06_features")
    site = load_site()
    plane = load_panel_plane()
    decision = load_target_decision()
    policy = yaml.safe_load(POLICY_FILE.read_text(encoding="utf-8"))
    share = policy["hourly"]["op_limited_min_share"]
    with RunLog("06_features", "python -m solarpv.features.build", out_dir=report_dir) as log:
        inputs = log.inputs([HOURLY_FILE, MINUTE_FILE])
        inputs.update({p.relative_to(CONFIGS_DIR.parent).as_posix(): text_sha256(p)
                       for p in (POLICY_FILE, PANEL_FILE)})
        log.note(f"panel plane {plane.tilt}/{plane.azimuth} ({plane.model}); D-11 decision: "
                 f"{decision['k_denominator'] if decision else 'not yet taken (both kept)'}")
        with log.step("load stage-5 hourly data and minute flags"):
            hourly = pd.read_parquet(HOURLY_FILE)
            flags = pd.read_parquet(MINUTE_FILE, columns=sorted(
                {f for v in INVERTER_FLAGS.values() for f in v}))
            log.value("hours", len(hourly))
        with log.step("geometry, clear sky, calendar, targets"):
            base = build_base(hourly, site, plane)
            day = base["daytime"].to_numpy()
            train_day = day & (base["split"] == "train").to_numpy()
            for den in DENOMINATORS:
                k, valid = base[f"k_{den}"], base[f"k_valid_{den}"]
                log.note(f"k_{den}: valid daytime train hours {int((valid & train_day).sum())};"
                         f" shares above 1 / 1.2 / 1.5: "
                         f"{share_above(k, valid.to_numpy() & train_day)}")
        with log.step("D-22 masks"):
            sources, masked = masked_sources(hourly, flags, share)
            for col, n in masked.items():
                log.note(f"{col}: {n} hours masked before past-day features")
        tables, imputed, guards = {}, {}, {}
        for h in HORIZONS:
            with log.step(f"past-day features and table for D+{h}"):
                table, shares, guard = build_horizon(base, sources, h)
                if decision:
                    table = add_decided_target(table, decision)
                tables[h], imputed[h], guards[h] = table, shares, guard
                log.value(f"D{h}_dst_guard_removed_values", guard)
                log.value(f"D{h}_max_imputed_share", round(float(shares.max()), 5))
                log.value(f"D{h}_features", len(features_for_horizon(h)))
        files = []
        with log.step("write outputs"):
            for h, table in tables.items():
                path = out_dir / f"features_1H_D{h}.parquet"
                table.to_parquet(path)
                files.append(path)
            dictionary = feature_dictionary(imputed)
            dict_path = report_dir / "feature_dictionary.csv"
            dictionary.to_csv(dict_path, index=False, lineterminator="\n")
            files.append(dict_path)
            derived, checks = None, None
            if decision:
                t = tables[1]
                mdk = mean_daytime_k(t["target_k"], t["k_valid"], t["daytime"].to_numpy(),
                                     t["split"].to_numpy())
                derived = {"mean_daytime_k": round(mdk, 6),
                           "k_denominator": decision["k_denominator"],
                           "panel_plane": {"tilt": plane.tilt, "azimuth": plane.azimuth},
                           "source": "python -m solarpv.features.build (stage 6)"}
                DERIVED_CONSTANTS_FILE.write_text(
                    "# Written by stage 6 (solarpv.features.build); do not edit by hand.\n"
                    + yaml.safe_dump(derived, sort_keys=False), encoding="utf-8", newline="\n")
                log.value("mean_daytime_k", derived["mean_daytime_k"])
                files.append(DERIVED_CONSTANTS_FILE)
                checks = target_checks(t, hourly["n_site_total_AC_kW"])
                log.note(f"target checks: {checks}")
                if checks["daytime_min_theo_kw"] <= THEO_MIN_KW:
                    log.warn("a daytime hour has theo at or below the k threshold (§6.5)")
        out_hashes = log.outputs(files)
        manifest = {
            "stage": "06_features",
            "git_commit": git_commit(),
            "run_log": log.relative_path,
            "inputs_sha256": inputs,
            "panel_plane": {"tilt": plane.tilt, "azimuth": plane.azimuth, "model": plane.model},
            "k_denominator": decision["k_denominator"] if decision else None,
            "rows_per_horizon": {f"D{h}": len(t) for h, t in tables.items()},
            "features_per_horizon": {f"D{h}": features_for_horizon(h) for h in HORIZONS},
            "masked_power_hours": masked,
            "dst_guard_removed_values": {f"D{h}": g for h, g in guards.items()},
            "imputed_share_max": {f"D{h}": round(float(s.max()), 5) for h, s in imputed.items()},
            "warmup_hours": int(base["warmup"].sum()),
            "k_shares_train_daytime": {
                den: share_above(base[f"k_{den}"], base[f"k_valid_{den}"].to_numpy() & train_day)
                for den in DENOMINATORS},
            "derived_constants": derived,
            "target_checks": checks,  # None until D-11 is decided
            "outputs": out_hashes,
            "warnings": log.warnings,
        }
        manifest_text = json.dumps(manifest, indent=2, default=str)
        for path in (out_dir / "features_manifest.json", report_dir / "features_manifest.json"):
            path.write_text(manifest_text, encoding="utf-8", newline="\n")
        log.note("manifest written to data/processed/06_features and outputs/06_features")
    return manifest


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Two data sources in the raw export, and reconstruction of polled data (finding F-18).

Until 2025-02-25 every tag is a 1-minute average written by the site loggers ("minute" source,
two drifting logger clocks, F-15). From 2025-02-26 the export switches, tag by tag and hour by
hour, to a second source: an instantaneous snapshot polled every 15 minutes (each tag on its own
phase) and held until the next poll ("polled" source, its own clock). The minute in which a poll
falls holds a time-weighted blend of the old and the new value. After the switch the minute
source still appears for some hours, still on its old clocks. Each tag-hour is therefore assigned
to its source, and each source gets its own clock correction (clock.py).

Held values lag the signal by half the poll period (~7.5 min) and turn an hourly mean into a mean
of four stale samples, so continuous tags are reconstructed: every snapshot is put back at its
poll instant and the minutes in between are interpolated linearly. Step-like tags (states, modes,
setpoints) are left as they are.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from solarpv.config.constants import TIMEZONE_LOCAL

POLL_PERIOD_MIN = 15
DETECT_COLUMNS = ("irradiance", "s01_inv1_DC_kW", "s01_inv2_DC_kW", "s01_inv1_AC_kW",
                  "s01_inv2_AC_kW")
# tags whose own change rate identifies the source; other tags follow a reference tag
CONTINUOUS_COLUMNS = (
    "irradiance", "pyranometer", "ambient_temp", "site_ambient_temp", "mod_temp1", "mod_temp2",
    "s01_inv1_AC_kW", "s01_inv2_AC_kW", "s01_inv1_DC_kW", "s01_inv2_DC_kW",
    "s01_inv1_AC_kVar", "s01_inv2_AC_kVar",
)
MIN_DETECT_MINUTES = 20  # daytime minutes an hour needs for its source to be classified
POLL_RUN = (10, 16)  # constant-run lengths [min] of a polled tag (14 between blend minutes)
POLLED_SHARE_MIN = 0.5  # share of an hour's minutes in such runs that marks it as polled
MINUTE_RATE_MIN = 0.25  # changes per minute above which a non-polled hour is minute-source
ERA_RUN_DAYS = 7
ERA_MAX_GAP_DAYS = 2


def local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar day (naive midnight) of each UTC timestamp."""
    return pd.DatetimeIndex(index.tz_convert(TIMEZONE_LOCAL).date)


def reference_column(column: str) -> str:
    """Tag whose sampling pattern decides the source of `column`."""
    if column in CONTINUOUS_COLUMNS:
        return column
    if column.startswith("s01_inv2_"):
        return "s01_inv2_AC_kW"
    if column.startswith(("s01_", "iec_")):
        return "s01_inv1_AC_kW"
    return "irradiance"


def hourly_source(df: pd.DataFrame, daytime: np.ndarray) -> pd.DataFrame:
    """'polled', 'minute' or 'other' (frozen or partly frozen) per hour and column, from the
    daytime minutes of the hour (NaN if too few). Indexed by the UTC hour; local offsets are
    whole hours, so these are also the local hours."""
    hour = df.index[daytime].floor("h")
    out = {}
    for col in df.columns:
        v = df[col].to_numpy(dtype=float)
        changed = np.r_[True, v[1:] != v[:-1]]
        run_id = np.cumsum(changed) - 1
        run_len = np.bincount(run_id)[run_id]
        in_poll_run = (run_len >= POLL_RUN[0]) & (run_len <= POLL_RUN[1])
        stats = pd.DataFrame({"rate": changed[daytime], "poll": in_poll_run[daytime]}
                             ).groupby(hour).agg(["mean", "size"])
        state = np.where(stats[("poll", "mean")] >= POLLED_SHARE_MIN, "polled",
                         np.where(stats[("rate", "mean")] >= MINUTE_RATE_MIN, "minute", "other"))
        out[col] = pd.Series(state, index=stats.index).where(
            stats[("rate", "size")] >= MIN_DETECT_MINUTES)
    return pd.DataFrame(out)


@dataclass(frozen=True)
class PolledEra:
    """first_day: first appearance of the polled source (polled hours from here on are polled);
    era_start: first day of a sustained polled presence (from here on, hours that are not clearly
    minute-source, e.g. unclassified night hours, count as polled). Local calendar days."""

    first_day: pd.Timestamp
    era_start: pd.Timestamp


def polled_era(source: pd.DataFrame, columns=DETECT_COLUMNS) -> PolledEra | None:
    """era_start: first day of the first run of ERA_RUN_DAYS consecutive days that each have a
    polled hour in a main tag; first_day: walk back from era_start over days with polled hours
    separated by at most ERA_MAX_GAP_DAYS."""
    s = source[list(columns)]
    per_day = s.eq("polled").sum(axis=1).groupby(local_day(s.index)).sum()
    days = per_day.index[per_day > 0]
    if len(days) == 0:
        return None
    consecutive = days.to_series().diff().dt.days.eq(1).to_numpy()
    run = np.zeros(len(days), dtype=int)
    for i in range(1, len(days)):
        run[i] = run[i - 1] + 1 if consecutive[i] else 0
    ends = np.flatnonzero(run >= ERA_RUN_DAYS - 1)
    if len(ends) == 0:
        return None
    k = ends[0] - (ERA_RUN_DAYS - 1)
    era_start = days[k]
    while k > 0 and (days[k] - days[k - 1]).days <= ERA_MAX_GAP_DAYS + 1:
        k -= 1
    return PolledEra(first_day=days[k], era_start=era_start)


def source_mask(df: pd.DataFrame, hourly: pd.DataFrame, era: PolledEra | None) -> pd.DataFrame:
    """Per minute and column: 'minute' or 'polled'.

    Before era.first_day everything is minute-source. From era.first_day, hours whose reference
    tag is polled are polled. From era.era_start, every hour that is not clearly minute-source
    (polled, frozen, unclassified night) is polled."""
    out = pd.DataFrame("minute", index=df.index, columns=df.columns)
    if era is None:
        return out
    day = local_day(df.index)
    from_first = np.asarray(day >= era.first_day)
    from_era = np.asarray(day >= era.era_start)
    hour = df.index.floor("h")
    for col in df.columns:
        ref = reference_column(col)
        h = hourly[ref].reindex(hour).to_numpy() if ref in hourly else np.full(len(df), None)
        state = np.full(len(df), "minute", dtype=object)
        state[(from_first & (h == "polled")) | (from_era & (h != "minute"))] = "polled"
        out[col] = state
    return out


def poll_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Poll instants (minutes from the first sample; label t covers [t, t+1)) and values.

    A held value starting at minute r is preceded either by a one-minute blend b at r-1, in
    which case the poll fell at r-1+f with f = (v_new - b) / (v_new - v_old), or by another held
    value, in which case the poll fell on the boundary r. Polls that did not change the value
    leave no trace, so long runs get extra points every POLL_PERIOD_MIN.
    """
    v = np.asarray(values, dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(v) != 0) + 1]
    lengths = np.diff(np.r_[starts, v.size])
    times, vals = [], []
    for k, (r, n) in enumerate(zip(starts, lengths, strict=True)):
        if n < 2 and 0 < k < len(starts) - 1:
            continue  # blend minute, handled with the run that follows
        t0 = float(r)
        if k >= 2 and lengths[k - 1] == 1:
            old, blend, new = v[starts[k - 2]], v[starts[k - 1]], v[r]
            f = (new - blend) / (new - old) if new != old else 0.5
            t0 = r - 1 + float(np.clip(f, 0.0, 1.0))
        t = t0
        while t < r + n:
            times.append(t)
            vals.append(v[r])
            t += POLL_PERIOD_MIN
    return np.asarray(times), np.asarray(vals)


def dehold(s: pd.Series) -> pd.Series:
    """Reconstruct a polled (sample-and-hold) series on its 1-minute grid. NaNs split blocks."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    v = s.to_numpy(dtype=float)
    ok = np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        t, x = poll_points(v[a:b])
        out.iloc[a:b] = np.interp(np.arange(b - a) + 0.5, t, x)
    return out


def harmonise_sources(df: pd.DataFrame, daytime: np.ndarray
                      ) -> tuple[pd.DataFrame, pd.DataFrame, PolledEra | None]:
    """Assign every tag-minute to its source and reconstruct polled continuous tags.

    Returns (data on the stamped-time grid, per-minute source mask, polled era)."""
    hourly = hourly_source(df[[c for c in df.columns if c in CONTINUOUS_COLUMNS]], daytime)
    era = polled_era(hourly)
    source = source_mask(df, hourly, era)
    out = df.astype(float)
    for col in df.columns:
        polled = source[col].eq("polled").to_numpy()
        if col in CONTINUOUS_COLUMNS and polled.any():
            # minute-source minutes become NaN, which splits the blocks dehold() works on
            rebuilt = dehold(out[col].where(polled)).to_numpy()
            out.loc[polled, col] = rebuilt[polled]
    return out, source, era

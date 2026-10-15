"""Estimate the clock offsets of both data sources and write configs/clock_offsets.csv (D-19).

    python scripts/estimate_clock_drift.py

1. Source of every tag-minute and reconstruction of polled tags (resolution.py, F-18).
2. Daily estimates per source at the measured orientation (F-13: tilt 20, azimuth 180).
3. Minute source: step (resync) days from the inter-clock difference (exact to the day) and from
   the meteo offset; rolling median within segments; inverter = meteo + delta.
4. Polled source: meteo offset from the irradiance sensor; the inverter clock is compared with
   the meteo clock through tau_dc - tau_met against the same quantity in the minute source.
5. Checks: (a) anchors - the offset at the start of the data and right after the 2023-05 meteo
   resync should be ~0; (b) internal - after the correction, |monthly median| <= 2 min for every
   estimate of both sources.
Report files: 00_data_audit/clock_drift_2026-10-14/.
"""

import json
import sys

import numpy as np
import pandas as pd

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, PROCESSED_DIR
from solarpv.config.site import load_site
from solarpv.data.clock import (
    DC_COLUMNS,
    MIN_ELEV,
    SOURCES,
    Orientation,
    apply_correction,
    clear_sky_components,
    coarse_steps,
    daily_estimates,
    refine_step,
    smooth_within_segments,
)
from solarpv.data.resolution import harmonise_sources

REPORT_DIR = AUDIT_DIR / "clock_drift_2026-10-14"
OFFSETS_FILE = CONFIGS_DIR / "clock_offsets.csv"
COLUMNS = ["irradiance", *DC_COLUMNS, "s01_inv1_AC_kW", "s01_inv2_AC_kW"]
CLEAR_RMSE_MAX = 0.03
DC_RMSE_MAX = 0.06
DELTA_CORR_MIN = 0.7
DELTA_VAR_MIN = 0.005
STEP_MIN = {"delta": 3.0, "tau": 4.0}
STEP_MERGE_DAYS = 10
ANCHOR_DAYS = 30
SINGLE_CLOCK_TOL = 1.0  # min; polled inverter vs meteo clock
TOLERANCE = 2.0  # min; acceptance for anchors and the internal check


def reliable_tau(est: pd.DataFrame, rmse_max: float = CLEAR_RMSE_MAX) -> pd.DataFrame:
    return est[(est["clear_rel_rmse"] <= rmse_max)
               & est["clear_ratio"].between(0.9, 1.25) & (est["tau_met"].abs() < 29)]


def reliable_delta(est: pd.DataFrame, corr_min: float = DELTA_CORR_MIN) -> pd.DataFrame:
    return est[(est["delta_corr"] >= corr_min) & (est["variability"] >= DELTA_VAR_MIN)
               & (est["delta"].abs() < 29)]


def reliable_dc(est: pd.DataFrame) -> pd.DataFrame:
    t = reliable_tau(est)
    return t[(t["dc_rel_rmse"] <= DC_RMSE_MAX) & (t["tau_dc"].abs() < 29)]


def find_steps(est: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """Steps seen in delta are dated by delta (sharper); tau steps are kept only if new.
    Detection uses the reliable days; dating uses every usable day, so a step is not pushed
    to a later day just because the first day of the new regime was calm."""
    rt, rd = reliable_tau(est), reliable_delta(est)
    ut = reliable_tau(est, 0.06)
    ud = est[(est["delta_corr"] >= 0.5) & (est["delta"].abs() < 29)]
    steps = [("delta", refine_step(ud["day"], ud["delta"], c))
             for c in coarse_steps(rd["day"], rd["delta"], STEP_MIN["delta"])]
    for c in coarse_steps(rt["day"], rt["tau_met"], STEP_MIN["tau"]):
        r = refine_step(ut["day"], ut["tau_met"], c)
        if all(abs((r - s).days) > STEP_MERGE_DAYS for _, s in steps):
            steps.append(("tau", r))
    return sorted(steps, key=lambda s: s[1])


def monthly(est: pd.DataFrame, dc_bias: float | None = None) -> pd.DataFrame:
    t, d = reliable_tau(est), reliable_delta(est)
    cols = {"n_tau": t.groupby(t["day"].dt.to_period("M")).size(),
            "tau_met": t.groupby(t["day"].dt.to_period("M"))["tau_met"].median()}
    if d["delta"].notna().any():
        cols["n_delta"] = d.groupby(d["day"].dt.to_period("M")).size()
        cols["delta"] = d.groupby(d["day"].dt.to_period("M"))["delta"].median()
    if dc_bias is not None:
        c = reliable_dc(est)
        gap = c["tau_dc"] - c["tau_met"] - dc_bias
        cols["dc_clock_gap"] = gap.groupby(c["day"].dt.to_period("M")).median()
    return pd.DataFrame(cols).round(2)


def usable(source: pd.DataFrame, name) -> pd.DataFrame:
    return pd.DataFrame({
        "irradiance": source["irradiance"].eq(name).to_numpy(),
        "dc": (source[DC_COLUMNS[0]].eq(name) & source[DC_COLUMNS[1]].eq(name)).to_numpy(),
    }, index=source.index)


def main() -> int:
    site = load_site()
    raw = pd.read_parquet(PROCESSED_DIR / "01_ingest" / "scada_1min_utc.parquet", columns=COLUMNS)
    comp = clear_sky_components(raw.index, site)
    data, source, era = harmonise_sources(raw, comp["elev"].to_numpy() > MIN_ELEV)
    print(f"polled source: first day {era.first_day.date()}, era from {era.era_start.date()}",
          flush=True)
    orient = Orientation(site.surface_tilt, site.surface_azimuth)
    est = {s: daily_estimates(data, comp, orient, usable(source, s), with_delta=s == "minute")
           for s in SOURCES}
    em, ep = est["minute"], est["polled"]
    print(f"daily estimates: minute {len(em)} days, polled {len(ep)} days", flush=True)

    # minute source
    steps = find_steps(em)
    print("steps:", [(k, str(s.date())) for k, s in steps])
    step_days = [s for _, s in steps]
    local = raw.index.tz_convert("Asia/Jerusalem")
    calendar = pd.date_range(str(local[0].date()), str(local[-1].date()), freq="D")
    rt, rd = reliable_tau(em), reliable_delta(em)
    m_meteo = smooth_within_segments(rt["day"], rt["tau_met"], step_days, calendar)
    m_delta = smooth_within_segments(rd["day"], rd["delta"], step_days, calendar, window=61,
                                     min_obs=3).fillna(0.0)

    # polled source
    pt = reliable_tau(ep)
    p_cal = calendar[calendar >= era.first_day]
    p_meteo = smooth_within_segments(pt["day"], pt["tau_met"], [], p_cal)
    mc = reliable_dc(em)
    mc_gap = mc["tau_dc"] - mc["tau_met"] - m_delta.reindex(mc["day"]).to_numpy()
    months = sorted(set(reliable_dc(ep)["day"].dt.month))
    dc_bias = float(mc_gap[mc["day"].dt.month.isin(months)].median())
    pc = reliable_dc(ep)
    p_gap = float((pc["tau_dc"] - pc["tau_met"]).median()) - dc_bias
    single_clock = abs(p_gap) <= SINGLE_CLOCK_TOL
    print(f"polled inverter-meteo gap {p_gap:+.2f} min (DC bias {dc_bias:+.2f} from minute-source "
          f"months {months}) -> {'one clock' if single_clock else 'separate offset'}")

    offsets = pd.DataFrame({
        "minute_meteo": m_meteo, "minute_delta": m_delta, "minute_inverter": m_meteo + m_delta,
        "polled_meteo": p_meteo.reindex(calendar),
        "polled_inverter": (p_meteo + (0.0 if single_clock else p_gap)).reindex(calendar),
    })
    offsets.index.name = "day"
    if offsets[["minute_meteo", "minute_inverter"]].isna().any().any():
        raise ValueError("minute-source offset undefined for some days")
    if offsets.loc[offsets.index >= era.first_day, ["polled_meteo", "polled_inverter"]
                   ].isna().any().any():
        raise ValueError("polled-source offset undefined for some days")
    offsets.round(3).to_csv(OFFSETS_FILE, lineterminator="\n")

    # (a) anchors
    first = rt["day"].min()
    meteo_resync = next(s for k, s in steps if s.year == 2023 and s.month == 5)
    anchors = {}
    for name, start in (("data start", first), ("meteo resync 2023-05", meteo_resync)):
        win = rt[(rt["day"] >= start) & (rt["day"] < start + pd.Timedelta(days=ANCHOR_DAYS))]
        anchors[name] = {"from": str(start.date()), "n_days": int(len(win)),
                         "median_offset_min": round(float(win["tau_met"].median()), 2)}
    print("anchors:", anchors)

    # (b) internal: apply, re-estimate on the corrected data (source mask moved the same way)
    code = source.replace({"minute": 0.0, "polled": 1.0}).astype(float)
    corrected = apply_correction(data, offsets, source)
    moved_source = apply_correction(code, offsets, source).replace({0.0: "minute",
                                                                    1.0: "polled"})
    after = {s: daily_estimates(corrected, comp, orient, usable(moved_source, s),
                                with_delta=s == "minute") for s in SOURCES}
    reports = {s: monthly(est[s], dc_bias if s == "polled" else None).join(
        monthly(after[s], dc_bias if s == "polled" else None), rsuffix="_after")
        for s in SOURCES}
    worst = {}
    for s, rep in reports.items():
        print(f"\n[{s} source]\n{rep.to_string()}")
        cols = [c for c in rep.columns if c.endswith("_after") and not c.startswith("n_")]
        worst[s] = {c: float(rep[c].abs().max()) for c in cols}
    print("\nafter correction, max |monthly median|:", worst)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SOURCES:
        est[s].to_csv(REPORT_DIR / f"daily_estimates_{s}_before.csv", index=False)
        after[s].to_csv(REPORT_DIR / f"daily_estimates_{s}_after.csv", index=False)
        reports[s].to_csv(REPORT_DIR / f"monthly_{s}_before_after.csv")
    ok_anchor = all(abs(a["median_offset_min"]) <= TOLERANCE for a in anchors.values())
    ok_internal = all(v <= TOLERANCE for w in worst.values() for v in w.values()
                      if np.isfinite(v))
    (REPORT_DIR / "clock_fit_summary.json").write_text(json.dumps({
        "orientation": {"tilt": orient.tilt, "azimuth": orient.azimuth},
        "polled_source": {"first_day": str(era.first_day.date()),
                          "era_start": str(era.era_start.date()),
                          "inverter_meteo_gap_min": round(p_gap, 2),
                          "dc_bias_min": round(dc_bias, 2), "single_clock": single_clock},
        "steps": [{"kind": k, "day": str(s.date())} for k, s in steps],
        "anchors": anchors,
        "validation_max_abs_monthly_after": worst,
        "passed": {"anchors": ok_anchor, "internal": ok_internal},
    }, indent=2), encoding="utf-8")
    print(f"\nanchors ok: {ok_anchor}, internal ok: {ok_internal}")
    return 0 if ok_anchor and ok_internal else 1


if __name__ == "__main__":
    sys.exit(main())

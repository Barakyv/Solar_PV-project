"""Stage 6b, S2-S3: GFS 1-degree runs -> hourly values at the site -> features per horizon
(docs/stage6b_plan.md; PROJECT_MASTER.md §6.12, D-12).

    python -m solarpv.nwp.gfs

Time rules (T-17, plan §2a N1-N4, N7):
- **Run choice.** The forecast for local day D+h is issued at T0 = 00:00 local of day D+1 (D-16,
  stage 6 `issuance_utc`). It uses the 12Z run of the UTC date of T0, or the flagged fallback
  of the downloader (06Z, then 00Z of that day).
- **Availability.** Every row satisfies init + 6 h <= T0, and the latest upload time of the
  run's files <= T0. A row that breaks this raises.
- **Labels.** The row labelled t covers [t, t+1). It receives GFS values for (t, t+1], i.e. the
  3-hour block that contains lead (t + 1 - init) for averaged fields, and the time t + 30 min
  for instantaneous fields.
- **Gaps.** Never backward fill. A row without a run stays NaN.

Reconstruction:
- **De-averaging (L-06).** GFS averages run from the start s of each 6-hour window. The 3-hour
  mean over (n-3, n] is ((n-s) A(s,n) - (n-3-s) A(s,n-3)) / 3, or A(s,n) when n-3 = s.
- **Irradiance to hours.** The clear-sky index of each 3-hour block (block mean / mean
  clear-sky GHI over the same hours) times the hourly clear-sky GHI. This keeps every block
  mean exactly. Below 20 W/m2 of clear sky the block mean is spread evenly.
- **Averaged cloud fields:** constant within the block.
- **Instantaneous fields:** linear in time.
- **Space:** bilinear from the 1-degree cell that contains the site; the 3x3 neighbourhood gives
  the cloud mean and std (L-08).
"""

import json
import sys

import numpy as np
import pandas as pd
import pvlib

from solarpv.config.constants import (
    AC_CAPACITY_KW,
    HORIZONS,
    NWP_MIN_AVAILABILITY_HOURS,
    THEO_MIN_KW,
    TIMEZONE_LOCAL,
)
from solarpv.config.paths import OUTPUTS_DIR, PROCESSED_DIR
from solarpv.config.provenance import git_commit
from solarpv.config.runlog import RunLog
from solarpv.config.site import load_site
from solarpv.features.build import issuance_utc
from solarpv.features.geometry import load_panel_plane
from solarpv.nwp.download import MESSAGES, OUT_DIR

FEATURES_D1 = PROCESSED_DIR / "06_features" / "features_1H_D1.parquet"
HOURLY_FILE = PROCESSED_DIR / "05_clean" / "scada_1H_clean.parquet"
STAGE_DIR = PROCESSED_DIR / "06b_gfs"
REPORT_DIR = OUTPUTS_DIR / "06b_gfs"
AVERAGED = tuple(name for name, _, _, kind in MESSAGES if kind == "ave")
INSTANT = tuple(name for name, _, _, kind in MESSAGES if kind == "inst")
CS_MIN_WM2 = 20.0          # below this clear-sky irradiance the clear-sky index is not used
GAMMA_PER_K = -0.0042      # temperature coefficient measured in stage 3 (table_physics)
PREFIX = "fcst_gfs_"


# ── reconstruction ─────────────────────────────────────────────────────────


def parse_step_range(step_range: str) -> tuple[int, int]:
    start, end = (int(x) for x in str(step_range).split("-"))
    return start, end


def three_hour_means(leads: pd.Series, step_ranges: pd.Series, values: pd.Series) -> pd.Series:
    """3-hour block means indexed by the block's end lead, from window-start averages (L-06)."""
    a = pd.DataFrame({"lead": leads.to_numpy(), "range": step_ranges.to_numpy(),
                      "value": values.to_numpy()}).set_index("lead").sort_index()
    out = {}
    for n, row in a.iterrows():
        s, end = parse_step_range(row["range"])
        if end != n:
            raise ValueError(f"step range {row['range']} does not end at lead {n}")
        if n - 3 == s:
            out[n] = row["value"]
        else:
            if n - 3 not in a.index:
                raise ValueError(f"lead {n - 3} is needed to de-average the window {s}-{n}")
            prev_s, prev_end = parse_step_range(a.loc[n - 3, "range"])
            if (prev_s, prev_end) != (s, n - 3):
                raise ValueError(f"lead {n - 3} has range {a.loc[n - 3, 'range']}, expected "
                                 f"{s}-{n - 3}")
            out[n] = ((n - s) * row["value"] - (n - 3 - s) * a.loc[n - 3, "value"]) / 3.0
    return pd.Series(out, name="mean").sort_index()


def bilinear_weights(lat: float, lon: float) -> dict[tuple[float, float], float]:
    """Weights of the four 1-degree grid points around (lat, lon)."""
    lat0, lon0 = np.floor(lat), np.floor(lon)
    wy, wx = lat - lat0, lon - lon0
    return {(lat0, lon0): (1 - wy) * (1 - wx), (lat0, lon0 + 1): (1 - wy) * wx,
            (lat0 + 1, lon0): wy * (1 - wx), (lat0 + 1, lon0 + 1): wy * wx}


def at_site(frame: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Bilinear value per (variable, lead, step_range) from the 3x3 point rows."""
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("bilinear weights do not sum to 1")
    w = pd.DataFrame([(lat, lon, wt) for (lat, lon), wt in weights.items()],
                     columns=["lat", "lon", "w"])
    joined = frame.merge(w, on=["lat", "lon"], how="inner")
    if len(joined) != 4 * frame.groupby(["variable", "lead"]).ngroups:
        raise ValueError("a corner of the bilinear cell is missing from the run")
    return (joined.assign(wv=joined["value"] * joined["w"])
            .groupby(["variable", "lead", "step_range"], as_index=False)["wv"].sum()
            .rename(columns={"wv": "value"}))


def hourly_from_run(run: pd.DataFrame, init: pd.Timestamp, hours: pd.DatetimeIndex,
                    ghi_cs: pd.Series, weights: dict) -> pd.DataFrame:
    """Hourly site values of one run for the target rows `hours` (row t covers [t, t+1))."""
    site = at_site(run, weights)
    lead_end = ((hours + pd.Timedelta(hours=1) - init) / pd.Timedelta(hours=1)).to_numpy()
    block_end = (np.ceil(lead_end / 3.0) * 3).astype(int)
    out = pd.DataFrame(index=hours)
    out["gfs_lead_end_h"] = lead_end
    for name in AVERAGED:
        v = site[site["variable"] == name]
        means = three_hour_means(v["lead"], v["step_range"], v["value"])
        out[name] = means.reindex(block_end).to_numpy()
    # irradiance: clear-sky index of each block applied to the hourly clear-sky GHI. The block
    # mean of the clear sky uses its three hours from the full series (also outside `hours`)
    cs = ghi_cs.reindex(hours).to_numpy()
    starts = pd.DatetimeIndex(init + pd.to_timedelta(block_end - 3, unit="h"))
    block_cs = np.nanmean(np.vstack([ghi_cs.reindex(starts + pd.Timedelta(hours=k)).to_numpy()
                                     for k in range(3)]), axis=0)
    kt_block = np.where(block_cs > CS_MIN_WM2, out["dswrf"] / np.maximum(block_cs, 1e-9), np.nan)
    out["ghi"] = np.where(block_cs > CS_MIN_WM2, kt_block * cs, out["dswrf"])
    # instantaneous fields: linear in time at the middle of the hour
    mid = lead_end - 0.5
    for name in INSTANT:
        v = site[site["variable"] == name].sort_values("lead")
        out[name] = np.interp(mid, v["lead"].to_numpy(float), v["value"].to_numpy(float),
                              left=np.nan, right=np.nan)
    out["wind10"] = np.hypot(out.pop("u10"), out.pop("v10"))
    out["t2m"] = out["t2m"] - 273.15  # K -> degC
    # cloud neighbourhood (L-08): block means at each of the 9 points
    tcc = run[run["variable"] == "tcc"]
    per_point = [three_hour_means(g["lead"], g["step_range"], g["value"]).reindex(block_end)
                 .to_numpy() for _, g in tcc.groupby(["lat", "lon"])]
    stack = np.vstack(per_point)
    out["tcc_mean3x3"], out["tcc_std3x3"] = stack.mean(axis=0), stack.std(axis=0, ddof=1)
    return out.drop(columns=["dswrf"])


# ── features ───────────────────────────────────────────────────────────────


def physical_features(h: pd.DataFrame, geo: pd.DataFrame, plane) -> pd.DataFrame:
    """POA, cell temperature and physical PV power from the GFS GHI (L-01, L-03)."""
    zen = geo["solar_zenith"].reindex(h.index)
    azi = geo["solar_azimuth"].reindex(h.index)
    ghi = h["ghi"].clip(lower=0)
    doy = h.index.dayofyear
    split = pvlib.irradiance.erbs(ghi, zen, doy)
    extra = pvlib.irradiance.get_extra_radiation(h.index + pd.Timedelta(minutes=30))
    poa = pvlib.irradiance.get_total_irradiance(
        plane.tilt, plane.azimuth, zen, azi, split["dni"], ghi, split["dhi"], dni_extra=extra,
        model="haydavies", albedo=plane.albedo)["poa_global"].fillna(0.0).clip(lower=0)
    ghi_cs = geo["ghi_clearsky"].reindex(h.index)
    poa_cs = geo["poa_global_clearsky"].reindex(h.index)
    theo = geo["theo_kw"].reindex(h.index)
    out = pd.DataFrame(index=h.index)
    out["kt"] = np.where(ghi_cs > CS_MIN_WM2, ghi / ghi_cs, 0.0)
    out["poa"] = poa
    out["kt_poa"] = np.where(poa_cs > CS_MIN_WM2, poa / poa_cs, 0.0)
    out["tcell"] = pvlib.temperature.faiman(poa, h["t2m"], h["wind10"])
    out["pv_kw"] = (poa * AC_CAPACITY_KW / 1000.0 * (1 + GAMMA_PER_K * (out["tcell"] - 25.0))
                    ).clip(0, AC_CAPACITY_KW)
    out["k_phys"] = np.where(theo > THEO_MIN_KW, out["pv_kw"] / theo, 0.0)
    return out


def run_table(run: pd.DataFrame, record: dict, geo: pd.DataFrame, plane, weights: dict
              ) -> pd.DataFrame:
    """All hourly GFS columns of one run, for every target row of its three forecast days."""
    day = pd.Timestamp(record["day"])
    t0 = (day + pd.Timedelta(days=1)).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    end = (day + pd.Timedelta(days=4)).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    # one hour either side for the centred mean (L-02); both are inside the run
    hours = pd.date_range(t0 - pd.Timedelta(hours=1), end, freq="h", inclusive="both")
    hours = hours[hours.isin(geo.index)]
    init = pd.Timestamp(record["init_utc"])
    h = hourly_from_run(run, init, hours, geo["ghi_clearsky"], weights)
    h = h.join(physical_features(h, geo, plane))
    h["kt_poa_ma3"] = h["kt_poa"].rolling(3, center=True).mean()
    h["gfs_init"] = init
    h["gfs_run"] = int(record["run"])
    h["gfs_fallback"] = bool(record["fallback"])
    h["gfs_uploaded_max"] = max(pd.Timestamp(f["uploaded_utc"]) for f in record["files"])
    return h


def horizon_rows(table: pd.DataFrame, record: dict, horizon: int) -> pd.DataFrame:
    """The rows of local day D+horizon of a run's table, with T0 and the T-17 check."""
    first_day = pd.Timestamp(record["day"]) + pd.Timedelta(days=1)  # local date of D+1
    local = table.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    rows = table[local == first_day + pd.Timedelta(days=horizon - 1)].copy()
    rows["t0_utc"] = issuance_utc(rows.index, horizon)
    available = rows["gfs_init"] + pd.Timedelta(hours=NWP_MIN_AVAILABILITY_HOURS)
    if not ((available <= rows["t0_utc"]) & (rows["gfs_uploaded_max"] <= rows["t0_utc"])).all():
        raise ValueError(f"T-17: run {record['day']} {record['run']}Z not available at T0")
    return rows


def daily_means(rows: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Daily mean of kt_poa and tcc over the daytime hours of the target local day (L-02);
    all from the same run, so known at T0."""
    day = geo["daytime"].reindex(rows.index).to_numpy()
    local = rows.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    out = pd.DataFrame(index=rows.index)
    for col in ("kt_poa", "tcc"):
        out[f"{col}_daymean"] = (rows[col].where(day).groupby(local).transform("mean")
                                 .to_numpy())
    return out


def clear_index_prevday(hourly: pd.DataFrame, geo: pd.DataFrame, horizon: int) -> pd.Series:
    """L-11a: p90 of measured in-plane irradiance / POA_cs over hours with the sun above 20 deg,
    on the last full local day before T0. Not from GFS; known at T0 by construction."""
    poa_cs = geo["poa_global_clearsky"]
    ratio = (hourly["irradiance"] / poa_cs).where((geo["sun_elevation"] > 20) & (poa_cs > 50))
    local = geo.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    daily = ratio.groupby(local).quantile(0.9)
    t0 = issuance_utc(geo.index, horizon)
    prev = t0.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize() - pd.Timedelta(days=1)
    return pd.Series(daily.reindex(prev).to_numpy(), index=geo.index,
                     name="irradiance_clear_index_prevday")


GFS_COLUMNS = ("ghi", "kt", "poa", "kt_poa", "kt_poa_ma3", "kt_poa_daymean", "tcc", "lcc", "mcc",
               "hcc", "tcc_mean3x3", "tcc_std3x3", "tcc_daymean", "t2m", "rh2m", "pwat",
               "wind10", "sunsd", "tcell", "pv_kw", "k_phys")
META_COLUMNS = ("t0_utc", "gfs_init", "gfs_run", "gfs_fallback", "gfs_uploaded_max",
                "gfs_lead_end_h")


def load_runs() -> list[tuple[pd.DataFrame, dict]]:
    runs = []
    for record_path in sorted(OUT_DIR.glob("*/*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            runs.append((pd.read_parquet(record_path.with_suffix(".parquet")), record))
    return runs


def build_tables(runs, geo: pd.DataFrame, hourly: pd.DataFrame, plane, weights: dict
                 ) -> dict[int, pd.DataFrame]:
    parts = {h: [] for h in HORIZONS}
    for run, record in runs:
        table = run_table(run, record, geo, plane, weights)
        for h in HORIZONS:
            rows = horizon_rows(table, record, h)
            parts[h].append(rows.join(daily_means(rows, geo)))
    out = {}
    for h in HORIZONS:
        t = pd.concat(parts[h]).sort_index() if parts[h] else pd.DataFrame()
        if t.index.duplicated().any():
            raise ValueError(f"D+{h}: a target hour comes from two runs")
        t = t.reindex(geo.index)
        frame = t[list(META_COLUMNS)].copy()
        for col in GFS_COLUMNS:
            frame[f"{PREFIX}{col}"] = t[col]
        frame["irradiance_clear_index_prevday"] = clear_index_prevday(hourly, geo, h)
        out[h] = frame
    return out


def run() -> dict:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with RunLog("06b_gfs", "python -m solarpv.nwp.gfs", out_dir=REPORT_DIR) as log:
        log.inputs([FEATURES_D1, HOURLY_FILE])
        geo = pd.read_parquet(FEATURES_D1, columns=["ghi_clearsky", "poa_global_clearsky",
                                                    "solar_zenith", "solar_azimuth",
                                                    "sun_elevation", "theo_kw", "daytime",
                                                    "split"])
        hourly = pd.read_parquet(HOURLY_FILE, columns=["irradiance"])
        site, plane = load_site(), load_panel_plane()
        weights = bilinear_weights(site.latitude, site.longitude)
        with log.step("load runs"):
            runs = load_runs()
            log.value("runs", len(runs))
            log.value("fallback_runs", sum(r["fallback"] for _, r in runs))
        with log.step("hourly reconstruction and features per horizon"):
            tables = build_tables(runs, geo, hourly, plane, weights)
        files = []
        for h, t in tables.items():
            path = STAGE_DIR / f"gfs_features_D{h}.parquet"
            t.to_parquet(path)
            files.append(path)
            log.value(f"D{h}_rows_with_gfs", int(t[f"{PREFIX}ghi"].notna().sum()))
        out_hashes = log.outputs(files)
        manifest = {"stage": "06b_gfs", "git_commit": git_commit(), "run_log": log.relative_path,
                    "runs": len(runs), "fallback_runs": int(sum(r["fallback"] for _, r in runs)),
                    "rows_with_gfs": {f"D{h}": int(t[f"{PREFIX}ghi"].notna().sum())
                                      for h, t in tables.items()},
                    "columns": [c for c in tables[1].columns if c not in META_COLUMNS],
                    "outputs": out_hashes, "warnings": log.warnings}
        for path in (STAGE_DIR / "gfs_manifest.json", REPORT_DIR / "gfs_manifest.json"):
            path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8",
                            newline="\n")
    return manifest


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

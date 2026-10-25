"""Stage 6b on synthetic data (docs/stage6b_plan.md §5).

T-42  de-averaging recovers 3-hour means; the hourly reconstruction keeps them; label convention
      (row t gets (t, t+1]); instantaneous fields interpolated at the middle of the hour
T-43  run choice: a run not available at T0 is refused (T-17), the D+h local day is selected
      also on DST days, and a later run changes nothing that an earlier run serves
      plus the downloader's helpers (index parsing, message selection, range merging)
"""

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.nwp import download as D
from solarpv.nwp import gfs as G

POINTS = [(lat, lon) for lat in (31.0, 30.0, 29.0) for lon in (34.0, 35.0, 36.0)]


def _window_start(n: int) -> int:
    return 6 * ((n - 1) // 6)


def _run_frame(leads, hourly_fields: dict[str, np.ndarray], inst_fields: dict[str, np.ndarray]):
    """A synthetic run in the downloader's layout. `hourly_fields[name][k]` is the true mean over
    lead hour (k-1, k]; averaged messages are built as GFS builds them (from the window start)."""
    rows = []
    for name, f in hourly_fields.items():
        for n in leads:
            s = _window_start(n)
            value = float(np.mean(f[s + 1:n + 1]))
            rows += [{"lead": n, "variable": name, "step_range": f"{s}-{n}", "step_type": "avg",
                      "lat": lat, "lon": lon, "value": value} for lat, lon in POINTS]
    for name, f in inst_fields.items():
        for n in leads:
            value = float(f[n])
            if name in G.ACCUMULATED_IN_WINDOW:  # seconds of sunshine since the window start
                value = float(np.sum(f[_window_start(n) + 1:n + 1]) * 3600)
            rows += [{"lead": n, "variable": name, "step_range": str(n), "step_type": "instant",
                      "lat": lat, "lon": lon, "value": value} for lat, lon in POINTS]
    return pd.DataFrame(rows)


def _fields(n_hours=100, seed=0):
    rng = np.random.default_rng(seed)
    ave = {name: rng.uniform(0, 100, n_hours) for name in G.AVERAGED}
    ave["dswrf"] = rng.uniform(0, 900, n_hours)
    inst = {name: np.linspace(1, 2, n_hours) * (i + 1) for i, name in enumerate(G.INSTANT)}
    inst["t2m"] = 280 + np.arange(n_hours, dtype=float)  # linear in lead: exact interpolation
    inst["sunsd"] = rng.uniform(0, 1, n_hours)  # sunshine fraction of each lead hour
    return ave, inst


# ── T-42 ────────────────────────────────────────────────────────────────────

def test_deaveraging_recovers_three_hour_means():
    f = np.random.default_rng(1).uniform(0, 1000, 100)
    leads = list(range(9, 85, 3))
    run = _run_frame(leads, {"dswrf": f}, {})
    v = run[(run["lat"] == 30.0) & (run["lon"] == 35.0)]
    means = G.three_hour_means(v["lead"], v["step_range"], v["value"])
    for n in leads:
        assert means[n] == pytest.approx(np.mean(f[n - 2:n + 1]))  # hours n-2, n-1, n


def test_deaveraging_rejects_inconsistent_windows():
    """Lead 12 averages from 6, so lead 9 must also average from 6; a 3-9 range breaks it."""
    leads = pd.Series([9, 12])
    with pytest.raises(ValueError, match="expected 3-9"):
        G.three_hour_means(leads, pd.Series(["6-9", "3-12"]), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="lead 6 is needed"):
        G.three_hour_means(leads, pd.Series(["3-9", "3-12"]), pd.Series([1.0, 2.0]))
    # a 3-hour range is a block mean by itself, whatever the window convention
    means = G.three_hour_means(leads, pd.Series(["6-9", "9-12"]), pd.Series([1.0, 2.0]))
    assert means.tolist() == [1.0, 2.0]


def test_bilinear_weights():
    w = G.bilinear_weights(29.25, 35.5)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w[(29.0, 35.0)] == pytest.approx(0.75 * 0.5)
    assert G.bilinear_weights(30.0, 35.0)[(30.0, 35.0)] == pytest.approx(1.0)


def _hours_and_cs(init, n=60, cs_level=600.0):
    hours = pd.date_range(init + pd.Timedelta(hours=9), periods=n, freq="h")
    cs = pd.Series(cs_level, index=pd.date_range(init, periods=120, freq="h"))
    return hours, cs


def test_hourly_reconstruction_keeps_block_means_and_labels():
    init = pd.Timestamp("2022-06-15 12:00", tz="UTC")
    ave, inst = _fields()
    run = _run_frame(range(9, 85, 3), ave, inst)
    hours, cs = _hours_and_cs(init)
    cs.iloc[:] = np.where(np.arange(len(cs)) % 7 == 0, 300.0, 700.0)  # non-constant clear sky
    out = G.hourly_from_run(run, init, hours, cs, G.bilinear_weights(29.9, 35.1))
    lead_end = (hours + pd.Timedelta(hours=1) - init) / pd.Timedelta(hours=1)
    block = np.ceil(lead_end / 3) * 3
    # label convention: row t carries the block that contains lead (t + 1 - init)
    np.testing.assert_array_equal(out["gfs_lead_end_h"].to_numpy(), lead_end.to_numpy())
    for b in np.unique(block):
        rows = block == b
        if rows.sum() == 3:  # complete blocks: the hourly GHI keeps the 3-hour mean
            assert out.loc[rows, "ghi"].mean() == pytest.approx(
                np.mean(ave["dswrf"][int(b) - 2:int(b) + 1]))
            assert out.loc[rows, "tcc"].nunique() == 1  # averaged cloud constant in the block
    # sunshine: accumulated in the window -> fraction of each 3-hour block
    expected = [np.mean(inst["sunsd"][int(b) - 2:int(b) + 1]) for b in block]
    np.testing.assert_allclose(out["sunsd"].to_numpy(), expected)
    # instantaneous fields at the middle of the hour (t2m linear in lead, K -> degC)
    np.testing.assert_allclose(out["t2m"].to_numpy(), 280 + (lead_end - 0.5) - 273.15)


def test_twilight_blocks_follow_the_clear_sky():
    """An hour without sun inside a block gets 0; the block mean is kept (R6b-03)."""
    init = pd.Timestamp("2022-06-15 12:00", tz="UTC")
    ave, inst = _fields()
    run = _run_frame(range(9, 85, 3), ave, inst)
    hours, cs = _hours_and_cs(init)
    cs.iloc[:] = np.tile([0.0, 0.0, 30.0], len(cs) // 3)  # dark, dark, first light per block
    out = G.hourly_from_run(run, init, hours, cs, G.bilinear_weights(30.0, 35.0))
    dark = cs.reindex(hours).to_numpy() == 0
    assert (out.loc[dark, "ghi"] == 0).all()
    lead_end = ((hours + pd.Timedelta(hours=1) - init) / pd.Timedelta(hours=1)).to_numpy()
    block = np.ceil(lead_end / 3) * 3
    for b in np.unique(block):
        rows = block == b
        if rows.sum() == 3:
            assert out.loc[rows, "ghi"].mean() == pytest.approx(
                np.mean(ave["dswrf"][int(b) - 2:int(b) + 1]))


def test_a_block_without_sun_is_spread_evenly():
    init = pd.Timestamp("2022-06-15 12:00", tz="UTC")
    ave, inst = _fields()
    run = _run_frame(range(9, 85, 3), ave, inst)
    hours, cs = _hours_and_cs(init, cs_level=0.0)
    out = G.hourly_from_run(run, init, hours, cs, G.bilinear_weights(30.0, 35.0))
    lead_end = ((hours + pd.Timedelta(hours=1) - init) / pd.Timedelta(hours=1)).to_numpy()
    expected = [np.mean(ave["dswrf"][int(b) - 2:int(b) + 1]) for b in np.ceil(lead_end / 3) * 3]
    np.testing.assert_allclose(out["ghi"].to_numpy(), expected)


# ── T-43 ────────────────────────────────────────────────────────────────────

def _table_for(day: str, run_hour: int, uploaded: str):
    """A minimal run table: every hour from T0 - 1 h to the end of D+3, with metadata."""
    t0 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    end = (pd.Timestamp(day) + pd.Timedelta(days=4)).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    hours = pd.date_range(t0 - pd.Timedelta(hours=1), end, freq="h")
    table = pd.DataFrame({"ghi": np.arange(len(hours), dtype=float)}, index=hours)
    table["gfs_init"] = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=run_hour)
    table["gfs_uploaded_max"] = pd.Timestamp(uploaded, tz="UTC")
    return table, {"day": day, "run": run_hour}


@pytest.mark.parametrize("horizon", (1, 2, 3))
def test_the_12z_run_serves_every_horizon(horizon):
    table, record = _table_for("2022-06-14", 12, "2022-06-14 16:00")
    rows = G.horizon_rows(table, record, horizon)
    local = rows.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize().unique()
    assert list(local) == [pd.Timestamp("2022-06-15") + pd.Timedelta(days=horizon - 1)]
    assert (rows["gfs_init"] + pd.Timedelta(hours=6) <= rows["t0_utc"]).all()


def test_a_run_not_available_at_t0_is_refused():
    table, record = _table_for("2022-06-14", 18, "2022-06-14 22:05")  # the 18Z run
    with pytest.raises(ValueError, match="T-17"):
        G.horizon_rows(table, record, 1)
    table, record = _table_for("2022-06-14", 12, "2022-06-14 21:30")  # uploaded after T0
    with pytest.raises(ValueError, match="T-17"):
        G.horizon_rows(table, record, 1)


def test_fall_back_day_has_25_rows():
    table, record = _table_for("2022-10-29", 12, "2022-10-29 16:00")
    assert len(G.horizon_rows(table, record, 1)) == 25  # 2022-10-30 local


def test_a_later_run_changes_nothing_served_by_an_earlier_run():
    """Two consecutive runs: D+2 of run A and D+1 of run B are different local days, so no
    target hour is served by both, and editing run B leaves run A's rows untouched."""
    a, rec_a = _table_for("2022-06-14", 12, "2022-06-14 16:00")
    b, rec_b = _table_for("2022-06-15", 12, "2022-06-15 16:00")
    rows_a = {h: G.horizon_rows(a, rec_a, h) for h in (1, 2, 3)}
    b_changed = b.assign(ghi=b["ghi"] + 999.0)
    for h in (1, 2, 3):
        again = G.horizon_rows(a, rec_a, h)
        pd.testing.assert_frame_equal(rows_a[h], again)
        assert not G.horizon_rows(b_changed, rec_b, h).index.isin(rows_a[h].index).any()


# ── downloader helpers ────────────────────────────────────────────────────

IDX = """1:0:d=2022061512:PRMSL:mean sea level:9 hour fcst:
2:1000:d=2022061512:DSWRF:surface:6-9 hour ave fcst:
3:2000:d=2022061512:TCDC:entire atmosphere:9 hour fcst:
4:3000:d=2022061512:TCDC:entire atmosphere:6-9 hour ave fcst:
5:4000:d=2022061512:LCDC:low cloud layer:6-9 hour ave fcst:
6:5000:d=2022061512:MCDC:middle cloud layer:6-9 hour ave fcst:
7:6000:d=2022061512:HCDC:high cloud layer:6-9 hour ave fcst:
8:7000:d=2022061512:TMP:2 m above ground:9 hour fcst:
9:8000:d=2022061512:RH:2 m above ground:9 hour fcst:
10:9000:d=2022061512:PWAT:entire atmosphere (considered as a single layer):9 hour fcst:
11:10000:d=2022061512:UGRD:10 m above ground:9 hour fcst:
12:11000:d=2022061512:VGRD:10 m above ground:9 hour fcst:
13:12000:d=2022061512:SUNSD:surface:9 hour fcst:
14:13000:d=2022061512:DLWRF:surface:6-9 hour ave fcst:
"""


def test_index_parsing_and_selection():
    entries = D.parse_idx(IDX)
    assert entries[1].offset == 1000 and entries[1].end == 1999 and entries[-1].end is None
    wanted = D.select_messages(entries)
    assert set(wanted) == {name for name, *_ in D.MESSAGES}
    assert wanted["tcc"].ftime == "6-9 hour ave fcst"  # the average, not the instantaneous
    with pytest.raises(ValueError):
        D.select_messages(entries[:5])  # a wanted message is missing


def test_range_merging():
    entries = list(D.select_messages(D.parse_idx(IDX)).values())
    assert D.merge_ranges(entries, gap=0) == [(1000, 1999), (3000, 12999)]
    assert D.merge_ranges(entries, gap=1500) == [(1000, 12999)]


def test_fallback_runs_cover_the_same_target_hours():
    for run in D.RUN_HOURS:
        leads = D.leads_for_run(run)
        assert leads[0] - (12 - run) == 9 and leads[-1] - (12 - run) == 84
    url = D.file_url("2022-06-15", 6, 15)
    assert url.endswith("gfs.20220615/06/atmos/gfs.t06z.pgrb2.1p00.f015")


def test_sunshine_accumulation_that_breaks_the_window_raises():
    leads = pd.Series([9, 12])
    with pytest.raises(ValueError, match="exceeds"):
        G.three_hour_totals(leads, pd.Series([20_000.0, 21_600.0]))  # > 3 h in a 3-h window


def test_physical_chain_on_a_clear_day():
    """GFS GHI equal to the clear sky: kt = 1 in daylight, kt_poa near 1 (Erbs + Hay-Davies vs
    the Ineichen POA), tcell above the air temperature at noon, pv_kw within the capacity, and
    the result aligned on the row index (a shifted index once broke this, R6b-02)."""
    from solarpv.config.site import EXAMPLE_FILE, load_site
    from solarpv.features.geometry import hourly_geometry, load_panel_plane

    plane = load_panel_plane()
    hours = pd.date_range("2022-06-15 00:00", periods=24, freq="h", tz="UTC")
    geo = hourly_geometry(hours, load_site(EXAMPLE_FILE), plane)
    geo["theo_kw"] = geo["poa_global_clearsky"] * 3449.0 / 1000.0
    h = pd.DataFrame({"ghi": geo["ghi_clearsky"], "t2m": 30.0, "wind10": 2.0}, index=hours)
    out = G.physical_features(h, geo, plane)
    assert out.index.equals(hours) and len(out) == 24
    day = geo["ghi_clearsky"] > 200
    np.testing.assert_allclose(out.loc[day, "kt"], 1.0)
    assert out.loc[day, "kt_poa"].between(0.85, 1.15).all()
    assert (out.loc[day, "tcell"] > 30.0).all()
    assert out["pv_kw"].between(0, 3449.0).all()


def test_a_run_with_a_file_uploaded_after_t0_is_rejected():
    """2022-01-24: one file re-uploaded at 22:06Z, after T0 (22:00Z in winter); R6b-05."""
    assert D.issuance_of_run_day("2022-01-24") == pd.Timestamp("2022-01-24 22:00", tz="UTC")
    assert D.issuance_of_run_day("2022-07-23") == pd.Timestamp("2022-07-23 21:00", tz="UTC")
    record = {"day": "2022-01-24", "files": [
        {"url": "f024", "uploaded_utc": "2022-01-24T15:41:32+00:00"},
        {"url": "f027", "uploaded_utc": "2022-01-24T22:06:32+00:00"}]}
    assert D.late_files(record) == ["f027"]
    record["files"].pop()
    assert D.late_files(record) == []

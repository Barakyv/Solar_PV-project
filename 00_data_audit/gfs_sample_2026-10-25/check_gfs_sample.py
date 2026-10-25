"""Is the GFS archive usable? A multi-period sample before the full download (stage 6b; the
user's request of 2026-10-25: "a sample from different periods, so that there is no problem as
with the previous forecasts").

    python 00_data_audit/gfs_sample_2026-10-25/check_gfs_sample.py

Ten blocks of three consecutive 12Z runs (30 runs, about 0.6 GB of transfer, point values kept).
Three consecutive runs X, X+1, X+2 all forecast day X+3, at D+3, D+2 and D+1: the same target
day from three leads. The blocks cover:
- train: every season, the spring-forward day 2022-03-25 and the fall-back day 2022-10-30;
- validation: two periods;
- test: two periods, including the polled-source period of 2025. **Test blocks get technical
  checks only**: no comparison with any measurement, so the test split informs no decision.

Checks, aimed at the failures of the original forecasts (docs/stage6b_plan.md §2a):
1. completeness: 26 files x 11 messages per run, the expected averaging windows;
2. availability (N1, T-17): every file uploaded before T0, and init + 6 h <= T0;
3. physics: ranges, no irradiance at night, GFS irradiance in phase with the clear-sky model
   (lag 0, N3/N7);
4. lead dependence (N2): the three forecasts of the same target day differ;
5. train and validation only: GFS against the measured in-plane irradiance:
   - lag of the best correlation (must be 0);
   - correlation of kt_poa with the measured k per horizon (plausible, not near-perfect, and
     not improving with the horizon).

Writes `sample_summary.json`, `sample_runs.csv`, `sample_horizons.csv` and
`sample_days.png` here, and a run log under `runs/`.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.config.runlog import RunLog
from solarpv.config.site import load_site
from solarpv.features.geometry import load_panel_plane
from solarpv.nwp import download as D
from solarpv.nwp import gfs as G

HERE = Path(__file__).resolve().parent
BLOCKS = {  # first run day of each block -> split of its target days
    "2021-10-12": "train", "2022-01-17": "train", "2022-03-22": "train",
    "2022-07-11": "train", "2022-10-27": "train", "2023-04-17": "train",
    "2024-03-11": "val", "2024-08-12": "val",
    "2024-12-09": "test", "2025-04-07": "test",
}
EXPECTED_ROWS = len(D.LEADS_12Z) * len(D.MESSAGES) * 9


def run_days() -> list[str]:
    return [(pd.Timestamp(b) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for b in BLOCKS for i in range(3)]


def lag_of_best_correlation(a: pd.Series, b: pd.Series, lags=range(-3, 4)) -> int:
    """Lag (hours) at which b shifted by `lag` correlates best with a."""
    scores = {lag: a.corr(b.shift(lag)) for lag in lags}
    return max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])


def main() -> int:
    geo = pd.read_parquet(G.FEATURES_D1, columns=["ghi_clearsky", "poa_global_clearsky",
                                                  "solar_zenith", "solar_azimuth",
                                                  "sun_elevation", "theo_kw", "daytime",
                                                  "split", "target_k", "k_valid"])
    hourly = pd.read_parquet(G.HOURLY_FILE, columns=["irradiance"])
    site, plane = load_site(), load_panel_plane()
    weights = G.bilinear_weights(site.latitude, site.longitude)
    command = "python 00_data_audit/gfs_sample_2026-10-25/check_gfs_sample.py"
    with RunLog("06b_gfs", command, out_dir=HERE) as log:
        days = run_days()
        log.note(f"{len(days)} runs in {len(BLOCKS)} blocks; test blocks: technical checks only")
        with log.step("download the sample (resumable)"):
            done = D.completed_runs()
            for day in days:
                if day not in done:
                    result = D.download_day(day, workers=8)
                    if result["status"] != "complete":
                        log.warn(f"{day}: no allowed run")
        runs = {r["day"]: (frame, r) for frame, r in G.load_runs() if r["day"] in days}
        log.value("runs_downloaded", len(runs))

        run_rows, tables = [], {}
        with log.step("completeness, availability and physics per run"):
            for day in days:
                frame, rec = runs[day]
                t0 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(
                    TIMEZONE_LOCAL).tz_convert("UTC")
                init = pd.Timestamp(rec["init_utc"])
                uploaded = max(pd.Timestamp(f["uploaded_utc"]) for f in rec["files"])
                ave = frame[frame["step_type"] == "avg"]
                windows_ok = all(int(r.split("-")[0]) % 6 == 0 and
                                 int(r.split("-")[1]) - int(r.split("-")[0]) in (3, 6)
                                 for r in ave["step_range"].unique())
                table = G.run_table(frame, rec, geo, plane, weights)
                tables[day] = table
                cs = geo["ghi_clearsky"].reindex(table.index)
                night = cs < 1.0
                run_rows.append({
                    "day": day, "run": rec["run"], "fallback": rec["fallback"],
                    "rows": len(frame), "rows_expected": EXPECTED_ROWS,
                    "files": len(rec["files"]), "averaging_windows_ok": windows_ok,
                    "init_plus_6h_le_t0": init + pd.Timedelta(hours=6) <= t0,
                    "uploaded_max_utc": uploaded.isoformat(),
                    "hours_before_t0": (t0 - uploaded) / pd.Timedelta(hours=1),
                    "ghi_min": float(table["ghi"].min()),
                    "ghi_night_max": float(table.loc[night, "ghi"].max()),
                    "ghi_over_clearsky_max": float((table["ghi"] / cs).where(cs > 200).max()),
                    "tcc_range": f"{table['tcc'].min():.0f}..{table['tcc'].max():.0f}",
                    "t2m_range": f"{table['t2m'].min():.1f}..{table['t2m'].max():.1f}",
                    "rh2m_range": f"{table['rh2m'].min():.0f}..{table['rh2m'].max():.0f}",
                    "pwat_range": f"{table['pwat'].min():.1f}..{table['pwat'].max():.1f}",
                    "lag_vs_clearsky_h": lag_of_best_correlation(table["ghi"], cs),
                })
        runs_df = pd.DataFrame(run_rows)
        runs_df.to_csv(HERE / "sample_runs.csv", index=False, lineterminator="\n")

        horizon_rows, block_rows = [], []
        with log.step("lead dependence and comparison with measurements"):
            for first, split in BLOCKS.items():
                b_days = [(pd.Timestamp(first) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                          for i in range(3)]
                target = pd.Timestamp(first) + pd.Timedelta(days=3)  # served by all three runs
                per_h = {}
                for i, day in enumerate(b_days):
                    h = 3 - i  # run X serves D+3, X+1 serves D+2, X+2 serves D+1
                    rows = G.horizon_rows(tables[day], runs[day][1], h)
                    local = rows.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
                    per_h[h] = rows[local == target]
                same_index = per_h[1].index.equals(per_h[2].index) and \
                    per_h[1].index.equals(per_h[3].index)
                day_mask = geo["daytime"].reindex(per_h[1].index).to_numpy()
                diffs = {f"max_abs_diff_ghi_D{a}_D{b}": float(
                    (per_h[a]["ghi"] - per_h[b]["ghi"]).abs()[day_mask].max())
                    for a, b in ((1, 2), (1, 3), (2, 3))}
                row = {"block": first, "split": split, "target_day": target.date().isoformat(),
                       "same_target_hours": same_index, **diffs}
                if split in ("train", "val"):  # never the test split
                    for h in (1, 2, 3):
                        r = per_h[h][day_mask]
                        k_obs = geo["target_k"].reindex(r.index)
                        horizon_rows.append({
                            "block": first, "split": split, "horizon": h, "hours": len(r),
                            "kt_poa_mean": float(r["kt_poa"].mean()),
                            "k_measured_mean": float(k_obs.mean()),
                            "abs_error_k": float((r["kt_poa"] - k_obs).abs().mean())})
                    # alignment against the measured in-plane irradiance, D+1 days of the block
                    d1 = pd.concat([G.horizon_rows(tables[d], runs[d][1], 1) for d in b_days])
                    measured = hourly["irradiance"].reindex(d1.index)
                    row["lag_vs_measured_h"] = lag_of_best_correlation(d1["poa"], measured)
                    row["corr_poa_vs_measured_D1"] = float(d1["poa"].corr(measured))
                block_rows.append(row)
        horizons = pd.DataFrame(horizon_rows)
        blocks = pd.DataFrame(block_rows)
        horizons.to_csv(HERE / "sample_horizons.csv", index=False, lineterminator="\n")
        blocks.to_csv(HERE / "sample_blocks.csv", index=False, lineterminator="\n")

        with log.step("figure"):
            fig, axes = plt.subplots(len(BLOCKS), 1, figsize=(13, 2.2 * len(BLOCKS)))
            for ax, (first, split) in zip(axes, BLOCKS.items(), strict=True):
                b_days = [(pd.Timestamp(first) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                          for i in range(3)]
                d1 = pd.concat([G.horizon_rows(tables[d], runs[d][1], 1) for d in b_days])
                x = d1.index.tz_convert(TIMEZONE_LOCAL).tz_localize(None)
                ax.plot(x, geo["poa_global_clearsky"].reindex(d1.index), color="0.6", lw=1,
                        label="clear-sky POA")
                ax.plot(x, d1["poa"], color="#1565c0", lw=1.2, label="GFS POA, D+1")
                if split != "test":
                    ax.plot(x, hourly["irradiance"].reindex(d1.index), color="k", lw=1,
                            ls="--", label="measured in-plane")
                ax.set_title(f"{split}: D+1 days of the runs from {first}"
                             + (" (test: no measurement shown)" if split == "test" else ""),
                             fontsize=8)
                ax.set_ylabel("W/m2", fontsize=7)
                ax.tick_params(labelsize=7)
            axes[0].legend(fontsize=7, ncol=3)
            fig.tight_layout()
            fig.savefig(HERE / "sample_days.png", dpi=110)
            plt.close(fig)

        summary = {
            "runs": len(runs_df), "fallback_runs": int(runs_df["fallback"].sum()),
            "complete_runs": int((runs_df["rows"] == EXPECTED_ROWS).sum()),
            "averaging_windows_ok": bool(runs_df["averaging_windows_ok"].all()),
            "all_available_before_t0": bool(runs_df["init_plus_6h_le_t0"].all()
                                            and (runs_df["hours_before_t0"] > 0).all()),
            "min_hours_uploaded_before_t0": float(runs_df["hours_before_t0"].min()),
            "ghi_never_negative": bool((runs_df["ghi_min"] >= -1e-6).all()),
            "max_ghi_at_night": float(runs_df["ghi_night_max"].max()),
            "lag_vs_clearsky_all_zero": bool((runs_df["lag_vs_clearsky_h"] == 0).all()),
            "lead_dependence_all_blocks": bool((blocks[[c for c in blocks if
                                                        c.startswith("max_abs_diff")]] > 0)
                                               .all().all()),
            "lag_vs_measured_train_val": blocks.dropna(subset=["lag_vs_measured_h"])
            ["lag_vs_measured_h"].astype(int).tolist(),
            "corr_poa_vs_measured_D1_train_val": blocks["corr_poa_vs_measured_D1"].dropna()
            .round(3).tolist(),
            "abs_error_k_by_horizon_train_val": horizons.groupby("horizon")["abs_error_k"]
            .mean().round(4).to_dict(),
        }
        (HERE / "sample_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                 encoding="utf-8", newline="\n")
        for key, value in summary.items():
            log.value(key, value)
        log.outputs([HERE / f for f in ("sample_runs.csv", "sample_blocks.csv",
                                        "sample_horizons.csv", "sample_days.png",
                                        "sample_summary.json")])
    return 0


if __name__ == "__main__":
    sys.exit(main())

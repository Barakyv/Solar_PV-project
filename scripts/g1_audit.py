"""Evidence for gate G-1 that no other step produces (docs/stage9_plan.md §20.4; X57, M35).

    python scripts/g1_audit.py            # writes outputs/g1_audit/audit/
    python scripts/g1_audit.py --only test_access

Read-only on the data and the runs; it writes only into `outputs/g1_audit/audit/`.

1. test_access: the test-access log of the evaluation layer (absent means no evaluation of the
   test split), its git history, the `test_access` records of every step of every run, and the
   timestamps of every forecast file of every run (predictions, uncorrected and ablation forecast
   files; a smoke or dev file may not hold a test hour).
2. lookahead_clean (X57): the stage-5 cleaning has two-sided rules (frozen runs that reach
   daytime, margins, the centred F3 window, state runs). For every issuance time T0 (local
   midnight) the stage-4 minutes of [T0 - 72 h, T0) are cleaned alone (causal: nothing after T0
   is known) and with one more day [T0 - 72 h, T0 + 24 h); the lag sources of the features (the
   seven series, after the D-22 masks) are compared on the 48 source hours before T0. The
   full-window values are also compared with the real stage-5 table (a check of the method).
3. lookahead_clock (X57): the clock offsets of stage 4 are a centred 31-day median of daily
   estimates (D-19); the same estimates with a trailing median (only days before) give the
   offset a forecaster would have had; compared in whole minutes, as applied.
4. lookahead_polled (X57): the polled reconstruction (F-18, test period only) interpolates the
   minutes after the last poll before T0 towards the first poll after it; the change of the
   hourly mean of the last source hour is bounded by 15/60 of the change across T0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solarpv.config.cli import parser  # noqa: E402
from solarpv.config.constants import TEST_START, TIMEZONE_LOCAL  # noqa: E402
from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR  # noqa: E402
from solarpv.config.provenance import find_git  # noqa: E402

OUT = OUTPUTS_DIR / "g1_audit" / "audit"
SOURCES = ("s01_inv1_AC_kW", "s01_inv2_AC_kW", "site_total_AC_kW", "irradiance", "mod_temp1",
           "mod_temp2", "ambient_temp")
WINDOW_BEFORE_H, COMPARED_H, AFTER_H = 72, 48, 24
TOL = 1e-9


def write(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False, lineterminator="\n")
    else:
        path.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8",
                        newline="\n")
    return path


# ── 1. test access ──────────────────────────────────────────────────────────


def _find_key(obj, key: str, found: list):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            _find_key(v, key, found)
    elif isinstance(obj, list):
        for v in obj:
            _find_key(v, key, found)


# every kind of file of a run that holds forecast hours (a glob of `predictions_*` alone missed the
# ablation forecasts of the new step, X76)
FORECAST_FILES = ("predictions_*.csv", "uncorrected_*.csv", "ablation_forecasts_seed*.csv")
# the quantile files are not parsed here (D-33): they hold the hours of the predictions file of the
# same step, which is read


def forecast_files(run: Path) -> list[Path]:
    """The forecast files of one run folder, without the folders a step moved aside."""
    return sorted(f for pattern in FORECAST_FILES for f in run.rglob(pattern)
                  if not any(part.startswith(".") for part in f.relative_to(run).parts))


def test_access() -> dict:
    import subprocess

    from solarpv.evaluation.evaluate import TEST_ACCESS_LOG

    git = find_git()
    rel = TEST_ACCESS_LOG.relative_to(REPO).as_posix()
    history = subprocess.run([git, "log", "--all", "--oneline", "--", rel], cwd=REPO,
                             capture_output=True, text=True).stdout.split("\n")
    log_rows = (pd.read_csv(TEST_ACCESS_LOG).to_dict("records")
                if TEST_ACCESS_LOG.exists() else [])
    runs, steps_with_access, prediction_files, test_rows = [], [], 0, 0
    test_start = pd.Timestamp(TEST_START, tz="UTC")
    for run in sorted((OUTPUTS_DIR / "runs").iterdir()):
        if not run.is_dir():
            continue
        records = list(run.glob("*/step.json"))
        access = []
        for r in records:
            found: list = []
            _find_key(json.loads(r.read_text(encoding="utf-8")), "test_access", found)
            entries = [e for v in found for e in (v if isinstance(v, list) else [v]) if e]
            if entries:
                steps_with_access.append({"run": run.name, "step": r.parent.name,
                                          "entries": entries})
            access += entries
        n_files, n_test, last = 0, 0, None
        for f in forecast_files(run):
            ts = pd.to_datetime(pd.read_csv(f, usecols=["timestamp"])["timestamp"], utc=True)
            n_files += 1
            n_test += int((ts >= test_start).sum())
            last = max(last, ts.max()) if last is not None else ts.max()
        prediction_files += n_files
        test_rows += n_test
        runs.append({"run": run.name, "step_records": len(records), "test_access_entries":
                     len(access), "prediction_files": n_files, "rows_at_or_after_test_start":
                     n_test, "last_prediction_hour": str(last) if last is not None else None})
    facts = {"log_file": rel, "log_exists": TEST_ACCESS_LOG.exists(), "log_rows": len(log_rows),
             "log_in_git_history": [h for h in history if h.strip()],
             "runs": len(runs), "step_records": sum(r["step_records"] for r in runs),
             "steps_with_test_access": steps_with_access,
             "prediction_files": prediction_files, "prediction_rows_in_test": test_rows,
             "test_start": str(test_start), "per_run": runs}
    write("test_access.json", facts)
    return facts


# ── 2. look-ahead of the cleaning ──────────────────────────────────────────


def effective_sources(minutes: pd.DataFrame, value_cols: list[str], policy: dict
                      ) -> pd.DataFrame:
    """The seven lag sources of stage 6 from cleaned minutes: the hourly means of stage 5
    (`clean.to_hourly`, on the stage-4 value columns as `clean.clean` passes them) with the D-22
    masks (`features.build.masked_sources`), as the feature table reads them."""
    from solarpv.data.clean import to_hourly
    from solarpv.features.build import masked_sources

    hourly = to_hourly(minutes, value_cols, policy)
    sources, _ = masked_sources(hourly, minutes, policy["hourly"]["op_limited_min_share"])
    return pd.DataFrame(sources)


def lookahead_clean() -> dict:
    from solarpv.config.site import load_site
    from solarpv.data.clean import clean_minutes, load_policy
    from solarpv.data.clock import Orientation, clear_sky_components, poa_from_components

    stage4 = pd.read_parquet(PROCESSED_DIR / "04_preprocess" / "scada_1min_true_utc.parquet")
    site = load_site()
    comp = clear_sky_components(stage4.index, site)
    elev = comp["elev"].to_numpy()
    poa = poa_from_components(comp, Orientation(site.surface_tilt, site.surface_azimuth))
    policy = load_policy()
    # the real stage-5 lag sources, as stage 6 builds them
    clean_min = pd.read_parquet(PROCESSED_DIR / "05_clean" / "scada_1min_clean.parquet")
    clean_hour = pd.read_parquet(PROCESSED_DIR / "05_clean" / "scada_1H_clean.parquet")
    from solarpv.features.build import masked_sources

    real, _ = masked_sources(clean_hour, clean_min, policy["hourly"]["op_limited_min_share"])
    real = pd.DataFrame(real)
    del clean_min

    index = stage4.index
    value_cols = [c for c in stage4.columns if not c.startswith("src_")]
    first = index[0].tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    last = index[-1].tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    # local midnights (never ambiguous in Israel), with 72 h before and 24 h after in the data
    local_days = pd.date_range(first + pd.Timedelta(days=4), last - pd.Timedelta(days=1),
                               freq="D").tz_localize(TIMEZONE_LOCAL)
    cells = []
    for day in local_days:
        t0 = day.tz_convert("UTC")
        a = index.searchsorted(t0 - pd.Timedelta(hours=WINDOW_BEFORE_H))
        b = index.searchsorted(t0)
        c = index.searchsorted(t0 + pd.Timedelta(hours=AFTER_H))
        causal = effective_sources(clean_minutes(stage4.iloc[a:b], elev[a:b], poa[a:b],
                                                 policy), value_cols, policy)
        full = effective_sources(clean_minutes(stage4.iloc[a:c], elev[a:c], poa[a:c],
                                               policy), value_cols, policy)
        hours = pd.date_range(t0 - pd.Timedelta(hours=COMPARED_H), t0, freq="h",
                              inclusive="left")
        for col in SOURCES:
            x = causal[col].reindex(hours).to_numpy(dtype=float)
            y = full[col].reindex(hours).to_numpy(dtype=float)
            r = real[col].reindex(hours).to_numpy(dtype=float)
            differ = ~((np.abs(x - y) <= TOL) | (np.isnan(x) & np.isnan(y)))
            method = ~((np.abs(y - r) <= TOL) | (np.isnan(y) & np.isnan(r)))
            for k in np.flatnonzero(differ | method):
                cells.append({"t0_utc": t0, "hours_before_t0": COMPARED_H - k,
                              "source": col, "hour": hours[k], "causal": x[k],
                              "with_next_day": y[k], "stage5": r[k],
                              "lookahead": bool(differ[k]),
                              "method_mismatch": bool(method[k])})
    detail = pd.DataFrame(cells, columns=["t0_utc", "hours_before_t0", "source", "hour",
                                          "causal", "with_next_day", "stage5", "lookahead",
                                          "method_mismatch"])
    n_days = len(local_days)
    rows = []
    for col in SOURCES:
        d = detail[(detail["source"] == col) & detail["lookahead"]]
        both = d.dropna(subset=["causal", "with_next_day"])
        rows.append({
            "source": col, "issuance_days": n_days, "cells": n_days * COMPARED_H,
            "cells_changed": len(d),
            "share_pct": round(100 * len(d) / (n_days * COMPARED_H), 4),
            "value_vs_missing": int(len(d) - len(both)),
            "max_abs_change": float(np.abs(both["causal"] - both["with_next_day"]).max())
            if len(both) else 0.0,
            "within_6h_of_t0": int((d["hours_before_t0"] <= 6).sum()),
            "days_affected": int(d["t0_utc"].nunique()),
            "test_days_affected": int((d["t0_utc"] >= pd.Timestamp(TEST_START, tz="UTC"))
                                      .groupby(d["t0_utc"]).any().sum()) if len(d) else 0,
            "method_mismatch_cells": int(detail[(detail["source"] == col)
                                                & detail["method_mismatch"]].shape[0]),
        })
    summary = pd.DataFrame(rows)
    write("lookahead_clean.csv", summary)
    write("lookahead_clean_cells.csv", detail)
    return {"issuance_days": n_days, "compared_hours_per_day": COMPARED_H,
            "by_source": summary.to_dict("records")}


# ── 3. look-ahead of the clock correction ──────────────────────────────────


def _trailing_within_segments(days, values, steps, calendar, window=31, min_obs=4):
    """As data.clock.smooth_within_segments, but the median of the window ending on the day
    (only earlier days), carried forward; days before the first causal value of a segment are
    counted as warm-up and take that first value."""
    edges = [calendar[0]] + sorted(steps) + [calendar[-1] + pd.Timedelta(days=1)]
    obs = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(days)).sort_index()
    out, warm = pd.Series(np.nan, index=calendar), pd.Series(False, index=calendar)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        seg_cal = calendar[(calendar >= lo) & (calendar < hi)]
        seg = obs[(obs.index >= lo) & (obs.index < hi)].dropna()
        if len(seg) == 0 or len(seg_cal) == 0:
            continue
        daily = seg.groupby(level=0).median().reindex(seg_cal)
        med = daily.rolling(window, center=False, min_periods=min_obs).median().ffill()
        warm.loc[seg_cal] = med.isna().to_numpy()
        out.loc[seg_cal] = med.bfill().to_numpy()
    return out, warm


def lookahead_clock() -> dict:
    sys.path.insert(0, str(REPO / "scripts"))
    import estimate_clock_drift as E

    folder = AUDIT_DIR / "clock_drift_2026-10-14"
    em = pd.read_csv(folder / "daily_estimates_minute_before.csv", parse_dates=["day"])
    summary = json.loads((folder / "clock_fit_summary.json").read_text(encoding="utf-8"))
    steps = [pd.Timestamp(s["day"]) for s in summary["steps"]]
    offsets = pd.read_csv(CONFIGS_DIR / "clock_offsets.csv", parse_dates=["day"]).set_index("day")
    calendar = pd.DatetimeIndex(offsets.index)
    rt, rd = E.reliable_tau(em), E.reliable_delta(em)
    meteo, warm = _trailing_within_segments(rt["day"], rt["tau_met"], steps, calendar)
    delta, _ = _trailing_within_segments(rd["day"], rd["delta"], steps, calendar, window=61,
                                         min_obs=3)
    delta = delta.fillna(0.0)
    causal = pd.DataFrame({"minute_meteo": meteo, "minute_inverter": meteo + delta})
    rows, per_day = [], pd.DataFrame(index=calendar)
    per_day.index.name = "day"
    per_day["warm_up"] = warm.to_numpy()
    for col in ("minute_meteo", "minute_inverter"):
        both = offsets[col].notna() & causal[col].notna()
        applied = offsets[col][both].round()  # apply_correction rounds to whole minutes
        trailing = causal[col][both].round()
        diff = (trailing - applied).abs()
        per_day[f"{col}_applied_min"] = applied
        per_day[f"{col}_trailing_min"] = trailing
        rows.append({"clock": col, "days": len(calendar), "days_compared": int(both.sum()),
                     "days_shift_differs": int((diff > 0).sum()),
                     "share_pct": round(100 * float((diff > 0).mean()), 2),
                     "max_shift_difference_min": int(diff.max()),
                     "days_differ_by_2_or_more": int((diff >= 2).sum()),
                     "warm_up_days": int(warm.sum())})
    table = pd.DataFrame(rows)
    write("lookahead_clock.csv", table)
    write("lookahead_clock_days.csv", per_day.reset_index())
    return {"steps_found_on_the_whole_series": [str(s.date()) for s in steps],
            "by_clock": table.to_dict("records")}


# ── 4. look-ahead of the polled reconstruction ─────────────────────────────


def lookahead_polled() -> dict:
    stage4 = pd.read_parquet(PROCESSED_DIR / "04_preprocess" / "scada_1min_true_utc.parquet")
    src_cols = [c for c in stage4.columns if c.startswith("src_")]
    polled_any = stage4[src_cols].eq(1).any(axis=1) if src_cols else pd.Series(False,
                                                                               stage4.index)
    days = sorted({t.tz_convert(TIMEZONE_LOCAL).normalize() for t in stage4.index[polled_any]})
    rows = []
    for col in SOURCES:
        if col == "site_total_AC_kW" or col not in stage4:
            continue
        bounds = []
        for day in days:
            t0 = day.tz_convert("UTC")
            before = stage4[col].asof(t0 - pd.Timedelta(minutes=15))
            after = stage4[col].asof(t0 + pd.Timedelta(minutes=15))
            if np.isfinite(before) and np.isfinite(after):
                bounds.append(15 / 60 * abs(after - before))
        rows.append({"source": col, "polled_issuance_days": len(days),
                     "max_bound_on_last_hour_mean": round(float(max(bounds)), 4) if bounds
                     else None,
                     "median_bound": round(float(np.median(bounds)), 4) if bounds else None})
    table = pd.DataFrame(rows)
    write("lookahead_polled.csv", table)
    return {"polled_issuance_days": len(days),
            "first_day": str(days[0].date()) if days else None,
            "all_in_test": bool(days and days[0].tz_convert("UTC")
                                >= pd.Timestamp(TEST_START, tz="UTC")),
            "by_source": table.to_dict("records")}


def smoke_final() -> dict:
    """5. The newest smoke run (the one at the commit to be tagged): every step ok, one clean
    commit, and its forecast files byte for byte against the smoke run before it."""
    import hashlib

    runs = sorted(p for p in (OUTPUTS_DIR / "runs").iterdir() if p.name.endswith("_smoke"))
    run, previous = runs[-1], runs[-2]
    steps = {}
    for sj in sorted(run.glob("*/step.json")):
        if "." in sj.parent.name:
            continue
        rec = json.loads(sj.read_text(encoding="utf-8"))
        steps[sj.parent.name] = {"status": rec.get("status"), "commit": rec.get("commit")}
    commits = sorted({s["commit"] for s in steps.values()})

    def forecast_files(r: Path) -> dict[str, str]:
        out = {}
        for pat in ("predictions_*.csv", "uncorrected_*.csv", "quantiles_*.csv", "*.npz"):
            for f in r.rglob(pat):
                rel = f.relative_to(r)
                if not any(part.startswith(".") or "failed" in part for part in rel.parts):
                    out[rel.as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
        return out
    a, b = forecast_files(previous), forecast_files(run)
    common = sorted(set(a) & set(b))
    facts = {"run": run.name, "previous": previous.name, "steps": len(steps),
             "all_ok": all(s["status"] == "ok" for s in steps.values()),
             "commits": commits, "commit": commits[0] if len(commits) == 1 else commits,
             "clean": len(commits) == 1 and not commits[0].endswith("+dirty"),
             "forecast_files": len(b), "byte_identical_to_previous": sum(a[k] == b[k]
                                                                         for k in common),
             "compared_with_previous": len(common), "per_step": steps}
    path = OUTPUTS_DIR / "g1_audit" / "smoke_final.json"
    path.write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8", newline="\n")
    return {k: v for k, v in facts.items() if k != "per_step"}


PARTS = {"test_access": test_access, "lookahead_clean": lookahead_clean,
         "lookahead_clock": lookahead_clock, "lookahead_polled": lookahead_polled,
         "smoke_final": smoke_final}


def main(argv: list[str] | None = None) -> int:
    p = parser("python scripts/g1_audit.py", __doc__)
    p.add_argument("--only", choices=sorted(PARTS), action="append",
                   help="run only these parts (default: all)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    facts = {}
    summary_path = OUT / "audit_summary.json"
    if summary_path.exists():
        facts = json.loads(summary_path.read_text(encoding="utf-8"))
    for name in args.only or list(PARTS):
        facts[name] = PARTS[name]()
        print(f"{name}: done", flush=True)
    write("audit_summary.json", facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

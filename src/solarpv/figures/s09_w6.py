"""Figures of stage 9, W6: gate G-1 (docs/stage9_plan.md §20). They read the evidence files of
the audit (`outputs/g1_audit/`), the code's own window constants, and train-period rows only; no
test-split row is read (the GFS table is filtered to the hours before VAL_START inside the reader).
"""

import ast
import subprocess

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from solarpv.config.constants import TEST_START, VAL_START
from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import find_git
from solarpv.figures import _data as D
from solarpv.figures.registry import register
from solarpv.figures.style import C, subplots

AUDIT = "outputs/g1_audit/audit"
CLEAN = f"{AUDIT}/lookahead_clean.csv"
CLEAN_CELLS = f"{AUDIT}/lookahead_clean_cells.csv"
CLOCK = f"{AUDIT}/lookahead_clock.csv"
CLOCK_DAYS = f"{AUDIT}/lookahead_clock_days.csv"
POLLED = f"{AUDIT}/lookahead_polled.csv"
ACCESS = f"{AUDIT}/test_access.json"
INCIDENT = "outputs/g1_audit/incident_2026-11-12/regenerated_vs_original_content.json"
# the reproduction from the raw data that counts for W6 (attempt 3, R9-27, R9-29)
REPRO = "outputs/g1_audit/reproduction_20261112T202955Z/record.json"
STEP_MODEL = {"baselines": "references", "xgb": "XGB", "bilstm": "BiLSTM", "tft96": "TFT-96",
              "tft168": "TFT-168", "tftbase": "TFT baseline", "patchtst": "PatchTST"}
BEFORE_X62 = "b2c3f1f"  # the commit W6 started from (R9-25)
SERIES = {"s01_inv1_AC_kW": "inverter 1 AC", "s01_inv2_AC_kW": "inverter 2 AC",
          "site_total_AC_kW": "site AC", "irradiance": "irradiance", "mod_temp1": "module temp 1",
          "mod_temp2": "module temp 2", "ambient_temp": "ambient temp"}


# ── S09-34: the checklist of gate G-1 ──────────────────────────────────────

SUITE_DIR = "outputs/g1_audit/pytest_final"  # the full suite at the commit to be tagged
SUITE = f"{SUITE_DIR}/junit.xml"
SUITE_META = f"{SUITE_DIR}/meta.txt"  # "start <time> HEAD <commit>"
SMOKE_FINAL = "outputs/g1_audit/smoke_final.json"  # the smoke run at that commit
REVIEW_LOG = "docs/g1_review_log.md"
SCOPE = "docs/scope_register.md"
LEAKAGE = "docs/leakage_audit.md"
ALLOWED_SKIP = {"test_after_the_tag_only_allowed_paths_changed"}  # active only from the tag (T-103)


def _dry_run() -> str:
    logs = sorted((REPO_ROOT / "outputs" / "g1_audit").glob("dry_run_final_*.log"))
    last = logs[-1] if logs else REPO_ROOT / "outputs" / "g1_audit" / "dry_run_final_.log"
    return last.relative_to(REPO_ROOT).as_posix()


def _code_unchanged_since(commit: str) -> bool:
    """Evidence gathered at `commit` holds for HEAD when only the paths of D-36 changed since:
    figures, documents, tests, outputs (`checks.after_tag_problems`)."""
    from solarpv.pipeline import checks

    out = subprocess.run([find_git(), "diff", "--name-only", f"{commit}..HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    return out.returncode == 0 and not checks.after_tag_problems(out.stdout.split())


def _checklist() -> list[tuple[str, str, str]]:
    """(item, state, evidence) per item of PROJECT_MASTER.md §7, from the evidence files."""
    import re
    import xml.etree.ElementTree as ET

    master = (REPO_ROOT / "PROJECT_MASTER.md").read_text(encoding="utf-8")
    reg = master.split("### 6.13")[1].split("## 7.")[0]
    rows = re.findall(r"^\| (A-\d+) \|.*\| ([^|]*) \|$", reg, re.M)
    open_rows = [a for a, s in rows if "✅" not in s and "מוצהרת כמגבלה" not in s]
    declared = [a for a, s in rows if "מוצהרת כמגבלה" in s]
    decided = {d: bool(re.search(rf"^\| {d} \|.*הוכרע", master, re.M))
               for d in ("D-34", "D-35", "D-36", "D-37")}
    suite = ET.parse(REPO_ROOT / SUITE).getroot()
    ts = suite if suite.tag == "testsuite" else suite.find("testsuite")
    n, bad = int(ts.get("tests")), int(ts.get("failures")) + int(ts.get("errors"))
    skipped = [c.get("name") for c in ts.iter("testcase") if c.find("skipped") is not None]
    other_skips = [s for s in skipped if s not in ALLOWED_SKIP]
    suite_commit = (REPO_ROOT / SUITE_META).read_text(encoding="utf-8").split("HEAD")[-1].split()[0]
    acc = D.jload(ACCESS)
    rec = D.jload(REPRO)
    unexplained = [(s["step"], r["path"]) for s in rec["steps"] for r in s["failing"]
                   if not explained_reason(s["step"], r["path"])]
    smoke = D.jload(SMOKE_FINAL)
    dry = _dry_run()
    dry_ok = "exit 0" in (REPO_ROOT / dry).read_text(encoding="utf-8")
    log = (REPO_ROOT / REVIEW_LOG).read_text(encoding="utf-8")
    review = [[c.strip() for c in line.strip("|").split("|")]
              for line in log.splitlines() if re.match(r"^\| R-G1-\d+", line)]
    review_open = [r for r in review if len(r) >= 7 and r[5].lower() != "note" and not r[6]]
    review_done = bool(re.search(r"^Review completed: \d{4}-\d{2}-\d{2}", log, re.M))
    scope = (REPO_ROOT / SCOPE).read_text(encoding="utf-8")
    x59 = sorted(set(re.findall(r"^\| (S-\d+) \|.*X59", scope, re.M)))
    tag = subprocess.run([find_git(), "tag", "--list", "g1-frozen"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()
    done, wait, todo, going = ("evidence complete", "waiting for the user", "at the tag",
                               "in progress")
    suite_ok = not bad and not other_skips and _code_unchanged_since(suite_commit)
    repro_ok = (not unexplained and all(s["exit"] == 0 for s in rec["steps"])
                and _code_unchanged_since(rec["commit"]) and decided["D-34"])
    smoke_ok = (smoke["all_ok"] and smoke["clean"] and dry_ok
                and _code_unchanged_since(smoke["commit"]))
    return [
        ("assumptions verified or declared", done if not open_rows else wait,
         f"{len(rows) - len(declared)} verified, {len(declared)} declared ({', '.join(declared)})"),
        ("full test suite green", done if suite_ok else wait,
         f"{n} tests: {bad} failed, {len(skipped)} skipped (T-103, from the tag); "
         f"at {suite_commit}"),
        ("manual leakage audit", done if decided["D-35"] else wait,
         "docs/leakage_audit.md §3; the clock look-ahead accepted (D-35)"),
        ("test split touched no decision", done if acc["prediction_rows_in_test"] == 0
         and not acc["log_exists"] else wait,
         f"{acc['step_records']} step records, {acc['prediction_files']} forecast files, "
         "0 test hours"),
        ("reproduction from scratch", done if repro_ok else wait,
         f"{len(rec['steps'])} steps exit 0; {len(unexplained)} unexplained differences (D-34)"),
        ("external review", done if review_done and not review_open else going,
         f"{len(review)} comments, {len(review_open)} open; the review of the new steps follows"),
        ("smoke run and dry run", done if smoke_ok else wait,
         f"{smoke['steps']} steps ok at {smoke['commit']}; dry run writes nothing"),
        ("scope register", done if decided["D-36"] else wait,
         f"every row has a producer or X59 ({len(x59)} rows after the tag, D-36)"),
        ("tag g1-frozen", done if tag else todo, "after the user's approval"),
    ]


@register(
    "S09-34", "9", "g1_checklist", "Gate G-1: the checklist and its evidence",
    "Which items of gate G-1 have their evidence, and which are still open?",
    "The nine items of the gate (PROJECT_MASTER.md §7), each with its state read from its "
    "evidence file (the assumption register, the JUnit file of the full suite, the test-access "
    "audit, the reproduction record, the smoke run, the dry run, the review log, the scope "
    "register, the git tags). Evidence gathered at an earlier commit counts for HEAD only when "
    "nothing outside the paths of D-36 changed since.",
    "{done} of {n} items have complete evidence, {going} is in progress and {todo} wait for the "
    "tag ({wait} wait for a decision). The full suite: {suite}.",
    "The user approves the tag; then the final run by the runbook.",
    (SUITE, SUITE_META, SMOKE_FINAL, ACCESS, REPRO, REVIEW_LOG, SCOPE, LEAKAGE))
def s09_g1_checklist():
    items = _checklist()
    colours = {"evidence complete": C["green"], "waiting for the user": C["orange"],
               "at the tag": C["grey"], "in progress": C["sky"]}
    marks = {"evidence complete": "✔", "waiting for the user": "?", "at the tag": "…",
             "in progress": "~"}
    fig, ax = subplots(height=3.6, width=8.4)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(items))
    for i, (item, state, evidence) in enumerate(items):
        y = len(items) - i - 0.5
        ax.add_patch(Rectangle((0.0, y - 0.42), 0.035, 0.84, color=colours[state]))
        ax.text(0.0175, y, marks[state], ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")
        ax.text(0.05, y, item, va="center", fontsize=8)
        ax.text(0.36, y, state, va="center", fontsize=7, color=colours[state])
        ax.text(0.55, y, evidence, va="center", fontsize=6.5)
    states = [s for _, s, _ in items]
    suite = next(e for i, _, e in items if i == "full test suite green")
    facts = {"n": len(items), "done": states.count("evidence complete"),
             "wait": states.count("waiting for the user"), "todo": states.count("at the tag"),
             "going": states.count("in progress"), "suite": suite}
    return fig, facts


# ── S09-35: when each input's information is available ─────────────────────


@register(
    "S09-35", "9", "issuance_windows", "When the information of each input is known",
    "For a forecast issued at T0, which hours does each family of inputs read, at each horizon?",
    "Hours relative to T0 (00:00 local) read by each input family, from the code's own "
    "constants: the kept lags and the rolling windows of the feature table, the encoder windows "
    "of the sequence models, the decoder of known inputs, and the GFS run (its initialisation "
    "and the latest upload of its files before T0 on the train rows); the shaded band is the "
    "forecast day of the horizon.",
    "No measured input reads an hour that ends after T0, at any horizon: the newest lag is "
    "{newest} days at D+1, D+2 and D+3, and it ends exactly at T0, as do the rolling windows "
    "and the encoders. The GFS run starts {init_min}–{init_max} h before T0 and its last file "
    "arrives at least {upload_margin:.1f} h before T0 on every train row. Only deterministic "
    "inputs (calendar, clear sky, geometry) and that GFS run describe the forecast days.",
    "The manual audit of G-1 (docs/leakage_audit.md §3) lists every family with its code and "
    "test; the steps before the hourly table are measured in S09-36.",
    ("data/processed/06b_gfs/gfs_features_D1.parquet",))
def s09_issuance_windows():
    from solarpv.features.build import ROLLING_WINDOWS_DAYS, kept_lags
    from solarpv.models.sequence_data import MODELS as ENCODERS
    from solarpv.models.sequence_data import N_DAYS

    g = ds.dataset(REPO_ROOT / "data/processed/06b_gfs/gfs_features_D1.parquet").to_table(
        columns=["timestamp_utc", "t0_utc", "gfs_init", "gfs_uploaded_max"],
        filter=ds.field("timestamp_utc") < pd.Timestamp(VAL_START, tz="UTC")).to_pandas()
    t0 = pd.to_datetime(g["t0_utc"], utc=True)
    init_h = (pd.to_datetime(g["gfs_init"], utc=True) - t0).dt.total_seconds() / 3600
    up_h = (pd.to_datetime(g["gfs_uploaded_max"], utc=True) - t0).dt.total_seconds() / 3600
    assert (up_h < 0).all(), "a GFS file uploaded after T0 on a train row"
    fig, axes = subplots(3, 1, height=6.2, sharex=True)
    names = {"bilstm": "BiLSTM", "tft96": "TFT-96", "tft168": "TFT-168 / baseline",
             "patchtst": "PatchTST"}
    newest = {}
    for ax, h in zip(axes, (1, 2, 3), strict=True):
        day0, day1 = 24 * (h - 1), 24 * h
        ax.axvspan(day0, day1, color=C["orange"], alpha=0.15, lw=0)
        ax.axvline(0, color=C["red"], lw=1.2)
        rows = []
        for d in kept_lags(h):
            rows.append((f"lag {d} d", day0 - 24 * d, day1 - 24 * d, C["blue"]))
        for n in ROLLING_WINDOWS_DAYS:
            rows.append((f"rolling {n} d", day0 - 24 * (h + n - 1), day1 - 24 * h, C["purple"]))
        for m, hours in ENCODERS.items():
            if m == "tftbase":
                continue
            rows.append((f"encoder {names[m]}", -hours, 0, C["green"]))
        rows.append(("decoder: known inputs", 0, 24 * N_DAYS, C["grey"]))
        rows.append(("GFS run (init to last upload)", float(init_h.min()), float(up_h.max()),
                     C["sky"]))
        for i, (label, a, b, col) in enumerate(rows):
            ax.barh(-i, b - a, left=a, height=0.6, color=col, edgecolor="white")
            ax.text(a, -i, f"{label} ", ha="right", va="center", fontsize=5.5)
            if col in (C["blue"], C["purple"], C["green"]):
                assert b <= 0, f"D+{h} {label} reads an hour after T0"
        newest[h] = min(kept_lags(h))
        assert max(b for _, _, b, c in rows if c == C["blue"]) == 0  # the newest lag ends at T0
        ax.set_yticks([])
        ax.set_title(f"D+{h}: the forecast day is shaded; T0 is the red line", fontsize=8)
        ax.set_xlim(-24 * 16, 24 * N_DAYS + 4)
    axes[-1].set_xlabel("hours relative to T0 (00:00 local of the first forecast day)")
    assert newest == {1: 1, 2: 2, 3: 3}, newest  # LAG_DROP: lag h d is the newest at D+h
    facts = {"newest": ", ".join(str(newest[h]) for h in (1, 2, 3)),
             "init_min": int(round(-init_h.max())), "init_max": int(round(-init_h.min())),
             "upload_margin": float(-up_h.max())}
    return fig, facts


# ── S09-36: look-ahead in the data preparation (X57) ───────────────────────


@register(
    "S09-36", "9", "lookahead_prep", "Look-ahead in the steps before the hourly table (X57)",
    "Do the clock correction, the polled reconstruction and the cleaning use data after T0, and "
    "how much does it change?",
    "Left: whole-minute clock shift of each day as applied (a centred median of daily estimates) "
    "against a trailing median of the earlier days only, as days per absolute difference. "
    "Middle: source cells of the 48 hours before T0 that change when the cleaning also sees the "
    "next day, per series, over every issuance day. Right: the bound on the last source hour's "
    "mean from the polled interpolation (test period only).",
    "The clock correction is the one look-ahead of size: a causal median would move the shift of "
    "{clock_share:.0f} % of the days, by at most {clock_max} min. The cleaning changes "
    "{clean_cells} of {clean_total:,} source cells on {clean_days} issuance days, none a scored "
    "hour, all a stale value at night; the full-window cleaning equals the stage-5 table on every "
    "compared cell. The polled interpolation is bounded by {polled_max:.2f} °C on a night hour.",
    "Declared in docs/deviations.md; whether to accept the centred clock or rerun stages 4–9 with "
    "a causal one is the user's decision (stage-9 plan §20.7).",
    (CLOCK, CLOCK_DAYS, CLEAN, CLEAN_CELLS, POLLED))
def s09_lookahead_prep():
    days = D.csv(CLOCK_DAYS)
    clock = D.csv(CLOCK).set_index("clock")
    clean = D.csv(CLEAN).set_index("source")
    cells = D.csv(CLEAN_CELLS)
    polled = D.csv(POLLED).set_index("source")
    fig, axes = subplots(1, 3, height=3.2, width=8.4)
    ax = axes[0]
    width = 0.38
    for k, (col, name, colour, hatch) in enumerate((("minute_meteo", "meteo clock", C["blue"], ""),
                                                    ("minute_inverter", "inverter clock",
                                                     C["orange"], "//"))):
        diff = (days[f"{col}_trailing_min"] - days[f"{col}_applied_min"]).abs()
        counts = diff.value_counts().reindex(range(0, int(diff.max()) + 1), fill_value=0)
        assert int(counts.iloc[1:].sum()) == int(clock.loc[col, "days_shift_differs"])
        ax.bar(counts.index + (k - 0.5) * width, counts.to_numpy(), width, color=colour,
               hatch=hatch, edgecolor="white", label=name)
    ax.set_xlabel("|trailing − applied| (whole minutes)")
    ax.set_ylabel("days")
    ax.set_title("clock correction (stage 4)", fontsize=8)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    order = list(SERIES)
    ax.barh(range(len(order)), clean.loc[order, "cells_changed"].to_numpy(), color=C["purple"])
    ax.set_yticks(range(len(order)), [SERIES[s] for s in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("cells changed by the next day")
    ax.set_title("cleaning (stage 5), every issuance day", fontsize=8)
    look = cells[cells["lookahead"]]
    hour = pd.to_datetime(look["hour"], utc=True)
    from solarpv.config.site import load_site
    from solarpv.evaluation.daytime import daytime_mask

    assert not np.asarray(daytime_mask(pd.DatetimeIndex(hour), load_site())).any()
    assert int(clean["method_mismatch_cells"].sum()) == 0
    assert (pd.to_datetime(look["t0_utc"], utc=True) < pd.Timestamp(VAL_START, tz="UTC")).all()
    ax = axes[2]
    ps = [s for s in order if s in polled.index]
    bound = polled.loc[ps, "max_bound_on_last_hour_mean"].to_numpy()
    ax.barh(range(len(ps)), bound, color=C["sky"])
    for i, b in enumerate(bound):
        if b == 0:
            ax.text(0, i, " 0 (no power at midnight)", va="center", fontsize=6)
    units = {"irradiance": "W/m²", "mod_temp1": "°C", "mod_temp2": "°C", "ambient_temp": "°C"}
    ax.set_yticks(range(len(ps)), [f"{SERIES[s]} ({units.get(s, 'kW')})" for s in ps],
                  fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("bound on the hourly mean")
    ax.set_title("polled source (test period only)", fontsize=8)
    temps = [s for s in ps if "temp" in s]
    facts = {"clock_share": float(clock["share_pct"].max()),
             "clock_max": int(clock["max_shift_difference_min"].max()),
             "clean_cells": int(clean["cells_changed"].sum()),
             "clean_total": int(clean["cells"].sum()),
             "clean_days": int(look["t0_utc"].nunique()),
             "polled_max": float(polled.loc[temps, "max_bound_on_last_hour_mean"].max())}
    return fig, facts


# ── S09-37: the test split was never read ──────────────────────────────────


@register(
    "S09-37", "9", "test_never_read", "No run has read the test split",
    "Has any run of the rebuild read or forecast a test hour before the gate?",
    "Every run of the runner so far: the last hour that any of its prediction files holds, "
    "against the start of the test split, with the number of its step records and prediction "
    "files; the evaluation layer's test-access log and its git history.",
    "Across {runs} runs, {records} step records hold no test access and {files} prediction files "
    "hold {test_rows} test hours; the latest forecast hour of any run is {last}, before the test "
    "split starts on {test_start}. The access log {log_state}.",
    "The test split is read once, by the final run (phase 1 forecasts it, phase 2 scores it, "
    "D-32), and every read is logged.",
    (ACCESS,))
def s09_test_never_read():
    acc = D.jload(ACCESS)
    runs = pd.DataFrame(acc["per_run"])
    runs["last"] = pd.to_datetime(runs["last_prediction_hour"], utc=True)
    test_start = pd.Timestamp(TEST_START, tz="UTC")
    assert acc["prediction_rows_in_test"] == 0 and not acc["steps_with_test_access"]
    assert not acc["log_exists"] and not acc["log_in_git_history"]
    assert (runs["last"] < test_start).all()
    fig, ax = subplots(height=3.6, width=7.6)
    y = np.arange(len(runs))
    kinds = runs["run"].str.endswith("_smoke")
    ax.scatter(runs.loc[kinds, "last"].dt.tz_localize(None), y[kinds.to_numpy()], marker="s",
               color=C["green"], label="smoke run")
    ax.scatter(runs.loc[~kinds, "last"].dt.tz_localize(None), y[~kinds.to_numpy()], marker="o",
               color=C["blue"], label="development run")
    ax.axvline(test_start.tz_localize(None), color=C["red"], lw=1.2)
    ax.axvspan(pd.Timestamp(VAL_START), test_start.tz_localize(None), color=C["orange"],
               alpha=0.12, lw=0)
    ax.axvspan(test_start.tz_localize(None), pd.Timestamp("2025-06-01"), color=C["grey"],
               alpha=0.15, lw=0)
    ax.text(test_start.tz_localize(None), len(runs) - 0.5, " test split", color=C["red"],
            fontsize=7, va="top")
    ax.text(pd.Timestamp(VAL_START), len(runs) - 0.5, " validation", color=C["orange"],
            fontsize=7, va="top")
    ax.set_yticks(y, [f"{r} ({n} files)" for r, n in zip(runs["run"], runs["prediction_files"],
                                                          strict=True)], fontsize=5.5)
    ax.set_xlabel("last forecast hour of the run (UTC)")
    ax.legend(fontsize=6.5, loc="center right")
    facts = {"runs": int(acc["runs"]), "records": int(acc["step_records"]),
             "files": int(acc["prediction_files"]),
             "test_rows": int(acc["prediction_rows_in_test"]),
             "last": str(runs["last"].max().strftime("%Y-%m-%d %H:%M UTC")),
             "test_start": str(test_start.date()),
             "log_state": "does not exist and was never committed"}
    return fig, facts


# ── S09-38: the reproduction from the raw data ─────────────────────────────

EXPLAINED_CLASS = "explained difference (D-34)"
UNEXPLAINED_CLASS = "unexplained difference"
CLASSES = (("identical bytes", C["green"], ""), ("same content, other bytes", C["sky"], "//"),
           ("floating-point noise", C["yellow"], ".."), (EXPLAINED_CLASS, C["orange"], "xx"),
           (UNEXPLAINED_CLASS, C["red"], "XX"), ("listed, not compared", C["light"], ""))
# D-34: a difference that follows from a documented fix is explained; nothing else is. A failing
# file that is not listed here is unexplained, and the figure refuses to be built with one.
EXPLAINED = (
    ("decide_k", None, "X66: a decision check run on the laptop; its numbers depend on the "
                       "machine and its decisions are equal"),
    ("nwp_check", None, "X66: a decision check run on the laptop; its numbers depend on the "
                        "machine and its decisions are equal"),
    ("dev_neural", ("report_steps.csv", "report_warnings.csv"),
     "X45: the step now emits the tft168 warning; and the row order of a resumed report"),
)


def explained_reason(step: str, path: str) -> str | None:
    for name, files, why in EXPLAINED:
        if step == name and (files is None or path.rsplit("/", 1)[-1] in files):
            return why
    return None


def _class(row: dict, step: str, failing: set) -> str:
    if row["path"] in failing:
        return EXPLAINED_CLASS if explained_reason(step, row["path"]) else UNEXPLAINED_CLASS
    if row["equal"] is True:
        return "identical bytes" if row["kind"] == "bytes" else "same content, other bytes"
    if row["equal"] is False and row.get("float_noise"):
        return "floating-point noise"
    return "listed, not compared"


def _model_style():
    from solarpv.figures.s07_s09 import NEURAL, NEURAL_STYLE, REF_STYLE

    style = {"XGB": REF_STYLE["xgb_sota"], "references": (C["grey"], "^:")}
    style.update({name: NEURAL_STYLE[model] for _, model, name in NEURAL})
    return style


@register(
    "S09-38", "9", "reproduction", "The pipeline reproduced from the raw data",
    "Does a fresh checkout, given only the raw data, the GFS point extracts and the site file, "
    "produce the same outputs at every stage and the same forecasts for every model?",
    "Left: every file each stage wrote in a fresh clone, compared with this checkout's: identical "
    "bytes, the same content in other bytes (a newer parquet writer), floating-point noise "
    "(numeric cells only, relative difference at most 1e-9, nothing missing on one side), a "
    "difference explained by a documented fix (D-34), an unexplained difference, or listed only "
    "(figures, logs, notebooks). Right: the forecast files of the two development runs per "
    "model, against the reference runs; colour and marker per model.",
    "The script's verdict is {result}, by its rule that any failing file fails: of {files:,} "
    "compared files, {identical:,} have identical bytes, {content} the same content, {noise} "
    "differ by floating-point noise only (at most {max_rel:.1g} relative), {explained} differ for "
    "a documented reason and {unexplained} for none. The explained files are the {n_laptop} of "
    "the decision checks run on the laptop (X66: the {decisions} decisions they record come out "
    "the same) and the {n_neural} reports of the neural development run (X45 and the row order "
    "of a resumed report). {models_equal} of {models} models give the same forecasts as the "
    "reference runs. The parse of the raw text differs by up to {ingest_rel:.1g} relative "
    "(X63).",
    "Evidence of gate G-1 (stage-9 plan §20.5, D-34); the final run starts from the same code "
    "and inputs.",
    (REPRO,))
def s09_reproduction():
    rec = D.jload(REPRO)
    steps = rec["steps"]
    dev_steps = [s for s in steps if s["step"].startswith("dev_")]
    unexplained = [(s["step"], r["path"]) for s in steps for r in s["failing"]
                   if not explained_reason(s["step"], r["path"])]
    assert not unexplained, f"unexplained differences: {unexplained}"  # D-34
    counts = []
    for s in steps:
        failing = {r["path"] for r in s["failing"]}
        c = {k: 0 for k, _, _ in CLASSES}
        for r in s["compared"]:
            c[_class(r, s["step"], failing)] += 1
        counts.append((s["step"], c))
    fig, axes = subplots(1, 2, height=4.2, width=8.6, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for i, (_, c) in enumerate(counts):
        left = 0
        for k, colour, hatch in CLASSES:
            if c[k]:
                ax.barh(i, c[k], left=left, color=colour, hatch=hatch, edgecolor="white")
                left += c[k]
    ax.set_yticks(range(len(counts)), [n for n, _ in counts], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("files written by the step (symmetric log scale)")
    handles = [Patch(facecolor=col, hatch=h, edgecolor="white", label=k)
               for k, col, h in CLASSES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.5)
    ax = axes[1]
    style = _model_style()
    per_model: dict[str, dict[str, int]] = {}
    for s in dev_steps:
        failing = {r["path"] for r in s["failing"]}
        for r in s["compared"]:
            parts = r["path"].split("/")
            if len(parts) < 3 or parts[1] not in STEP_MODEL:
                continue
            name = parts[-1]
            if not (name.startswith(("predictions_", "quantiles_", "uncorrected_"))
                    or name.endswith(".npz")):
                continue
            m = per_model.setdefault(STEP_MODEL[parts[1]], {"same": 0, "noise": 0, "differs": 0})
            k = _class(r, s["step"], failing)
            m["same" if k in ("identical bytes", "same content, other bytes") else
              "noise" if k == "floating-point noise" else "differs"] += 1
    order = [m for m in ("XGB", "BiLSTM", "TFT-96", "TFT-168", "TFT baseline", "PatchTST",
                         "references") if m in per_model]
    for i, m in enumerate(order):
        colour, fmt = style[m]
        v = per_model[m]
        ax.barh(i, v["same"], color=colour, edgecolor="black", lw=0.4)
        if v["noise"]:
            ax.barh(i, v["noise"], left=v["same"], color=C["yellow"], hatch="..",
                    edgecolor="white")
        if v["differs"]:
            ax.barh(i, v["differs"], left=v["same"] + v["noise"], color=C["red"], hatch="XX",
                    edgecolor="white")
        ax.plot(v["same"] + v["noise"] + v["differs"], i, fmt[0], color=colour, ms=6,
                mec="black", mew=0.4)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("forecast files of the development runs")
    ax.set_title("per model", fontsize=8)
    total = {k: sum(c[k] for _, c in counts) for k, _, _ in CLASSES}
    rels = [r.get("max_rel", 0.0) for s in steps for r in s["compared"]
            if r["equal"] is False and r.get("float_noise")]
    ingest = next((r.get("max_rel", 0.0) for s in steps if s["step"] == "ingest"
                   for r in s["compared"] if r["path"].endswith("scada_1min_utc.parquet")), 0.0)
    models = [m for m in order if m != "references"]
    assert all(per_model[m]["differs"] == 0 for m in per_model), per_model  # no forecast differs
    # X66: a failing decision file of a stage run on the laptop: are its decisions the same?
    import yaml

    decisions = []
    for s in steps:
        for r in s["failing"]:
            if not (r["path"].startswith("configs/") and r.get("kept")):
                continue
            ref = yaml.safe_load((REPO_ROOT / r["path"]).read_text(encoding="utf-8"))
            new = yaml.safe_load((REPO_ROOT / r["kept"]).read_text(encoding="utf-8"))
            if "k_denominator" in ref:
                keys = ("k_denominator", "training_min_theo_kw", "evaluation_min_theo_kw",
                        "wins")
                same = all(ref[k] == new[k] for k in keys)
            else:
                same = ref["admitted"] == new["admitted"] and all(
                    ref["results"][k]["passes"] == new["results"][k]["passes"]
                    for k in ref["results"])
            decisions.append((r["path"], same))
    assert decisions and all(same for _, same in decisions), decisions
    in_step = {name: sum(1 for s in steps if s["step"] == name for _ in s["failing"])
               for name in ("decide_k", "nwp_check", "dev_neural")}
    facts = {"result": "NOT reproduced" if rec.get("result") != "reproduced" else "reproduced",
             "decisions": len(decisions), "stages": len(steps),
             "files": sum(total[k] for k in total if k != "listed, not compared"),
             "identical": total["identical bytes"], "content": total["same content, other bytes"],
             "noise": total["floating-point noise"], "explained": total[EXPLAINED_CLASS],
             "unexplained": total[UNEXPLAINED_CLASS],
             "n_laptop": in_step["decide_k"] + in_step["nwp_check"],
             "n_neural": in_step["dev_neural"],
             "max_rel": max(rels) if rels else 0.0,
             "models_equal": sum(per_model[m]["differs"] == 0 and per_model[m]["noise"] == 0
                                 for m in models),
             "models": len(models), "ingest_rel": ingest}
    assert facts["unexplained"] == 0
    return fig, facts


# ── S09-40 and S09-41: the two analyses that train models (D-36) ────────────

W6DEV = D.W6_DEV_RUN
CV_FOLDS_F, CV_SUMMARY_F = f"{W6DEV}/cv/cv_folds.csv", f"{W6DEV}/cv/cv_summary.csv"
AB_RESULTS_F, AB_SUMMARY_F = (f"{W6DEV}/ablation/ablation_results.csv",
                              f"{W6DEV}/ablation/ablation_summary.csv")
HORIZON_STYLE = {1: ("o", "-"), 2: ("s", "--"), 3: ("^", ":")}  # XGB is black in every figure


@register(
    "S09-40", "9", "xgb_cv", "Cross-validation of the XGB models on the train rows (S-38)",
    "How much does the error of the XGB models vary between five expanding date folds of the "
    "train rows, and how does it compare with their validation value?",
    "nRMSE in kW (% of 3,449 kW) of each of the five folds of the stage-7 expanding-window split "
    "(3-day gap) for each target and horizon, seed {seed}; the model of the development run "
    "refitted on the fold's earlier days with its rounds fixed, scored on the fold's daytime hours "
    "with AC truth. Right of each panel: the validation value of the same model (its "
    "early-stopping set, raw forecast). XGB only: S-38 is a v6 analysis of XGB-Opt.",
    "The fold values range from {lo:.1f} to {hi:.1f} % (mean {mean_lo:.1f}–{mean_hi:.1f} %, std "
    "{std_lo:.1f}–{std_hi:.1f} pp over the five folds); the fits grow from {fit_lo:,} to "
    "{fit_hi:,} rows. The validation value is {gap_lo:.1f}–{gap_hi:.1f} pp below the CV mean in "
    "all six cases. Validation tuned the model, the folds did not, and the early folds train on "
    "less: the gap is reported and not explained (M43). One seed.",
    "The step `cv` repeats this for the three seeds of the final run, and the step `test` adds "
    "the test value of the uncorrected forecast (phase 2).",
    (CV_FOLDS_F, CV_SUMMARY_F))
def s09_xgb_cv():
    folds, summary = D.csv(CV_FOLDS_F), D.csv(CV_SUMMARY_F)
    assert set(folds["seed"]) == {summary["seed"].iloc[0]} and folds["fold"].nunique() == 5
    assert (pd.to_datetime(folds["last_day"]) < pd.Timestamp(VAL_START)).all()  # train only
    assert (summary["validation_minus_cv_pp"] < 0).all()
    fig, axes = subplots(1, 2, height=3.4, width=8.4, sharey=True)
    for ax, target in zip(axes, ("indirect", "direct"), strict=True):
        for h in (1, 2, 3):
            g = folds[(folds["target"] == target) & (folds["horizon"] == h)].sort_values("fold")
            marker, line = HORIZON_STYLE[h]
            ax.plot(g["fold"], g["nrmse_kw_pct"], marker=marker, ls=line, color="black",
                    label=f"D+{h}")
            v = summary[(summary["target"] == target) & (summary["horizon"] == h)].iloc[0]
            ax.plot([5.2], [v["validation_uncorrected"]], marker=marker, ls="", color=C["red"],
                    mec="black", ms=7)
        first = folds[(folds["target"] == target) & (folds["horizon"] == 1)].sort_values("fold")
        ax.set_xticks([*first["fold"], 5.2], [f"{i + 1}\n{str(a)[:7]}" for i, a in
                                              zip(first["fold"], first["first_day"],
                                                  strict=True)] + ["validation"], fontsize=6.5)
        ax.set_title(target, fontsize=8)
        ax.set_xlabel("fold, and the month it starts")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("nRMSE in kW (% of 3,449 kW)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", ls="", color=C["red"], mec="black", ms=7))
    fig.legend(handles=handles, labels=[*labels, "validation value of the same model"],
               loc="outside lower center", ncol=4, fontsize=6.5)
    ys = folds["nrmse_kw_pct"]
    facts = {"seed": int(summary["seed"].iloc[0]), "lo": float(ys.min()), "hi": float(ys.max()),
             "mean_lo": float(summary["cv_mean"].min()), "mean_hi": float(summary["cv_mean"].max()),
             "std_lo": float(summary["cv_std"].min()), "std_hi": float(summary["cv_std"].max()),
             "fit_lo": int(folds["fit_rows"].min()), "fit_hi": int(folds["fit_rows"].max()),
             "gap_lo": float(-summary["validation_minus_cv_pp"].max()),
             "gap_hi": float(-summary["validation_minus_cv_pp"].min())}
    return fig, facts


GROUP_LABELS = {"nwp_gfs": "GFS forecasts", "clearsky": "clear sky", "other": "geometry, other",
                "time_encodings": "time encodings", "lag_power": "lags",
                "rolling_stats": "rolling statistics", "history_other": "previous-day index",
                "history": "all measured history"}
VERDICT_STYLE = {"helpful": (C["red"], "//"), "neutral": (C["grey"], ""),
                 "harmful": (C["blue"], "xx")}  # helpful: removing the group raises the error


def _mode_label(mode: str) -> str:
    if mode == "minimal":
        return "only time encodings"
    return "without " + GROUP_LABELS.get(mode.removeprefix("no_"), mode.removeprefix("no_"))


@register(
    "S09-41", "9", "xgb_ablation", "Feature-group ablation of the XGB models (S-41)",
    "How much does the validation error of the XGB models change when one feature group is "
    "removed and the model is refitted?",
    "For each target and horizon, the change in nRMSE in kW (pp) of the model refitted without "
    "one feature group of the contract (and without all measured history, and with only the "
    "time encodings), against the full model of the development run, seed {seed}. Bars right of "
    "zero: removing the group raises the error (hatched red); the dotted lines are the registered "
    "margin of {margin:.2f} pp (with one seed, the floor). XGB only: S-41 is a v6 analysis of "
    "XGB-Opt.",
    "Removing the GFS forecasts raises the error by {nwp_lo:.2f}–{nwp_hi:.2f} pp in every case "
    "and is the largest change of any single group; with only the time encodings the error is "
    "{min_lo:.1f}–{min_hi:.1f} pp higher. Of {n_modes} changes, {helpful} exceed the margin on one "
    "seed, {neutral} do not and {harmful} are below minus it. With one seed the margin is only "
    "the floor, so these verdicts are provisional (M41); the final run adds two seeds and the "
    "pooled seed std.",
    "The step `ablation` repeats this for the three seeds of the final run, and the step `test` "
    "scores the same fits on the test window in phase 2.",
    (AB_RESULTS_F, AB_SUMMARY_F))
def s09_xgb_ablation():
    summary, res = D.csv(AB_SUMMARY_F), D.csv(AB_RESULTS_F)
    assert summary["single_seed"].all() and res["seed"].nunique() == 1
    margin = float(summary["margin_pp"].iloc[0])
    assert (summary["margin_pp"] == margin).all()
    singles = summary[~summary["mode"].isin(["minimal", "no_history"])]
    for (_, _), g in singles.groupby(["target", "horizon"]):  # the largest single group: GFS
        assert g.loc[g["mean_delta_pp"].idxmax(), "mode"] == "no_nwp_gfs"
    order = ["no_nwp_gfs", "no_clearsky", "no_other", "no_time_encodings", "no_lag_power",
             "no_rolling_stats", "no_history_other", "no_history", "minimal"]
    fig, axes = subplots(2, 3, height=5.4, width=8.8, sharex=True)
    for i, target in enumerate(("indirect", "direct")):
        for j, h in enumerate((1, 2, 3)):
            ax = axes[i][j]
            g = summary[(summary["target"] == target) & (summary["horizon"] == h)]
            g = g.set_index("mode").reindex([m for m in order if m in set(g["mode"])])
            for k, (_, r) in enumerate(g.iterrows()):
                col, hatch = VERDICT_STYLE[r["verdict"]]
                ax.barh(-k, r["mean_delta_pp"], color=col, hatch=hatch, edgecolor="white")
                ax.text(max(r["mean_delta_pp"], 0) + 0.03, -k, f"{r['mean_delta_pp']:+.2f}",
                        va="center", fontsize=5)
            ax.set_yticks(-np.arange(len(g)), [_mode_label(m) for m in g.index], fontsize=5.5)
            for s in (margin, -margin):
                ax.axvline(s, color="black", ls=":", lw=0.8)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_title(f"{target}, D+{h}", fontsize=8)
            if i == 1:
                ax.set_xlabel("change in nRMSE (pp)", fontsize=7)
    handles = [Patch(facecolor=col, hatch=h, edgecolor="white", label=f"{v} (one seed)")
               for v, (col, h) in VERDICT_STYLE.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.5)
    nwp = summary[summary["mode"] == "no_nwp_gfs"]["mean_delta_pp"]
    only = summary[summary["mode"] == "minimal"]["mean_delta_pp"]
    counts = summary["verdict"].value_counts()
    facts = {"seed": int(res["seed"].iloc[0]), "margin": margin, "nwp_lo": float(nwp.min()),
             "nwp_hi": float(nwp.max()), "min_lo": float(only.min()), "min_hi": float(only.max()),
             "n_modes": int(len(summary)), "helpful": int(counts.get("helpful", 0)),
             "neutral": int(counts.get("neutral", 0)), "harmful": int(counts.get("harmful", 0))}
    return fig, facts


# ── S09-39: the entry points before and after X62 ──────────────────────────


def _entry_points_at(commit: str | None) -> dict[str, bool]:
    """Every entry point of src/solarpv and scripts/ at a commit (None: the working tree), and
    whether it parses its arguments before any work (the static rule of T-98)."""
    git = find_git()
    if commit:
        names = subprocess.run([git, "ls-tree", "-r", "--name-only", commit, "src/solarpv",
                                "scripts"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.split()
    else:
        names = [p.relative_to(REPO_ROOT).as_posix() for p in
                 [*(REPO_ROOT / "src" / "solarpv").rglob("*.py"),
                  *(REPO_ROOT / "scripts").glob("*.py")]]
    out = {}
    for rel in sorted(n for n in names if n.endswith(".py")):
        text = (subprocess.run([git, "show", f"{commit}:{rel}"], cwd=REPO_ROOT,
                               capture_output=True, text=True, encoding="utf-8",
                               check=True).stdout if commit
                else (REPO_ROOT / rel).read_text(encoding="utf-8"))
        tree = ast.parse(text)
        guarded = any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)
                      for n in tree.body)
        builder = rel.startswith("scripts/make_")
        if not (guarded or rel.endswith("__main__.py") or builder):
            continue
        if rel in ("src/solarpv/config/constants.py", "src/solarpv/models/neural_worker.py"):
            continue  # exempt (T-98): they write nothing without their exact arguments
        out[rel] = _parses_first(tree)
    return out


def _parses_first(tree: ast.Module) -> bool:
    def calls(node) -> set[str]:
        return {(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
                for n in ast.walk(node) if isinstance(n, ast.Call)}
    parse = {"no_options", "ArgumentParser", "parser"}
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
                None)
    if main is not None:
        body = [s for s in main.body if not (isinstance(s, ast.Expr)
                                             and isinstance(s.value, ast.Constant))]
        return bool(body) and bool(calls(body[0]) & parse)
    rest = [s for s in tree.body if not isinstance(s, (ast.Import, ast.ImportFrom))
            and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return bool(rest) and (bool(calls(rest[0]) & parse) or "main" in calls(rest[0]))


@register(
    "S09-39", "9", "entry_points", "Entry points that run nothing on --help (X62)",
    "Could reading a module's help rewrite data, and did the accidental rerun change anything?",
    "Left: the entry points of the code (stage modules, scripts, notebook builders) at the "
    "start of W6 and now, by whether they parse their arguments before any work. Right: the "
    "four stage-4 and stage-5 tables that the accidental run rewrote, cell by cell against the "
    "originals restored from the hand-off package.",
    "At the start of W6 {before_bad} of {before_n} entry points ran their work on any argument, "
    "--help included; now {after_bad} of {after_n} do. The accidental rerun rewrote {tables} "
    "tables with {cells:,} cells, every one equal to the original; only the parquet writer's "
    "bytes differed ({writer_old} against {writer_new}), and the originals are back.",
    "T-98 checks every entry point statically before it calls any; the reproduction compares "
    "content, not bytes (M33).",
    (INCIDENT,))
def s09_entry_points():
    before = _entry_points_at(BEFORE_X62)
    after = _entry_points_at(None)
    inc = D.jload(INCIDENT)
    assert all(r["content_equal"] for r in inc.values())
    assert all(after.values()), [k for k, v in after.items() if not v]
    fig, axes = subplots(1, 2, height=3.0, width=7.6)
    ax = axes[0]
    for x, (_, d) in enumerate((("start of W6", before), ("after X62", after))):
        ok, bad = sum(d.values()), sum(not v for v in d.values())
        ax.bar(x, ok, color=C["green"], label="parses first" if x == 0 else None)
        ax.bar(x, bad, bottom=ok, color=C["red"], hatch="//", edgecolor="white",
               label="runs its work on any argument" if x == 0 else None)
        ax.text(x, ok + bad, f"{bad} of {ok + bad} unsafe", ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks([0, 1], ["start of W6", "after X62"])
    ax.set_ylabel("entry points")
    ax.set_ylim(0, 1.5 * max(len(before), len(after)))
    ax.legend(fontsize=6.5, loc="upper center", ncol=1)
    ax = axes[1]
    names = list(inc)
    n_cells = [r["shape_orig"][0] * r["shape_orig"][1] for r in inc.values()]
    ax.barh(range(len(names)), n_cells, color=C["blue"])
    ax.set_yticks(range(len(names)), [n.split("/", 2)[-1] for n in names], fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("cells compared (all equal; log scale)")
    ax.invert_yaxis()
    r0 = next(iter(inc.values()))
    facts = {"before_bad": sum(not v for v in before.values()), "before_n": len(before),
             "after_bad": sum(not v for v in after.values()), "after_n": len(after),
             "tables": len(inc), "cells": int(sum(n_cells)),
             "writer_old": r0["created_by_orig"].replace("parquet-cpp-arrow version ", "pyarrow "),
             "writer_new": r0["created_by_new"].replace("parquet-cpp-arrow version ", "pyarrow ")}
    return fig, facts

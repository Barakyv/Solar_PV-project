"""Stage 6b, S1: download the GFS 1-degree runs from the NOAA archive on AWS
(docs/stage6b_plan.md; PROJECT_MASTER.md §6.12, D-12).

    python -m solarpv.nwp.download --estimate --sample 5   # read only .idx files: size estimate
    python -m solarpv.nwp.download --start 2022-06-15 --end 2022-06-15   # one run
    python -m solarpv.nwp.download                          # the whole period, resumable

For every UTC day, the 12Z run is fetched; if a file of it is missing, the 06Z and then the 00Z
run of the same day (both published earlier, so allowed at the issuance time) are used, and the
run is flagged.

For each forecast file f009-f084 (every 3 h):
- read the .idx;
- select the 11 messages of MESSAGES;
- fetch them with HTTP Range requests, merging neighbours when the gap is small (S3 serves one
  range per request);
- decode in memory and keep only the 3x3 grid points around 30N 35E.

The Last-Modified time of every file is recorded: T-17 checks that each file was public before
the issuance time it serves. Output, per run:
- data/external/gfs_1p00/YYYY/YYYYMMDD_HHz.parquet: the values (lead, variable, step range,
  lat, lon);
- YYYYMMDD_HHz.json: the record (urls, ranges, bytes, upload times, messages).

A run with a complete record is skipped (resume).
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from solarpv.config.paths import DATA_DIR, OUTPUTS_DIR
from solarpv.config.runlog import RunLog

BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT_DIR = DATA_DIR / "external" / "gfs_1p00"
FIRST_RUN_DAY = "2021-08-29"  # the D+3 issuance for the first data day (2021-09-01 local)
LAST_RUN_DAY = "2025-05-30"   # the D+1 issuance for the last data day (2025-05-31 local)
RUN_HOURS = (12, 6, 0)        # preferred run first; all are published before T0 (A-13)
LEADS_12Z = tuple(range(9, 85, 3))  # hour ends 10-82 h for D+1..D+3, plus the first block start
MERGE_GAP_BYTES = 50_000  # 19.6 MB per run; larger gaps fetch unwanted messages
TIMEOUT_S = 60
RETRIES = 4

# (canonical name, GRIB name, level text in the .idx, kind) - plan S1, ideas L-07
MESSAGES = (
    ("dswrf", "DSWRF", "surface", "ave"),
    ("tcc", "TCDC", "entire atmosphere", "ave"),
    ("lcc", "LCDC", "low cloud layer", "ave"),
    ("mcc", "MCDC", "middle cloud layer", "ave"),
    ("hcc", "HCDC", "high cloud layer", "ave"),
    ("t2m", "TMP", "2 m above ground", "inst"),
    ("rh2m", "RH", "2 m above ground", "inst"),
    ("pwat", "PWAT", "entire atmosphere (considered as a single layer)", "inst"),
    ("u10", "UGRD", "10 m above ground", "inst"),
    ("v10", "VGRD", "10 m above ground", "inst"),
    ("sunsd", "SUNSD", "surface", "inst"),
)
FTIME = {"ave": re.compile(r"^\d+-\d+ hour ave fcst$"), "inst": re.compile(r"^\d+ hour fcst$")}
_LOCAL = threading.local()


def _session() -> requests.Session:
    """One HTTP session per thread (connection reuse without sharing a session)."""
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


@dataclass(frozen=True)
class IdxEntry:
    number: int
    offset: int
    end: int | None  # last byte (inclusive); None for the last message of the file
    var: str
    level: str
    ftime: str


def file_url(day: str, run: int, lead: int) -> str:
    d = day.replace("-", "")
    return f"{BASE_URL}/gfs.{d}/{run:02d}/atmos/gfs.t{run:02d}z.pgrb2.1p00.f{lead:03d}"


def leads_for_run(run: int) -> tuple[int, ...]:
    """Leads that cover the same target hours as f009-f084 of the 12Z run."""
    return tuple(lead + (12 - run) for lead in LEADS_12Z)


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse a GFS .idx file: 'n:offset:d=YYYYMMDDHH:VAR:LEVEL:FTIME:' per line."""
    raw = [line.split(":") for line in text.strip().splitlines() if line.strip()]
    entries = []
    for i, f in enumerate(raw):
        end = int(raw[i + 1][1]) - 1 if i + 1 < len(raw) else None
        entries.append(IdxEntry(int(f[0]), int(f[1]), end, f[3], f[4], f[5]))
    return entries


def select_messages(entries: list[IdxEntry]) -> dict[str, IdxEntry]:
    """The entry of each wanted message; exactly one must match, otherwise ValueError."""
    out = {}
    for name, var, level, kind in MESSAGES:
        hits = [e for e in entries if e.var == var and e.level == level
                and FTIME[kind].match(e.ftime)]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} index entries for {var}:{level} ({kind})")
        out[name] = hits[0]
    return out


def merge_ranges(entries: list[IdxEntry], gap: int = MERGE_GAP_BYTES) -> list[tuple[int, int]]:
    """Byte ranges (inclusive) that cover the entries, merging neighbours closer than `gap`."""
    spans = sorted((e.offset, e.end) for e in entries)
    merged = []
    for start, end in spans:
        if end is None:
            raise ValueError("a wanted message is the last of its file; not supported")
        if merged and start - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _get(session: requests.Session, url: str, headers: dict | None = None) -> requests.Response:
    for attempt in range(RETRIES):
        try:
            r = session.get(url, headers=headers or {}, timeout=TIMEOUT_S)
            if r.status_code in (200, 206, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {RETRIES} attempts: {url}")


def fetch_file(day: str, run: int, lead: int) -> dict | None:
    """The wanted messages of one forecast file, decoded at the 3x3 points. None if missing."""
    from solarpv.nwp.grib import decode_message  # imports ecCodes

    session = _session()
    url = file_url(day, run, lead)
    r = _get(session, url + ".idx")
    if r.status_code == 404:
        return None
    wanted = select_messages(parse_idx(r.text))
    by_offset = {e.offset: (name, e) for name, e in wanted.items()}
    rows, uploaded, nbytes, ranges = [], None, 0, merge_ranges(list(wanted.values()))
    for start, end in ranges:
        resp = _get(session, url, {"Range": f"bytes={start}-{end}"})
        if resp.status_code != 206:
            raise RuntimeError(f"range request not honoured ({resp.status_code}): {url}")
        uploaded = parsedate_to_datetime(resp.headers["Last-Modified"]).astimezone(UTC)
        blob, nbytes = resp.content, nbytes + len(resp.content)
        for offset, (name, e) in by_offset.items():
            if start <= offset <= end:
                msg = decode_message(blob[offset - start:e.end - start + 1])
                for (lat, lon), value in msg["points"].items():
                    rows.append({"lead": lead, "variable": name, "step_range": msg["stepRange"],
                                 "step_type": msg["stepType"], "lat": lat, "lon": lon,
                                 "value": value})
    return {"url": url, "uploaded_utc": uploaded.isoformat(), "bytes": nbytes,
            "ranges": ranges, "rows": rows}


def fetch_run(day: str, run: int, workers: int) -> dict | None:
    """All files of one run; None if any file is missing."""
    leads = leads_for_run(run)
    with ThreadPoolExecutor(workers) as pool:
        files = list(pool.map(lambda lead: fetch_file(day, run, lead), leads))
    if any(f is None for f in files):
        return None
    return {"day": day, "run": run, "init_utc": f"{day}T{run:02d}:00:00+00:00",
            "files": [{k: v for k, v in f.items() if k != "rows"} for f in files],
            "rows": [row for f in files for row in f["rows"]]}


def run_paths(day: str, run: int) -> tuple[Path, Path]:
    stem = OUT_DIR / day[:4] / f"{day.replace('-', '')}_{run:02d}z"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def completed_runs() -> dict[str, int]:
    """Days whose run record is complete -> the run hour used."""
    done = {}
    for path in OUT_DIR.glob("*/*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") == "complete":
            done[rec["day"]] = rec["run"]
    return done


def download_day(day: str, workers: int) -> dict:
    for run in RUN_HOURS:
        result = fetch_run(day, run, workers)
        if result is not None:
            parquet, record = run_paths(day, run)
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result.pop("rows")).to_parquet(parquet, index=False)
            result.update({"status": "complete", "fallback": run != RUN_HOURS[0],
                           "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds")})
            record.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
            return result
    return {"day": day, "status": "missing"}


def estimate(days: list[str]) -> dict:
    """Transfer estimate from .idx files only (no GRIB data is downloaded)."""
    total, files = 0, 0
    with requests.Session() as session:
        for day in days:
            for lead in LEADS_12Z:
                r = _get(session, file_url(day, 12, lead) + ".idx")
                if r.status_code == 404:
                    continue
                for start, end in merge_ranges(list(select_messages(parse_idx(r.text)).values())):
                    total += end - start + 1
                files += 1
    return {"days_sampled": len(days), "files": files, "bytes": total,
            "bytes_per_run": total / max(len(days), 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download GFS 1-degree runs (stage 6b).")
    parser.add_argument("--start", default=FIRST_RUN_DAY)
    parser.add_argument("--end", default=LAST_RUN_DAY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--estimate", action="store_true", help="read .idx files only")
    parser.add_argument("--sample", type=int, default=0, help="with --estimate: days sampled")
    args = parser.parse_args(argv)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start, args.end, freq="D")]
    command = "python -m solarpv.nwp.download " + " ".join(argv or sys.argv[1:])
    if args.estimate:
        sample = days[:: max(len(days) // args.sample, 1)][: args.sample] if args.sample else days
        est = estimate(sample)
        est["runs_in_period"] = len(days)
        est["estimated_total_gb"] = est["bytes_per_run"] * len(days) / 1e9
        print(json.dumps(est, indent=1))
        return 0
    with RunLog("06b_gfs", command.strip(), out_dir=OUTPUTS_DIR / "06b_gfs") as log:
        done = completed_runs()
        todo = [d for d in days if d not in done]
        log.note(f"{len(days)} days requested; {len(done)} complete; {len(todo)} to download")
        counts = {"complete": 0, "fallback": 0, "missing": 0, "bytes": 0}
        for i, day in enumerate(todo, 1):
            result = download_day(day, args.workers)
            if result["status"] == "complete":
                counts["complete"] += 1
                counts["fallback"] += int(result["fallback"])
                counts["bytes"] += sum(f["bytes"] for f in result["files"])
                if result["fallback"]:
                    log.warn(f"{day}: 12Z missing, used {result['run']:02d}Z")
            else:
                counts["missing"] += 1
                log.warn(f"{day}: no allowed run available")
            if i % 50 == 0 or i == len(todo):
                log.note(f"{i}/{len(todo)} days; {counts}")
        for key, value in counts.items():
            log.value(key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())

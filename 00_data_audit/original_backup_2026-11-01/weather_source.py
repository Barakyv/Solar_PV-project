"""What was the original "GFS forecast", and how good was it? (feeds F-28, C-06, D-12)

    python 00_data_audit/original_backup_2026-11-01/inventory.py        # recovers the blobs
    python 00_data_audit/original_backup_2026-11-01/weather_source.py

1. **The request.** The backup keeps the HTTP cache of its feature notebook (two versions, working
   copy and git history). The request URL is read from the raw bytes of the cache rows; the
   cached response is never unpickled. The coordinates in the URL are private and are not written
   out (configs/site.private.yaml).
2. **The file** `ketura_gfs_forecast.parquet`: rows, period, columns.
3. **Skill against the measurement, train and validation only** (the test split is not read):
   clear-sky index of the original GHI (fcst_ghi / ghi_clearsky) and of the real D+1 GFS run
   (stage 6b, fcst_gfs_kt) against the measured in-plane index (irradiance / poa_global_clearsky),
   daytime hours with sun elevation > 10°. First the hourly lag that aligns the original series
   (the API labels the mean of the preceding hour at the hour end, N7); then daily means per
   local day. The original pipeline joined it on a clock 2-3 h off (F-01, N3): the same scan
   shows what that cost.
"""

import json
import re
import sqlite3
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import _backup as B
import numpy as np
import pandas as pd

from solarpv.config.paths import PROCESSED_DIR
from solarpv.config.runlog import RunLog

HERE = Path(__file__).resolve().parent
CACHES = [B.BACKUP / "04_feature_engineering/.cache/http_cache.sqlite"]
ORIGINAL = B.BACKUP / "04_feature_engineering/data/raw/ketura_gfs_forecast.parquet"
FEATURES = PROCESSED_DIR / "06_features" / "features_1H_D1.parquet"
GFS = PROCESSED_DIR / "06b_gfs" / "gfs_features_D1.parquet"
HOURLY = PROCESSED_DIR / "05_clean" / "scada_1H_clean.parquet"
PRIVATE = {"latitude", "longitude"}


def requests_in(cache: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{cache.as_posix()}?mode=ro", uri=True)
    urls = set()
    for (value,) in con.execute("select value from responses"):
        urls |= set(re.findall(rb"https?://[\x21-\x7e]+?(?=[^\x21-\x7e]|$)", bytes(value)))
    con.close()
    out = []
    for raw in sorted(urls):
        parts = urlsplit(raw.decode("ascii"))
        query = parse_qsl(parts.query)
        out.append({"endpoint": f"{parts.scheme}://{parts.netloc}{parts.path}",
                    "parameters": {k: v for k, v in query if k not in PRIVATE and k != "hourly"},
                    "hourly_variables": [v for k, v in query if k == "hourly"],
                    "coordinates": "present, not written out (private)"})
    unique = {json.dumps(r, sort_keys=True) for r in out}
    return [json.loads(r) for r in sorted(unique)]


def corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> int:
    with RunLog("audit", "python 00_data_audit/original_backup_2026-11-01/weather_source.py",
                out_dir=HERE) as log:
        caches = CACHES + [B.recovered("bb194fca9d7e")]
        log.inputs(caches + [ORIGINAL, FEATURES, GFS, HOURLY])
        out = {"requests": {}, "file": {}, "skill": {}}
        with log.step("request"):
            for cache in caches:
                reqs = requests_in(cache)
                out["requests"][cache.name] = reqs
                for r in reqs:
                    log.note(f"{cache.name}: {r['endpoint']} {r['parameters']}")
        with log.step("file"):
            old = pd.read_parquet(ORIGINAL)
            out["file"] = {"rows": len(old), "start_utc": str(old.index.min()),
                           "end_utc": str(old.index.max()), "columns": list(old.columns),
                           "series": "one hourly series for the whole period, no issue time"}
        with log.step("skill on train and validation"):
            base = pd.read_parquet(FEATURES, columns=["split", "daytime", "sun_elevation",
                                                      "ghi_clearsky", "poa_global_clearsky"])
            t = base.join(pd.read_parquet(GFS, columns=["fcst_gfs_kt"]))
            t = t[t["split"].isin(["train", "val"])]
            assert not (t["split"] == "test").any()  # the test split is never read further
            t = t[t["daytime"] & (t["sun_elevation"] > 10) & (t["ghi_clearsky"] > 50)]
            measured = (pd.read_parquet(HOURLY, columns=["irradiance"])["irradiance"]
                        .reindex(t.index) / t["poa_global_clearsky"])
            ghi = old["fcst_ghi"]
            scan = pd.DataFrame({"shift_h": range(-3, 4)})
            scan["hourly_r"] = [corr(ghi.shift(s).reindex(t.index) / t["ghi_clearsky"], measured)
                                for s in scan["shift_h"]]
            scan.round(4).to_csv(HERE / "weather_lag_scan.csv", index=False, lineterminator="\n")
            best = int(scan.loc[scan["hourly_r"].idxmax(), "shift_h"])
            frame = pd.DataFrame({
                "measured": measured,
                "original": ghi.shift(best).reindex(t.index) / t["ghi_clearsky"],
                "gfs_d1": t["fcst_gfs_kt"], "split": t["split"]}).dropna()
            day = frame.index.tz_convert("Asia/Jerusalem").date
            daily = frame.drop(columns="split").groupby(day).mean()
            split_of_day = frame["split"].groupby(day).first()
            daily_r = {}
            for name, rows in (("train_and_val", daily.index),
                               ("train", daily.index[split_of_day == "train"]),
                               ("val", daily.index[split_of_day == "val"])):
                d = daily.loc[rows]
                daily_r[name] = {"days": len(d),
                                 "original": round(corr(d["original"], d["measured"]), 3),
                                 "gfs_d1": round(corr(d["gfs_d1"], d["measured"]), 3)}
            cloudy = daily[daily["measured"] < 0.7]
            out["skill"] = {
                "hours": len(frame), "best_shift_h": best,
                "hourly_r_by_shift": dict(zip(scan["shift_h"].astype(str),
                                              scan["hourly_r"].round(3), strict=True)),
                "hourly_r": {"original": round(corr(frame["original"], frame["measured"]), 3),
                             "gfs_d1": round(corr(frame["gfs_d1"], frame["measured"]), 3)},
                "daily_r": daily_r,
                "cloudy_days": {"rule": "measured daily index < 0.7", "days": len(cloudy),
                                "original": round(corr(cloudy["original"],
                                                       cloudy["measured"]), 3),
                                "gfs_d1": round(corr(cloudy["gfs_d1"], cloudy["measured"]), 3)},
                "original_join_shifts_h": [best - 1, best - 2],
                "note": "the original pipeline joined this series 2-3 h off the true clock "
                        "(F-01, N3); relative to the best shift that is 1-2 h further",
            }
            at_joins = scan.set_index("shift_h").loc[[best - 1, best - 2], "hourly_r"]
            log.note(f"best shift {best:+d} h; daily r train+val: original "
                     f"{daily_r['train_and_val']['original']}, GFS D+1 "
                     f"{daily_r['train_and_val']['gfs_d1']}; hourly r at the original joins: "
                     f"{at_joins.round(3).tolist()}")
        path = HERE / "weather_source.json"
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
        log.outputs([path, HERE / "weather_lag_scan.csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

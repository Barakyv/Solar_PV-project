"""Write notebooks/01_data_ingest.ipynb (stage 2: raw files, tags, true UTC) and execute it.

    python -m solarpv.data.ingest
    python scripts/make_data_ingest_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-2
manifest and QA table and from the stage-3 time table, never typed.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import AUDIT_DIR, OUTPUTS_DIR, PROCESSED_DIR, REPO_ROOT

no_options("python scripts/make_data_ingest_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "01_data_ingest.ipynb"
MANIFEST = PROCESSED_DIR / "01_ingest" / "ingest_manifest.json"
QA = PROCESSED_DIR / "01_ingest" / "ingest_qa.csv"
CHECKSUMS = AUDIT_DIR / "raw_checksums.sha256"
TIME_BUG = OUTPUTS_DIR / "03_eda" / "tables" / "table_time_bug.csv"

M = json.loads(MANIFEST.read_text(encoding="utf-8"))
Q = pd.read_csv(QA).set_index("column")
TB = pd.read_csv(TIME_BUG)
orig = TB[TB["reading"].str.startswith("original")]["median_centroid_minus_solar_noon_min"]
new = TB[TB["reading"].str.startswith("rebuild")]["median_centroid_minus_solar_noon_min"]
AC_MAX = Q.loc[["s01_inv1_AC_kW", "s01_inv2_AC_kW"], "max"].max()
NEG270 = [c for c in Q.index if Q.loc[c, "min"] == -270.0]

cells = [
    md(f"""# 01 — Data ingest and the true time base (stage 2)

**Purpose.** Merge the {len(M['raw_files'])} monthly SCADA exports of the Ktora section into one
minute-level table, give every tag a canonical name, and put every timestamp on true UTC.

**Inputs.** `Raw data/*.CSV` (read-only; byte-identical to the stage-0 checksums).
**Outputs.** `data/processed/01_ingest/`: `scada_1min_utc.parquet`, `ingest_manifest.json`,
`ingest_qa.csv`, `tag_map.csv`.

**Rules.** No value is changed at this stage: anomalies are only counted, and cleaning belongs to
stage 5. The time rule fixes finding F-01 of the review: the original pipeline read Israel local
time as if it were UTC.

**Where it is documented.** `docs/data_processing_record.md`, OP-01 to OP-05; module
`solarpv.data.ingest`; tests T-05 and T-08."""),
    md("## 1. The raw files"),
    code("""import pandas as pd

from solarpv.data import ingest

files = ingest.raw_files()
print(len(files), "files:", files[0].name, "…", files[-1].name)
print("byte-identical to the stage-0 checksums:", ingest._verify_against_checksums(files))"""),
    md(f"""**What it shows.** {len(M['raw_files'])} monthly files, from {M['raw_files'][0]} to
{M['raw_files'][-1]}, each byte-identical to the checksums taken when the data were received
(`00_data_audit/raw_checksums.sha256`). Every run of the stage repeats this check
({M['raw_files_verified_against_checksums']} files verified in the last run), so a changed raw
file stops the pipeline instead of silently changing the results."""),
    code("""raw = pd.read_csv(files[1], nrows=3, dtype=str)
print(raw.shape[1], "columns")
raw.iloc[:, :6]"""),
    md(f"""**What it shows.** One raw file: a local-time column `t_stamp` and {len(M['columns'])}
SCADA tag paths, one value per minute. The stamps carry no time zone: they are Israel local time,
with daylight-saving time."""),
    md("## 2. Canonical names"),
    code("""pd.DataFrame({"raw tag": list(ingest.TAG_MAP),
              "name": list(ingest.TAG_MAP.values())})"""),
    md("""**What it shows.** The 27 tags and their names in the rebuild (PROJECT_MASTER.md §6.1).

**What it means.** `irradiance` is the in-plane (POA) sensor, not a horizontal one, and
`pyranometer` is an identical second in-plane sensor kept for quality checks (F-13, D-09). The
site has no horizontal GHI measurement, which matters for the definition of the clear-sky
reference in stage 6."""),
    md("## 3. Daylight-saving time: the repeated October hour and the missing March hour"),
    code("""def local_stamps(name):
    path = next(p for p in files if p.name == name)
    s = pd.read_csv(path, usecols=["t_stamp"], dtype=str)["t_stamp"]
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f")

october = local_stamps("2110.CSV")
repeated = october[october.duplicated(keep=False)]
print("October 2021: local stamps that occur twice:", repeated.size,
      "| from", repeated.min(), "to", repeated.max())
utc = ingest.to_utc(october)
print("after conversion: duplicates", utc.duplicated().sum(),
      "| the two passes of 01:00 local ->", utc[october == repeated.min()].tolist())

march = local_stamps("2203.CSV")
gap = pd.date_range("2022-03-25 02:00", "2022-03-25 02:59", freq="min")
print("March 2022: local stamps between 02:00 and 02:59 on 25 March:", march.isin(gap).sum())"""),
    md("""**What it shows.** When the clocks go back in October, local 01:00–01:59 occurs twice: the
file holds both passes, one after the other. When the clocks go forward in March, local 02:00–02:59
does not exist and the file has no rows for it.

**What it means.** The rebuild localises the stamps to `Asia/Jerusalem` (the repeated hour is
resolved by file order) and converts them to UTC; afterwards every minute is unique. **In the
original** (notebook 03) the local stamps were simply labelled UTC, so the whole data set was
shifted by 2 hours in winter and 3 in summer against the sun (F-01); the repeated October hour
kept two values under one label, and the March hour stayed missing (F-10)."""),
    code("""from solarpv.config.paths import PROCESSED_DIR

STAGE = PROCESSED_DIR / "01_ingest"
minutes = pd.read_parquet(STAGE / "scada_1min_utc.parquet", columns=["s01_inv1_AC_kW"])
ingest.check_time_axis(minutes.index)
print(len(minutes), "minutes |", minutes.index.min(), "->", minutes.index.max())"""),
    md(f"""**What it shows.** The merged table passes the time-axis check: {M['rows']:,} minutes,
from {M['utc_start']} to {M['utc_end']} UTC, with no duplicate and no missing minute.

**Rule used** (manifest): `{M['time_rule']}`."""),
    md("## 4. What the raw data contain (counted, not changed)"),
    code("""qa = pd.read_csv(STAGE / "ingest_qa.csv")
qa[["column", "min", "p50", "max", "n_negative", "n_above_inverter_flag"]].round(1)"""),
    md(f"""**What it shows.** Per tag: minimum, median, maximum, and counts of negative values
and of inverter values above {M['anomalies']['inverter_ac_flag_kw']:.0f} kW (110 % of one inverter's
share of the 3,449 kW AC capacity).

**What it means.** The raw data hold impossible values that the original pipeline left in the
target and the features (F-14):
- inverter AC power up to {AC_MAX:,.0f} kW ({M['anomalies']['inverter_ac_above_flag']} minutes
  above the flag);
- the fault code −270 °C in {', '.join(NEG270)};
- wind speeds up to {Q.loc['wind_speed', 'max']:.0f} m/s.

They are removed in stage 5 by documented rules, never here."""),
    md("## 5. Evidence that the time base is right"),
    code("""pd.read_csv("../outputs/03_eda/tables/table_time_bug.csv")"""),
    md(f"""**What it shows.** On clear days, the centre of the production curve should coincide with
solar noon. Read the original way (local stamps as UTC), it falls {orig.min():.0f}–{orig.max():.0f}
minutes after solar noon (winter and summer). In the rebuild, after the true-UTC conversion and
the logger-clock correction of stage 4, it falls {new.min():.1f}–{new.max():.1f} minutes after.

**What it means.** Production and solar geometry are in phase. Every solar-position, clear-sky
and irradiance-based feature of the later stages depends on this.

**Next stage.** Stage 4 (`03_preprocessing.ipynb`) corrects the drift of the logger clocks and
the 2025 change of data source, and aggregates to hours.

**Reproduce.** `python -m solarpv.data.ingest`, then
`python scripts/make_data_ingest_notebook.py`."""),
]

write_notebook(NB, cells, [MANIFEST, QA, CHECKSUMS, TIME_BUG])

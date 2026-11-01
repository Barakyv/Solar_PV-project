"""Write notebooks/03_preprocessing.ipynb (stage 4: clocks, data sources, hours) and execute it.

    python scripts/estimate_clock_drift.py
    python -m solarpv.data.preprocess
    python scripts/make_preprocessing_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-4
manifest and from the clock and polling audits, never typed.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, PROCESSED_DIR, REPO_ROOT

NB = REPO_ROOT / "notebooks" / "03_preprocessing.ipynb"
MANIFEST = PROCESSED_DIR / "04_preprocess" / "preprocess_manifest.json"
CLOCK = AUDIT_DIR / "clock_drift_2026-10-14" / "clock_fit_summary.json"
MONTHLY = AUDIT_DIR / "clock_drift_2026-10-14" / "monthly_minute_before_after.csv"
OFFSETS = CONFIGS_DIR / "clock_offsets.csv"
POLLED = AUDIT_DIR / "polled_reconstruction_2026-10-15" / "summary.csv"

M = json.loads(MANIFEST.read_text(encoding="utf-8"))
C = json.loads(CLOCK.read_text(encoding="utf-8"))
MON = pd.read_csv(MONTHLY)
OFF = pd.read_csv(OFFSETS)
P = pd.read_csv(POLLED).set_index(["days", "method"])
worst = OFF[["minute_meteo", "minute_inverter"]].min().min()
after = C["validation_max_abs_monthly_after"]["minute"]
lost = M["minutes_lost_by_clock_correction"]
site = M["site_total_AC_kW"]

cells = [
    md(f"""# 03 — Preprocessing: logger clocks, data sources and hourly means (stage 4)

**Purpose.** Put every tag of every minute on the true time, reconstruct the data of the second
source that appears in 2025, and aggregate to hourly means.

**Inputs.** The stage-2 minute table (true UTC), `configs/clock_offsets.csv` (estimated by
`scripts/estimate_clock_drift.py`). **Outputs.** `data/processed/04_preprocess/`:
`scada_1min_true_utc.parquet` ({M['rows_1min']:,} minutes) and `scada_1H_utc.parquet`
({M['rows_1H']:,} hours), with `preprocess_manifest.json`.

**Why the stage exists.** Two problems the original pipeline never saw:
- the two data loggers' clocks drift by minutes and are reset at different dates (F-15);
- from {M['polled_source']['first_day']} the export switches to a second source that holds a
  15-minute snapshot for 15 minutes (F-18).

Both shift production against the sun, and therefore against every solar feature.

**Where it is documented.** `docs/data_processing_record.md`, OP-08 to OP-17; decisions D-19,
D-20 and D-21; module `solarpv.data.preprocess` (with `clock` and `resolution`)."""),
    md("## 1. The clocks drift"),
    code("""import json

import pandas as pd
from IPython.display import Image, display

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, PROCESSED_DIR

offsets = pd.read_csv(CONFIGS_DIR / "clock_offsets.csv", parse_dates=["day"])
ax = offsets.set_index("day")[["minute_meteo", "minute_inverter"]].plot(
    figsize=(10, 3.5), title="Estimated clock offset (stamped − true), minutes")
ax.axhline(0, color="grey", lw=0.5)
offsets[["minute_meteo", "minute_inverter"]].describe().round(2)"""),
    md(f"""**What it shows.** The daily offset of each logger's clock, stamped time minus true time:
the meteorological logger (irradiance, temperatures) and the inverter logger (power). The largest
offset is {worst:.1f} minutes. The steps are resynchronisations:
{', '.join(s['day'] for s in C['steps'])}.

**How it was measured (OP-09).** The meteo clock from the fit of the in-plane sensor to the
clear-sky irradiance on clear days; the inverter clock relative to the meteo clock from the
cross-correlation of minute-to-minute changes of DC power and irradiance. A 31-day rolling median
within the segments between steps."""),
    code("""monthly = pd.read_csv(AUDIT_DIR / "clock_drift_2026-10-14"
                      / "monthly_minute_before_after.csv")
monthly[["day", "tau_met", "tau_met_after", "delta", "delta_after"]].tail(12)"""),
    md(f"""**What it shows.** Per month, the residual offset before and after the correction
(`tau_met`: meteo clock against the sun; `delta`: inverter clock against meteo clock).

**What it means.** After the correction no monthly median exceeds {after['tau_met_after']}
minutes for the meteo clock and {after['delta_after']} minutes between the loggers; the absolute
uncertainty is about ±2 minutes. Correcting the clocks costs a few minutes at the edges of the
data ({min(lost.values())}–{max(lost.values())} minutes per tag), shifted out of the range."""),
    md("## 2. The second data source of 2025"),
    code("""display(Image("../outputs/03_eda/figures/fig04_polled.png", width=900))"""),
    md(f"""**What it shows.** From {M['polled_source']['first_day']} (era from
{M['polled_source']['era_start']}), most minutes repeat the previous value: each tag is a snapshot
taken every 15 minutes on its own phase and held until the next one.

**What it means.** An hourly mean of held snapshots lags by about 7.5 minutes and is built from
four samples. The rebuild detects the source of every tag and hour (OP-10), puts each sample
back at its poll instant on its own clock, and interpolates between samples (D-20)."""),
    code("""pd.read_csv(AUDIT_DIR / "polled_reconstruction_2026-10-15" / "summary.csv")"""),
    md(f"""**What it shows.** The reconstruction methods were compared by simulating the 2025
polling on train data, where the true minutes are known: hold the snapshot (M0), linear
interpolation (M1), interpolation of the clear-sky index (M2) and a fused source (M3).

**What it means.** On all simulated hours linear interpolation has an hourly error of
{P.loc[('all', 'M1 linear'), 'rmse_pct_capacity']} % of capacity against
{P.loc[('all', 'M0 hold'), 'rmse_pct_capacity']} % for holding the snapshot. M2 is no better.
**Decision (D-21):** M1, chosen on train before any model result. The test months that come
from this source are also reported separately (stage 8)."""),
    md("## 3. Hourly means"),
    code("""hourly = pd.read_parquet(PROCESSED_DIR / "04_preprocess" / "scada_1H_utc.parquet",
                         columns=["site_total_AC_kW", "irradiance"])
manifest = json.loads((PROCESSED_DIR / "04_preprocess" / "preprocess_manifest.json")
                      .read_text(encoding="utf-8"))
print(len(hourly), "hours | rule:", manifest["hourly_rule"])
hourly.describe().round(1)"""),
    md(f"""**What it shows.** {M['rows_1H']:,} hourly rows, labelled at the start of the hour (row t
covers t to t+1 h), valid when at least 45 of the 60 minutes are present. No hour of the target
or of the main sensors is missing ({sum(M['hourly_nan'].values())} empty hours).

**What it means.** The site total peaks at {site['max_hourly']:,} kW in one hour, above its 99.99th
percentile ({site['p99_99_hourly']:,} kW). Such hours still contain logger spikes; they are
removed in stage 5 by documented rules, not here.

**Next stage.** Stage 5 (`04_cleaning.ipynb`) applies the cleaning rules and the operation flags.

**Reproduce.** `python scripts/estimate_clock_drift.py`, `python -m solarpv.data.preprocess`, then
`python scripts/make_preprocessing_notebook.py`."""),
]

write_notebook(NB, cells, [MANIFEST, CLOCK, MONTHLY, OFFSETS, POLLED])

"""Write notebooks/02_eda.ipynb (the stage-3 narrative) and execute it.

    python -m solarpv.eda
    python scripts/make_eda_notebook.py
"""

import nbformat
import pandas as pd
from nbclient import NotebookClient

from solarpv.config.paths import OUTPUTS_DIR, REPO_ROOT

NB = REPO_ROOT / "notebooks" / "02_eda.ipynb"
FIG = "../outputs/03_eda/figures"
# numbers quoted in the text are read from the stage-3 tables, never typed
SPLITS = pd.read_csv(OUTPUTS_DIR / "03_eda" / "tables" / "table_splits.csv").set_index("split")
TEST = SPLITS.loc["test"]

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("""# 02 — Exploratory data analysis (stage 3)

**Purpose.** Describe the data the corrected models are trained on, show every correction of the
rebuild with evidence, and give the descriptive basis for the stage-6 decisions. The submitted
report had no EDA chapter and the original EDA notebook did not survive.

**Inputs.** Stage-5 output (clock-corrected, source-harmonised and cleaned data), the raw files
(for the time-zone figure) and the stage-4 / clock audit outputs. Figures and tables are produced
by `python -m solarpv.eda`; this notebook reads them and interprets them.

**Rules.** EDA changes no data. Anything that could inform a modelling choice is computed on the
**train split only** and labelled so; whole-period overviews mark validation and test.
Plan: `docs/stage3_eda_plan.md`. Every operation is listed in `docs/data_processing_record.md`."""),
    code("""import json

import pandas as pd

from solarpv.config.paths import OUTPUTS_DIR

EDA = OUTPUTS_DIR / "03_eda"
manifest = json.loads((EDA / "eda_manifest.json").read_text(encoding="utf-8"))
tables = {p.stem: pd.read_csv(p) for p in sorted((EDA / "tables").glob("*.csv"))}
print("produced by commit", manifest["git_commit"])"""),
    md(f"""## 1. The data at a glance

![overview]({FIG}/fig01_overview.png)

Two of the 25 inverters of the 40 MW Ketura Solar park (3,449 kW AC), one minute resolution,
September 2021 to May 2025, aggregated to 32,856 hourly rows. Daily energy follows the seasons
(about 20 MWh on clear summer days, 12–15 MWh in winter) with cloudy-day dips concentrated in
winter and spring. The 30-day maximum hourly power stays between about 2,500 and 3,000 kW for the
whole period, with no step that would indicate a change of equipment or capacity (assumption
A-16). The hatched area is the second data source of 2025 (F-18); dashed lines mark the two
logger-clock resyncs (F-15)."""),
    code("""tables["table_splits"]"""),
    md(f"""The split is temporal and defined in UTC. Exact counts replace the approximate values of
the submitted report (train ~21,049 h there; {SPLITS.loc["train", "hours"]:,} h here). The test
period has {TEST["daytime_hours"]:,} daytime hours (sun above 5° at mid-hour, F-02), of which
{TEST["daytime_op_limited"]} are op-limited (inverter stop, curtailment or derated inverter, D-22)
and {TEST["daytime_polled"]:,} come from the polled source (D-21). Validation and training contain
no polled data, so the second source never enters training."""),
    md(f"""## 2. Time base: the original read local time as UTC (F-01)

![time bug]({FIG}/fig02_time_bug.png)"""),
    code("""tables["table_time_bug"]"""),
    md("""On six clear days of a winter and a summer training month, the power-weighted centre of
production lies **+120 min** (winter, UTC+2) and **+174 min** (summer, UTC+3) after solar noon
when the raw stamps are read as UTC, as the original pipeline did. With the rebuild's true UTC
and clock correction it lies **+1.9 and +2.6 min** after solar noon. The summer value is not
exactly +180 min because the logger clock was already about 9 min slow then (section 3). In the
original, solar geometry and clear-sky features were therefore 2–3 h out of phase with the
target."""),
    md(f"""## 3. Logger clocks (F-15, D-19)

![clocks]({FIG}/fig03_clock.png)

The meteo and inverter loggers drifted up to 16 minutes slow, mostly in the warm months, and were
resynchronised independently: on 2023-05-09 only the meteo clock was reset; on 2024-01-21 it was
set to the (drifted) inverter clock. The 2025 data source has its own clock, about 3 minutes slow.
After the correction, every monthly residual lies within ±2 minutes (lower panel). The flat
segment of the minute-source offsets after February 2025 is the last estimate carried forward:
only a few hours still come from that source, and a direct measurement on 2025-04-29 suggests
about 1.5 minutes more drift (a stated limitation)."""),
    md(f"""## 4. The second data source of 2025 (F-18)

![polled]({FIG}/fig04_polled.png)

From 2025-02-26 each tag is an instantaneous snapshot taken every 15 minutes, on its own phase,
and repeated until the next one. Held values lag by half a poll period and make an hourly mean a
mean of four stale samples. The rebuild puts each snapshot back at its poll instant and
interpolates linearly. A simulation that applies the same sampling to real minute data of the
training period gives an hourly error of 2.6 % of capacity for the held values and 1.6 % for the
rebuild (0.9 % on clear days); fusing the six tags naively was worse and was rejected."""),
    md(f"""## 5. Cleaning (stage 5, D-22)

![cleaning]({FIG}/fig05_cleaning.png)

Value rules replace wrong measurements; flags mark real plant events without changing values.
The communication freeze of 11–12 September 2021 repeated the last values for hours, including
about 1,300 kW in the middle of the night; the frozen run is removed. This figure also exposed
a 247 kW blend minute at sunrise when the data resumed, which led to a margin around frozen
runs (R3). Daily energy is unchanged on all days except the few with spikes or freezes
(right panel)."""),
    md(f"""## 6. Which irradiance sensor is right? (F-19)

![sensors]({FIG}/fig10_sensors.png)

The site has two in-plane irradiance sensors. Against modelled clear-sky irradiance on the panel
plane, the sensor used by the models (`irradiance`, meteo station st01) is stable in every month,
while the redundant pyranometer drops to exactly ×0.75 from 2023-05-28 to 2023-12-31 and on days
between December 2024 and April 2025 — a calibration-constant fault. The pyranometer is not a
model input; its faulty days are removed by rule R7. This confirms decision D-09.

At the clearest hours the in-plane sensor reads about 10–20 % above the Ineichen clear-sky model
with default turbidity; the scale of the clear-sky denominator is examined in stage 6 (D-11)."""),
    code("""tables["table_sensors"].head(24)"""),
    md(f"""## 7. Production patterns (train only)

![diurnal]({FIG}/fig06_diurnal.png)

Mean production by local hour and month: a long, flat summer plateau and a short winter day.
Production starts around 06:00–07:00 local time all year, as expected for a fixed, south-facing
array at 30° N.

![power vs irradiance]({FIG}/fig07_power_irradiance.png)"""),
    code("""tables["table_physics"]"""),
    md("""Hourly AC power is close to linear in the in-plane irradiance; at high irradiance the
spread is explained by module temperature. On normal-operation hours with 700–900 W/m², AC per unit
irradiance falls by **0.42 % per kelvin** of module temperature, within the usual range for
crystalline silicon (about −0.3 to −0.5 %/K). Temperature is therefore a physically meaningful
input, and clear-sky irradiance alone cannot explain summer noon output.

![variability]({FIG}/fig08_variability.png)"""),
    code("""tables["table_variability"]"""),
    md("""The daily clear-sky index (daily AC energy over a clear-sky reference) is high and narrow
in summer (median about 0.77, few cloudy days) and wide in winter and spring, where most
forecasting error will come from. Day-to-day persistence is modest: the autocorrelation of the
daily index is 0.33 at one day and near zero from two days on. Yesterday's production carries
some information about tomorrow, older days carry little — consistent with the small weight of
long lags and with the value of weather forecasts (stage 6b)."""),
    md(f"""## 8. Operational events

![operations]({FIG}/fig09_operations.png)

Inverter stops, grid curtailment and derated inverters (one inverter running at about 75 % of
the other for hours or days; flag F3, found in stage 3b) affect from a few up to about 90 daytime
hours per month. Curtailment is frequent in spring 2024 (validation) and autumn 2024 (test);
derates cluster in 2023. These hours stay in the data with their flags: they are excluded from the
training loss and from past-day features, included in the main evaluation and excluded in a
secondary one (D-22). Missing-target hours come from communication freezes. The data-quality
notebook (`02b_data_quality.ipynb`) gives the evidence for every flag and threshold."""),
    md("""## 9. What this means for the next stages

- All geometry is now in phase with production (section 2); the clear-sky and solar-position
  features of stage 6 can be trusted.
- The irradiance sensor is reliable over the whole period (section 6); its lags are valid inputs.
- Module temperature has a clear physical effect (section 7) — keep its lags.
- Persistence beyond one day is weak (section 7) — horizon-aware rolling features (D-04) and NWP
  (stage 6b) matter more than long lags.
- The clear-sky reference scale and the denominator of k (GHI vs POA, D-11) are decided in stage
  6 on train only.
- The 2025 test months need the per-source reporting of D-21.

**Reproduce:** `python -m solarpv.eda` (figures, tables, manifest), then run this notebook."""),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
NotebookClient(nb, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": str(NB.parent)}}).execute()
NB.write_bytes(nbformat.writes(nb).encode("utf-8"))
print("wrote", NB)

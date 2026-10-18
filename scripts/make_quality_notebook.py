"""Write notebooks/02b_data_quality.ipynb (stage 3b) and execute it.

    python -m solarpv.eda.quality          # tables, figures and the manifest with the facts
    python scripts/make_quality_notebook.py

Every code cell runs one analysis of solarpv.eda.quality on the data and is followed by a
markdown note: what the cell shows, what it means, and which cleaning rule or decision it
supports. Numbers in the notes are formatted from outputs/03b_quality/quality_manifest.json
(produced by the same functions), never typed; the notebook records the manifest's hash and
T-31 checks that it is current. Plan: docs/stage3b_quality_eda_plan.md.
"""

import json

import nbformat
from nbclient import NotebookClient

from solarpv.config.paths import REPO_ROOT
from solarpv.config.provenance import text_sha256
from solarpv.eda.quality import MANIFEST

NB = REPO_ROOT / "notebooks" / "02b_data_quality.ipynb"
M = json.loads(MANIFEST.read_text(encoding="utf-8"))
F = {name.split("_")[0]: entry["facts"] for name, entry in M["analyses"].items()}
NAMES = {name.split("_")[0]: name for name in M["analyses"]}

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f} %"


def run(qid: str, comment: str) -> nbformat.NotebookNode:
    return code(f"r = Q.{NAMES[qid]}(d)  # {comment}\nr.show()")


def analysis(qid: str, title: str, intro: str, comment: str, note: str) -> list:
    """Heading and question, the code cell, and the note under it."""
    return [md(f"### {qid.upper()} — {title}\n\n{intro}"), run(qid, comment), md(note)]


a, b = F["q01"], F["q02"]
cells = [
    md("""# 02b — Data quality before and after cleaning (stage 3b)

**Purpose.** A full examination of the quality of all 27 tags before and after cleaning, with the
evidence behind every cleaning threshold. Stage 3 (`02_eda.ipynb`) covered the corrections of the
rebuild (time base, clocks, the 2025 data source) and the target; this notebook covers the data
themselves: what is wrong in them, why each rule of `configs/cleaning_policy.yaml` is set where it
is, what cleaning changed, and what the cleaned data imply for modelling.

**Before and after.** "Before cleaning" is the stage-4 output: clock-corrected, true-UTC minutes
with the 2025 polled data rebuilt — exactly what the cleaning rules see. "After cleaning" is the
stage-5 output. The raw export (stage 2) is used where the export format itself is the subject.

**Rules.** EDA changes no data. Every threshold is judged on the **train split only**
(UTC ≤ 2024-01-31); analyses over the whole period are descriptive and say so; the split
comparison (Q31) is used for no decision. Each code cell runs one analysis of
`solarpv.eda.quality` and is followed by a note: *what the cell shows*, *what it means*, and
*the rule or decision it supports*. The numbers in the notes come from the manifest written by the
same functions (`python -m solarpv.eda.quality`).

**Findings fed back into cleaning.** This analysis found three problems, which were fixed in
stage 5 before this notebook was run (`docs/stage3b_quality_eda_plan.md`, PROJECT_MASTER.md):
F-20, a drifting curtailment setpoint that did not limit output (Q15); F-21, pyranometer faults
within a day that a daily check missed (Q13); and F-22, partial inverter outages that no flag
marked (Q16, new flag F3).

**Structure.** Part A — the export as delivered (Q01–Q04). Part B — the evidence for each cleaning
threshold (Q05–Q17). Part C — what cleaning changed (Q18–Q22). Part D — the cleaned data for
modelling (Q23–Q31). A summary table closes the notebook."""),
    code("""import pandas as pd

from solarpv.eda import quality as Q

pd.set_option("display.max_columns", 40)
d = Q.load()  # stage-4 and stage-5 minutes and hours, sun position, clear-sky POA, policy
print(f"stage 4 minutes: {d.s4.shape}, stage 5 minutes: {d.s5.shape}, hours: {d.h5.shape}")
print(f"train minutes: {int(d.train.sum()):,}; daytime minutes (sun > 10 deg): "
      f"{int(d.daytime.sum()):,}")"""),
    md("""**Loaded.** The minute data before (stage 4) and after (stage 5) cleaning on the full
true-UTC grid, the hourly tables, the sun elevation and clear-sky in-plane irradiance of every
minute (pvlib, measured orientation 20°/180°), the split of every minute and the cleaning policy.
"Daytime" in Parts A–C is the policy's definition (sun above 10° at the minute); the hourly
analyses use the evaluation mask (sun above 5° at mid-hour, F-02)."""),
    md("## Part A — The export as delivered"),
    *analysis(
        "q01", "Inventory of the 27 tags",
        "**Question.** What does each tag measure, what is its role in the project, and how "
        "complete is it? **Data.** Raw export (stage 2) for the value statistics; NaN counts in "
        "stages 2, 4 and 5.",
        "units, roles, ranges and gaps of every tag",
        f"""**What it shows.** The export has {a['n_tags']} tags on a complete one-minute grid
({a['minutes']:,} minutes = {a['hours']:,} hours; every stamp unique and one minute apart).
**Not one value is missing in the raw export** ({a['raw_nan_total']} NaN in all columns).

**What it means.** When a logger loses communication, the historian repeats the last value, so
missing data are disguised as constant values. NaN counts say nothing about quality here; the
frozen-value rule R3 (Q08–Q10) is what finds the gaps. Stage 4 adds at most
{a['stage4_nan_max']} NaN per column (minutes shifted out by the clock correction, at night or at
a change of data source). Coded tags are averaged within the minute, so a change of code shows
as a non-integer value (e.g. {pct(a['state_non_integer_share'], 2)} of the inverter-status
minutes); this matters for F2 (Q15) and F3 (Q16).

**Roles.** {a['roles']['target']} target columns (the site total and the two inverters),
{a['roles']['model input']} model inputs (only as past-day lags: irradiance, two module and one
ambient temperature), {a['roles']['QA only']} QA-only columns, {a['roles']['flag']} flag sources
(setpoint and the two status tags) and {a['roles']['not used']} unused tags (reasons in Q30)."""),
    *analysis(
        "q02", "Held values over time",
        "**Question.** Where and when do values stop changing? **Data.** Stage 4, all periods "
        "(descriptive). A minute is *held* if it is part of a run of one value lasting at least "
        "30 minutes (power and reactive power: non-zero runs only; exact zeros are stops).",
        "share of daytime minutes held >= 30 min, per tag and month",
        f"""**What it shows.** For the main sensors held values are rare
({b['held_pct_train']['irradiance']:.2f} % of train daytime minutes for irradiance,
{b['held_pct_train']['s01_inv1_AC_kW']:.2f} % for inverter AC) and they appear as vertical
stripes: the same months in all tags at once — communication freezes (Q09). Three tags are dead
or partly dead: precipitation is held in all {b['dead_months']['precipitation'][2]} months (it
never reports rain, Q30), soil is held from {b['dead_months']['soil'][0]}
({b['dead_months']['soil'][2]} months) and soiling01 from {b['dead_months']['soiling01'][0]}.
Wind speed has its own stuck periods ({b['held_pct_all']['wind_speed']:.1f} % of daytime
minutes, worst in the first two months).

**Consequence.** Freezes of the main sensors are removed by R3; the dead tags are not used
(Q30) rather than repaired."""),
]
c = F["q03"]
cells += analysis(
    "q03", "Two sampling regimes (stage-4 source detection)",
    "**Question.** Can stage 4 tell, for every tag and hour, a one-minute average from a held "
    "15-minute snapshot (F-18, D-21)? **Data.** Raw export, all periods: this is the detection "
    "of the export format, not a modelling choice.",
    "detection statistics per tag-hour, before and after the switch",
    f"""**What it shows.** Before {c['first_day']} only {pct(c['polled_share_before'], 2)} of
tag-hours look polled (freezes and partial freezes; stage 4's era logic keeps them on the minute
source); from {c['first_day']} {pct(c['polled_share_after'])} are polled. Only
{pct(c['ambiguous_share'])} of the {c['tag_hours']:,} classified tag-hours have a poll share
between 0.3 and 0.7.

**What it means.** The two regimes are separated by a wide, almost empty gap, so the detection
thresholds of stage 4 (0.5 for the share of minutes in 10–16-minute runs, 0.25 changes per
minute) are not critical: any value inside the gap gives the same classification.""")
c = F["q04"]["outside"]
cells += analysis(
    "q04", "Value distributions before cleaning",
    "**Question.** What do the distributions of the ranged columns look like, and where are "
    "the R1 limits? **Data.** Stage 4, all periods (descriptive; the limits are judged on train "
    "in Q05).",
    "histograms with the R1 limits",
    f"""**What it shows.** The body of every distribution is physically plausible; the values
outside the limits form separate, isolated groups: the temperature fault code −270 °C
({c['ambient_temp']:,} minutes in ambient_temp, {c['mod_temp1']:,} and {c['mod_temp2']:,} in the
module sensors), power spikes up to 40,825 kW ({c['s01_inv1_AC_kW']}–{c['s01_inv1_DC_kW']}
minutes per power column) and wind speeds up to 629 m/s ({c['wind_speed']} minutes). Irradiance,
pyranometer and site_ambient_temp never leave their limits.

**Note.** A few isolated counts between −20 and 0 °C in the module sensors are one-minute blends
of a fault code with a real value: they are inside the limits, which is why R1 has a margin
(Q06).""")
cells += [md("""## Part B — The evidence for each cleaning threshold

Each analysis below takes one rule of `configs/cleaning_policy.yaml`, looks at the stage-4 data of
the **train split**, and asks whether the threshold sits where the data say it should. Value rules
R1–R7 replace a wrong measurement (NaN or 0); flags F1–F3 mark a real plant event and never change
a value (D-22).""")]
c = F["q05"]
pw_in, pw_out = c["power_largest_inside_over_upper"], c["power_smallest_above_over_upper"]
t_in, t_low = c["temperature_smallest_inside_not_next"], c["temperature_largest_below"]
cells += analysis(
    "q05", "R1 — are the physical limits in empty regions?",
    "**Question.** Is there a gap between normal values and faulty values, and does the limit "
    "sit in it? **Data.** Stage 4, train.",
    "normal range vs values outside the limits, per column (train)",
    f"""**What it shows.** For power, the largest normal value is
{min(pw_in.values()):.2f}–{max(pw_in.values()):.2f} of the upper limit (1,897 kW = 1.1 × the
1,724.5 kW rating), and the smallest value above the limit is
{min(v for v in pw_out.values() if v):.1f}–{max(v for v in pw_out.values() if v):.1f} times the
limit. For temperatures, the coldest normal readings are {t_in['mod_temp2']} °C (module, radiative
cooling on clear winter nights) and {t_in['ambient_temp']} °C (ambient) — far above −20 °C — while
the faults lie at −270 °C and blends down to {t_low['mod_temp1']} °C. Irradiance never exceeds
{c['irradiance_largest_inside']:,.0f} W/m² (cloud enhancement in the panel plane), below the
1,600 limit.

**What it means.** The limits could move a long way without changing a single decision: they are
not tuned. Two caveats: a blended minute next to a fault can lie inside the limits
({c['temperature_smallest_inside']['mod_temp2']} °C in mod_temp2), which the margin handles (Q06);
and wind speed has no clean gap (values up to {c['wind_largest_inside']} m/s inside the limit are
themselves spikes), which does not matter because wind is not used (Q30).""")
c = F["q06"]
cells += analysis(
    "q06", "R1 margin — are the neighbours of a faulty minute contaminated?",
    "**Question.** Should the minutes next to an out-of-range value be removed too? **Data.** "
    "Stage 4, train, minute source. A neighbour is contaminated if it deviates from the next "
    "three minutes by more than the 99.9th percentile of normal one-minute changes.",
    "neighbours of R1 minutes against normal one-minute changes (train)",
    f"""**What it shows.** Power spikes are single-minute transmission errors: none of their
neighbours is contaminated ({pct(c['share_beyond_normal']['power'], 0)}). Next to temperature
fault codes, {pct(c['share_beyond_normal']['temperatures'])} of the neighbours deviate beyond
normal: a one-minute average that contains part of the fault period is a blend (left panel: a
minute at about 12 °C between 21 °C and −270 °C).

**What it means.** The ±1-minute margin on the minute source is needed for the temperature faults
and harmless for power spikes (two extra minutes per spike). On the polled source the margin is
±15 minutes by construction: stage 4 interpolates linearly between snapshots 15 minutes apart, so
one bad snapshot contaminates up to one poll period on each side.""")
c = F["q07"]
cells += analysis(
    "q07", "R2 — negative irradiance at night",
    "**Question.** Are negative irradiance readings an offset or a fault? **Data.** Stage 4, "
    "train, by sun elevation.",
    "negative irradiance by sun elevation (train)",
    f"""**What it shows.** {pct(c['irradiance_negative_share_night'])} of night minutes are
negative, the minimum is {c['irradiance_min_train']:.2f} W/m², and no value is ever below
−20 W/m². With the sun above 10° no value is negative ({pct(c['irradiance_negative_share_sun_above_10'], 0)}).
The pyranometer never reads below {c['pyranometer_min_train']:.0f} (its logger clamps at zero).

**What it means.** The negative values are the sensor's dark offset. Setting −20…0 W/m² to 0 (R2)
removes an artefact with no physical meaning and never touches daylight; R1 never sees these
values because the offset is far from −20.""")
c = F["q08"]
r15 = [v for k, v in c["ratio_15_60"].items() if k not in ("ambient_temp",)]
cells += analysis(
    "q08", "R3 — how long does a value naturally stay constant?",
    "**Question.** Where does natural constancy end and a freeze begin? **Data.** Stage 4, "
    "train, daytime; power: non-zero runs. The curve gives the share of daytime minutes that "
    "lie in a run of at least k identical values.",
    "share of daytime minutes in runs >= k (train)",
    f"""**What it shows.** Real measurements change almost every minute. The curves fall steeply
over the first minutes and then flatten: for irradiance {c['share_ge_5']['irradiance']:.2f} % of
daytime minutes lie in runs of 5 minutes or more and {c['share_ge_30']['irradiance']:.2f} % in
runs of 30 minutes or more — the plateau is the freezes. Between 15 and 60 minutes the share
changes by a factor of only {min(r15):.2f}–{max(r15):.2f} for every R3 column except
ambient_temp.

**What it means.** The 30-minute threshold sits on the plateau, so it is not sensitive. The
meteo ambient sensor has a resolution of {c['value_step']['ambient_temp']} °C and changes slowly:
it has genuine runs of 10–30 minutes ({c['share_ge_5']['ambient_temp']:.1f} % of minutes in runs
of 5 minutes or more) and its plateau starts later (≥ 30 / ≥ 120 min =
{c['ratio_30_120']['ambient_temp']:.2f}), hence 60 minutes for the two ambient sensors. Runs of
exactly 0 kW are inverter stops (F1), not frozen sensors, and are excluded from R3.""")
c = F["q09"]
cells += analysis(
    "q09", "R3 — sensor faults or communication freezes?",
    "**Question.** Do long runs hit one sensor or the whole logger? **Data.** Stage 4; the "
    "share is computed on train, the event list covers all periods (descriptive).",
    "columns frozen at the same minute, and the freeze events",
    f"""**What it shows.** {pct(c['share_multi_column_train'])} of the frozen daytime
minute-columns in train have at least five of the ten main columns frozen at the same minute.
There are {c['events_all']} freeze events in the whole period ({c['events_by_split']['train']}
train, {c['events_by_split']['val']} val, {c['events_by_split']['test']} test),
{c['events_reaching_daytime']} of which reach daylight; the longest lasts
{c['longest_hours']:.0f} hours (2021-09-10/12).

**What it means.** Long constant runs are communication freezes: the historian repeats stale
values of every tag. R3 therefore removes copies of old values, not measurements.""")
c = F["q10"]
cells += analysis(
    "q10", "R3 margin — what happens when data resume?",
    "**Question.** Is the first minute after a freeze a real measurement? **Data.** Stage 4, "
    "minute source; resumes with a clear jump. Position 0 = the stale value, 1 = the next "
    "minute's value.",
    "position of the first minute after each freeze",
    f"""**What it shows.** In {pct(c['blend_share_train'])} of the {c['resumes_train']} train
resumes the first new value lies between the stale value and the next minute — a blend. Example:
{c['example_blend_kw']:.0f} kW at sunrise on 2021-09-12 after a night frozen at 1,296 kW (left
panel; found in stage 3).

**What it means.** The resume minute is neither the old nor the new truth, so R3 removes one
minute on each side of a frozen run on the minute source and one poll period (15 minutes) on the
polled source, where the reconstruction interpolates from the stale snapshot.""")
c, kwh = F["q11"], F["q18"]["kwh_by_rule"]
cells += analysis(
    "q11", "R4 and R5 — what is real at night and in twilight?",
    "**Question.** Which readings below the horizon are noise, stale values or production? "
    "**Data.** Stage 4, train, by sun elevation; frozen = in or within 15 minutes of a run "
    "of at least 30 minutes.",
    "inverter AC by sun elevation (train)",
    f"""**What it shows.** Far below the horizon (sun below −6°) the inverters read exactly 0
whenever the data are live ({c['deep_night_nonzero_not_frozen']['s01_inv1_AC_kW']} non-zero minutes
outside freezes): there is no night noise. Every reading above 5 kW below −6°
({c['r5_minutes_all']:,} minutes, all periods) is in or next to a frozen run
({pct(c['r5_share_in_or_next_to_freeze'], 0)}). Between −3° and 0° the plant produces a little —
up to {c['max_not_frozen_twilight']['s01_inv1_AC_kW']:.0f} kW (diffuse light at dusk and dawn).

**What it means.** R5 (power above 5 kW with the sun below −6° → NaN) removes only stale values.
R4 (|power| ≤ 5 kW below the horizon → 0) was described as removing night noise, but the data show
none: {pct(c['r4_candidates_in_twilight_share'], 0)} of its candidates are twilight readings
between −3° and 0°, i.e. tiny real production. Its effect is negligible
({kwh['R4_night_zero']:.0f} kWh in 3¾ years, about 0.001 % of the energy) and it falls where k is
undefined and no evaluation takes place, so R4 is kept to make the target exactly 0 when the sun
is down; its justification is corrected in the policy and the record.""")
c = F["q12"]
cells += analysis(
    "q12", "R6 — dead DC sensor, and the consistency of DC with AC",
    "**Question.** Is DC consistent with AC, and what does R6 remove? **Data.** Stage 4, train "
    "daytime for the efficiency; R6 minutes over all periods (descriptive).",
    "AC/DC efficiency and the dead-DC minutes",
    f"""**What it shows.** The median conversion efficiency is {pct(c['efficiency_median']['inv1'])}
(inverter 1) and {pct(c['efficiency_median']['inv2'])} (inverter 2). AC above DC — physically
impossible — appears in {pct(c['share_ac_above_dc']['inv1'])} of minutes, but in only
{pct(c['share_ac_above_dc_stable']['inv1'], 2)} of steady minutes: it is a timing effect of two
one-minute averages during fast ramps, not a sensor fault. R6 (DC = 0 while AC > 100 kW) hits
{c['r6_minutes']['inv1']:,} minutes of inverter 1, {pct(c['r6_share_top_month']['inv1'])} of
them in {c['r6_top_month']['inv1']}, and {c['r6_minutes']['inv2']} minutes of inverter 2.

**What it means.** One dead DC sensor in March 2023; DC needs no other rule. DC is not a model
input, so R6 matters for QA and the clock check only. The lower band in the left panel
(AC ≈ 0.78 × DC) is the derated inverter of Q16.""")
c = F["q13"]
gap_day, gap_ratio = next(iter(c["train_gap_days"].items())) if c["train_gap_days"] else ("-", 0)
cells += analysis(
    "q13", "R7 — the pyranometer's scale fault",
    "**Question.** Is the redundant pyranometer consistent with the irradiance sensor, and how "
    "should a fault be detected? **Data.** Stage 4 pyranometer against the cleaned irradiance; "
    "bright minutes only (sun > 30°, irradiance > 300 W/m²); train for the thresholds.",
    "daily and hourly ratio pyranometer / irradiance",
    f"""**What it shows.** Per local day the ratio forms two populations in train:
{c['train_good_median']:.3f} (inter-quartile range {c['train_good_iqr'][0]:.3f}–{c['train_good_iqr'][1]:.3f})
and {c['train_faulty_median']:.3f} — a calibration constant switched to exactly ×0.75 on
{c['train_faulty_days']} train days (and {c['faulty_days_by_split']['test']} test days). The band
0.9–1.1 lies in the empty space between them. The only train day inside the gap, {gap_day}
(ratio {gap_ratio:.2f}), is a day on which the fault switched on and off (third panel: ×0.75
from 07:00 to 09:00 UTC, ×1.00 afterwards).

**What it means.** A daily median misses faults that cover part of a day, so R7 now also judges
every UTC hour (finding F-21); in train this adds {c['train_extra_hours']} hours
({c['train_extra_hours_near_075']} of them at ≈ 0.75, the others hours where the two sensors
disagree by more than 10 % for other reasons). A period is judged only with enough usable minutes
(at least 20): a freeze day had been decided on 2 minutes. The sun must be above
30° because at low sun the two sensors differ by a few percent through their angular (cosine)
response (fourth panel). The pyranometer is not a model input; the irradiance sensor that the
models use was verified against clear-sky irradiance in stage 3 (F-19).""")
c = F["q14"]
cells += analysis(
    "q14", "F1 — what does an inverter stop look like?",
    "**Question.** Is 20 kW the right line between a stop and low production? **Data.** Train, "
    "after the value rules; sun above 5° and irradiance above 150 W/m².",
    "inverter AC when the sun is up and bright (train)",
    f"""**What it shows.** Even at the lowest irradiance of the rule (150–200 W/m²) an inverter
normally produces about {c['median_at_150_200']['s01_inv1_AC_kW']:.0f} kW (1st percentile
{c['p1_at_150_200']['s01_inv1_AC_kW']:.0f} kW). Stops form a separate mode below 20 kW
({pct(c['share_below_20']['s01_inv1_AC_kW'], 2)} and {pct(c['share_below_20']['s01_inv2_AC_kW'], 2)}
of these minutes), and the region between 20 and 100 kW is almost empty
({pct(c['share_20_to_100']['s01_inv1_AC_kW'], 3)}).

**What it means.** The 20 kW threshold sits in a valley far below normal output; F1 marks stops
without changing any value (D-22).""")
c = F["q15"]
rel = c["relative_output_by_setpoint_train"]
drift = c["relative_output_drift_days"]
removed = c["minutes_below_100_by_split"]["train"] - c["flagged_minutes_by_split"]["train"]
cells += analysis(
    "q15", "F2 — does the setpoint really limit output?",
    "**Question.** Does a park setpoint below 100 % limit our two inverters, and does `iec_mode`? "
    "**Data.** Stage 4, train. Relative output = AC per unit irradiance over its median in the "
    "same calendar month in normal operation (1.0 = normal).",
    "output under each setpoint, setpoint runs, and iec_mode (train)",
    f"""**What it shows.** Under a commanded step the output falls to {rel['0']:.2f} (setpoint
0 %), {rel['25']:.2f} (25 %), {rel['50']:.2f} (50 %) and {rel['75']:.2f} (75 %) of normal —
roughly the setpoint plus ten points (the setpoint applies to the whole 40 MW park, not exactly
to our segment). On 2022-02-09…12, however, the setpoint drifted through non-integer values from
98 % to 28 % for {c['train_drift_runs'][0]:,} minutes while output stayed normal
({min(drift.values()):.2f}–{max(drift.values()):.2f}). A real step shows at most
{c['train_transition_run_max']} non-integer minutes (the minute that contains the step is a
blend).

**What it means.** The drift was not a command. F2 now ignores runs of non-integer setpoint values
longer than 10 minutes (finding F-20): this removes {removed:,} wrongly flagged train minutes and
leaves validation and test unchanged. `iec_mode` does not change output (modes 0/1/2:
{c['mode_relative_output']['0']:.3f}/{c['mode_relative_output']['1']:.3f}/{c['mode_relative_output']['2']:.3f}),
which confirms F-17: it is not used.""")
c = F["q16"]
rs = c["ratio_by_state"]
cells += analysis(
    "q16", "F3 — a derated inverter (new flag)",
    "**Question.** Are there periods in which one inverter produces only part of its normal "
    "output, unseen by F1? **Data.** Stage 5 AC (after the value rules), train; the ratio "
    "AC1/AC2 in a centred 31-minute window, with the sun above 15° and both inverters above "
    "100 kW.",
    "windowed inverter ratio, its stability, and the status codes (train)",
    f"""**What it shows.** The windowed ratio has its main mode at {c['modes']['normal']:.2f}
and secondary modes at {c['modes']['inv1_low']:.2f} and {c['modes']['inv2_low']:.2f} (inverter 1
or inverter 2 derated, for hours or whole days); only
{pct(c['train_valley_share'], 2)} of the windows lie in the valleys where the thresholds 0.85
and 1/0.85 sit. A derate is a stable fraction: {pct(c['train_low_stable_share'], 0)} of the
low windows have an inter-quartile range of the log ratio below 0.01, while clouds over one field
and the 2025 polling phases give 0.1–0.5 ({pct(c['train_low_unstable_share'], 0)} of low windows,
not flagged). The status codes agree: in code 1 the ratio is {rs['inv1 code 1 all']:.2f} /
{rs['inv2 code 1 all']:.2f}; in codes 5 and 6 sustained for 30 minutes or more it is
{rs['inv1 code 5 run >= 30 min']:.2f} / {rs['inv2 code 5 run >= 30 min']:.2f}. Short start-up
episodes of code 5 are not flagged; code 3 is ambiguous (derated on inverter 2,
{rs['inv2 code 3 all']:.2f}; normal on inverter 1, {rs['inv1 code 3 all']:.2f}), so it is left
to the ratio rule.

**What it means.** A partial outage is an operational limitation like a stop or curtailment, so it
gets a flag (finding F-22, flag F3; values unchanged, D-22): {c['hours_by_split']['train']} train,
{c['hours_by_split']['val']} validation and {c['hours_by_split']['test']} test daytime hours, with
about {c['lost_mwh_by_split']['train']:.0f} / {c['lost_mwh_by_split']['val']:.0f} /
{c['lost_mwh_by_split']['test']:.0f} MWh not produced. Without it, the models would learn these
hours as weather.""")
c = F["q17"]
op = c["op_hours_at"]
cells += analysis(
    "q17", "Hourly thresholds — coverage 45/60 and op-limited share 0.05",
    "**Question.** Are the two hourly thresholds sensitive? **Data.** Train daytime hours "
    "(evaluation mask).",
    "valid minutes per hour and flagged share per hour (train)",
    f"""**What it shows.** Hours are either complete or empty: of {c['daytime_train_hours']:,}
train daytime hours, {c['hours_partial_coverage_1_44']} have between 1 and 44 valid target
minutes and {c['hours_empty']} have none. Flagged hours are mostly fully flagged
({pct(c['flagged_hours_full_share'], 0)}), and the number of op-limited hours hardly depends on
the share threshold: {op['0.01']} at 0.01, {op['0.05']} at 0.05 and {op['0.1']} at 0.10.

**What it means.** Neither the coverage threshold (A-09) nor the op-limited share (D-22) is a
sensitive choice; any reasonable value gives the same hours.""")
c = F["q18"]
total_min = sum(c["minutes_by_split"].values())
cells += [md("""## Part C — What cleaning changed (stage 5 against stage 4)

Descriptive analyses over the whole period: what each rule changed, whether the body of the data
stayed intact, whether energy is conserved, what is missing, and whether anything is left.""")]
cells += analysis(
    "q18", "What each rule changed, where and when",
    "**Question.** How much did each rule change, in which split and in which months? **Data.** "
    "Rule codes of stage 5, all periods.",
    "minutes and energy changed by rule, column and split",
    f"""**What it shows.** The counts equal the cleaning log ({'yes' if c['matches_cleaning_log'] else 'NO'}).
In minutes, R7 dominates ({c['minutes_by_rule']['R7_scale_fault']:,} pyranometer minutes: whole
faulty days and hours), followed by R3 ({c['minutes_by_rule']['R3_frozen']:,}) and the R2 night
offset ({c['minutes_by_rule']['R2_offset']:,}); R1, R4 and R6 are small. Of the {total_min:,}
changed minute-columns, {c['minutes_by_split']['train']:,} are in train, where the 2023
pyranometer fault and most freezes fall. Each rule acts in a few months (heatmap): R6 in
{c['busiest_month_by_rule']['R6_dc_dead']}, R7 mostly in {c['busiest_month_by_rule']['R7_scale_fault'][:4]}.

**Energy.** Cleaning removed {c['kwh_by_rule']['R3_frozen'] / 1000:.0f} MWh of inverter AC under
R3 and {c['kwh_by_rule']['R1_range'] / 1000:.1f} MWh under R1 — stale values held through
freezes (e.g. 1,296 kW repeated through a night) and spikes, not production.""")
c = F["q19"]
model_cols = ("s01_inv1_AC_kW", "s01_inv2_AC_kW", "irradiance", "mod_temp1", "mod_temp2",
              "ambient_temp")
cells += analysis(
    "q19", "Did the distributions change only in the tails?",
    "**Question.** Did cleaning change the body of the data? **Data.** Daytime values of every "
    "cleaned column, all periods; shift of the 1st, 50th and 99th percentiles as a share of the "
    "1–99 % range.",
    "quantiles before and after cleaning",
    f"""**What it shows.** For the model-relevant columns the percentiles move by at most
{max(c['max_bulk_shift_pct'][k] for k in model_cols):.2f} % of their range, and after cleaning no
value lies outside the policy limits ({sum(c['outside_range_after'].values())} minutes). Two QA
columns shift more, by design: the pyranometer ({c['max_bulk_shift_pct']['pyranometer']:.1f} %,
whole ×0.75 days removed by R7) and DC of inverter 1
({c['max_bulk_shift_pct']['s01_inv1_DC_kW']:.1f} %, the dead-sensor zeros removed by R6).

**What it means.** Cleaning changes the tails, not the body of the data. (The plan expected less
than 1 % for every column; the two exceptions are explained by rules that remove whole faulty
populations.)""")
c = F["q20"]
by = c["days_changed_by_rule"]
cells += analysis(
    "q20", "Is energy conserved?",
    "**Question.** Does cleaning change daily energy? **Data.** Daily site energy (local days) "
    "before and after, all periods.",
    "daily energy before and after cleaning",
    f"""**What it shows.** Daily energy is unchanged on {c['days'] - c['days_changed']:,} of
{c['days']:,} days. The {c['days_changed']} days that change by more than 0.1 % are all freeze
({by.get('R3_frozen', 0)}) or spike ({by.get('R1_range', 0)}) days; the largest change
({c['largest_decrease_pct']:.1f} %, 2021-09-11) is a day whose recorded energy was almost entirely
a frozen value. Over the splits the change is {c['total_change_pct_by_split']['train']:+.2f} %
(train), {c['total_change_pct_by_split']['val']:+.3f} % (validation) and
{c['total_change_pct_by_split']['test']:+.2f} % (test).

**What it means.** Cleaning never removes real production (T-16, T-27). The plan expected less
than 0.5 % in every split; train exceeds it only because two long freezes (2021-09-11/12 and
2023-08-26/27) held high values for many hours — energy that was never produced.""")
c = F["q21"]
mp = c["missing_pct_all"]
cells += analysis(
    "q21", "Where is data missing after cleaning?",
    "**Question.** What is missing after cleaning, and where? **Data.** Stage 5, daytime minutes "
    "and hours, all periods.",
    "missing values by column and month; target gaps by split",
    f"""**What it shows.** {mp['site_total_AC_kW']:.2f} % of daytime target minutes are missing —
{c['target_nan_daytime_hours']['train']} / {c['target_nan_daytime_hours']['val']} /
{c['target_nan_daytime_hours']['test']} daytime hours in train / validation / test, all from
communication freezes. The model inputs miss {min(mp[k] for k in ('irradiance', 'mod_temp1', 'ambient_temp')):.2f}–{max(mp[k] for k in ('irradiance', 'mod_temp1', 'ambient_temp')):.2f} %
of daytime minutes; the pyranometer {mp['pyranometer']:.1f} % (R7) and DC of inverter 1
{mp['s01_inv1_DC_kW']:.1f} % (R6), both QA columns.

**What it means.** Missing hours stay NaN. How past-day features see them is decided by the
imputation policy of stage 6 (expanding median and forward fill; never backward fill).""")
c = F["q22"]
cells += analysis(
    "q22", "Is anything left?",
    "**Question.** After cleaning, are there remaining frozen runs, impossible values or "
    "unexplained outliers? **Data.** Stage 5, all periods (descriptive; a finding would go back "
    "to stage 5).",
    "residual checks on the cleaned data",
    f"""**What it shows.** No frozen daytime run of 30 minutes or more is left in any main column
({sum(c['frozen_daytime_minutes_left'].values())} minutes). {c['csi_above_1_2']} daytime hours
have AC above 1.2 × the clear-sky AC; {pct(c['csi_above_1_2_low_sun_share'], 0)} of them have the
sun below 15° (03 and 14–15 UTC). {c['robust_outliers']} hours are robust outliers of AC per unit
irradiance; {c['robust_outliers_by_utc_hour'].get('05', 0)} of them are at 05 UTC in December and
January. Without flag F3, {c['imbalance_hours_without_f3']} daytime hours would show an inverter
imbalance (ratio outside 0.8–1.25); with it, {c['imbalance_hours']} remain.

**What it means.** None of these is a data error. At sunrise and sunset the isotropic clear-sky
model underestimates the light on the panels, so the clear-sky index is unstable at low sun —
relevant to the denominator of k and its theo > 50 kW rule (D-11, stage 6). On winter mornings
the rows of panels shade each other while the irradiance sensor is not shaded — a physical effect
the models must learn from sun elevation and azimuth. The few remaining imbalance hours are
short or cloudy episodes below the F3 thresholds.""")
cells += [md("""## Part D — The cleaned data for modelling

Properties of the cleaned data that bear on the next stages. Anything that could inform a modelling
choice is computed on train only.""")]
c = F["q23"]
cells += analysis(
    "q23", "The real ceiling of each inverter",
    "**Question.** How close does the plant come to the assumed capacity of 3,449 kW (A-08)? "
    "**Data.** Stage 5, train.",
    "AC against DC at the top, and daily peaks (train)",
    f"""**What it shows.** Neither inverter approaches the assumed rating of 1,724.5 kW: the
maximum is {c['ac_max']['inv1']:,.0f} / {c['ac_max']['inv2']:,.0f} kW (99.99th percentile
{c['ac_p99_99']['inv1']:,.0f} / {c['ac_p99_99']['inv2']:,.0f} kW), and the highest hourly site
mean in train is {c['site_hourly_max_train']:,.0f} kW, {pct(c['share_of_capacity'], 0)} of
3,449 kW. There is no hard clipping plateau; AC grows more slowly than DC near the top
(efficiency {c['efficiency_at_dc_1600']:.2f} at 1,600–1,650 kW DC, against 0.97 in the middle
of the range).

**What it means.** The nRMSE of the direct target is normalised by 3,449 kW, as in the original
(kept for comparability); relative to the observed peak it would be about 15 % larger. The report
must state the normalisation, and A-08 stays open until the inverter model is known (Q-11).""")
c = F["q24"]
amb, mod = c["pairs"]["ambient_temp - site_ambient_temp"], c["pairs"]["mod_temp1 - mod_temp2"]
cells += analysis(
    "q24", "Temperature sensors — which agree, which is reliable?",
    "**Question.** Which ambient sensor should the models use (the open part of D-09), and are "
    "the two module sensors redundant? **Data.** Stage 5, train.",
    "sensor pairs, fault rates, and module heating (train)",
    f"""**What it shows.** The two ambient sensors agree closely (r = {amb['pearson_r']:.3f},
median difference {amb['median_diff']:+.2f} K; 98 % of differences between {amb['p1_diff']:+.1f}
and {amb['p99_diff']:+.1f} K). The meteo sensor has {c['faults']['ambient_temp']['R1']:,} fault-code
minutes that the SiteData sensor does not have, but it explains module heating slightly better
(R² {c['physics']['ambient_temp']['r2']:.2f} against {c['physics']['site_ambient_temp']['r2']:.2f}
for module − ambient against irradiance) and shares the logger and clock of the irradiance and
module sensors. The module sensors are practically identical (r = {mod['pearson_r']:.4f}, median
difference {mod['median_diff']:+.2f} K).

**What it means.** Recommendation for D-09 (temperature): keep `ambient_temp` as the model input
(its fault codes are removed by R1) and `site_ambient_temp` for QA only. One module sensor would
be enough; stage 7 can confirm it.""")
c = F["q25"]["target_vs"]
cells += analysis(
    "q25", "How are the hourly variables related?",
    "**Question.** Which measured variables move with the target? **Data.** Hourly, train, "
    "daytime, normal operation; Spearman correlation.",
    "correlation matrix (train)",
    f"""**What it shows.** The target follows the in-plane irradiance almost one-to-one
(Spearman {c['irradiance']:.3f}), much more closely than clear-sky irradiance
({c['poa_clearsky']:.3f}) or sun elevation ({c['elev']:.2f}): clouds matter. Module temperature
correlates {c['mod_temp1']:.2f} with output, mostly because both rise with irradiance; ambient
temperature ({c['ambient_temp']:.2f}) and wind ({c['wind_speed']:.2f}) hardly.

**What it means.** All of these are measured at the target time and may enter the models only as
past-day lags (§6.6), so these correlations are an upper bound on what weather information could
give — not what the lags will give (Q26).""")
c = F["q26"]
cells += analysis(
    "q26", "How much does the same hour on past days tell?",
    "**Question.** How informative are the lag features (1, 2, 3, 7, 14 days) of the contract? "
    "**Data.** Hourly, train, daytime, normal operation.",
    "correlation with the same hour 1-14 days earlier (train)",
    f"""**What it shows.** The same hour yesterday correlates {c['target']['1']:.2f} with today's
output, and still {c['target']['14']:.2f} two weeks earlier — but this is mostly the daily and
seasonal cycle, which the clear-sky features already describe. Once output is divided by
clear-sky output, the day-to-day persistence of the sky condition is weak:
{c['clear_sky_index']['1']:.2f} at one day, {c['clear_sky_index']['2']:.2f} at two,
{c['clear_sky_index']['7']:.2f} at seven and {c['clear_sky_index']['14']:.2f} at fourteen days.

**What it means.** The lags carry little information beyond the clear-sky features (consistent
with the daily persistence of 0.33 found in stage 3); weather forecasts (stage 6b) are the main
source of skill, and long lags are expected to be weak in feature selection (stage 7).""")
c = F["q27"]
cells += analysis(
    "q27", "How cloudy is it, by season?",
    "**Question.** How often is the sky clear, partly cloudy or overcast? **Data.** Hourly, train, "
    "daytime, normal operation; kt = measured irradiance / clear-sky POA.",
    "hourly sky condition by season (train)",
    f"""**What it shows.** Most daytime hours are clear (kt ≥ 0.8): {c['clear_pct']['DJF']} % in
winter, {c['clear_pct']['MAM']} % in spring, {c['clear_pct']['JJA']} % in summer and
{c['clear_pct']['SON']} % in autumn; overcast hours (kt < 0.3) are rare ({c['overcast_pct']['DJF']} %
in winter). The median kt is about {c['median_kt']['JJA']:.2f}: the in-plane sensor reads about
8 % above the Ineichen clear-sky model on typical clear hours.

**What it means.** Forecast errors will concentrate in winter and spring; summer is nearly
deterministic. The scale of the clear-sky reference is decided in stage 6 (D-11).""")
c = F["q28"]
cells += analysis(
    "q28", "Is there a weekday effect?",
    "**Question.** Does production depend on the day of the week (`is_weekend`, `day_of_week` "
    "are in the feature contract)? **Data.** Daily clear-sky index, train, complete days in "
    "normal operation.",
    "daily index by day of week (train)",
    f"""**What it shows.** The daily clear-sky index does not differ between days of the week
(Kruskal–Wallis p = {c['kruskal_p']:.2f} over {c['days']} days; the medians lie within
{c['median_range']:.3f}).

**What it means.** A PV plant does not follow a work week. `is_weekend` and `day_of_week` can only
add noise; stage 7 is expected to drop them.""")
c = F["q29"]
sm = c["same_month_plant"]
cells += analysis(
    "q29", "Long-term trend: degradation, soiling or sensor drift?",
    "**Question.** Does the plant's performance change over the years, and is it the plant or "
    "the sensor? **Data.** Hourly, normal operation; trends estimated on train, validation and "
    "test shown for description only.",
    "monthly performance index and clear-sky checks",
    f"""**What it shows.** At 700–900 W/m², corrected to 25 °C, output per unit irradiance
declines by {abs(c['slope_pct_per_year_train']):.1f} % per year over train
(p = {c['p_value_train']:.3f}), on top of a seasonal swing of about
{c['seasonal_range_pct']:.0f} %. Against the clear-sky model, which does not depend on the
sensor, the irradiance sensor is stable ({c['sensor_vs_clear_sky_slope_train']:+.1f} % per year,
p = {c['sensor_vs_clear_sky_p']:.2f}: no drift), while the plant's clearest-hour output points the
same way ({c['plant_vs_clear_sky_slope_train']:+.1f} % per year) but is dominated by the seasons
(p = {c['plant_vs_clear_sky_p']:.2f}). In January the clearest-hour output was
{' → '.join(f"{v:.3f}" for v in sm['1'].values())} over the four winters.

**What it means.** A slow decline of about 1 % per year, plus soiling or cleaning cycles, is
plausible; it is small next to weather errors and not monotonic into the test period, so no trend
feature is proposed. It is recorded as a limitation (a possible small level bias of the direct
target in test).""")
c = F["q30"]
cells += analysis(
    "q30", "Why are the remaining tags not used?",
    "**Question.** Could any unused tag help, and does any need cleaning? **Data.** Stage 4, all "
    "periods (descriptive).",
    "evidence for each unused tag",
    f"""**What it shows.** Precipitation never records rain in 45 months (maximum
{c['precipitation_max']}, {c['precipitation_positive_minutes']} positive minutes; the site does
get rain), so the gauge is not reporting. The three soiling sensors freeze or fall to zero during
the data (last active {c['soiling_last_active_month']['soil']}, {c['soiling_last_active_month']['soiling01']}
and {c['soiling_last_active_month']['soiling02']}) and have no documented units. Wind speed and
direction are measured at the target time (EXCL) and stick for long periods
({c['wind_held_pct']['wind_speed']:.1f} % of daytime minutes). Reactive power and power factor
follow the grid operator's setpoint (PF exactly 1.000 in {c['pf_one_pct']:.0f} % of minutes);
the IEC Q/V/PF setpoints are grid control. Inverter status codes are used only by flag F3.

**What it means.** None of these is a candidate model input, and cleaning them beyond R1 on wind
speed would serve no purpose.""")
c = F["q31"]
cells += analysis(
    "q31", "How do the splits differ? (descriptive only)",
    "**Question.** What do the three splits contain? **Data.** Hourly, daytime, all splits. "
    "*Descriptive only — used for no decision.*",
    "months and production levels by split",
    f"""**What it shows.** Test covers calendar months {c['test_calendar_months']} — it has no
summer hours ({c['test_hours_JJA']} hours in June–August), the season in which forecasting is
easiest (Q27). The mean daytime output is similar in the three splits
({c['mean_ac_kw']['train']:,.0f} / {c['mean_ac_kw']['val']:,.0f} / {c['mean_ac_kw']['test']:,.0f} kW),
as is the share of clear hours ({c['clear_hour_pct']['train']:.0f} / {c['clear_hour_pct']['val']:.0f} /
{c['clear_hour_pct']['test']:.0f} %).

**What it means.** Test errors will be higher than a full-year average would be; comparisons with
the original remain fair because the split is the same, and per-season reporting (§6.9) makes the
difference visible.""")
cells += [md(f"""## Summary — the evidence for each cleaning rule

| Rule | Threshold | Evidence (train) | Verdict |
|---|---|---|---|
| R1 range | temperatures −20…60/90 °C, irradiance −20…1,600 W/m², power −20…1,897 kW | limits in empty regions (Q04, Q05) | ✅ not tuned |
| R1 margin | ±1 min (minute source), ±15 min (polled) | blends next to temperature faults (Q06); polled: interpolation over one poll period | ✅ |
| R2 offset | −20…0 W/m² → 0 | night offset ≥ {F['q07']['irradiance_min_train']:.1f} W/m², never negative in daylight (Q07) | ✅ |
| R3 frozen | 30 min (ambient 60), non-zero power | plateau of run lengths (Q08); freezes hit all sensors at once (Q09) | ✅ insensitive |
| R3 margin | ±1 / ±15 min | resume minutes are blends (Q10) | ✅ |
| R4 night zero | ≤ 5 kW below the horizon → 0 | no night noise exists; acts on tiny twilight output, {F['q18']['kwh_by_rule']['R4_night_zero']:.0f} kWh in total (Q11) | ⚠️ kept as a definition; justification corrected |
| R5 night stale | > 5 kW below −6° → NaN | all such minutes are frozen values (Q11) | ✅ |
| R6 dead DC | DC = 0 with AC > 100 kW | one dead sensor, March 2023 (Q12) | ✅ |
| R7 scale fault | ratio outside 0.9–1.1, per day **and per hour** | two populations, 1.00 and 0.75; the fault also switches within a day (Q13) | ✅ **changed: F-21** |
| F1 downtime | AC < 20 kW, sun > 5°, irradiance > 150 | stops are a separate mode; 20–100 kW is empty (Q14) | ✅ |
| F2 curtailment | setpoint < 100 %, **not a non-integer drift > 10 min** | steps cut output; the 2022 drift did not (Q15) | ✅ **changed: F-20** |
| F3 derate | status 5/6 ≥ 30 min, or stable windowed ratio < 0.85 | modes 0.99 and 0.74/1.30 with empty valleys; stable fractions (Q16) | 🆕 **new: F-22** |
| Hourly | coverage 45/60, op-limited share 0.05 | hours complete or empty; few hours near the threshold (Q17) | ✅ insensitive |

**For the next stages.** Clear-sky index unstable at low sun → denominator of k (D-11, Q22);
winter-morning row shading (Q22); the observed peak is {pct(F['q23']['share_of_capacity'], 0)} of
the 3,449 kW capacity (A-08, Q23);
keep `ambient_temp`, one module sensor suffices (D-09, Q24); past-day lags carry little beyond
the clear-sky features (Q26); no weekday effect (Q28); a slow decline of about 1 % per year
(Q29); test has no summer (Q31).

**Reproduce.** `python -m solarpv.data.clean` (stage 5), `python -m solarpv.eda.quality`, then
`python scripts/make_quality_notebook.py`. Checks: T-31 (`tests/contracts/test_quality_eda_output.py`).""")]

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "solarpv": {"quality_manifest_sha256": text_sha256(MANIFEST)}})
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(NB.parent)}}).execute()
NB.write_bytes(nbformat.writes(nb).encode("utf-8"))
print("wrote", NB)

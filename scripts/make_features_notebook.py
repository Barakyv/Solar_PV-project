"""Write notebooks/04b_features_target.ipynb (stage 6) and execute it.

    python -m solarpv.features.build
    python -m solarpv.features.decide_k
    python -m solarpv.features.build
    python scripts/make_features_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-6
outputs (features manifest, configs/target.yaml, derived constants, orientation summary, report
functions), never typed. The notebook records the hashes of those files; the stage-6 contract
test checks that they are current. Plan: docs/stage6_plan.md.
"""

import json

import nbformat
import pandas as pd
import yaml
from nbclient import NotebookClient

from solarpv.config.cli import no_options
from solarpv.config.constants import DERIVED_CONSTANTS_FILE
from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR, REPO_ROOT
from solarpv.config.provenance import text_sha256
from solarpv.features import report as R

no_options("python scripts/make_features_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "04b_features_target.ipynb"
MANIFEST = OUTPUTS_DIR / "06_features" / "features_manifest.json"  # the tracked copy
TARGET = CONFIGS_DIR / "target.yaml"
ORIENT = AUDIT_DIR / "orientation_2026-10-20" / "orientation_summary.json"
QUALITY = OUTPUTS_DIR / "03b_quality" / "quality_manifest.json"  # stage 3b (Q26)
SOURCES = (MANIFEST, TARGET, DERIVED_CONSTANTS_FILE, ORIENT, QUALITY)

M = json.loads(MANIFEST.read_text(encoding="utf-8"))
D = yaml.safe_load(TARGET.read_text(encoding="utf-8"))
K = yaml.safe_load(DERIVED_CONSTANTS_FILE.read_text(encoding="utf-8"))
ORI = json.loads(ORIENT.read_text(encoding="utf-8"))
Q26 = json.loads(QUALITY.read_text(encoding="utf-8"))["analyses"]["q26_lags"]["facts"]
T1 = R.load(1)
KS = M["k_shares_train_daytime"]  # unrounded, as in the D-11 criteria
COR = R.target_correlations(T1).facts
INV = R.kw_from_k(T1)
TC = M["target_checks"]
ORES = pd.read_csv(AUDIT_DIR / "orientation_2026-10-20" / "orientation_results.csv")
CRIT = D["criteria"]
L04 = D["l04_validation_nrmse_by_threshold"]
L04S = D["l04_seed_std_by_threshold"]

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f} %"


m1 = ORES.set_index("method")
cells = [
    md("""# 04b — Target and features (stage 6)

**Purpose.** Build the modelling table: the two targets (§6.5), direct AC power and the clear-sky
index k, and the feature contract (§6.6), one table for each horizon D+1, D+2, D+3. The stage
also takes four open decisions: the orientation of the panels (A-04), the denominator of k
(D-11), the duplicate azimuth (Q-05), and the treatment of low sun (L-04, from the literature
review).

**Inputs.** The cleaned hourly data of stage 5 and the stage-5 minute flags. **Outputs.**
`data/processed/06_features/features_1H_D{h}.parquet`, the feature dictionary, the D-11 decision
(`configs/target.yaml`) and `MEAN_DAYTIME_K` (`configs/derived_constants.yaml`).

**Rules.**
- Decisions on train; D-11's third criterion on validation, as specified; the test split is used
  for nothing.
- A feature may use only hours that ended by the issuance time T0 (00:00 local of the first
  forecast day, D-16).
- Op-limited hours never feed past-day power features (D-22).

Each code cell is followed by a note: what it shows, what it means, and what was decided. The
numbers come from the stage-6 outputs. Plan: `docs/stage6_plan.md`; every run is described in
`docs/run_journal.md` and logged under `outputs/06_features/runs/`."""),
    code("""import pandas as pd
import yaml
from IPython.display import Image, display

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR
from solarpv.features import report as R

t1 = R.load(1)  # the D+1 table; D+2 and D+3 differ only in the past-day features
manifest = R.manifest()
print(t1.shape, "| horizons:", manifest["rows_per_horizon"])"""),
    md(f"""**Loaded.** The D+1 table has {M['rows_per_horizon']['D1']:,} hourly rows (one per hour,
September 2021 – May 2025, true UTC), each with its issuance time, split, daytime mask, flags, the
targets and the {len(M['features_per_horizon']['D1'])} features of D+1. D+2 has
{len(M['features_per_horizon']['D2'])} features and D+3 {len(M['features_per_horizon']['D3'])}:
the lags that are not yet known at T0 are left out (lag_1d at D+2; lag_1d and lag_2d at D+3)."""),
    md("## 1. Which way do the panels face? (A-04)"),
    code("""display(pd.read_csv(AUDIT_DIR / "orientation_2026-10-20" / "orientation_results.csv"))
display(Image(str(AUDIT_DIR / "orientation_2026-10-20" / "orientation_fit.png")))"""),
    md(f"""**What it shows.** On {ORI['clear_days']} clear train days, temperature-corrected DC power
fits the clear-sky irradiance best on a plane of tilt {ORI['panel_tilt_m1']:.0f}° and azimuth
{m1.loc['M1 DC one scale', 'azimuth']:.0f}° (one scale for all days), azimuth
{m1.loc['M2 DC scale per day', 'azimuth']:.1f}° from the daily shape alone, and
{m1.loc['M3 DC centre vs clear-sky centre', 'azimuth']:.1f}° from the centre of the production
curve. The irradiance sensor, fitted the same way, faces {ORI['sensor_plane_m0']['azimuth']}°.

**What it means.** The panels face about 5° west of south. The sensor faces south, and the
earlier result of F-13 (180°) came from clocks that were still slow. Methods agree, and the
difference exceeds the 2° rule of the plan. **Decision (A-04, F-23):** the panel plane
{ORI['panel_tilt_m1']:.0f}°/{ORI['panel_azimuth_median_m1_m3']:.0f}° is used for the theoretical
power and the POA features. The clock correction keeps the sensor plane on which it was framed.
External confirmation (satellite row orientation or the operator) is still open (Q-11)."""),
    md("## 2. The target k and its denominator (D-11)"),
    code("""r = R.k_distributions(t1)
r.show()"""),
    md(f"""**What it shows.** k = AC / theoretical clear-sky power on valid daytime train hours.
- **With GHI_cs:** {pct(KS['ghi']['k>1.0'])} of hours have k above 1, and
  {pct(KS['ghi']['k>1.2'])} above 1.2.
- **With POA_cs on the panel plane:** {pct(KS['poa']['k>1.0'])} and
  {pct(KS['poa']['k>1.2'], 2)}.
- No value sits exactly at 1.2.
- {sum(TC['k_suspect_hours'].values())} hours over all splits exceed 1.5 (`k_suspect`). All are
  at low sun (elevation {TC['k_suspect_sun_elevation_deg'][0]}–{TC['k_suspect_sun_elevation_deg'][1]}°,
  theo at most {TC['k_suspect_max_theo_kw']} kW), and {TC['k_suspect_op_limited']} are
  op-limited. Each is a ratio of two small numbers; the hours are flagged, not removed (§6.5).

**What it means.** k is not clipped (F-06 fixed). In the original, 16.8 % of daytime test hours
were cut at exactly 1.2. With the right reference, almost nothing is near that value, because
k > 1 is now rare instead of being an artefact of the reference."""),
    code("""r = R.weeks(t1)
r.show()"""),
    md("""**What it shows.** A clear summer week and a winter week: measured AC, and the
theoretical power with each reference.

**What it means.** In winter the sun is low and the panels are tilted towards it. A horizontal
reference (GHI_cs) underestimates what they receive, so measured AC exceeds it and k with GHI
rises above 1. The plane-of-array reference follows the panel geometry. In summer the two
references are close."""),
    code("""crit = pd.read_csv(OUTPUTS_DIR / "06_features" / "d11_criteria.csv")
display(crit)
display(Image(str(OUTPUTS_DIR / "06_features" / "d11_l04.png")))"""),
    md(f"""**What it shows.** The three D-11 criteria, all fixed before they were computed:

| Criterion | GHI_cs | POA_cs |
|---|---|---|
| 1. share of k > 1 (train) | {pct(CRIT['ghi']['share_k_above_1'])} | {pct(CRIT['poa']['share_k_above_1'])} |
| 2. spread of the monthly clear-day median k (train) | {CRIT['ghi']['clear_day_k_spread']:.3f} | {CRIT['poa']['clear_day_k_spread']:.3f} |
| 3. validation nRMSE in kW (fixed XGBoost, 3 seeds) | {CRIT['ghi']['val_nrmse_kw_pct']:.2f} ± {CRIT['ghi']['val_nrmse_kw_seed_std']:.2f} % | {CRIT['poa']['val_nrmse_kw_pct']:.2f} ± {CRIT['poa']['val_nrmse_kw_seed_std']:.2f} % |

**Decision (D-11): {D['k_denominator'].upper()}_cs**, which wins
{D['wins'][D['k_denominator']]} of 3 criteria. It removes the season from clear-day k almost
completely (criterion 2) and gives the lower error in kW (criterion 3). Validation was used only
for criterion 3, as specified."""),
    md("## 3. Low sun (L-04, from the literature review)"),
    code("""display(pd.read_csv(OUTPUTS_DIR / "06_features" / "l04_k_by_elevation.csv"))
display(pd.read_csv(OUTPUTS_DIR / "06_features" / "l04_training_threshold.csv"))"""),
    md(f"""**What it shows.** k by sun elevation on train. Below that, the validation nRMSE in kW
when hours with a small theoretical power are left out of the training loss (3 seeds each):

| Threshold | nRMSE |
|---|---|
| 50 kW | {L04['50.0']:.3f} ± {L04S['50.0']:.3f} % |
| 100 kW | {L04['100.0']:.3f} ± {L04S['100.0']:.3f} % |
| 150 kW | {L04['150.0']:.3f} ± {L04S['150.0']:.3f} % |

**What it means.** k is least stable at low sun, as the literature (C2) and stage 3b (Q22)
suggested. But leaving those hours out of the training loss does not help reliably.
- The best gain (150 kW) is {L04['50.0'] - L04['150.0']:.3f} pp.
- The rule requires more than {D['l04_needed_gain_pp']:.3f} pp: twice the seed-to-seed noise.
- 100 kW is worse than both.

**Decision: the training threshold stays {D['training_min_theo_kw']:.0f} kW** (§6.5).

**A note on method.** The first rule (a gain of 0.05 pp on one fit) would have chosen 150 kW
from noise: on one seed, 100 kW was 0.18 pp worse and 150 kW 0.05 pp better. The rule was made
seed-robust *before* the threshold was adopted, and the change is recorded in the run journal
(R6-03, R6-04) and in `decide_k.py`."""),
    code("""derived = yaml.safe_load((CONFIGS_DIR / "derived_constants.yaml").read_text(encoding="utf-8"))
print(derived)
print(R.kw_from_k(t1))"""),
    md(f"""**What it shows.** `MEAN_DAYTIME_K` = {K['mean_daytime_k']:.4f}: the mean k over valid
daytime train hours, the denominator of the indirect nRMSE (§6.9). The inversion k × theo − AC
is at most {INV['max_abs_error_kw']:.1e} kW over {INV['valid_hours']:,} valid hours (T-11).

**What it means.**
- The original's 0.6279 came from the clipped, wrongly timed k; it is kept only for the
  evaluation fixture (T-15).
- A forecast of k converts to kW exactly through the same theoretical power. The route
  comparison can therefore score both targets against actual AC (C-16)."""),
    md("## 4. Past-day features: issuance, DST, masks and gaps"),
    code("""from solarpv.config.constants import TIMEZONE_LOCAL
from solarpv.features import build as B

s = t1["site_total_AC_kW"]
guarded = pd.Series(B.past_value(s, 1, B.issuance_utc(t1.index, 1)), index=t1.index)
h = t1.loc["2022-10-30 19:00":"2022-10-30 23:00"].index  # around the end of the 25-hour day
src = h - pd.Timedelta(hours=24)
pd.DataFrame({
    "local time": h.tz_convert(TIMEZONE_LOCAL).strftime("%d.%m %H:%M %Z"),
    "T0 (UTC)": t1.loc[h, "t0_utc"].dt.strftime("%d.%m %H:%M").to_numpy(),
    "24 h back (UTC)": src.strftime("%d.%m %H:%M"),
    "source ended by T0": (src + pd.Timedelta(hours=1) <= t1.loc[h, "t0_utc"]).to_numpy(),
    "plain 24 h shift": s.shift(24).loc[h].to_numpy(),
    "lag_1d after the guard": guarded.loc[h].to_numpy(),
    "lag_1d in the table": t1.loc[h, "site_total_AC_kW_lag_1d"].to_numpy()},
    index=h.strftime("%d.%m %H:%M UTC"))"""),
    md(f"""**What it shows.** The last hours of 30 October 2022 local, the day the clock falls
back (25 local hours), with the issuance time T0 of each row. For the 25th hour (23:00 local),
the hour 24 h earlier is 29.10 21:00 UTC. That is T0 itself, the first hour of the forecast day,
so its value is not known when the forecast is issued.

**What it means.**
- A plain 24-hour shift would read it. The guard turns it into NaN, and the imputation then
  fills it from earlier days.
- Over the four autumns the guard removes {M['dst_guard_removed_values']['D1']} values at D+1,
  {M['dst_guard_removed_values']['D2']} at D+2 and {M['dst_guard_removed_values']['D3']} at D+3.
- At this site the 25th hour is at night, so the leaked value would have been 0 anyway. The
  guard keeps the rule exact rather than lucky.
- The perturbation tests (T-09, T-18) change every value from T0 on and check that no feature
  of the day moves."""),
    code("""r = R.imputation(manifest)
r.show(rows=12)"""),
    md(f"""**What it shows.** The share of imputed values per past-day feature at D+1 (at most
{pct(M['imputed_share_max']['D1'], 2)}).

**What it means.**
- **D-22 masks.** Before any lag, each inverter's AC is masked in the hours of its own stops,
  derates or curtailment: {M['masked_power_hours']['s01_inv1_AC_kW']} and
  {M['masked_power_hours']['s01_inv2_AC_kW']} hours. The site total is masked in
  {M['masked_power_hours']['site_total_AC_kW']} op-limited hours. A limitation is not weather,
  and must not teach the model that yesterday was cloudy.
- **Imputation.** The gaps are filled with the median of the same UTC hour over the previous 30
  days. The original used the median of the whole column, nights included.
- **Warm-up.** The first {M['warmup_hours'] // 24} days ({M['warmup_hours']} hours) have too
  little history and are marked as warm-up, excluded from training."""),
    code("""d = pd.read_csv(OUTPUTS_DIR / "06_features" / "feature_dictionary.csv")
with pd.option_context("display.max_rows", 80, "display.max_colwidth", 120):
    display(d[["feature", "group", "definition", "unit", "horizons"]])"""),
    md(f"""**What it shows.** The feature dictionary: for each of the
{len(M['features_per_horizon']['D1'])} features, its definition, unit and the horizons that use
it. The CSV also gives the method of each group, *why the feature is known at T0*, and the
imputed share at each horizon.

**What changed from the original:**
- `sun_azimuth` is gone: it equals `solar_azimuth` (Q-05).
- Solar noon comes from the computed transit, not "local hour − 12".
- `daylight_fraction` is computed from sunrise and sunset, not from an approximation.
- The weekend is Friday–Saturday (Israel), not Saturday–Sunday.
- The POA features use the measured panel plane, not 30°/180°.
- The rolling statistics follow the horizon, which fixes F-07."""),
    md("## 5. A first look at what predicts k"),
    code("""r = R.target_correlations(t1)
r.show(rows=10)"""),
    md(f"""**What it shows.** Spearman correlation of each D+1 feature with k (train, daytime,
normal operation). The strongest are geometric: {', '.join(list(COR['top'])[:3])}.

**What it means.**
- k is lower when the clear-sky irradiance on the panels is high (summer, midday) and higher
  when it is low (|r| about {abs(list(COR['top'].values())[0]):.1f}). Possible causes, not tested
  here: heat losses of the modules and the inverters at high irradiance, and the share of
  diffuse light at low sun, which the clear-sky model estimates least well. Even after dividing
  by the clear sky, geometry still carries information about k.
- Yesterday's *power* at the same hour correlates {COR['k_lag_1d_power']:+.2f} with today's k.
  Power carries the daily and seasonal cycle that k removes.
- **Proposal for stage 7** (to be decided on validation, not part of the contract now): give the
  indirect route lags of k itself, which carry yesterday's sky condition without the cycle.
- In stage 3b the clear-sky index persisted only weakly from one day to the next (r =
  {Q26['clear_sky_index']['1']:.2f}, Q26). The weather forecasts of stage 6b are expected to
  carry most of the skill.

These are correlations on train, not causes; the choice of features belongs to stage 7."""),
    md("## 6. Checks"),
    code("""print((AUDIT_DIR / "features_2026-10-20" / "mutation_check_output.txt").read_text())"""),
    md(f"""**What it shows.** Five deliberate leaks were planted in the feature code, and each was
caught by the tests:
- (a) the original's rolling window (F-07);
- (b) backward fill;
- (b2) a median from later days;
- (c) lags without the DST guard, in two forms.

The unmodified code passes the same tests.

**What it means.** The leakage tests do more than run: they detect the errors they are written
for. The first run of this check found two weak tests: the synthetic data had no gaps, and the
test window skipped the DST day. Both were fixed before this table was built.

## Summary

| Decision | Result |
|---|---|
| A-04, panel orientation | {ORI['panel_tilt_m1']:.0f}°/{ORI['panel_azimuth_median_m1_m3']:.0f}° (panel plane; the sensor plane 20°/180° stays the clock frame) |
| D-11, k denominator | {D['k_denominator'].upper()}_cs: {D['wins'][D['k_denominator']]} of 3 criteria |
| L-04, low-sun threshold | stays {D['training_min_theo_kw']:.0f} kW (no robust gain) |
| Q-05 | `sun_azimuth` removed; {len(M['features_per_horizon']['D1'])} features |
| MEAN_DAYTIME_K | {K['mean_daytime_k']:.4f} (original 0.6279) |

**Next:** the evaluation layer reads truth and theoretical power from these tables (stage 8,
part B). Then the GFS forecasts (stage 6b), whose literature ideas (L-05–L-11) start there.

**Reproduce:** `python -m solarpv.features.build`, `python -m solarpv.features.decide_k`,
`python -m solarpv.features.build`, `python scripts/make_features_notebook.py`. Tests:
`pytest tests/unit/test_features_build.py tests/unit/test_target.py
tests/contracts/test_features_output.py`."""),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "solarpv": {"sources_sha256": {p.name: text_sha256(p) for p in SOURCES}}})
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(NB.parent)}}).execute()
NB.write_bytes(nbformat.writes(nb).encode("utf-8"))
print("wrote", NB)

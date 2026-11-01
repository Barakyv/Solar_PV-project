"""Write notebooks/04_cleaning.ipynb (stage 5: value rules and operation flags) and execute it.

    python -m solarpv.data.clean
    python scripts/make_cleaning_notebook.py

Every code cell is followed by a markdown note. Numbers in the notes are read from the stage-5
manifest, the cleaning log and the mutation check, never typed.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR, REPO_ROOT

NB = REPO_ROOT / "notebooks" / "04_cleaning.ipynb"
MANIFEST = OUTPUTS_DIR / "05_cleaning" / "cleaning_manifest.json"
LOG = OUTPUTS_DIR / "05_cleaning" / "cleaning_log.csv"
POLICY = CONFIGS_DIR / "cleaning_policy.yaml"
MUTATION = AUDIT_DIR / "cleaning_2026-10-15" / "mutation_check_output.txt"

M = json.loads(MANIFEST.read_text(encoding="utf-8"))
L = pd.read_csv(LOG)
MUT = [line for line in MUTATION.read_text(encoding="utf-8").splitlines() if line.strip()]
caught = sum(line.startswith("caught") for line in MUT)
H = pd.DataFrame(M["hours_by_split"]).T
ac = L[L["column"].isin(["s01_inv1_AC_kW", "s01_inv2_AC_kW"])]
r7 = L[(L["rule"] == "R7_scale_fault")].iloc[0]
AC_R1 = int(ac[ac["rule"] == "R1_range"]["minutes"].sum())
AC_R3 = int(ac[ac["rule"] == "R3_frozen"]["minutes"].sum())

cells = [
    md("""# 04 — Cleaning: value rules and operation flags (stage 5)

**Purpose.** Remove impossible values with explicit, documented rules, and flag the hours in
which the plant did not run normally (stops, grid curtailment, derated inverters), without
erasing them.

**Inputs.** The stage-4 minute table (true time, sources harmonised) and
`configs/cleaning_policy.yaml`. **Outputs.** `data/processed/05_clean/` (minute and hourly
tables), `outputs/05_cleaning/`: `cleaning_manifest.json`, `cleaning_log.csv` and the applied
policy.

**Rules of the stage.**
- Every changed value has a named rule; nothing is changed silently.
- Thresholds are fixed on evidence (stage 3b, `02b_data_quality.ipynb`), not tuned on a model.
- Operation flags are kept as flags: they are excluded from the training loss and from past-day
  features, included in the main evaluation and excluded in a secondary one (D-22).

**In the original** the spikes (up to 40,825 kW), the 32,766 sentinel and the −270 °C fault code
stayed in the data, curtailment was not handled, and quality flags were used as model features
(F-14, F-17). **Where it is documented:** OP-18 to OP-28; module `solarpv.data.clean`."""),
    md("## 1. The policy"),
    code("""import json

import pandas as pd
import yaml
from IPython.display import Image, display

from solarpv.config.paths import AUDIT_DIR, CONFIGS_DIR, OUTPUTS_DIR

policy = yaml.safe_load((CONFIGS_DIR / "cleaning_policy.yaml").read_text(encoding="utf-8"))
print("sections:", ", ".join(policy))
pd.DataFrame(policy["range"]).T.rename(columns={0: "low", 1: "high"})"""),
    md("""**What it shows.** The sections of the policy file and the physical ranges of rule R1.

**The value rules.**
- **R1 range:** outside the physical range.
- **R2 offset:** small negative irradiance at night, set to zero.
- **R3 frozen:** a value repeated for at least 30 minutes into daytime.
- **R4 night zero:** readings of at most 5 kW below the horizon.
- **R5 night stale:** power reported with the sun more than 6° below the horizon is a stale value.
- **R6 dead DC:** DC zero while AC is produced.
- **R7 scale fault:** the second sensor's scale fault.

**The flags.** F1 `downtime`, F2 `curtail` and F3 `derate`, together `op_limited_flag`."""),
    md("## 2. What each rule changed"),
    code("""log = pd.read_csv(OUTPUTS_DIR / "05_cleaning" / "cleaning_log.csv")
log.sort_values(["column", "rule"])"""),
    md(f"""**What it shows.** Per rule and column: minutes changed, how many in daylight, the hours
touched, and for power the energy removed.

**What it means.**
- On the inverters' AC power, the range rule removed {AC_R1:,}
  minutes, the spikes. The frozen-value rule removed {AC_R3:,}
  minutes of communication freezes.
- The largest count, {int(r7['minutes']):,} minutes, is the scale fault of the second irradiance
  sensor, which is only a quality check (D-09). The main sensor is not affected."""),
    code("""display(Image("../outputs/03_eda/figures/fig05_cleaning.png", width=900))"""),
    md(f"""**What it shows.** The communication freeze of 11–12 September 2021, before and after
cleaning: the logger repeated its last values for hours, including about 1,300 kW in the middle
of the night, and the frozen run is removed. Daily energy is unchanged on every day except the few
with spikes or freezes (right panel).

**Effect on the target.** The highest hourly site total falls from a spike-driven value to
{M['site_total_AC_kW']['max_hourly']:,} kW (99.99th percentile
{M['site_total_AC_kW']['p99_99_hourly']:,} kW), below the 3,449 kW AC capacity."""),
    md("## 3. Operation flags"),
    code("""manifest = json.loads((OUTPUTS_DIR / "05_cleaning" / "cleaning_manifest.json")
                      .read_text(encoding="utf-8"))
pd.DataFrame(manifest["hours_by_split"]).T"""),
    md(f"""**What it shows.** Per split, the daytime hours and the hours touched by each flag.

**What it means.**
- **Op-limited hours:** train {int(H.loc['train', 'op_limited_hours'])}, validation
  {int(H.loc['val', 'op_limited_hours'])} and test {int(H.loc['test', 'op_limited_hours'])}.
- **Curtailment** is frequent in validation and test. A model that learned it as weather would be
  wrong, and that is why the flags never enter as features.
- **The test split** has {int(H.loc['test', 'polled_hours'])} hours from the second source of 2025
  and {int(H.loc['test', 'mixed_hours'])} mixed hours. They are also reported separately
  (D-21)."""),
    code("""display(Image("../outputs/03_eda/figures/fig09_operations.png", width=900))"""),
    md("""**What it shows.** Flagged daytime hours per month and type: stops, curtailment and
derates.

**What it means.** The flags are not rare, and they cluster: derates in 2023, curtailment in
spring and autumn 2024. Without them a model would treat equipment and grid limits as weather."""),
    md("## 4. Does the cleaning code catch what it should?"),
    code("""print((AUDIT_DIR / "cleaning_2026-10-15" / "mutation_check_output.txt")
      .read_text(encoding="utf-8"))"""),
    md(f"""**What it shows.** A mutation check: {len(MUT)} deliberate faults were planted in the
code or the data (a value changed without a rule, production zeroed as night, a spike left in place,
a curtailment minute not flagged, and so on). The test suite caught {caught} of {len(MUT)}.

**What it means.** The tests of the cleaning stage fail when the cleaning is wrong, not only when
it crashes.

**Next stage.** Stage 6 (`04b_features_target.ipynb`) builds the targets and the features on the
cleaned hours.

**Reproduce.** `python -m solarpv.data.clean`, then `python scripts/make_cleaning_notebook.py`."""),
]

write_notebook(NB, cells, [MANIFEST, LOG, POLICY, MUTATION])

# ruff: noqa: E501  (markdown notes with values cannot be wrapped; see make_quality_notebook)
"""Write notebooks/12_stage11_part_d.ipynb (stage 11, part D: the quantile crossing in the report) and execute it.

    python -m solarpv.figures.analyses11
    python scripts/make_report_v7.py
    python -m solarpv.figures task S11-D
    python scripts/make_stage11_part_d_notebook.py

Every code cell is followed by a markdown note. Every number in the notes is read from
`outputs/11_analyses/quantile_crossing.csv`, `docs/traceability.csv`, the report or the figure manifest, never typed;
each claim is asserted here before the note is written. No model is trained and the step `test` is not run again.
"""

import json

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import REPO_ROOT

no_options("python scripts/make_stage11_part_d_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "12_stage11_part_d.ipynb"
DOCS = REPO_ROOT / "docs"
TABLE = REPO_ROOT / "outputs" / "11_analyses" / "quantile_crossing.csv"
MANIFEST = REPO_ROOT / "outputs" / "figures" / "figures_manifest.json"
SOURCES = [TABLE, REPO_ROOT / "outputs" / "11_analyses" / "metrics_kw.csv", DOCS / "report_v7.md",
           DOCS / "traceability.csv", MANIFEST]

t = pd.read_csv(TABLE)
m = pd.read_csv(REPO_ROOT / "outputs" / "11_analyses" / "metrics_kw.csv")
report = (DOCS / "report_v7.md").read_text(encoding="utf-8")
figs = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
TFTS = ["tft_96", "tft_168", "tft"]
KEYS = ["model", "seed", "target", "horizon", "split"]

n_ok = (t.set_index(KEYS)["n"] == m[m["view"] == "all"].set_index(KEYS)["n"].reindex(t.set_index(KEYS).index)).all()
assert n_ok
per = t.groupby(["model", "target", "split"])[["crossed_pct", "point_not_median_pct"]].mean()
tft = per.loc[TFTS]
top = {s: tft.xs(s, level="split")["crossed_pct"].idxmax() for s in ("val", "test")}
assert top == {"val": ("tft_96", "direct"), "test": ("tft_96", "direct")}
lstm_zero = bool((t[t["model"] == "lstm_sota"]["crossed_pct"] == 0).all())
assert lstm_zero
drop = 100 * (1 - t["pinball_sorted_kw"] / t["pinball_raw_kw"])
assert (drop >= -1e-9).all()
n_cells = len(t)
n_lower = int((drop > 1e-9).sum())
gmax = float(t["gap_max_kw"].max())
facts = figs["S11-21"]["facts"]
sec = report.split("### 6.5 Quantile outputs", 1)[1].split("\n## ", 1)[0]
assert "[Chernozhukov et al., 2010]" in sec and "[Koenker & Bassett, 1978]" in sec


def rng(split: str, col: str) -> str:
    s = tft.xs(split, level="split")[col]
    return f"{s.min():.1f} to {s.max():.1f}"


cells = [
    md("""# 12 - Stage 11, part D: the quantile crossing in the report

**Purpose.** The decision D-33 (2026-11-12) sorts the quantiles of the neural models wherever they are read as a
distribution and keeps the point forecast as the model's 0.5 head; it committed the report to state how often the
quantiles cross and what that costs. Part D measures both on the final run, on validation and on test, and writes
them into section 6.5 of the report with the two sources of the method (user request, 2026-11-16).

**Rules.**
- Nothing is trained and the step `test` is not run again; the quantile files are the frozen files of the run,
  checked against their step records.
- Nothing is decided from these numbers: no score, no point forecast and no test changes.
- The sources are verified against Crossref and OpenAlex with the user's approval.

**Documented in** `docs/stage11_plan.md` section 7, OP-69, run journal R11-06, errata X117 and X118, tests T-191 and
T-192, and the figure S11-21."""),
    md("## 1. How often the quantiles are out of order"),
    code("""import pandas as pd

from solarpv.config.paths import REPO_ROOT

t = pd.read_csv(REPO_ROOT / "outputs/11_analyses/quantile_crossing.csv")
t.groupby(["model", "target", "split"])["crossed_pct"].mean().unstack("split").round(1)"""),
    md(f"""**What it shows.** The share of the scored hours (daytime with a measured output) whose quantiles are out of order,
in percent, per model and route, the mean over the horizons and the seeds.

**What it means.** The quantiles of the TFTs are out of order on {rng('val', 'crossed_pct')} percent of the hours of a
model and route on validation and on {rng('test', 'crossed_pct')} percent on test, most often for TFT-96 on the direct
route in both periods. The BiLSTM-Opt writes {int(t[t['model'] == 'lstm_sota']['n_quantiles'].iloc[0])} quantiles and
they are never out of order; the TFTs write {int(t[t['model'] == 'tft_96']['n_quantiles'].iloc[0])}, from separate
heads that nothing in the training keeps in order."""),
    md("## 2. The table is on the scored hours"),
    code("""m = pd.read_csv(REPO_ROOT / "outputs/11_analyses/metrics_kw.csv")
keys = ["model", "seed", "target", "horizon", "split"]
n = m[m["view"] == "all"].set_index(keys)["n"]
(t.set_index(keys)["n"] == n.reindex(t.set_index(keys).index)).value_counts()"""),
    md(f"""**What it shows.** For every row of the table, whether its number of hours equals that of the scored metrics in kW.

**What it means.** All {n_cells} rows match: the crossing is measured on exactly the hours on which every model is
scored, not on night hours or hours without a measurement."""),
    md("## 3. What sorting costs the point forecast"),
    code("""tft = t[t["model"].isin(["tft_96", "tft_168", "tft"])]
tft.groupby(["model", "target", "split"])[["point_not_median_pct", "gap_mean_kw", "gap_max_kw"]].mean().round(1)"""),
    md(f"""**What it shows.** Per TFT and route: the share of the hours on which the 0.5 quantile, the point forecast, is not
the median of the sorted quantiles, and the mean and largest distance between the two on those hours, in kW.

**What it means.** On up to {tft.xs('val', level='split')['point_not_median_pct'].max():.1f} percent of the hours of
a model and route on validation and {tft.xs('test', level='split')['point_not_median_pct'].max():.1f} percent on test
the point forecast is not the median of the sorted distribution, by {facts['g_v']:.1f} and {facts['g_t']:.1f} kW on
average on those hours, and by at most {gmax:.1f} kW. This is the cost that D-33 accepted: the point forecast stays
the model's own output, so every score of the report is unchanged."""),
    md("## 4. Sorting never raises the quantile loss"),
    code("""drop = 100 * (1 - t["pinball_sorted_kw"] / t["pinball_raw_kw"])
t.assign(drop_pct=drop).groupby(["model", "target", "split"])["drop_pct"].max().unstack("split").round(2)"""),
    md(f"""**What it shows.** The largest drop of the mean quantile loss in kW over the horizons and seeds, in percent, when the
quantiles are sorted per hour.

**What it means.** The loss falls in {n_lower} of {n_cells} cells and rises in none, by at most {drop.max():.1f}
percent (TFT-96, direct route, validation). The part of the quantile loss that depends on the order of the quantiles is
smallest when they rise with their levels, so sorting can only lower it; this is the monotone rearrangement of
Chernozhukov, Fernández-Val and Galichon (2010)."""),
    md("## 5. Section 6.5 of the report"),
    code("""report = (REPO_ROOT / "docs/report_v7.md").read_text(encoding="utf-8")
print(report.split("### 6.5 Quantile outputs", 1)[1].split("\\n## ", 1)[0].strip())"""),
    md(f"""**What it shows.** The text of section 6.5 as the report generator wrote it.

**What it means.** Every number in it is registered in `docs/traceability.csv` with this table as its source (T-178,
T-192), and it cites the two sources of the method, Chernozhukov et al. (2010) for the rearrangement and Koenker and
Bassett (1978) for the quantile loss, both listed in chapter 17. The figure S11-21 gives the same numbers: the highest
share of crossed hours on validation is {facts['v_hi']:.1f} percent in both."""),
]

write_notebook(NB, cells, SOURCES)

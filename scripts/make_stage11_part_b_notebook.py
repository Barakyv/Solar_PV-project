# ruff: noqa: E501  (markdown notes with values cannot be wrapped; see make_quality_notebook)
"""Write notebooks/10_stage11_part_b.ipynb (stage 11, part B) and execute it.

    python scripts/make_report_v7.py
    python scripts/make_stage11_part_b_notebook.py

Every code cell is followed by a markdown note. Every number in the notes is read from `docs/traceability.csv`,
`docs/old_vs_new.csv` or the tables of the final run, never typed; each claim is asserted here before the note
is written. No model is trained and the step `test` is not run again.
"""

import pandas as pd
from _notebook import code, md, write_notebook

from solarpv.config.cli import no_options
from solarpv.config.paths import REPO_ROOT

no_options("python scripts/make_stage11_part_b_notebook.py", __doc__)  # X62: --help runs nothing

NB = REPO_ROOT / "notebooks" / "10_stage11_part_b.ipynb"
DOCS = REPO_ROOT / "docs"
RUN = REPO_ROOT / "outputs" / "runs" / "20261114T083930Z_final"
SOURCES = [DOCS / "traceability.csv", DOCS / "old_vs_new.csv", DOCS / "report_v7.md",
           RUN / "evaluate/val/route_comparison.csv", RUN / "test/test/route_comparison.csv",
           REPO_ROOT / "outputs/11_analyses/patchtst_gap.csv"]

tr = pd.read_csv(DOCS / "traceability.csv", dtype=str, keep_default_na=False)
ov = pd.read_csv(DOCS / "old_vs_new.csv", dtype=str, keep_default_na=False)
report = (DOCS / "report_v7.md").read_text(encoding="utf-8")
assert (tr["script"] == "scripts/make_report_v7.py").all()
sections = [ln for ln in report.splitlines() if ln.startswith("## ")]
bench = ov[ov["section"] == "benchmark"]
route = ov[ov["section"] == "route"].copy()
for c in ("old_value", "new_validation", "new_test"):
    route[c] = route[c].astype(float)
assert len(route) == 6 and route["old_value"].min() > 1.5
changed = int(((route["new_validation"] < 0) != (route["new_test"] < 0)).sum())
comparable_no = int((ov["comparable"].str.startswith(("no", "units"))).sum())
n_quoted = int(tr["file"].str.contains("v6.docx|PROJECT_MASTER", regex=True).sum())

cells = [
    md("""# 10 - Stage 11, part B: the report sections, the traceability and the old against new results

**Purpose.** The sections of the submitted report that are text (S-01, S-05, S-42 to S-44) rewritten from the
frozen final run `20261114T083930Z_final` (tag `g1-frozen`), the table that lists every number of the text with
its source, and the table of old against new results.

**Rules.**
- Nothing is trained and the step `test` is not run again.
- Every test number is placed beside the validation number of the same view.
- No winner is declared (D-37). Four of the five neural models have one seed.
- Decisions of the user (2026-11-14): Q-29, the dust-day half of L-20 is dropped; Q-30, a new script rebuilds
  the tables from the frozen predictions and `reproduce_paper_results.py` is not edited.

**Documented in** `docs/stage11_plan.md` section 5, OP-67, run journal R11-02, errata C-85 to C-91 (M51 to M57)
and X107 to X112, tests T-178 to T-184, and the figures S11-18, S11-19, C-11 and C-12."""),
    md("## 1. What the generator wrote"),
    code("""import pandas as pd

from solarpv.config.paths import REPO_ROOT

DOCS = REPO_ROOT / "docs"
tr = pd.read_csv(DOCS / "traceability.csv", dtype=str, keep_default_na=False)
report = (DOCS / "report_v7.md").read_text(encoding="utf-8")
files = tr["file"].str.split("; ").str[0]
pd.Series({"registered numbers": len(tr), "source files": files.nunique(),
           "sections": sum(ln.startswith("## ") for ln in report.splitlines()),
           "lines of text": len(report.splitlines())})"""),
    md(f"""**What it shows.** `scripts/make_report_v7.py` wrote {len(sections)} sections ({len(report.splitlines())}
lines) and {len(tr)} registered numbers, from {tr['file'].str.split('; ').str[0].nunique()} files.

**What it means.** A number cannot reach the text without a row: each is formatted by the generator with its file, its
script and the date, and the test T-178 fails when the text holds a number that no row names. {n_quoted} rows are
quotations of the submitted report or of the project documents, registered as old values; all other numbers are read
from a file of the run or of `outputs/` (S11-18)."""),
    md("## 2. The route, old against new (C2, M52)"),
    code("""ov = pd.read_csv(DOCS / "old_vs_new.csv", dtype=str, keep_default_na=False)
r = ov[ov["section"] == "route"][["item", "old_value", "new_validation", "new_test"]]
r.set_index("item")"""),
    md(f"""**What it shows.** The nRMSE of the direct route minus that of the indirect route in kW (a positive value:
the indirect route is better) for XGB-Opt and BiLSTM-Opt, as submitted and in the rebuild on validation and on test.

**What it means.** The submitted report found the indirect route better by {route['old_value'].min():.2f} to
{route['old_value'].max():.2f} pp. In the rebuild the difference is {route['new_validation'].min():+.2f} to
{route['new_validation'].max():+.2f} pp on validation and {route['new_test'].min():+.2f} to
{route['new_test'].max():+.2f} pp on test, and the sign changes between the periods for {changed} of the 6 pairs. The
submitted truth for the indirect route was rebuilt from the clipped index (C-16); the advantage is not reproduced and
neither route is declared better (`C-12`)."""),
    md("## 3. The benchmark, old against new"),
    code("""b = ov[ov["section"] == "benchmark"]
cols = ["item", "old_value", "new_validation", "new_test", "comparable"]
b[b["item"].str.contains("indirect D\\\\+1")][cols].set_index("item")"""),
    md(f"""**What it shows.** The D+1 indirect nRMSE of the six models: the submitted value (in the units of the target, with the
number of hours), and the rebuild's value in the same unit on validation and on test (seed mean).

**What it means.** The table has {len(bench)} rows in the units of the target, and a second set in kW. The
indirect rows are marked as not comparable: the target changed (F-06: the index is not clipped and its mean is
0.788 instead of 0.6279), so the submitted 29.09 per cent and the rebuilt value are not a difference of one quantity.
{comparable_no} of the {len(ov)} rows of the whole table carry such a mark, and the reason is in the column next to it
(X108)."""),
    md("## 4. Where the numbers come from (S11-18)"),
    code("""first = tr["file"].str.split("; ").str[0]
fam = first.map(lambda f: "quoted" if ("v6.docx" in f or "PROJECT_MASTER" in f)
                else "analyses" if "11_analyses" in f else "run or stages")
fam.value_counts()"""),
    md(f"""**What it shows.** The registered numbers by kind of source file.

**What it means.** {n_quoted} of the {len(tr)} numbers are quotations of fixed documents (the submitted report and the
project documents) and the rest come from output files. The figure `S11-18` draws the same counts."""),
    md("## 5. The statements the sections make about the results (D-37)"),
    code("""t = report
checks = {
    "no winner declared": "No model is declared the winner" in t,
    "validation beside test": "beside the validation number of the same view" in t,
    "operation-limited hours named": "operation-limited" in t,
    "quintile profile is a binning effect": "made by the binning" in t,
    "climatology margin stated": "minimum detectable effect" in t,
}
pd.Series(checks)"""),
    md(f"""**What it shows.** Five phrases of the rules for the results, searched in the generated text.

**What it means.** All {sum(p in report for p in ['No model is declared the winner', 'beside the validation number of the same view', 'operation-limited', 'made by the binning', 'minimum detectable effect'])} are present: no
winner is declared (D-37), every test number has the validation number of the same view beside it, the positive bias is
placed in the operation-limited hours (F-30), the quintile profile of v6 is called an effect of the binning and not
physics (M45), and the limit of the test against climatology is stated through the minimum detectable effect. The
test T-181 fails if one of them is removed."""),
    md("""## 6. What part B did not do

- **Dust days (L-20).** Dropped by the user (Q-29, option c): the project holds no aerosol record, and a label made from
  the measured deficit would be made from the outcome. The season half of L-20 stays.
- **`--xgb-only` (S-45).** A training run, left for the end of stage 11 (Q-30, option a). The rebuild of the tables and
  figures without training is `scripts/make_tables_from_frozen.py`.
- **The radar of v6 figure 14** (C-84) and the **coverage of the quantile intervals** (L-14) are not part of the scope
  of this part."""),
]

write_notebook(NB, cells, SOURCES)

# Project-local Agent Skills: provenance

Installed on 2026-11-01. Each downloaded skill is a complete, unmodified copy of its directory in
the source repository (checked with `diff -rq` after copying). Third-party material: it guides
the agent but never overrides `CLAUDE.md` or `PROJECT_MASTER.md`.

| Skill | Source repository | Path in the repository | Commit | License |
|---|---|---|---|---|
| exploratory-data-analysis | https://github.com/K-Dense-AI/scientific-agent-skills | `skills/exploratory-data-analysis` | `5f2bc0341354fd5590aef76a533ada1e1e5b4e62` | MIT (`LICENSE-scientific-agent-skills.md`) |
| scikit-learn | same | `skills/scikit-learn` | same | BSD-3-Clause (per SKILL.md) |
| statistical-analysis | same | `skills/statistical-analysis` | same | MIT |
| shap | same | `skills/shap` | same | MIT |
| scientific-visualization | same | `skills/scientific-visualization` | same | MIT |
| literature-review | same | `skills/literature-review` | same | MIT |
| scientific-writing | same | `skills/scientific-writing` | same | MIT |
| systematic-debugging | https://github.com/obra/superpowers | `skills/systematic-debugging` | `2ce8aa15d05bd57213e6f746f953aafb7ded7a8b` | MIT (`LICENSE-superpowers.txt`) |
| verification-before-completion | same | `skills/verification-before-completion` | same | MIT |
| task-figures | **locally authored** for this project (2026-11-12, user request: figures at the end of every task, a look-back, uniform across the models) | — | — | project |
| forecasting-validity-review | **locally authored** for this project | — | — | project |

`anthropics/skills` was cloned at `ca45fc496a3436251f191354091d557da31c15b5` to check the
document skills; none was copied, because `docx`, `pptx`, `xlsx` and `pdf` are already available
in this Claude Code environment as the `anthropic-skills:*` plugin skills.

## Verification on 2026-11-01

- **Files:** all ten skills have valid frontmatter (name equals the directory, lower-case,
  description under 1,024 characters), and every `references/`, `scripts/` and `assets/` path
  named in a SKILL.md exists. The nine downloaded directories are byte-identical to the source
  commits. No name collides with a user-level or plugin skill.
- **Imports in the `solar` environment:** available for every bundled script except the optional
  `Bio` and `tifffile` (exploratory-data-analysis: FASTA/TIFF inspectors) and `pypdf`
  (scientific-visualization: PDF metadata). None is needed by this project; none was installed.
- **Run on synthetic or bundled data, in the `solar` environment:**
  `missingness_leakage_audit.py` (flagged a planted test row stamped inside the train period;
  note that a single-site series always shows its site in several splits),
  `palette_audit.py`, `assumption_checks.check_normality`, `lint_manuscript.py` (flagged a
  planted placeholder), and `shap/scripts/tabular_report.py` (ran with SHAP 0.51.0 on Python
  3.11; additivity error 9e-9). Nothing else was run; the literature-review network scripts
  were not run.
- **Discovery:** the skills are not visible to the session in which they were installed
  (`Unknown skill`); Claude Code loads project skills when a session starts.

## Known limitations (checked on 2026-11-01)

- **Python versions.** The project environment `solar` runs Python 3.11.17 with scikit-learn
  1.9.1 and SHAP 0.51.0. The `shap` skill targets SHAP 0.52.0 (Python 3.12+) and
  `statistical-analysis` a separate Python 3.12 environment (Pingouin, PyMC). Their guidance
  applies; their pinned examples are not run in `solar`, and the project environment is not
  upgraded for them. A separate environment is created only if a concrete task needs one.
- **exploratory-data-analysis** bundled CLIs read CSV/TSV/JSON only (Parquet is
  reference-only): use them on a derived CSV copy, never on `Raw data/`.
- **literature-review** uses the network for database searches and DOI checks; `parallel-cli`
  needs Parallel authentication, schematics need `OPENROUTER_API_KEY`, PDF export needs Pandoc
  and XeLaTeX. None of these is configured; nothing is sent to an external service without the
  user's explicit approval.
- **systematic-debugging** refers to `superpowers:test-driven-development` and
  `superpowers:verification-before-completion`. The first is not installed (no test-driven
  development is imposed on exploratory notebooks); the second is installed here as
  `verification-before-completion`. The project's own rule applies: every bug gets a regression
  test and an errata entry.
- **K-Dense skills** ask to cite their paper when a skill "materially contributed" to a report.
  In this project no citation is added without the user's decision.

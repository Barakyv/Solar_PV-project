"""T-37: the approved scope of the submitted report (v6) stays complete in the plan.

docs/scope_register.md lists every deliverable of v6 (S-xx). The user decided on 2026-10-25 to
rebuild the whole scope. This test fails when:
- a figure or a numbered section of v6 has no row, or has two;
- an ID is missing;
- a row names a stage that does not exist;
- one of the six v6 models is dropped from decision D-08 in PROJECT_MASTER.md.
"""

import re

from solarpv.config.paths import REPO_ROOT

REGISTER = (REPO_ROOT / "docs" / "scope_register.md").read_text(encoding="utf-8")
MASTER = (REPO_ROOT / "PROJECT_MASTER.md").read_text(encoding="utf-8")
ROWS = re.findall(r"^\| (S-\d{2}) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| "
                  r"([^|]+) \|$", REGISTER, flags=re.MULTILINE)
V6_FIGURES = range(1, 26)  # Figures 1-25 of the submitted report
# numbered sections of v6 that carry work (2 Related Work and 17 References carry none)
V6_SECTIONS = ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6.1", "3.6.2", "3.6.3", "3.6.4", "3.6.5",
               "3.6.6", "3.6.7", "3.6.8", "3.7", "3.8", "3.9", "4.1", "4.2", "5.1", "5.2",
               "5.3", "5.4", "5.5", "5.6", "6.1", "6.2", "6.3", "6.4", "7.1", "7.2", "7.3",
               "7.4", "7.5", "8.1", "8.2", "9.1", "9.2", "10", "11.1", "11.2", "11.3", "11.4",
               "11.5", "12.1", "12.2", "12.3", "12.4", "12.5", "13.1", "13.2", "13.3", "13.4",
               "13.5", "13.6", "14", "15", "16"]
STAGES = {"1", "2", "3", "3b", "4", "5", "6", "6b", "7", "8", "9", "10", "11", "G-1"}
MODELS = ["XGB-Opt", "BiLSTM-Opt", "TFT-96", "TFT-168", "TFT baseline", "PatchTST"]


def _expand_sections(cell: str) -> set[str]:
    """'3.6.1–3.6.3' -> {3.6.1, 3.6.2, 3.6.3}; '4.2, 6.1–6.4' -> {4.2, 6.1, ..., 6.4}."""
    out = set()
    for part in (p.strip() for p in cell.split(",")):
        if "–" in part:
            a, b = part.split("–")
            head, first = a.rsplit(".", 1) if "." in a else ("", a)
            last = b.rsplit(".", 1)[-1]
            for n in range(int(first), int(last) + 1):
                out.add(f"{head}.{n}" if head else str(n))
        else:
            out.add(part)
    return out


def test_ids_are_complete():
    ids = [int(r[0][2:]) for r in ROWS]
    assert ids == list(range(1, len(ids) + 1)) and len(ids) >= 45, ids


def test_every_v6_figure_has_exactly_one_row():
    owners = {}
    for row in ROWS:
        for n in re.findall(r"\d+", row[3]):
            owners.setdefault(int(n), []).append(row[0])
    for fig in V6_FIGURES:
        assert len(owners.get(fig, [])) == 1, (fig, owners.get(fig))


def test_every_v6_section_is_covered():
    covered = set()
    for row in ROWS:
        covered |= _expand_sections(row[1])
    # a whole chapter row ("5", "13.1–13.6") covers its subsections
    missing = [s for s in V6_SECTIONS if s not in covered and s.split(".")[0] not in covered]
    assert not missing, missing


def test_every_row_names_known_stages():
    for row in ROWS:
        for stage in re.split(r"[,–]\s*|\s+", row[5].strip()):
            if stage:
                assert stage in STAGES or stage.isdigit(), (row[0], stage)


def test_the_six_v6_models_stay_in_scope():
    d08 = next(line for line in MASTER.splitlines() if line.startswith("| D-08 |"))
    for model in MODELS:
        assert model in d08, model
        assert model in REGISTER, model

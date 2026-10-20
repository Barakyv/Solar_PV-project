"""T-32: the literature ideas stay consistent (docs/literature_review.md, PROJECT_MASTER.md §16b).

Fails when an idea L-xx exists in one document but not the other, when the register misses an
ID, or when a stage pointer in §7 names an idea that does not exist.
"""

import re

from solarpv.config.paths import REPO_ROOT

REVIEW = (REPO_ROOT / "docs" / "literature_review.md").read_text(encoding="utf-8")
MASTER = (REPO_ROOT / "PROJECT_MASTER.md").read_text(encoding="utf-8")
SECTION = MASTER.split("## 16b.")[1].split("## 17.")[0]


def defined(text: str) -> list[int]:
    """IDs defined as table rows (| L-xx |)."""
    return [int(n) for n in re.findall(r"^\| L-(\d{2}) \|", text, flags=re.MULTILINE)]


def test_register_is_complete_and_in_order():
    ids = defined(SECTION)
    assert ids == list(range(1, len(ids) + 1)) and len(ids) >= 20, ids


def test_review_and_register_define_the_same_ideas():
    assert sorted(defined(REVIEW)) == sorted(defined(SECTION))


def test_stage_pointers_name_existing_ideas():
    known = set(defined(SECTION))
    for line in MASTER.splitlines():
        if line.startswith("> **מהספרות (§16b):**"):
            for n in re.findall(r"L-(\d{2})", line):
                assert int(n) in known, line


def test_every_core_paper_has_a_doi_or_arxiv_id():
    core = re.findall(r"^### (C\d)\. .*$", REVIEW, flags=re.MULTILINE)
    assert len(core) >= 3  # the user asked for at least three papers read in depth
    for block in REVIEW.split("\n### ")[1:]:
        if re.match(r"C\d\.", block):
            assert re.search(r"doi:10\.\S+|arXiv:\d{4}\.\d{4,5}", block), block[:60]

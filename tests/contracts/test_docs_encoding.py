"""Project documents are clean UTF-8 (PROJECT_MASTER.md v1.8: a cp1252 round trip once turned
the Hebrew text into mojibake)."""

import re

import pytest

from solarpv.config.paths import REPO_ROOT

DOCS = ["PROJECT_MASTER.md", "CLAUDE.md", "README.md", "tests/README.md",
        *[p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "docs").glob("*.md")]]
# UTF-8 bytes of Hebrew letters or typographic punctuation read as cp1252
MOJIBAKE = re.compile("â€|×[\u0080-¿ŒœŠšŸ"
                      "Žžƒˆ˜–-™]")


@pytest.mark.parametrize("name", DOCS)
def test_document_is_clean_utf8(name):
    path = REPO_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    text = path.read_bytes().decode("utf-8")  # raises on invalid UTF-8
    hits = MOJIBAKE.findall(text)
    assert not hits, f"{name}: {len(hits)} mojibake sequences, e.g. {hits[:3]}"

"""Phase 2 acceptance tests (PLAN.md section 2.9) plus boilerplate pattern tests.

Corpus tests read data/parsed/*.json (run `make parse` first). data/raw is
gitignored, so they skip where the corpus isn't available.
"""

import re

import pytest
from rapidfuzz import fuzz

from rag.config import settings
from rag.manifest import load_manifest
from rag.parse.clean import BOILERPLATE_PATTERNS, join_lines
from rag.parse.math import math_density
from rag.schemas import ParsedDoc

PAPERS = [p.id for p in load_manifest()]
HAVE_PARSED = all((settings.parsed_dir / f"{pid}.json").exists() for pid in PAPERS)
corpus = pytest.mark.skipif(not HAVE_PARSED, reason="run `make parse` first")


def load(pid: str) -> ParsedDoc:
    return ParsedDoc.model_validate_json(
        (settings.parsed_dir / f"{pid}.json").read_text(encoding="utf-8")
    )


def indexed_blocks(doc: ParsedDoc):
    return [b for s in doc.sections for b in s.blocks]


def indexed_text(doc: ParsedDoc) -> str:
    return "\n".join(b.text for b in indexed_blocks(doc))


# --- boilerplate patterns: one case each -----------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "6314, 2018, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2018/4762485",
        "Electronic copy available at: https://ssrn.com/abstract=3414134",
        "Volume 55, Issue 6, June 2025, Pages 1502-1511",
        "IAENG International Journal of Applied Mathematics",
        "30 September 2025 12:24:34",
        "030005-3",
        "a1111111111",
        "<Insert Figure 1>",
        "Publisher’s Note: MDPI stays neutral with regard to jurisdictional claims",
        "Attribution (CC BY) license. Creative Commons",
        "Copyright © 2018 Meng-Rong Li et al.",
        "This article is licensed under a Creative Commons Attribution 4.0",
        "______________________________________",
        "IJRBS VOL 12 NO 6 (2023) ISSN: 2147-4478",
        "This is an open access article distributed under the terms",
    ],
)
def test_boilerplate_pattern_matches(line: str) -> None:
    assert any(p.search(line) for p in BOILERPLATE_PATTERNS), line


@pytest.mark.parametrize(
    "line",
    ["The volatility of oil price increased in 2020.", "Table 3 reports the MAPE values."],
)
def test_boilerplate_pattern_spares_prose(line: str) -> None:
    assert not any(p.search(line) for p in BOILERPLATE_PATTERNS)


def test_hyphenation_uses_document_vocabulary() -> None:
    from collections import Counter

    vocab = Counter({"forecasting": 3, "second": 4, "order": 5})
    assert (
        join_lines(["stock price fore-", "casting works"], vocab) == "stock price forecasting works"
    )
    assert join_lines(["a second-", "order equation"], vocab) == "a second-order equation"


def test_math_density_flags_greek_heavy_text() -> None:
    assert math_density("dS/dt = βS + αS²") > 0.25
    assert math_density("The logistic model fits the data well.") < 0.05


# --- plan section 2.9 --------------------------------------------------------------


@corpus
def test_p01_no_wiley_stamp() -> None:
    assert "onlinelibrary.wiley.com" not in indexed_text(load("p01"))


@corpus
def test_p10_left_column_before_right_column() -> None:
    blocks = [b for b in indexed_blocks(load("p10")) if b.page == 2]
    text = " ".join(" ".join(b.text.split()) for b in blocks)
    left = text.find("less for volatility forecasting purposes.")
    right = text.find("This study focuses on the 15 individual S&P 500")
    assert left != -1 and right != -1
    assert left < right


@corpus
def test_p12_cover_page_and_page_ids_gone() -> None:
    text = indexed_text(load("p12"))
    assert "Articles You May Be Interested In" not in text
    assert "030005-" not in text


@corpus
def test_p08_ligatures_recovered() -> None:
    text = indexed_text(load("p08"))
    assert len(re.findall(r"\bfinancial\b", text, re.I)) >= 5
    assert "fnancial" not in text


@corpus
def test_p18_no_consecutive_near_duplicate_lines() -> None:
    blocks = [b.text for b in indexed_blocks(load("p18"))]
    for a, b in zip(blocks, blocks[1:], strict=False):
        if len(a) >= 15:
            assert fuzz.ratio(a, b) <= 95, a[:80]


@corpus
@pytest.mark.parametrize("pid", PAPERS)
def test_references_extracted_and_not_indexed(pid: str) -> None:
    doc = load(pid)
    assert doc.references, "no references extracted"
    assert all(s.label != "references" for s in doc.sections)
    for s in doc.sections:
        assert not re.search(r"^\s*References\s*$", s.heading, re.I)
        for b in s.blocks:
            assert not re.fullmatch(r"\s*(References|REFERENCES)\s*", b.text)


@corpus
@pytest.mark.parametrize("pid", PAPERS)
def test_has_abstract_and_conclusion(pid: str) -> None:
    labels = {s.label for s in load(pid).sections}
    assert "abstract" in labels
    assert "conclusion" in labels


@corpus
@pytest.mark.parametrize("pid", PAPERS)
def test_no_backmatter_leaks_into_index(pid: str) -> None:
    text = indexed_text(load(pid))
    for lead in ["Author Contributions:", "Conflicts of Interest:", "Data Availability Statement"]:
        assert lead.lower() not in text.lower(), lead

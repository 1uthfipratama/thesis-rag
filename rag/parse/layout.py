"""Sidebar and column detection, reading order within a page."""

import re
from collections import Counter

from rag.parse.extract import Page, RawBlock

SIDEBAR_MAX_X1 = 0.35  # of page width; MDPI/PLOS/Frontiers sidebars end at 0.28-0.33
SIDEBAR_PAGES = 2  # sidebars only appear on the first 1-2 pages
SIDEBAR_KEYWORDS = re.compile(
    r"citation|received|accepted|published|copyright|academic editor|edited by|"
    r"reviewed by|open access|competing interests|funding|data availability|licensee",
    re.I,
)
FOOTNOTE_START = re.compile(r"^\s*[\d*†‡§]{1,2}\s*\S")
MIN_COLUMN_SHARE = 0.15


def _substantial(b: RawBlock) -> bool:
    # Equation debris and stray labels shouldn't decide whether a page has columns.
    return len(b.lines) >= 2 or len(b.text) >= 40


def drop_sidebar(page: Page, dropped: Counter[str]) -> bool:
    """Drop the metadata column left of the body text on the first pages."""
    if page.number > SIDEBAR_PAGES:
        return False
    # Body text starts at the most common left edge of the wide paragraphs,
    # weighted by lines (MDPI x=166, PLOS x=200). Not the minimum: p06 has a
    # footer and an "Article" label merged into the title block starting at x=35.
    wide = [b for b in page.blocks if b.bbox[2] >= SIDEBAR_MAX_X1 * page.width and _substantial(b)]
    if not wide:
        return False
    edges: Counter[int] = Counter()
    for b in wide:
        edges[round(b.bbox[0] / 4) * 4] += len(b.lines)
    body_x0 = edges.most_common(1)[0][0]
    side = [b for b in page.blocks if b.bbox[2] <= body_x0 + 2]
    if not side or not SIDEBAR_KEYWORDS.search(" ".join(b.text for b in side)):
        return False
    for b in side:
        b.kind = "sidebar"
    dropped["sidebar_block"] += len(side)
    page.blocks = [b for b in page.blocks if b.kind != "sidebar"]
    return True


def mark_footnotes(page: Page, body_size: float) -> None:
    """Small-font numbered notes in the lower part of the page (p10)."""
    for b in page.blocks:
        if (
            b.size <= body_size - 0.5
            and b.bbox[1] > 0.6 * page.height
            and FOOTNOTE_START.match(b.text)
            and max(len(ln.text) for ln in b.lines) >= 30
            and b.kind == "text"
        ):
            b.kind = "footnote"


def order_page(page: Page) -> str:
    """Sort page.blocks into reading order; return "one_column" or "two_column".

    Two-column pages are cut into horizontal bands by full-width blocks (title,
    abstract, wide tables, figure captions). Within a band: left column top-down,
    then right column top-down. Footnotes go last so they don't split sentences.
    """
    notes = [b for b in page.blocks if b.kind == "footnote"]
    blocks = [b for b in page.blocks if b.kind != "footnote"]
    if not blocks:
        page.blocks = notes
        return "one_column"
    x0 = min(b.bbox[0] for b in blocks)
    x1 = max(b.bbox[2] for b in blocks)
    mid = (x0 + x1) / 2
    area_w = x1 - x0

    subst = [b for b in blocks if _substantial(b)]
    left_n = sum(1 for b in subst if b.bbox[2] <= mid + 10)
    right_n = sum(1 for b in subst if b.bbox[0] >= mid - 10)
    two_col = (
        len(subst) > 0
        and left_n >= max(1, MIN_COLUMN_SHARE * len(subst))
        and right_n >= max(1, MIN_COLUMN_SHARE * len(subst))
    )
    if not two_col:
        # Round y so fragments on the same visual line sort left-to-right.
        blocks.sort(key=lambda b: (round(b.bbox[1] / 3), b.bbox[0]))
        page.blocks = blocks + notes
        return "one_column"

    full = sorted(
        [b for b in blocks if b.bbox[2] - b.bbox[0] > 0.6 * area_w],
        key=lambda b: b.bbox[1],
    )
    cols = [b for b in blocks if b not in full]
    ordered: list[RawBlock] = []
    band_top = -1.0
    # Band edges sit at full-width block centres, so every column block lands in
    # exactly one band.
    for fw in [*full, None]:
        band_bottom = (fw.bbox[1] + fw.bbox[3]) / 2 if fw is not None else float("inf")
        in_band = [b for b in cols if band_top <= (b.bbox[1] + b.bbox[3]) / 2 < band_bottom]
        left = [b for b in in_band if (b.bbox[0] + b.bbox[2]) / 2 < mid]
        right = [b for b in in_band if (b.bbox[0] + b.bbox[2]) / 2 >= mid]
        ordered += sorted(left, key=lambda b: (b.bbox[1], b.bbox[0]))
        ordered += sorted(right, key=lambda b: (b.bbox[1], b.bbox[0]))
        if fw is not None:
            ordered.append(fw)
            band_top = (fw.bbox[1] + fw.bbox[3]) / 2
    page.blocks = ordered + notes
    return "two_column"

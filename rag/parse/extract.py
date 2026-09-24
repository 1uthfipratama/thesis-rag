"""PyMuPDF -> lines and blocks with bbox, font size and bold flag."""

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

BOLD_FLAG = 16  # PyMuPDF span flag bit for bold
MIN_FONT_SIZE = 4.0  # p17 has 1pt invisible text (a SAGE tracking string) on page 1

# Private-use glyphs (BMP and supplementary planes) are bracket/radical pieces
# from math fonts (p03, p04; p01 uses U+F5128).
_PUA = re.compile("[\ue000-\uf8ff\U000f0000-\U0010ffff]")
_CTRL = re.compile("[\x00-\x08\x0b-\x1f\x7f]")


@dataclass
class Line:
    text: str  # NFKC-normalized, what gets indexed
    raw: str  # as extracted; math scoring needs pre-NFKC codepoints (NFKC folds 𝑥 -> x)
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool


@dataclass
class RawBlock:
    page: int
    bbox: tuple[float, float, float, float]
    lines: list[Line]
    kind: str = "text"
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def raw(self) -> str:
        return "\n".join(ln.raw for ln in self.lines)

    @property
    def size(self) -> float:
        # char-weighted dominant size, so one big drop-cap doesn't move it
        c: Counter[float] = Counter()
        for ln in self.lines:
            c[round(ln.size, 1)] += max(1, len(ln.text))
        return c.most_common(1)[0][0] if c else 0.0

    @property
    def bold(self) -> bool:
        n = sum(len(ln.text) for ln in self.lines)
        b = sum(len(ln.text) for ln in self.lines if ln.bold)
        return n > 0 and b / n > 0.5

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def refresh_bbox(self) -> None:
        xs0, ys0, xs1, ys1 = zip(*(ln.bbox for ln in self.lines), strict=True)
        self.bbox = (min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    blocks: list[RawBlock]
    images: list[tuple[float, float, float, float]] = field(default_factory=list)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ﬁ -> fi, ﬀ -> ff, full-width digits
    text = text.replace("\u00ad", "")  # soft hyphen U+00AD, e.g. inside p17's page range "1–8"
    text = _PUA.sub("", text)
    # Control characters from symbol fonts (p07 renders Δ-brackets as /).
    text = _CTRL.sub("", text)
    return text


def extract(path: Path, drop_pages: list[int], dropped: Counter[str]) -> list[Page]:
    doc = pymupdf.open(path)
    pages: list[Page] = []
    for i, page in enumerate(doc, start=1):
        if i in drop_pages:
            dropped["drop_page"] += 1
            continue
        blocks: list[RawBlock] = []
        images: list[tuple[float, float, float, float]] = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:  # image
                x0, y0, x1, y1 = b["bbox"]
                if x1 > x0 and y1 > y0:  # p04 has a degenerate logo bbox
                    images.append((x0, y0, x1, y1))
                continue
            lines: list[Line] = []
            for ln in b["lines"]:
                spans = [s for s in ln["spans"] if s["text"]]
                if not spans:
                    continue
                raw = "".join(s["text"] for s in spans)
                if not raw.strip():
                    continue
                # Vertical text is stamps and chart axis labels in this corpus
                # (Wiley download stamp p01, AIP timestamp p12, axis titles p14/p17).
                if tuple(round(x) for x in ln["dir"]) != (1, 0):
                    dropped["rotated_line"] += 1
                    continue
                size = statistics.median(s["size"] for s in spans)
                if size < MIN_FONT_SIZE:
                    dropped["hidden_text"] += 1
                    continue
                nchars = sum(len(s["text"]) for s in spans)
                nbold = sum(len(s["text"]) for s in spans if s["flags"] & BOLD_FLAG)
                lines.append(
                    Line(
                        text=normalize(raw),
                        raw=raw,
                        bbox=tuple(ln["bbox"]),
                        size=size,
                        bold=nbold / nchars > 0.5,
                    )
                )
            lines = [ln for ln in lines if ln.text.strip()]
            if lines:
                blk = RawBlock(page=i, bbox=tuple(b["bbox"]), lines=lines)
                blk.refresh_bbox()
                blocks.append(blk)
        pages.append(Page(i, page.rect.width, page.rect.height, blocks, images))
    return pages


def body_font_size(pages: list[Page]) -> float:
    """Dominant font size of prose.

    Measured only on multi-line blocks that are mostly letters: p04's most common
    size overall is its 7pt table text, not its 9pt body.
    """
    c: Counter[float] = Counter()
    for p in pages:
        for b in p.blocks:
            t = b.text
            alpha = sum(ch.isalpha() for ch in t)
            if len(b.lines) >= 3 and alpha > 0.6 * max(1, len(t.replace(" ", ""))):
                for ln in b.lines:
                    c[round(ln.size, 1)] += len(ln.text)
    return c.most_common(1)[0][0] if c else 10.0

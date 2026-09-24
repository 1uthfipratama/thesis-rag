"""Shared glue for ML parser backends (MinerU, Marker).

Each adapter converts its tool's output into ExtBlocks per page. From there the
path is common: our boilerplate/licence/near-duplicate cleanup, then bylines and
back-matter lead-ins, then the same section assembly the PyMuPDF backend uses.
"""

import html
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from rapidfuzz import fuzz

from rag.manifest import Paper
from rag.parse import clean
from rag.parse.extract import Line, Page, RawBlock
from rag.parse.pipeline import BACKMATTER_LEAD, assemble
from rag.parse.tables import FIG_CAPTION, Table
from rag.schemas import ParsedDoc


@dataclass
class ExtBlock:
    kind: str  # text | heading | title | equation | caption | footnote | table
    text: str
    bbox: tuple[float, float, float, float]  # PDF points
    latex: str | None = None  # equations
    rows: list[list[str]] = field(default_factory=list)  # tables
    caption: str = ""  # tables


class TableHTML(HTMLParser):
    """<table> HTML -> rows of cell text. colspan is expanded so columns line up."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None
        self._span = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []
            self._span = int(dict(attrs).get("colspan", "1") or 1)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = " ".join(html.unescape("".join(self._cell)).split())
            if not self.rows:
                self.rows.append([])
            self.rows[-1].extend([text] + [""] * (self._span - 1))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_rows(content: str) -> list[list[str]]:
    p = TableHTML()
    p.feed(content)
    return [r for r in p.rows if r]


def assemble_external(
    paper: Paper,
    pdf_path: Path,
    pages_in: list[tuple[int, float, float, list[ExtBlock]]],
    references: list[str],
    dropped: Counter[str],
    backend: str,
) -> ParsedDoc:
    """pages_in: (page number, width, height, blocks in the tool's reading order)."""
    pages: list[Page] = []
    tables: list[Table] = []
    for number, w, h, ext in pages_in:
        if number in paper.drop_pages:
            dropped["drop_page"] += 1
            continue
        blocks: list[RawBlock] = []
        for e in ext:
            if e.kind == "table":
                if len(e.rows) < 2:
                    dropped["table_unparsed"] += 1
                    continue
                tables.append(Table(number, e.bbox, e.caption, e.rows))
                marker = RawBlock(number, e.bbox, [Line("[table]", "[table]", e.bbox, 10.0, False)])
                marker.kind = "table"
                marker.meta["table"] = len(tables) - 1
                blocks.append(marker)
                continue
            text = e.text.strip() if e.kind != "equation" else "[equation]"
            lines = [
                Line(t, t, e.bbox, 10.0, e.kind == "heading") for t in text.split("\n") if t.strip()
            ]
            if not lines:
                continue
            meta = {"latex": [e.latex]} if e.kind == "equation" and e.latex else {}
            kind = e.kind if (e.kind != "equation" or meta) else "equation"
            blocks.append(RawBlock(number, e.bbox, lines, kind=kind, meta=meta))
        pages.append(Page(number, w, h, blocks))

    # Our cleanup still runs on top of the tool's own header/footer tagging:
    # journal boilerplate patterns, licence notices, repeated lines, duplicates.
    clean.remove_boilerplate(pages, dropped)
    clean.remove_near_duplicates(pages, dropped)
    vocab = clean.build_vocab(pages)
    stream = [b for p in pages for b in p.blocks]
    _classify(stream, paper)
    layouts = {p.number: backend for p in pages}
    doc = assemble(paper, stream, tables, vocab, layouts, dropped, backend)
    # Prefer the tool's own reference entries only when they're at least as
    # complete as our split (Marker's "Reference" blocks can be bare anchors).
    if len(references) >= len(doc.references):
        doc.references = references
    return doc


def _classify(stream: list[RawBlock], paper: Paper) -> None:
    """Rules the ML tools don't cover: bylines, back-matter lead-ins, captions."""
    surnames = [a.split()[-1] for a in paper.authors if len(a.split()[-1]) > 2]
    first_page = stream[0].page if stream else 1
    seen_heading = False
    for b in stream:
        flat = " ".join(b.text.split())
        if (
            b.kind == "heading"
            and b.page <= first_page + 1
            and (fuzz.partial_ratio(flat.lower(), paper.title.lower()) > 90 and len(flat) > 15)
        ):
            b.kind = "title"  # Marker tags the paper title as an <h1> section header
            continue
        if b.kind == "heading":
            seen_heading = True
            if BACKMATTER_LEAD.match(flat) and len(flat.split()) > 6:
                b.kind = "backmatter"  # "Funding: This research..." tagged as a title
            continue
        if b.kind != "text":
            continue
        n_names = sum(s in flat for s in surnames)
        if not seen_heading and b.page <= first_page + 1 and n_names >= 1 and b.n_words <= 40:
            b.kind = "byline"
        elif BACKMATTER_LEAD.match(flat):
            b.kind = "backmatter"
        elif FIG_CAPTION.match(flat):
            b.kind = "caption"

"""Tables -> Markdown.

pdfplumber's default `find_tables()` only sees fully ruled tables. Most tables in
this corpus are booktabs style (horizontal rules only), and it finds nothing in
p01, p02, p10, p13, p16, p17, p18. So detection is anchored on captions instead:

1. A block starting "Table N" is a caption.
2. The table region is the run of blocks below it (same column span) up to the
   first prose paragraph, heading, figure caption or image.
3. pdfplumber extracts that cropped region if it finds a ruled grid (>= 3 rows);
   otherwise rows/columns are rebuilt from word positions.

Uncaptioned ruled tables that pdfplumber finds on its own are kept as well.
"""

import re
import statistics
from dataclasses import dataclass

import pdfplumber
import pymupdf

from rag.parse.extract import Page, RawBlock, normalize
from rag.parse.sections import is_heading

# A caption is "Table 3." / "Table 3:" / "TABLE III" / "Table 1 Parameter settings";
# "Table 4 gives information..." is prose (p11), so a lower-case word can't follow.
# "Table Cont'd" / "Table 6. Cont." start continuation pages (p19, p18).
TABLE_CAPTION = re.compile(
    r"^\s*(Table|TABLE)\s+(([\dIVX]+|[A-Z]\d+)(\s*[.:]|\s*$|\s+[A-Z(“\"])|Cont[’'`]?d\b|continued\b)"
)
FIG_CAPTION = re.compile(r"^\s*(Fig\.?|Figure|FIGURE)\s*[\dIVX]+")
NOTE_LINE = re.compile(r"^\s*(Note|Notes|Source|Sources|\*)", re.I)


@dataclass
class Table:
    page: int
    bbox: tuple[float, float, float, float]
    caption: str
    rows: list[list[str]]

    @property
    def markdown(self) -> str:
        return to_markdown(self.rows)


def _is_prose(b: RawBlock, body_size: float) -> bool:
    """A paragraph of running text: long lines of mostly letters, body size."""
    t = b.text
    chars = t.replace(" ", "").replace("\n", "")
    if not chars:
        return False
    alpha = sum(c.isalpha() for c in chars) / len(chars)
    avg_len = statistics.mean(len(ln.text) for ln in b.lines)
    words = t.split()
    if NOTE_LINE.match(t):
        return False
    return alpha > 0.72 and avg_len > 38 and len(words) >= 12 and b.size >= body_size - 1.0


def _cells_from_words(page: pymupdf.Page, region: tuple) -> list[list[str]]:
    """Rebuild rows/columns from word positions inside region."""
    words = [w for w in page.get_text("words") if _center_in(w[:4], region)]
    if not words:
        return []
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    for w in words:
        yc, h = (w[1] + w[3]) / 2, w[3] - w[1]
        if rows:
            ryc = statistics.mean((x[1] + x[3]) / 2 for x in rows[-1])
            if abs(yc - ryc) < 0.5 * h:
                rows[-1].append(w)
                continue
        rows.append([w])
    # Split each row into cells at gaps wider than ~0.8 em.
    cell_rows: list[list[tuple[float, float, str]]] = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        cells: list[list] = [[r[0]]]
        for prev, w in zip(r, r[1:], strict=False):
            em = w[3] - w[1]
            if w[0] - prev[2] > 0.8 * em:
                cells.append([w])
            else:
                cells[-1].append(w)
        cell_rows.append([(c[0][0], c[-1][2], " ".join(x[4] for x in c)) for c in cells])
    # Column anchors come from the row with the most cells; every other cell goes
    # to the nearest anchor, so empty cells don't shift values left.
    ref = max(cell_rows, key=len)
    anchors = [(c[0] + c[1]) / 2 for c in ref]
    out: list[list[str]] = []
    for cr in cell_rows:
        row = [""] * len(anchors)
        for x0, x1, text in cr:
            j = min(range(len(anchors)), key=lambda k: abs(anchors[k] - (x0 + x1) / 2))
            row[j] = f"{row[j]} {text}".strip()
        out.append(row)
    return out


def _center_in(bbox: tuple, region: tuple) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return region[0] - 1 <= cx <= region[2] + 1 and region[1] - 1 <= cy <= region[3] + 1


def _plumber_rows(ppage: pdfplumber.page.Page, region: tuple) -> list[list[str]]:
    try:
        crop = ppage.crop(region, strict=False)
        tables = crop.extract_tables()
    except ValueError:
        return []
    best = max(tables, key=len, default=[])
    return [[(c or "").replace("\n", " ").strip() for c in r] for r in best]


def _has_number(rows: list[list[str]]) -> bool:
    return any(re.search(r"\d", c) for r in rows for c in r)


def to_markdown(rows: list[list[str]]) -> str:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    # Same NFKC/ligature/private-use cleanup as prose; "|" would break the Markdown row.
    rows = [[normalize(c).replace("|", "/") for c in r] + [""] * (n - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def find_tables(
    pdf_path, pages: list[Page], body_size: float
) -> tuple[list[Table], dict[int, list[tuple]]]:
    """Return tables and, per page, the regions whose text must leave the prose stream."""
    doc = pymupdf.open(pdf_path)
    tables: list[Table] = []
    regions: dict[int, list[tuple]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pages:
            mpage = doc[pg.number - 1]
            # dedupe_chars: some PDFs (p18) draw characters twice for fake bold
            ppage = pdf.pages[pg.number - 1].dedupe_chars()
            images = [b["bbox"] for b in mpage.get_text("dict")["blocks"] if b["type"] == 1]
            blocks = sorted(pg.blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
            page_regions: list[tuple] = []
            for cap in blocks:
                if not TABLE_CAPTION.match(cap.text):
                    continue
                n_cap = _caption_line_count(cap)
                caption = " ".join(" ".join(ln.text for ln in cap.lines[:n_cap]).split())
                region = _region_below(cap, n_cap, blocks, images, body_size, pg.width)
                if region is None:
                    continue
                rows = _plumber_rows(ppage, region)
                if len(rows) < 3 or not _has_number(rows):
                    rows = _cells_from_words(mpage, region)
                if len(rows) < 2 or max(len(r) for r in rows) < 2 or not _has_number(rows):
                    continue
                full = (
                    min(cap.bbox[0], region[0]),
                    cap.bbox[1],
                    max(cap.bbox[2], region[2]),
                    region[3],
                )
                tables.append(Table(pg.number, full, caption, rows))
                page_regions.append(full)
                cap.kind = "table_caption"
            # Ruled tables without a detectable caption (e.g. p14 p.21).
            for t in ppage.find_tables():
                if any(_overlap(t.bbox, r) for r in page_regions):
                    continue
                rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
                if len(rows) >= 3 and max(len(r) for r in rows) >= 2 and _has_number(rows):
                    cap = _nearby_caption(t.bbox, blocks)
                    caption = " ".join(cap.lines[0].text.split()) if cap else ""
                    box = tuple(t.bbox)
                    if cap is not None:
                        cap.kind = "table_caption"
                        box = (
                            min(box[0], cap.bbox[0]),
                            min(box[1], cap.bbox[1]),
                            max(box[2], cap.bbox[2]),
                            max(box[3], cap.bbox[3]),
                        )
                    tables.append(Table(pg.number, box, caption, rows))
                    page_regions.append(box)
            regions[pg.number] = page_regions
    return tables, regions


def _nearby_caption(bbox: tuple, blocks: list[RawBlock]) -> RawBlock | None:
    """Caption within ~40pt above or below a ruled table (p12 puts some below)."""
    best, dist = None, 40.0
    for b in blocks:
        if b.kind != "text" or not TABLE_CAPTION.match(b.text):
            continue
        d = min(abs(bbox[1] - b.bbox[3]), abs(b.bbox[1] - bbox[3]))
        if d < dist:
            best, dist = b, d
    return best


def _caption_line_count(cap: RawBlock) -> int:
    """How many leading lines of the caption block are caption, not table.

    PyMuPDF often merges a caption with the table under it into one block (p07).
    The caption is the first line, plus the next one when the first is only the
    label ("TABLE II" / "MAPE AND ACCURACY...", IEEE style) or when the sentence
    visibly wraps (next line starts lower-case).
    """
    lines = cap.lines
    if len(lines) == 1:
        return 1
    first = lines[0].text.strip()
    nxt = lines[1].text.strip()
    if re.fullmatch(r"(Table|TABLE)\s+\S+[.:]?", first) or (nxt[:1].islower() and len(first) > 40):
        return 2
    return 1


def _region_below(cap, n_cap, blocks, images, body_size, page_width):
    """Table body: the rest of the caption block plus the blocks under it, within
    the caption's column, until prose, a figure, another caption or a big gap."""
    half = page_width / 2
    if cap.bbox[2] < half + 10:
        xr = (0, half + 10)
    elif cap.bbox[0] > half - 10:
        xr = (half - 10, page_width)
    else:
        xr = (0, page_width)
    top = cap.lines[n_cap - 1].bbox[3]
    seed = cap.lines[n_cap:]
    members: list[tuple] = [ln.bbox for ln in seed]
    last_y1 = max((b[3] for b in members), default=top)
    for b in blocks:
        if b is cap or b.bbox[1] < top - 2:
            continue
        if b.bbox[2] < xr[0] or b.bbox[0] > xr[1]:
            continue
        if TABLE_CAPTION.match(b.text) or FIG_CAPTION.match(b.text) or _is_prose(b, body_size):
            break
        if is_heading(b, body_size):
            break  # "IV. CONCLUSION" right under Table IV (p03)
        if any(_overlap(b.bbox, im) for im in images):
            break
        if b.bbox[1] - last_y1 > 3 * body_size:
            break  # large vertical gap: table ended
        members.append(b.bbox)
        last_y1 = max(last_y1, b.bbox[3])
    if not members:
        return None
    return (
        min(m[0] for m in members),
        top,
        max(m[2] for m in members),
        max(m[3] for m in members),
    )


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

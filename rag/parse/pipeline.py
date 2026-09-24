"""One PDF -> ParsedDoc. Same code path for core papers and user uploads."""

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from rapidfuzz import fuzz

from rag.manifest import Paper
from rag.parse import clean, layout
from rag.parse.extract import Line, RawBlock, body_font_size, extract
from rag.parse.latex import tidy
from rag.parse.math import is_equation, placeholder, split_mixed
from rag.parse.sections import (
    CAPTION,
    RawSection,
    despace,
    heading_level,
    is_heading,
    label_for,
    split_abstract_prefix,
    split_leading_heading,
)
from rag.parse.tables import FIG_CAPTION, find_tables
from rag.schemas import Block, ParsedDoc, Section, TableBlock

ABSTRACT_MIN_WORDS = 60
# MDPI/Springer/PLOS back matter is often bold lead-in paragraphs inside the last
# section, not headings (p18 "Funding: ...", "Data Availability Statement: ...").
BACKMATTER_LEAD = re.compile(
    r"^(Author Contributions?|Authors[’'] contributions|Funding|Institutional Review Board "
    r"Statement|Informed Consent Statement|Data Availability Statement|Data Availability|"
    r"Availability of data and materials|Conflicts? of Interest|Competing interests|"
    r"Acknowledge?ments?|Abbreviations|Supplementary Materials|Supporting information|"
    r"Declarations?|Ethics approval[^:]*|Consent for publication|Author details|"
    r"Publisher[’']s Note)\s*[:.]",
    re.I,
)
STRONG_LABEL = re.compile(
    r"\b(introduction|literature|related work|data|dataset|methods?|methodology|results?|"
    r"discussion|conclusions?)\b",
    re.I,
)
DATA_AVAIL = re.compile(r"^(Data Availability|Availability of data)", re.I)
DROP_LABELS = {"frontmatter", "backmatter", "references"}


def _center_in(bbox, box) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


MathLookup = Callable[[int, tuple[float, float, float, float]], list[str] | None]


def parse_pdf(
    paper: Paper, pdf_path: Path, math: MathLookup | None = None, backend: str = "pymupdf"
) -> ParsedDoc:
    """PyMuPDF backend. With `math`, equation placeholders get LaTeX (hybrid)."""
    dropped: Counter[str] = Counter()
    pages = extract(pdf_path, paper.drop_pages, dropped)
    clean.remove_boilerplate(pages, dropped)
    body = body_font_size(pages)

    # Tables first, so their cells never reach the prose stream.
    tables, regions = find_tables(pdf_path, pages, body)
    for pg in pages:
        regs = regions.get(pg.number, [])
        keep = []
        for b in pg.blocks:
            if any(_center_in(b.bbox, r) for r in regs):
                dropped["table_text_block"] += 1
            elif any(_center_in(b.bbox, im) for im in pg.images) and not CAPTION.match(b.text):
                dropped["text_inside_figure"] += 1
            else:
                keep.append(b)
        # A marker holds each table's place in reading order, so the table can be
        # assigned to the section it appears in.
        for k, t in enumerate(tables):
            if t.page == pg.number:
                marker = RawBlock(
                    pg.number, t.bbox, [Line("[table]", "[table]", t.bbox, body, False)]
                )
                marker.kind = "table"
                marker.meta["table"] = k
                keep.append(marker)
        pg.blocks = keep

    page_layouts: dict[int, str] = {}
    has_sidebar = paper.layout.endswith("_sidebar")
    for pg in pages:
        side = has_sidebar and layout.drop_sidebar(pg, dropped)
        layout.mark_footnotes(pg, body)
        pg.blocks = [piece for b in pg.blocks for piece in split_leading_heading(b, body)]
        page_layouts[pg.number] = layout.order_page(pg) + ("+sidebar" if side else "")

    for pg in pages:
        _fix_drop_caps(pg.blocks, body)
    clean.remove_near_duplicates(pages, dropped)
    clean.remove_figure_labels(pages, dropped)
    vocab = clean.build_vocab(pages)
    stream = [piece for pg in pages for b in pg.blocks for piece in split_mixed(b)]

    first_page = pages[0].number if pages else 1
    # Author bylines are short, large and bold, so they pass as headings (p04).
    surnames = [a.split()[-1] for a in paper.authors if len(a.split()[-1]) > 2]
    for b in stream:
        if b.kind != "text":
            continue
        flat = " ".join(b.text.split())
        on_front_pages = b.page <= first_page + 1
        title_like = len(flat) > 15 or b.size >= 1.4 * body  # "System", p02's last title line
        n_names = sum(s in flat for s in surnames)
        if (
            on_front_pages
            and title_like
            and fuzz.partial_ratio(flat.lower(), paper.title.lower()) > 90
        ):
            b.kind = "title"
        elif on_front_pages and (n_names >= 2 or (n_names == 1 and b.n_words <= 12)):
            b.kind = "byline"  # ACM gives each author a block of their own (p02)
        elif is_heading(b, body):
            b.kind = "heading"
        elif BACKMATTER_LEAD.match(flat):
            b.kind = "backmatter"
        elif FIG_CAPTION.match(flat):
            b.kind = "caption"
        elif is_equation(b):
            b.kind = "equation"

    if math is not None:
        attach_latex(stream, math, dropped)
    return assemble(paper, stream, tables, vocab, page_layouts, dropped, backend)


def assemble(paper, stream, tables, vocab, page_layouts, dropped, backend) -> ParsedDoc:
    """Shared by every backend: sections, labels, references, back matter, tables."""
    raw_sections = _build_sections(stream, dropped)
    doc = _to_parsed(paper, raw_sections, tables, vocab, page_layouts, dropped)
    doc.backend = backend
    return doc


def attach_latex(stream: list[RawBlock], math: "MathLookup", dropped: Counter[str]) -> None:
    """Hybrid mode: give each run of equation fragments the LaTeX a math-aware
    backend recognised in the same page region. Unmatched runs stay placeholders."""
    i = 0
    while i < len(stream):
        if stream[i].kind != "equation":
            i += 1
            continue
        j = i
        while (
            j + 1 < len(stream)
            and stream[j + 1].kind == "equation"
            and stream[j + 1].page == stream[i].page
        ):
            j += 1
        run = stream[i : j + 1]
        box = (
            min(b.bbox[0] for b in run),
            min(b.bbox[1] for b in run),
            max(b.bbox[2] for b in run),
            max(b.bbox[3] for b in run),
        )
        found = math(run[0].page, box)
        if found is None:  # fragment of an equation already matched to an earlier run
            for b in run:
                b.meta["merged"] = True
            dropped["equation_fragment_merged"] += 1
        elif found:
            run[0].meta["latex"] = found
            for b in run[1:]:
                b.meta["merged"] = True
            dropped["equation_latex_matched"] += 1
        else:
            dropped["equation_latex_unmatched"] += 1
        i = j + 1


def _fix_drop_caps(blocks: list[RawBlock], body: float) -> None:
    """IEEE drop cap: a lone oversized capital ("S") before "TOCKS are..." (p05).

    Remove the letter from wherever PyMuPDF put it (often the heading's block)
    and glue it back onto the first word of the next text block.
    """
    for i, b in enumerate(blocks):
        for ln in list(b.lines):
            if re.fullmatch(r"[A-Z]", ln.text.strip()) and ln.size >= 1.5 * body:
                nxt = next((x for x in blocks[i + 1 :] if x.kind == "text" and x.lines), None)
                if nxt is None or not re.match(r"[A-Z]{2,}", nxt.lines[0].text.strip()):
                    continue
                b.lines.remove(ln)
                first = nxt.lines[0]
                first.text = ln.text.strip() + first.text.strip()
                first.raw = ln.raw.strip() + first.raw.strip()
        if b.lines:
            b.refresh_bbox()
    blocks[:] = [b for b in blocks if b.lines]


def _build_sections(stream: list[RawBlock], dropped: Counter[str]) -> list[RawSection]:
    front = RawSection("frontmatter", "", 1, stream[0].page if stream else 1)
    sections = [front]
    top = front  # current level-1 section, for label inheritance
    in_refs = False
    for b in stream:
        cur = sections[-1]
        if b.kind == "heading" and not in_refs:
            text = despace(" ".join(b.text.split()))
            level = heading_level(text, b.bold)
            label = label_for(text)
            # A subsection keeps its own label only when its heading names a
            # section type outright ("3.2 Methodology" under "3. Data and
            # Methodology", p07). Weak keywords inherit the parent: "OLS Models"
            # under "3. Empirical Results" (p14) is results, not methodology.
            if (
                level > 1
                and label not in ("references", "backmatter")
                and (label == "other" or not STRONG_LABEL.search(text))
                and top.label not in ("other", "frontmatter")
            ):
                label = top.label
            sec = RawSection(label, text, level, b.page)
            sections.append(sec)
            if level == 1:
                top = sec
            if label == "references":
                in_refs = True
            continue
        if b.kind in ("title", "byline"):
            dropped[f"{b.kind}_block"] += 1
            continue
        # "Abstract. This research..." with no separate heading (p12, p13, p03)
        if not in_refs and cur.label in ("frontmatter", "other") and b.kind == "text":
            body = split_abstract_prefix(b)
            if body is not None and not any(s.label == "abstract" for s in sections):
                b.lines = [Line(body, body, b.bbox, b.lines[0].size, False)]
                sec = RawSection("abstract", "Abstract", 1, b.page, [b])
                sections.append(sec)
                top = sec
                continue
        # Untitled introduction (p19): abstracts are set in their own font size, so
        # a long body-size paragraph after abstract text starts a new section.
        if (
            cur.label == "abstract"
            and b.kind == "text"
            and b.n_words >= ABSTRACT_MIN_WORDS
            and sum(x.n_words for x in cur.blocks if x.kind == "text") >= ABSTRACT_MIN_WORDS
            and abs(b.size - cur.blocks[0].size) >= 0.5
        ):
            sec = RawSection("introduction", "Introduction (untitled)", 1, b.page, [b])
            sections.append(sec)
            top = sec
            continue
        cur.blocks.append(b)

    # Anything before the abstract that isn't a real section is front matter:
    # article-type labels ("RESEARCH", p08) and mastheads styled like headings.
    abs_idx = next((i for i, s in enumerate(sections) if s.label == "abstract"), None)
    if abs_idx is None:  # no printed abstract: the introduction bounds the front matter
        abs_idx = next((i for i, s in enumerate(sections) if s.label == "introduction"), None)
    if abs_idx is not None:
        for s in sections[:abs_idx]:
            if s.label == "other":
                s.label = "frontmatter"

    # Abstract without any heading (p01, p10, p02): the first long block in the
    # front matter before the introduction, plus its continuation in the next
    # column (a block starting lower-case, p02).
    if not any(s.label == "abstract" for s in sections):
        fronts = []
        for s in sections:
            if s.label != "frontmatter":
                break
            fronts.append(s)
        stream_front = [b for s in fronts for b in s.blocks if b.kind == "text"]
        start = next(
            (i for i, b in enumerate(stream_front) if b.n_words >= ABSTRACT_MIN_WORDS), None
        )
        if start is not None:
            picked = [stream_front[start]]
            for b in stream_front[start + 1 :]:
                if b.n_words >= ABSTRACT_MIN_WORDS or b.text[:1].islower():
                    picked.append(b)
                else:
                    break
            for s in fronts:
                s.blocks = [b for b in s.blocks if b not in picked]
            sections.insert(
                len(fronts), RawSection("abstract", "Abstract", 1, picked[0].page, picked)
            )
    return sections


def _to_parsed(paper, raw_sections, tables, vocab, page_layouts, dropped) -> ParsedDoc:
    out_sections: list[Section] = []
    references: list[str] = []
    data_availability: list[str] = []
    table_blocks = [
        TableBlock(
            page=t.page,
            bbox=t.bbox,
            caption=t.caption,
            markdown=t.markdown,
            n_rows=len(t.rows),
            n_cols=max(len(r) for r in t.rows),
        )
        for t in tables
    ]

    for s in raw_sections:
        # Table markers: record where the table sits, then leave the prose stream.
        for b in s.blocks:
            if b.kind == "table":
                tb = table_blocks[b.meta["table"]]
                tb.section = s.label if s.label not in DROP_LABELS else "other"
                tb.heading = s.heading
        for b in s.blocks:
            if b.kind == "backmatter":
                dropped["backmatter_paragraph"] += 1
                if DATA_AVAIL.match(b.text):
                    data_availability.append(clean.join_lines([ln.text for ln in b.lines], vocab))
        content = [b for b in s.blocks if b.kind not in ("table", "backmatter")]

        if s.label == "references":
            references += _split_references([ln.text for b in content for ln in b.lines])
            continue
        if s.label in ("frontmatter", "backmatter"):
            if DATA_AVAIL.search(s.heading):
                data_availability.append(
                    " ".join(clean.join_lines([ln.text for ln in b.lines], vocab) for b in content)
                )
            dropped[f"{s.label}_block"] += len(content)
            continue

        blocks = _section_blocks(content, vocab, dropped)
        if blocks or s.heading:
            out_sections.append(
                Section(
                    label=s.label,
                    heading=s.heading or s.label.title(),
                    page_start=s.page_start,
                    blocks=blocks,
                )
            )

    return ParsedDoc(
        paper_id=paper.id,
        title=paper.title,
        sections=out_sections,
        tables=table_blocks,
        references=references,
        data_availability=" ".join(data_availability),
        page_layouts=page_layouts,
        dropped=dict(dropped),
    )


def _section_blocks(content: list[RawBlock], vocab, dropped: Counter[str]) -> list[Block]:
    """Prose blocks as joined paragraphs; each run of equation blocks -> one placeholder."""
    blocks: list[Block] = []
    eq_run: list[RawBlock] = []
    for b in [*content, None]:
        if b is not None and b.meta.get("merged"):
            continue  # fragment already covered by its run's LaTeX
        if b is not None and b.kind == "equation" and not b.meta.get("latex"):
            eq_run.append(b)
            continue
        if eq_run:
            dropped["equation_block"] += len(eq_run)
            blocks.append(_block(eq_run[0], placeholder(eq_run), "equation"))
            eq_run = []
        if b is None:
            break
        if b.kind == "equation":
            latex = b.meta["latex"]
            blocks.append(
                # indexed text: tidied LaTeX; Block.latex keeps the recogniser's raw form
                _block(b, " ".join(f"$$ {tidy(x)} $$" for x in latex), "equation").model_copy(
                    update={"latex": "\n".join(latex)}
                )
            )
            continue
        text = clean.join_lines([ln.text for ln in b.lines], vocab)
        kind = b.kind if b.kind in ("caption", "footnote", "heading") else "text"
        blocks.append(_block(b, text, kind))
    return blocks


def _block(b: RawBlock, text: str, kind: str) -> Block:
    return Block(page=b.page, bbox=b.bbox, text=text, font_size=b.size, is_bold=b.bold, kind=kind)


_REF_START = re.compile(
    r"^\s*(\[\d+\]"  # [12]
    r"|\d{1,3}\.\s+[A-Z]"  # 12. Author
    r"|[A-ZÀ-Ž][\w'’\-]+(?: [A-Z][\w'’\-]+)*,\s+(?:[A-Z]\.|[A-Z][a-z]+))"  # Surname, X.
)


def _split_references(lines: list[str]) -> list[str]:
    """Split on [n], 'n.' or 'Surname, X.' line starts. Metadata only, never indexed.

    Numbered labels often sit on their own line ("2." then "M. Ali, ...", p13,
    p09, p11), so a label-only line starts a new reference too. Marker can merge
    several entries into one line ("...2024. [2] M. S. Wilson..."), so lines are
    first cut before every "[n] " label.
    """
    lines = [piece for ln in lines for piece in re.split(r"\s(?=\[\d{1,3}\]\s)", ln)]
    refs: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # "... 79–94. [CrossRef]" (MDPI) also ends a reference
        prev_done = (
            not refs or refs[-1].rstrip().endswith((".", ")", "]")) or re.search(r"\d$", refs[-1])
        )
        if _LABEL_ONLY.match(s):
            refs.append(s)
        elif refs and _LABEL_ONLY.match(refs[-1]):
            refs[-1] += " " + s
        elif _REF_START.match(s) and (prev_done or re.match(r"^(\[\d+\]|\d{1,3}\.\s)", s)):
            refs.append(s)
        elif refs:
            refs[-1] = (refs[-1][:-1] if refs[-1].endswith("-") else refs[-1] + " ") + s
        else:
            refs.append(s)
    return [" ".join(r.split()) for r in refs if len(r) > 10]


_LABEL_ONLY = re.compile(r"^(\[\d{1,3}\]|\d{1,3}\.)$")

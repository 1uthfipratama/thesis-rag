"""Heading detection, canonical section labels, references and back matter."""

import re
from dataclasses import dataclass, field

from rag.parse.extract import RawBlock

HEADING_MAX_WORDS = 12
HEADING_SIZE_RATIO = 1.1
PROMINENT_RATIO = 1.15  # size-only headings need a clearer jump than bold ones

NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+[A-Z]")
ROMAN = re.compile(r"^([IVX]+)\.\s+\S")

# First match wins, in this order (plan section 2.5), with two changes:
# - backmatter is checked early: its phrases are specific, and otherwise "data"
#   claims "Availability of data and materials" (p08);
# - "frontmatter" is added: keyword and article-info boxes before the abstract.
# Matched on word starts, so "data" doesn't fire inside "update".
LABEL_KEYWORDS: list[tuple[str, list[str]]] = [
    ("abstract", ["abstract"]),
    (
        "backmatter",
        [
            "acknowledg",
            "author contribution",
            "funding",
            "conflict",
            "data availability",
            "availability of data",
            "appendix",
            "publisher",
            "competing interest",
            "institutional review",
            "informed consent",
            "declaration",
            "abbreviations",
            "supporting information",
            "authors’ information",
            "author details",
            "ethics",
        ],
    ),
    (
        "frontmatter",
        [
            "keywords",
            "index terms",
            "article info",
            "jel",
            "ccs concepts",
            "acm reference",
            "highlights",
        ],
    ),
    ("introduction", ["introduction"]),
    ("literature_review", ["literature", "related work", "review"]),
    ("data", ["data", "sample", "dataset"]),
    (
        "methodology",
        [
            "method",
            "methodology",
            "model",
            "approach",
            "algorithm",
            "framework",
            "solution",
            "numerical",
        ],
    ),
    ("results", ["result", "simulation", "empirical", "experiment", "forecast", "fit", "analysis"]),
    ("discussion", ["discussion"]),
    ("conclusion", ["conclusion", "concluding"]),
    ("references", ["references", "bibliography", "daftar pustaka"]),
]
_KEYWORD_RES = [
    (label, re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")", re.I))
    for label, kws in LABEL_KEYWORDS
]
# An unnumbered, non-caps heading is only accepted if it starts with one of these,
# e.g. "Introduction", "Results and discussion". Otherwise every bold phrase
# would become a section.
KNOWN_START = re.compile(
    r"^(abstract|introduction|background|literature|related work|data|methods?|methodology|"
    r"materials|model|results?|discussion|conclusions?|concluding|references|bibliography|"
    r"acknowledg|appendix|funding|author contributions?|conflicts? of interest|"
    r"data availability|availability of data|competing interests|declarations?|"
    r"keywords|abbreviations|empirical|findings|summary|limitations)\b",
    re.I,
)
MONTH = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"
)
STRUCTURAL_LABEL = re.compile(
    r"^(abstract|article info|keywords|references|bibliography|acknowledge?ments?|"
    r"data availability( statement)?|conflicts? of interest|funding)\s*:?$",
    re.I,
)
ABSTRACT_PREFIX = re.compile(r"^\s*(Abstract|ABSTRACT)\s*[.:—–-]?\s*")
CAPTION = re.compile(r"^\s*(Table|TABLE|Fig\.?|Figure|FIGURE)\s*[\dIVX]+")


def despace(text: str) -> str:
    """'A B S T R A C T' -> 'ABSTRACT' (Elsevier p10, IJRBS p19)."""
    parts = re.split(r"\s{2,}", text.strip())
    out = []
    for part in parts:
        if re.fullmatch(r"(?:[A-Z] )+[A-Z]", part):
            part = part.replace(" ", "")
        out.append(part)
    return " ".join(out)


def label_for(heading: str) -> str:
    h = re.sub(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s*", "", despace(heading))
    # Deviation from the plan's order: an explicit "result(s)" wins, otherwise
    # "Numerical Results" (p03) lands in methodology via "numerical".
    if re.search(r"\bresults?\b", h, re.I) and not dict(_KEYWORD_RES)["backmatter"].search(h):
        return "results"
    for label, rx in _KEYWORD_RES:
        if rx.search(h):
            return label
    return "other"


def heading_level(text: str, bold: bool = True) -> int:
    """ "2.1 Euler method" -> 2. Unnumbered: bold or caps is top level, plain/italic
    (p17 "The Heston Model") is a subsection."""
    m = NUMBERED.match(text)
    if m:
        return m.group(1).count(".") + 1
    if ROMAN.match(text) or bold or text.isupper() or STRUCTURAL_LABEL.match(text):
        return 1
    return 2


def is_heading(b: RawBlock, body_size: float) -> bool:
    if b.kind != "text":
        return False
    text = despace(" ".join(b.text.split()))
    if re.match(r"^[^:]{2,40}:\s+\S", text):
        return False  # "Funding: This research..." is a lead-in paragraph, not a heading
    if not text or b.n_words > HEADING_MAX_WORDS or len(b.lines) > 3:
        return False
    if CAPTION.match(text) or text.endswith((",", ";")):
        return False
    # Bold table header cells look like headings (p16 "Method Used for the",
    # "31 October 2022"): reject unfinished phrases and dates.
    if re.search(r"\b(the|of|for|and|in|to|a|an|with|on|by)$", text, re.I):
        return False
    m = NUMBERED.match(text)
    if m and (int(m.group(1).split(".")[0]) > 20 or MONTH.match(text[m.end() - 1 :])):
        return False
    # Bare structural labels count whatever their style: p19 sets "A B S T R A C T"
    # in plain 8pt, p03/p05 set "DATA AVAILABILITY STATEMENT" at body size.
    if STRUCTURAL_LABEL.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    all_caps = (
        bool(letters) and all(c.isupper() for c in letters) and bool(re.search(r"[A-Z]{4,}", text))
    )
    # IEEE/IAENG headings ("II. DISCRETE TIME LOGISTIC MODEL") are small caps at
    # body size, neither bold nor larger.
    if ROMAN.match(text) and all_caps:
        return True
    styled = b.size >= HEADING_SIZE_RATIO * body_size or b.bold
    if not styled:
        return False
    if NUMBERED.match(text) or ROMAN.match(text):
        return True
    if all_caps:
        return True
    if KNOWN_START.match(text) and b.n_words <= 6:
        return True
    # Unnumbered journals (SAGE p17: "Mathematical Model", "The Heston Model")
    # mark headings by size alone: one short line, clearly larger than body.
    return (
        len(b.lines) == 1
        and b.size >= PROMINENT_RATIO * body_size
        and 1 <= b.n_words <= 8
        and text[:1].isupper()
        and not text.endswith(".")
        and bool(re.search(r"[A-Za-z]{3,}", text))
    )


def split_leading_heading(b: RawBlock, body_size: float) -> list[RawBlock]:
    """PyMuPDF often puts a heading and its first paragraph in one block (p17)."""
    if len(b.lines) < 2 or b.kind != "text":
        return [b]
    first = b.lines[0]
    rest = b.lines[1:]
    # The first line doesn't need to be bold/large here: IEEE headings
    # ("III. NUMERICAL RESULTS", p03) are body-size small caps. is_heading() below
    # still requires styling, a Roman-numbered caps line, or a structural label.
    rest_plain = all(not ln.bold and ln.size < HEADING_SIZE_RATIO * body_size for ln in rest)
    if rest_plain and len(first.text.split()) <= HEADING_MAX_WORDS:
        head = RawBlock(b.page, first.bbox, [first])
        body = RawBlock(b.page, b.bbox, rest)
        body.refresh_bbox()
        if is_heading(head, body_size):
            return [head, body]
    return [b]


@dataclass
class RawSection:
    label: str
    heading: str
    level: int
    page_start: int
    blocks: list[RawBlock] = field(default_factory=list)


def split_abstract_prefix(b: RawBlock) -> str | None:
    """'Abstract. This research...' -> body text, else None."""
    m = ABSTRACT_PREFIX.match(b.text)
    if m and len(b.text) - m.end() > 40:
        return b.text[m.end() :]
    return None

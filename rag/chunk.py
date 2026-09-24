"""ParsedDoc -> chunks (PLAN.md Phase 3).

- One paper card per paper: title, authors, year, venue, abstract. Answers
  "which paper..." questions and carries the abstract.
- Prose: packed within a section, never across sections. Target 450 tokens,
  max 700, 60-token overlap. Paragraph boundaries first, then sentences.
- Tables: one chunk each (caption + Markdown). Over 700 tokens, split by rows
  with the header row repeated, so every piece is readable on its own.
- embed_text = context header + text. The header ("Smales 2021 — <title> >
  3. Empirical Results") is contextual retrieval: an isolated chunk like
  "reduce the step size..." becomes findable by paper and section.
"""

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from rag.manifest import Paper
from rag.parse.sections import heading_level
from rag.schemas import Chunk, ParsedDoc, Section, TableBlock

TARGET_TOKENS = 450
MAX_TOKENS = 700
OVERLAP_TOKENS = 60
# A subsection this small ("A. Classical Solution" + one line) is folded into the
# next section with the same label instead of becoming a near-empty chunk.
MIN_SECTION_TOKENS = 40

# cl100k is not the embedding model's tokenizer; it's used only to size chunks
# consistently (plan section 3).
_enc = tiktoken.get_encoding("cl100k_base")

# Sentence boundary: ., ! or ? then space and a capital/digit/bracket, but not
# after common abbreviations ("et al.", "Fig.", "e.g.", "Eq.").
_ABBREV = (
    r"(?<!\bet al\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)"
    r"(?<!\bNo\.)(?<!\bpp\.)(?<!\bal\.)"
)
_SENT = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")


def n_tokens(text: str) -> int:
    return len(_enc.encode(text))


def header(paper: Paper, heading: str) -> str:
    return f"{paper.short_cite} — {paper.title} > {heading}\n\n"


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort for a single 'sentence' over the limit (e.g. an 800-token LaTeX block)."""
    toks = _enc.encode(text)
    return [_enc.decode(toks[i : i + limit]) for i in range(0, len(toks), limit)]


@dataclass
class Unit:
    text: str
    page: int
    tokens: int


def _units(blocks, limit: int) -> list[Unit]:
    """Paragraph units; oversized paragraphs become sentence units."""
    out: list[Unit] = []
    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        k = n_tokens(t)
        if k <= limit:
            out.append(Unit(t, b.page, k))
            continue
        for s in sentences(t):
            ks = n_tokens(s)
            if ks <= limit:
                out.append(Unit(s, b.page, ks))
            else:
                out += [Unit(x, b.page, n_tokens(x)) for x in _hard_split(s, limit)]
    return out


def _overlap_tail(text: str) -> str:
    """Trailing sentences of the previous chunk, up to OVERLAP_TOKENS."""
    tail: list[str] = []
    total = 0
    for s in reversed(sentences(text)):
        k = n_tokens(s)
        if total + k > OVERLAP_TOKENS:
            break
        tail.insert(0, s)
        total += k
    return " ".join(tail)


def pack(units: list[Unit]) -> list[tuple[str, int, int]]:
    """Greedy packing to ~TARGET_TOKENS (never over MAX), with sentence overlap.
    Returns (text, page_start, page_end)."""
    chunks: list[tuple[str, int, int]] = []
    cur: list[Unit] = []
    cur_tokens = 0
    for u in units:
        # +1 per join ("\n\n") is close enough for sizing
        if cur and (cur_tokens + u.tokens > MAX_TOKENS or cur_tokens >= TARGET_TOKENS):
            chunks.append(_emit(cur))
            tail = _overlap_tail(cur[-1].text)
            tail_tokens = n_tokens(tail) if tail else 0
            # overlap only if it still fits with the incoming unit under MAX
            if tail and tail_tokens + u.tokens + 1 <= MAX_TOKENS:
                cur, cur_tokens = [Unit(tail, cur[-1].page, tail_tokens)], tail_tokens
            else:
                cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += u.tokens + 1
    if cur and any(x.text for x in cur):
        chunks.append(_emit(cur))
    return chunks


def _emit(units: list[Unit]) -> tuple[str, int, int]:
    text = "\n\n".join(u.text for u in units if u.text)
    return text, min(u.page for u in units), max(u.page for u in units)


def _section_groups(doc: ParsedDoc) -> list[tuple[Section, str, list]]:
    """(section, heading path, blocks). Subsection headings get their parent's
    heading as a prefix ("2. Data and methodology > 2.1. Data source"); tiny
    sections fold into the next one with the same label."""
    groups: list[tuple[Section, str, list]] = []
    parent = ""
    carry: list = []
    carry_heading = ""
    for s in doc.sections:
        if s.label == "abstract":
            continue  # lives in the paper card
        level = heading_level(s.heading)
        if level == 1:
            parent = s.heading
        path = f"{parent} > {s.heading}" if level > 1 and parent else s.heading
        blocks = carry + list(s.blocks)
        if carry_heading:
            path = f"{carry_heading} / {path}"
        carry, carry_heading = [], ""
        size = sum(n_tokens(b.text) for b in blocks)
        if size == 0:
            continue
        nxt = _next_same_label(doc.sections, s)
        if size < MIN_SECTION_TOKENS and nxt:
            carry, carry_heading = blocks, s.heading
            continue
        groups.append((s, path, blocks))
    if carry:  # trailing tiny section: keep it rather than lose text
        groups.append((doc.sections[-1], carry_heading, carry))
    return groups


def _next_same_label(sections: list[Section], s: Section) -> bool:
    i = sections.index(s)
    return i + 1 < len(sections) and sections[i + 1].label == s.label


def paper_card(paper: Paper, doc: ParsedDoc) -> str:
    authors = ", ".join(paper.authors)
    lines = [paper.title, f"Authors: {authors}"]
    if paper.alt_authors:
        lines.append(f"Also cited as: {', '.join(paper.alt_authors)}")
    lines.append(f"Year: {paper.year}. Venue: {paper.venue}.")
    if paper.doi:
        lines.append(f"DOI: {paper.doi}")
    abstract = " ".join(b.text for s in doc.sections if s.label == "abstract" for b in s.blocks)
    return "\n".join(lines) + ("\n\nAbstract: " + abstract if abstract else "")


def table_texts(t: TableBlock) -> list[str]:
    """Caption + Markdown; split by rows with the header repeated if too long."""
    caption = t.caption or f"Table (page {t.page})"
    full = f"{caption}\n\n{t.markdown}"
    if n_tokens(full) <= MAX_TOKENS:
        return [full]
    lines = t.markdown.split("\n")
    head, rows = lines[:2], lines[2:]  # header row + |---| separator
    budget = MAX_TOKENS - n_tokens(caption + "\n\n" + "\n".join(head)) - 20
    parts: list[list[str]] = [[]]
    used = 0
    for r in rows:
        k = n_tokens(r) + 1
        if parts[-1] and used + k > budget:
            parts.append([])
            used = 0
        if k > budget:  # a single monster row: cut it
            r = _enc.decode(_enc.encode(r)[:budget])
            k = budget
        parts[-1].append(r)
        used += k
    n = len(parts)
    return [
        f"{caption} (part {i + 1} of {n})\n\n" + "\n".join(head + p) for i, p in enumerate(parts)
    ]


def chunk_doc(paper: Paper, doc: ParsedDoc) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}

    def add(section: str, heading: str, kind: str, text: str, p0: int, p1: int) -> None:
        n = counters.get(section, 0)
        counters[section] = n + 1
        chunks.append(
            Chunk(
                chunk_id=f"{paper.id}:{section}:{n:03d}",
                paper_id=paper.id,
                section=section,
                heading=heading,
                page_start=p0,
                page_end=p1,
                kind=kind,
                text=text,
                embed_text=header(paper, heading) + text,
                n_tokens=n_tokens(text),
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )

    card = paper_card(paper, doc)
    abs_pages = [s.page_start for s in doc.sections if s.label == "abstract"] or [1]
    if n_tokens(card) <= MAX_TOKENS:
        add("card", "Paper overview", "paper_card", card, abs_pages[0], abs_pages[0])
    else:  # very long abstract: card head + abstract packed as prose
        head, _, abstract = card.partition("\n\nAbstract: ")
        add("card", "Paper overview", "paper_card", head, abs_pages[0], abs_pages[0])
        for text, p0, p1 in pack(_units([_Tmp(abstract, abs_pages[0])], MAX_TOKENS)):
            add("abstract", "Abstract", "prose", text, p0, p1)

    for s, path, blocks in _section_groups(doc):
        for text, p0, p1 in pack(_units(blocks, MAX_TOKENS)):
            add(s.label, path, "prose", text, p0, p1)

    for t in doc.tables:
        for text in table_texts(t):
            add(t.section, t.caption or f"Table (page {t.page})", "table", text, t.page, t.page)
    return chunks


@dataclass
class _Tmp:
    text: str
    page: int

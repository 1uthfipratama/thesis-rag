"""ParsedDoc -> chunks (PLAN.md Phase 3, sized for the embedder).

- One paper card per paper: title, authors, year, venue, abstract. Answers
  "which paper..." questions and carries the abstract.
- Prose: packed within a section, never across sections, paragraph boundaries
  first, then sentences.
- Tables: one chunk each (caption + Markdown), split by rows with the header
  repeated when too long, so every piece is readable on its own.
- embed_text = context header + text. The header ("Smales 2021 — <title> >
  3. Empirical Results") is contextual retrieval: an isolated chunk like
  "reduce the step size..." becomes findable by paper and section.

Sizes are counted in the embedding model's tokenizer (rag/tokens.py) and the
budget includes the header: embed_text never exceeds the model's 512 tokens.
The plan's 450/700 cl100k sizes overflowed that for 44% of chunks.

Chunks are deliberately small for precise matching ("small-to-big"): each has
a `seq` so the generator can hand the LLM its neighbours from the same section.
"""

import hashlib
import re
from dataclasses import dataclass

from rag.manifest import Paper
from rag.parse.sections import heading_level
from rag.schemas import Chunk, ParsedDoc, Section, TableBlock
from rag.tokens import EMBED_MAX_TOKENS, SPECIAL_TOKENS, count, truncate

TARGET_FRACTION = 0.7  # aim for ~70% of the budget (~340 tokens), leave room to finish a paragraph
OVERLAP_TOKENS = 50
# A subsection this small ("A. Classical Solution" + one line) is folded into the
# next section with the same label instead of becoming a near-empty chunk.
MIN_SECTION_TOKENS = 40

# Sentence boundary: ., ! or ? then space and a capital/digit/bracket, but not
# after common abbreviations ("et al.", "Fig.", "e.g.", "Eq.").
_ABBREV = (
    r"(?<!\bet al\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)"
    r"(?<!\bNo\.)(?<!\bpp\.)(?<!\bal\.)"
)
_SENT = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")


def header(paper: Paper, heading: str) -> str:
    return f"{paper.short_cite} — {paper.title} > {heading}\n\n"


def budget_for(head: str) -> int:
    """Body tokens available once the header and [CLS]/[SEP] are in."""
    return EMBED_MAX_TOKENS - SPECIAL_TOKENS - count(head)


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


@dataclass
class Unit:
    text: str
    page: int
    tokens: int


def _units(blocks, limit: int) -> list[Unit]:
    """Paragraph units; oversized paragraphs become sentence units; oversized
    sentences (an 800-token LaTeX block) are cut by tokens."""
    out: list[Unit] = []
    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        k = count(t)
        if k <= limit:
            out.append(Unit(t, b.page, k))
            continue
        for s in sentences(t):
            ks = count(s)
            if ks <= limit:
                out.append(Unit(s, b.page, ks))
            else:
                out += [Unit(x, b.page, count(x)) for x in truncate(s, limit)]
    return out


def _overlap_tail(text: str) -> str:
    """Trailing sentences of the previous chunk, up to OVERLAP_TOKENS."""
    tail: list[str] = []
    total = 0
    for s in reversed(sentences(text)):
        k = count(s)
        if total + k > OVERLAP_TOKENS:
            break
        tail.insert(0, s)
        total += k
    return " ".join(tail)


def pack(units: list[Unit], budget: int) -> list[tuple[str, int, int]]:
    """Greedy packing to ~TARGET_FRACTION of budget, never over it, with sentence
    overlap. Returns (text, page_start, page_end)."""
    target = int(budget * TARGET_FRACTION)
    chunks: list[tuple[str, int, int]] = []
    cur: list[Unit] = []
    cur_tokens = 0
    for u in units:
        if cur and (cur_tokens + u.tokens > budget or cur_tokens >= target):
            chunks.append(_emit(cur))
            tail = _overlap_tail(cur[-1].text)
            tail_tokens = count(tail) if tail else 0
            # overlap only if it still fits with the incoming unit
            if tail and tail_tokens + u.tokens <= budget:
                cur, cur_tokens = [Unit(tail, cur[-1].page, tail_tokens)], tail_tokens
            else:
                cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += u.tokens  # "\n\n" joins add no WordPiece tokens
    if cur:
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
    for i, s in enumerate(doc.sections):
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
        size = sum(count(b.text) for b in blocks)
        if size == 0:
            continue
        next_same = i + 1 < len(doc.sections) and doc.sections[i + 1].label == s.label
        if size < MIN_SECTION_TOKENS and next_same:
            carry, carry_heading = blocks, s.heading
            continue
        groups.append((s, path, blocks))
    return groups


def paper_card(paper: Paper, doc: ParsedDoc) -> tuple[str, str]:
    """(card head, abstract)."""
    lines = [paper.title, f"Authors: {', '.join(paper.authors)}"]
    if paper.alt_authors:
        lines.append(f"Also cited as: {', '.join(paper.alt_authors)}")
    lines.append(f"Year: {paper.year}. Venue: {paper.venue}.")
    if paper.doi:
        lines.append(f"DOI: {paper.doi}")
    abstract = " ".join(b.text for s in doc.sections if s.label == "abstract" for b in s.blocks)
    return "\n".join(lines), abstract


def table_texts(t: TableBlock, budget: int) -> list[str]:
    """Caption + Markdown; split by rows with the header repeated if too long."""
    caption = t.caption or f"Table (page {t.page})"
    full = f"{caption}\n\n{t.markdown}"
    if count(full) <= budget:
        return [full]
    lines = t.markdown.split("\n")
    head, rows = lines[:2], lines[2:]  # header row + |---| separator
    # room for rows once caption, "(part i of n)" and the header row are in
    room = budget - count(f"{caption} (part 99 of 99)\n\n" + "\n".join(head))
    parts: list[list[str]] = [[]]
    used = 0
    for r in rows:
        k = count(r)
        if k > room:  # a single monster row: cut it
            r, k = truncate(r, room)[0], room
        if parts[-1] and used + k > room:
            parts.append([])
            used = 0
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
        embed_text = header(paper, heading) + text
        chunks.append(
            Chunk(
                chunk_id=f"{paper.id}:{section}:{n:03d}",
                paper_id=paper.id,
                seq=len(chunks),
                section=section,
                heading=heading,
                page_start=p0,
                page_end=p1,
                kind=kind,
                text=text,
                embed_text=embed_text,
                n_tokens=count(embed_text) + SPECIAL_TOKENS,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )

    head, abstract = paper_card(paper, doc)
    page = next((s.page_start for s in doc.sections if s.label == "abstract"), 1)
    card = head + ("\n\nAbstract: " + abstract if abstract else "")
    card_budget = budget_for(header(paper, "Paper overview"))
    if count(card) <= card_budget:
        add("card", "Paper overview", "paper_card", card, page, page)
    else:  # long abstract: card keeps the metadata, the abstract is packed as prose
        add("card", "Paper overview", "paper_card", head, page, page)
        budget = budget_for(header(paper, "Abstract"))
        for text, p0, p1 in pack(_units([_Text(abstract, page)], budget), budget):
            add("abstract", "Abstract", "prose", text, p0, p1)

    for s, path, blocks in _section_groups(doc):
        budget = budget_for(header(paper, path))
        for text, p0, p1 in pack(_units(blocks, budget), budget):
            add(s.label, path, "prose", text, p0, p1)

    for t in doc.tables:
        heading = t.caption or f"Table (page {t.page})"
        for text in table_texts(t, budget_for(header(paper, heading))):
            add(t.section, heading, "table", text, t.page, t.page)
    return chunks


@dataclass
class _Text:
    text: str
    page: int

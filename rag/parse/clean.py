"""Boilerplate removal, near-duplicate lines, hyphenation."""

import re
from collections import Counter

from rapidfuzz import fuzz

from rag.parse.extract import Page

# Belt and braces on top of the repeated-line detector. Each has a test.
BOILERPLATE_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"Downloaded from https://onlinelibrary\.wiley\.com",  # p01
        r"Electronic copy available at: ?https://ssrn\.com",  # p14
        r"Volume \d+, Issue \d+, .*Pages \d+-\d+",  # p03, p05
        r"^IAENG International Journal of",  # p03, p05
        r"^\s*\d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}\s*$",  # p12 vertical timestamp
        r"\b030005-\d+",  # p12 page ids, alone and inside the AIP footer line
        r"Permission to make digital or hard copies",  # ACM notice (p02)
        r"^\s*a1{10}\s*$",  # p09 margin art
        r"<\s*Insert (Figure|Table) \d+\s*>",  # p14 placeholders
        r"Publisher[’']s Note",
        r"Creative Commons",
        r"Copyright ©",
        r"This article is licensed under",
        r"^_{10,}\s*$",  # IAENG header rule drawn with underscores
        r"\bISSN\W{0,3}\d{4}\W?\d{3}[\dX]",  # journal mastheads (p07, p19 "... ISSN: 2146-4553")
        r"open access article",
        r"Corresponding author|^\s*\*?\s*Correspondence:",
        r"^\s*E-?mail( address)?:",
        r"^\s*(doi:\s*|https?://doi\.org/)\S+\s*$",  # a bare DOI line
        r"^\s*Academic Editors?:",
    ]
]

# Licence and publisher notices wrap over several lines; once one line matches,
# the whole (short) block goes, so tails like "creativecommons.org/licenses/by/"
# or "tional affiliations." don't survive.
LICENCE_BLOCK = re.compile(
    r"Creative Commons|creativecommons\.org|Publisher[’']s Note|jurisdictional claims|"
    r"Licensee [A-Z]|open access article|This article is licensed under|Copyright ©",
    re.I,
)
LICENCE_BLOCK_MAX_WORDS = 120

NEAR_DUP_RATIO = 95
NEAR_DUP_MIN_LEN = 15  # short lines ("(1)", "0.05") legitimately repeat


def _line_key(text: str) -> str:
    t = text.lower()
    # Digit runs collapse to one "#", so page "2 / 16" and "14 / 16" share a key.
    t = re.sub(r"\d+", "#", t)
    return re.sub(r"\s+", " ", t).strip()


def remove_boilerplate(pages: list[Page], dropped: Counter[str]) -> None:
    """Drop lines that repeat on >= 50% of pages, plus explicit patterns."""
    n_pages = len(pages)
    seen_on: Counter[str] = Counter()
    for p in pages:
        keys = {_line_key(ln.text) for b in p.blocks for ln in b.lines}
        seen_on.update(keys)
    repeated = {
        k
        for k, n in seen_on.items()
        # min length 6 so e.g. "where" or "(#)" can't be flagged as a running header
        if n_pages >= 3 and n >= 0.5 * n_pages and len(k) >= 6
    }
    for p in pages:
        margin = 0.08 * p.height
        kept_blocks = []
        for b in p.blocks:
            if LICENCE_BLOCK.search(b.text) and len(b.text.split()) <= LICENCE_BLOCK_MAX_WORDS:
                dropped["licence_block"] += 1
            else:
                kept_blocks.append(b)
        p.blocks = kept_blocks
        for b in p.blocks:
            keep = []
            for ln in b.lines:
                in_margin = ln.bbox[1] < margin or ln.bbox[3] > p.height - margin
                if in_margin and re.fullmatch(r"\s*\d{1,4}\s*", ln.text):
                    dropped["page_number"] += 1
                elif _line_key(ln.text) in repeated:
                    dropped["repeated_line"] += 1
                elif any(pat.search(ln.text) for pat in BOILERPLATE_PATTERNS):
                    dropped["boilerplate_pattern"] += 1
                else:
                    keep.append(ln)
            b.lines = keep
            if keep:
                b.refresh_bbox()
        p.blocks = [b for b in p.blocks if b.lines]


def remove_near_duplicates(pages: list[Page], dropped: Counter[str]) -> None:
    """Drop a line nearly identical to the previous kept line (duplicated text layers)."""
    prev = ""
    for p in pages:
        for b in p.blocks:
            keep = []
            for ln in b.lines:
                t = ln.text.strip()
                if len(t) >= NEAR_DUP_MIN_LEN and fuzz.ratio(t, prev) > NEAR_DUP_RATIO:
                    dropped["near_duplicate_line"] += 1
                    continue
                keep.append(ln)
                prev = t
            b.lines = keep
            if keep:
                b.refresh_bbox()
        p.blocks = [b for b in p.blocks if b.lines]


def remove_figure_labels(pages: list[Page], dropped: Counter[str]) -> None:
    """Axis labels of vector charts ("Time" x20 in p01): tiny blocks that repeat.

    Image-based figures are handled by bbox; vector charts have no image bbox, so
    their labels are recognised by being short and recurring across the document.
    """
    counts: Counter[str] = Counter(
        b.text.strip() for p in pages for b in p.blocks if len(b.text.split()) <= 3
    )
    for p in pages:
        keep = []
        for b in p.blocks:
            t = b.text.strip()
            if len(t.split()) <= 3 and counts[t] >= 4 and b.kind == "text":
                dropped["figure_label"] += 1
            else:
                keep.append(b)
        p.blocks = keep


_WORD = re.compile(r"[A-Za-z]+")


def build_vocab(pages: list[Page]) -> Counter[str]:
    c: Counter[str] = Counter()
    for p in pages:
        for b in p.blocks:
            c.update(w.lower() for w in _WORD.findall(b.text))
    return c


def join_lines(lines: list[str], vocab: Counter[str]) -> str:
    """Join wrapped lines into one paragraph, repairing end-of-line hyphenation.

    "fore-" + "casting" -> "forecasting", but "second-" + "order" stays
    "second-order". The decision uses the document's own vocabulary: join if the
    joined word occurs elsewhere; keep the hyphen if both halves are words that
    occur on their own elsewhere ("second", "order"); otherwise join, since most
    line-end hyphens are wraps. No external dictionary needed.
    """
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
            continue
        m_prev = re.search(r"([A-Za-z]+)-$", out)
        m_next = re.match(r"([a-z]+)", ln)
        if m_prev and m_next:
            a, b = m_prev.group(1), m_next.group(1)
            joined = (a + b).lower()
            if vocab.get(joined, 0) > 0 or not _hyphenated_known(a, b, vocab):
                out = out[:-1] + ln
            else:
                out = out + ln
        else:
            out = out + " " + ln
    return out


def _hyphenated_known(a: str, b: str, vocab: Counter[str]) -> bool:
    # both halves are words in their own right, e.g. "second" and "order"
    return vocab.get(a.lower(), 0) > 1 and vocab.get(b.lower(), 0) > 1

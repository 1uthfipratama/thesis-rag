"""Parser backends. Every backend returns the same ParsedDoc, so chunking,
indexing and evaluation don't care which one produced it.

- pymupdf: our rule-based parser. Fast (seconds/paper), no ML, runs anywhere,
  including the free HF Space for user uploads. Equations become placeholders.
- mineru:  MinerU's ML layout/table/formula models + our shared assembly.
  ~8 s/page on CPU; run offline for the core corpus.
- marker:  Marker (surya models via a local llama.cpp server) + our shared
  assembly. Strongest on equations per its benchmarks.
- hybrid:  pymupdf structure, with each dropped equation replaced by the LaTeX
  MinerU recognised in the same page region. Keeps the tuned structure and
  tests of the rule-based parser, adds math.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pymupdf

from rag.manifest import Paper
from rag.parse.pipeline import MathLookup, parse_pdf
from rag.schemas import ParsedDoc

MIN_OVERLAP = 0.3  # of the smaller box; equation fragment bboxes are loose


def _overlap_ratio(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def mineru_math(paper: Paper, pdf_path: Path) -> MathLookup:
    """Page-region -> LaTeX lookup over MinerU's display equations."""
    from rag.parse.mineru import ensure_output

    sc = json.loads(ensure_output(paper.id, pdf_path).read_text(encoding="utf-8"))
    pdf = pymupdf.open(pdf_path)
    by_page: dict[int, list[tuple[tuple, str]]] = {}
    for pg in sc["pages"]:
        n = pg["page_idx"] + 1
        w, h = pdf[n - 1].rect.width, pdf[n - 1].rect.height
        for b in pg["blocks"]:
            if b.get("type") == "equation" and (b.get("content") or "").strip():
                bb = b["bbox"]
                box = (bb[0] * w, bb[1] * h, bb[2] * w, bb[3] * h)
                by_page.setdefault(n, []).append((box, b["content"].strip()))

    used: set[tuple[int, int]] = set()

    def lookup(page: int, box: tuple) -> list[str] | None:
        """LaTeX for the equations overlapping `box`. Each MinerU equation is handed
        out once: a later fragment run overlapping only used equations returns None
        (drop it, it's a piece of an equation already emitted); no overlap -> []."""
        hits = [
            (i, eb, tex)
            for i, (eb, tex) in enumerate(by_page.get(page, []))
            if _overlap_ratio(eb, box) >= MIN_OVERLAP
        ]
        fresh = [(i, eb, tex) for i, eb, tex in hits if (page, i) not in used]
        if hits and not fresh:
            return None
        used.update((page, i) for i, _, _ in fresh)
        return [tex for _, eb, tex in sorted(fresh, key=lambda x: x[1][1])]

    return lookup


def _pymupdf(paper: Paper, pdf_path: Path) -> ParsedDoc:
    return parse_pdf(paper, pdf_path)


def _mineru(paper: Paper, pdf_path: Path) -> ParsedDoc:
    from rag.parse.mineru import parse_mineru

    return parse_mineru(paper, pdf_path)


def _marker(paper: Paper, pdf_path: Path) -> ParsedDoc:
    from rag.parse.marker import parse_marker

    return parse_marker(paper, pdf_path)


def _hybrid(paper: Paper, pdf_path: Path) -> ParsedDoc:
    return parse_pdf(paper, pdf_path, math=mineru_math(paper, pdf_path), backend="hybrid")


BACKENDS: dict[str, Callable[[Paper, Path], ParsedDoc]] = {
    "pymupdf": _pymupdf,
    "mineru": _mineru,
    "marker": _marker,
    "hybrid": _hybrid,
}

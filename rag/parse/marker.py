"""Marker backend: Marker's JSON block tree -> ExtBlocks -> shared assembly.

Marker runs in its own venv (.venvs/marker). Its surya model is served by a
local llama.cpp `llama-server` (SURYA_INFERENCE_BACKEND=llamacpp) since this
machine has no Docker/vLLM; set LLAMA_CPP_BINARY if it isn't on PATH. Output is
cached in data/parsed_raw/marker/{id}/{id}.json.

Licence note: Marker's code is Apache-2.0 but its model weights are under a
modified OpenRAIL-M licence (free for personal/research use and small
startups). Fine for this portfolio project; check before any commercial use.
"""

import html
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pymupdf

from rag.config import ROOT, settings
from rag.manifest import Paper
from rag.parse.external import ExtBlock, assemble_external, html_table_rows
from rag.parse.extract import normalize
from rag.parse.latex import tidy
from rag.schemas import ParsedDoc

MARKER_BIN = ROOT / ".venvs" / "marker" / "Scripts" / "marker_single"

KIND = {
    "Text": "text",
    "TextInlineMath": "text",
    "ListItem": "text",
    "Code": "text",
    "Handwriting": "text",
    "SectionHeader": "heading",
    "Caption": "caption",
    "Footnote": "footnote",
    "Equation": "equation",
    "Table": "table",
    "PageHeader": None,
    "PageFooter": None,
    "Picture": None,
    "Figure": None,
    "Diagram": None,
    "TableOfContents": None,
    "Form": None,
}
GROUPS = {"FigureGroup", "TableGroup", "ListGroup", "PictureGroup"}
REFERENCES = {"Bibliography", "Reference"}


def raw_path(paper_id: str) -> Path:
    return settings.data_dir / "parsed_raw" / "marker" / paper_id / f"{paper_id}.json"


def ensure_output(paper_id: str, pdf_path: Path) -> Path:
    out = raw_path(paper_id)
    if out.exists():
        return out
    out.parent.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"SURYA_INFERENCE_BACKEND": "llamacpp", "PYTHONUTF8": "1"}
    cmd = [
        str(MARKER_BIN), str(pdf_path), "--output_format", "json",
        "--output_dir", str(out.parent.parent), "--disable_image_extraction",
    ]  # fmt: skip
    with open(out.parent.parent / f"{paper_id}_log.txt", "w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return out


_MATH = re.compile(r"<math(?P<attrs>[^>]*)>(?P<tex>.*?)</math>", re.S)


def _text(fragment: str) -> str:
    """Block HTML -> plain text; inline <math> -> $tidied LaTeX$."""
    s = _MATH.sub(lambda m: f"${tidy(m.group('tex'))}$", fragment)
    # list items / paragraphs end lines: a reference list arrives as one block of <li>s
    s = re.sub(r"<br\s*/?>|</li>|</p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return normalize(html.unescape(s)).strip()


def _flatten(block: dict) -> list[dict]:
    if block["block_type"] in GROUPS and block.get("children"):
        return [x for c in block["children"] for x in _flatten(c)]
    return [block]


def parse_marker(paper: Paper, pdf_path: Path) -> ParsedDoc:
    doc = json.loads(ensure_output(paper.id, pdf_path).read_text(encoding="utf-8"))
    pdf = pymupdf.open(pdf_path)
    dropped: Counter[str] = Counter()
    references: list[str] = []
    pages_in = []
    for idx, page in enumerate(doc["children"]):
        number = idx + 1
        w, h = pdf[idx].rect.width, pdf[idx].rect.height
        ext: list[ExtBlock] = []
        pending_caption = ""
        for b in [x for c in page.get("children") or [] for x in _flatten(c)]:
            typ = b["block_type"]
            bbox = tuple(b["bbox"])  # Marker bboxes are already in PDF points
            if typ in REFERENCES:
                if number not in paper.drop_pages:
                    references += [r for r in _text(b["html"]).split("\n") if r.strip()]
                continue
            kind = KIND.get(typ, "text")
            if kind is None:
                dropped[f"{typ}_block"] += 1
                continue
            if kind == "equation":
                m = _MATH.search(b["html"])
                ext.append(
                    ExtBlock("equation", "", bbox, latex=m.group("tex").strip() if m else None)
                )
            elif kind == "table":
                ext.append(
                    ExtBlock(
                        "table", "", bbox, rows=html_table_rows(b["html"]), caption=pending_caption
                    )
                )
                pending_caption = ""
            elif kind == "caption" and re.match(r"\s*(Table|TABLE)\s", _text(b["html"])):
                pending_caption = _text(b["html"])  # table captions attach to the next table
            else:
                ext.append(ExtBlock(kind, _text(b["html"]), bbox))
        pages_in.append((number, w, h, ext))
    return assemble_external(paper, pdf_path, pages_in, references, dropped, "marker")

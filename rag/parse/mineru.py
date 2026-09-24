"""MinerU backend: MinerU's typed blocks -> ExtBlocks -> shared assembly.

MinerU runs in its own venv (.venvs/mineru, heavy ONNX deps) and is called as a
CLI. Its raw output is cached in data/parsed_raw/mineru/{id}/, so the slow step
(~8 s/page on CPU at the "basic" tier) runs once per paper.

What MinerU adds over the PyMuPDF backend: ML layout detection (headers,
footers, footnotes, captions tagged for us), table structure recognition, and
formula recognition to LaTeX, both display and inline ($...$).
"""

import html
import json
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import pymupdf

from rag.config import ROOT, settings
from rag.manifest import Paper
from rag.parse.external import ExtBlock, assemble_external, html_table_rows
from rag.parse.extract import normalize
from rag.parse.latex import tidy
from rag.schemas import ParsedDoc

MINERU_BIN = ROOT / ".venvs" / "mineru" / "Scripts" / "mineru-kit"
TIER = "basic"  # ONNX models, CPU-friendly; "standard" needs a 1.2B VLM server

# MinerU block type -> our kind. None = drop (counted).
KIND = {
    "text": "text",
    "code": "text",  # p05's fit/forecast algorithm is meaningful prose-like content
    "list": "text",
    "paragraph_title": "heading",
    "title": "heading",
    "doc_title": "title",
    "page_footnote": "footnote",
    "header": None,
    "footer": None,
    "page_number": None,
    "aside_text": None,  # margin notes, e.g. MDPI sidebars
}


def raw_dir(paper_id: str) -> Path:
    return settings.data_dir / "parsed_raw" / "mineru" / paper_id


def ensure_output(paper_id: str, pdf_path: Path) -> Path:
    """Run MinerU once per paper; return the path to structured_content.json."""
    out = raw_dir(paper_id)
    sc = out / "structured_content.json"
    if sc.exists():
        return sc
    out.mkdir(parents=True, exist_ok=True)
    # Local parsing only: never pass --remote (several papers aren't open access).
    cmd = [str(MINERU_BIN), "parse", str(pdf_path), "-o", str(out), "--tier", TIER, "-f", "zip"]
    with open(out / "log.txt", "w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    with zipfile.ZipFile(next(out.glob("*.zip"))) as z:
        z.extractall(out)
    return sc


def table_rows(content: str) -> list[list[str]]:
    """MinerU 4 emits Markdown pipe tables (with HTML entities); older versions HTML."""
    if re.search(r"<t(able|r)\b", content):
        return html_table_rows(content)
    rows = []
    for line in content.splitlines():
        if line.strip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", line):
            rows.append([html.unescape(c.strip()) for c in line.strip().strip("|").split("|")])
    return rows


def _plain(text: str) -> str:
    """Drop MinerU's inline Markdown emphasis and HTML spans; tidy inline math."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\$([^$]+)\$", lambda m: f"${tidy(m.group(1))}$", text)
    return normalize(text)


def _captions(b: dict) -> list[dict]:
    return [c for c in b.get("captions", []) if c.get("content", "").strip()]


def parse_mineru(paper: Paper, pdf_path: Path) -> ParsedDoc:
    sc = json.loads(ensure_output(paper.id, pdf_path).read_text(encoding="utf-8"))
    pdf = pymupdf.open(pdf_path)
    dropped: Counter[str] = Counter()
    references: list[str] = []
    pages_in = []
    for pg in sc["pages"]:
        number = pg["page_idx"] + 1
        w, h = pdf[number - 1].rect.width, pdf[number - 1].rect.height

        def pts(bb, w=w, h=h):  # MinerU bboxes are 0-1 page fractions
            return (bb[0] * w, bb[1] * h, bb[2] * w, bb[3] * h)

        ext: list[ExtBlock] = []
        for b in pg["blocks"]:
            typ = b.get("type")
            content = b.get("content") or ""
            if typ == "ref_text":
                if number not in paper.drop_pages:
                    references.append(" ".join(_plain(content).split()))
            elif typ == "table":
                rows = [[_plain(c) for c in r] for r in table_rows(content)]
                caption = " ".join(_plain(c["content"]) for c in _captions(b))
                ext.append(ExtBlock("table", "", pts(b["bbox"]), rows=rows, caption=caption))
            elif typ in ("image", "chart"):
                dropped[f"{typ}_block"] += 1
                ext += [
                    ExtBlock("caption", _plain(c["content"]), pts(c["bbox"])) for c in _captions(b)
                ]
            elif typ == "equation":
                ext.append(ExtBlock("equation", "", pts(b["bbox"]), latex=content.strip() or None))
            elif typ in KIND:
                if KIND[typ] is None:
                    dropped[f"{typ}_block"] += 1
                else:
                    ext.append(ExtBlock(KIND[typ], _plain(content), pts(b["bbox"])))
            else:
                dropped[f"unknown_{typ}"] += 1
                ext.append(ExtBlock("text", _plain(content), pts(b["bbox"])))
        pages_in.append((number, w, h, ext))
    return assemble_external(paper, pdf_path, pages_in, references, dropped, "mineru")

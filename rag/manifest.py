"""Load and validate data/manifest.yaml."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from rag.config import settings

Layout = Literal["one_column", "two_column", "one_column_sidebar", "two_column_sidebar"]


class Paper(BaseModel):
    id: str = Field(pattern=r"^[pu]\d{2}$")  # p = core paper, u = user upload (Phase 13)
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str
    authors: list[str] = Field(min_length=1)
    alt_authors: list[str] = []  # other author orders seen in citations (p13)
    year: int
    venue: str
    doi: str = ""
    short_cite: str
    layout: Layout
    open_access: bool
    drop_pages: list[int] = []  # 1-based
    notes: str = ""


def load_manifest(path: Path | None = None) -> list[Paper]:
    raw = yaml.safe_load((path or settings.manifest_path).read_text(encoding="utf-8"))
    return [Paper.model_validate(entry) for entry in raw]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(papers: list[Paper], raw_dir: Path | None = None) -> list[str]:
    """Return a list of problems; empty means the manifest matches data/raw."""
    raw_dir = raw_dir or settings.raw_dir
    problems: list[str] = []
    ids = [p.id for p in papers]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    for p in papers:
        path = raw_dir / p.file
        if not path.exists():
            problems.append(f"{p.id}: missing file {path}")
        elif sha256_file(path) != p.sha256:
            problems.append(f"{p.id}: sha256 mismatch for {p.file}")
    return problems

"""Pydantic models shared across parsing, chunking and serving."""

from typing import Literal

from pydantic import BaseModel

BlockKind = Literal["text", "heading", "table", "equation", "caption", "sidebar", "footnote"]

SectionLabel = Literal[
    "abstract",
    "introduction",
    "literature_review",
    "methodology",
    "data",
    "results",
    "discussion",
    "conclusion",
    "other",
]


class Block(BaseModel):
    page: int  # 1-based, as printed by a PDF viewer
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    is_bold: bool
    # "footnote" is an addition to the plan's list: p10 has long footnotes at the
    # bottom of the left column, and leaving them in place splits sentences that
    # run from the left column into the right one. They are moved to page end.
    kind: BlockKind


class Section(BaseModel):
    label: SectionLabel
    heading: str  # as printed, e.g. "3.2. Methodology"
    page_start: int
    blocks: list[Block]


class TableBlock(BaseModel):
    page: int
    bbox: tuple[float, float, float, float]
    caption: str
    section: str = "other"  # label of the section the table sits in
    heading: str = ""
    markdown: str
    n_rows: int
    n_cols: int


class ParsedDoc(BaseModel):
    paper_id: str
    title: str
    sections: list[Section]
    tables: list[TableBlock]
    references: list[str]  # kept as metadata, never chunked
    data_availability: str = ""  # p03/p05 link GitHub repos here
    page_layouts: dict[int, str]  # page -> "one_column" | "two_column" (+ "+sidebar")
    dropped: dict[str, int]  # counts per drop reason, for the report

"""Phase 3 acceptance: chunk sizes, boundaries, ids, headers."""

import json

import pytest

from rag.chunk import MAX_TOKENS, Unit, n_tokens, pack, sentences, table_texts
from rag.config import settings
from rag.manifest import load_manifest
from rag.schemas import TableBlock


def test_sentences_respect_abbreviations() -> None:
    s = sentences("Li et al. (2018) fit it. See Fig. 3 now. Eq. (2) holds. Next.")
    assert s == ["Li et al. (2018) fit it.", "See Fig. 3 now.", "Eq. (2) holds.", "Next."]


def test_pack_never_exceeds_max_even_with_overlap() -> None:
    para = "This is a sentence about stock prices and volatility. " * 60  # ~600 tokens
    units = [Unit(para.strip(), 1, n_tokens(para))] * 4
    out = pack(units)
    assert len(out) >= 4
    assert all(n_tokens(t) <= MAX_TOKENS for t, _, _ in out)


def test_long_table_splits_by_rows_with_header_repeated() -> None:
    rows = "\n".join(f"| r{i} | {i * 1.2345:.4f} | {i * 9.87:.3f} |" for i in range(400))
    md = "| name | a | b |\n|---|---|---|\n" + rows
    t = TableBlock(
        page=1, bbox=(0, 0, 1, 1), caption="Table 9. Big", markdown=md, n_rows=401, n_cols=3
    )
    parts = table_texts(t)
    assert len(parts) > 1
    for p in parts:
        assert n_tokens(p) <= MAX_TOKENS
        assert "| name | a | b |" in p and p.startswith("Table 9. Big (part ")


CHUNKS = settings.chunks_path
corpus = pytest.mark.skipif(not CHUNKS.exists(), reason="run `make chunks` first")


def _chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


@corpus
def test_chunk_limits_and_sections() -> None:
    cs = _chunks()
    assert all(c["n_tokens"] <= MAX_TOKENS for c in cs)
    assert not [c for c in cs if c["section"] in ("references", "frontmatter", "backmatter")]
    assert len({c["chunk_id"] for c in cs}) == len(cs)


@corpus
def test_every_paper_has_one_card_with_abstract() -> None:
    cs = _chunks()
    papers = {p.id: p for p in load_manifest()}
    for pid, paper in papers.items():
        cards = [c for c in cs if c["paper_id"] == pid and c["kind"] == "paper_card"]
        assert len(cards) == 1, pid
        assert paper.title in cards[0]["text"]


@corpus
def test_embed_text_carries_context_header() -> None:
    papers = {p.id: p for p in load_manifest()}
    for c in _chunks():
        p = papers[c["paper_id"]]
        assert c["embed_text"].startswith(f"{p.short_cite} — {p.title} > {c['heading']}\n\n")
        assert c["embed_text"].endswith(c["text"])

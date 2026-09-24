"""Phase 3 acceptance: chunk sizes (in embedder tokens), boundaries, ids, headers."""

import json

import pytest

from rag.chunk import Unit, pack, sentences, table_texts
from rag.config import settings
from rag.manifest import load_manifest
from rag.schemas import TableBlock
from rag.tokens import EMBED_MAX_TOKENS, count, tokenizer


def test_sentences_respect_abbreviations() -> None:
    s = sentences("Li et al. (2018) fit it. See Fig. 3 now. Eq. (2) holds. Next.")
    assert s == ["Li et al. (2018) fit it.", "See Fig. 3 now.", "Eq. (2) holds.", "Next."]


def test_pack_never_exceeds_budget_even_with_overlap() -> None:
    para = "This is a sentence about stock prices and volatility. " * 30  # ~300 tokens
    units = [Unit(para.strip(), 1, count(para))] * 4
    out = pack(units, budget=450)
    assert len(out) >= 4
    assert all(count(t) <= 450 for t, _, _ in out)


def test_long_table_splits_by_rows_with_header_repeated() -> None:
    rows = "\n".join(f"| r{i} | {i * 1.2345:.4f} | {i * 9.87:.3f} |" for i in range(400))
    md = "| name | a | b |\n|---|---|---|\n" + rows
    t = TableBlock(
        page=1, bbox=(0, 0, 1, 1), caption="Table 9. Big", markdown=md, n_rows=401, n_cols=3
    )
    parts = table_texts(t, budget=450)
    assert len(parts) > 1
    for p in parts:
        assert count(p) <= 450
        assert "| name | a | b |" in p and p.startswith("Table 9. Big (part ")


CHUNKS = settings.chunks_path
corpus = pytest.mark.skipif(not CHUNKS.exists(), reason="run `make chunks` first")


def _chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


@corpus
def test_no_chunk_is_truncated_by_the_embedder() -> None:
    for c in _chunks():
        n = len(tokenizer().encode(c["embed_text"]).ids)  # with [CLS]/[SEP]
        assert n <= EMBED_MAX_TOKENS, (c["chunk_id"], n)
        assert n == c["n_tokens"]


@corpus
def test_chunk_sections_and_ids() -> None:
    cs = _chunks()
    assert not [c for c in cs if c["section"] in ("references", "frontmatter", "backmatter")]
    assert len({c["chunk_id"] for c in cs}) == len(cs)
    for pid in {c["paper_id"] for c in cs}:
        seqs = [c["seq"] for c in cs if c["paper_id"] == pid]
        assert seqs == list(range(len(seqs)))


@corpus
def test_every_paper_has_one_card_with_title() -> None:
    cs = _chunks()
    for paper in load_manifest():
        cards = [c for c in cs if c["paper_id"] == paper.id and c["kind"] == "paper_card"]
        assert len(cards) == 1, paper.id
        assert paper.title in cards[0]["text"]


@corpus
def test_embed_text_carries_context_header() -> None:
    papers = {p.id: p for p in load_manifest()}
    for c in _chunks():
        p = papers[c["paper_id"]]
        assert c["embed_text"].startswith(f"{p.short_cite} — {p.title} > {c['heading']}\n\n")
        assert c["embed_text"].endswith(c["text"])

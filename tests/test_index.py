"""Phase 4 acceptance: index integrity, FTS query safety, embedder guard, smoke query."""

import sqlite3

import numpy as np
import pytest

from rag.config import settings
from rag.index import check_meta, connect
from rag.retrieve import bm25, dense, describe, fts_query


@pytest.mark.parametrize(
    "raw",
    [
        "ARDL(4,4,1)",
        "Adams-Bashforth",
        'what is "MAPE"',
        "NOT this AND that",
        "p = 0.05*",
        "Lyócsa",
    ],
)
def test_fts_query_is_always_valid_fts5(raw: str) -> None:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')")
    db.execute("INSERT INTO t VALUES ('ARDL 4 1 Adams Bashforth MAPE Lyócsa this that')")
    db.execute("SELECT * FROM t WHERE t MATCH ?", (fts_query(raw),)).fetchall()  # must not raise


def test_fts_query_drops_stopwords_and_quotes_tokens() -> None:
    assert (
        fts_query("Which papers use the ARDL(4,4,1) model?")
        == '"use" OR "ardl" OR "4" OR "1" OR "model"'
    )
    assert fts_query("the of which") == ""


INDEX = settings.index_path
corpus = pytest.mark.skipif(not INDEX.exists(), reason="run `make index` first")


@pytest.fixture(scope="module")
def db():
    conn = connect(readonly=True)
    yield conn
    conn.close()


@corpus
def test_row_counts_match_chunks_file(db) -> None:
    n = len(settings.chunks_path.read_text(encoding="utf-8").splitlines())
    for table in ("chunks", "fts_chunks", "vec_chunks"):
        assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == n, table


@corpus
def test_vec_rowids_are_chunk_ids(db) -> None:
    missing = db.execute(
        "SELECT count(*) FROM vec_chunks v LEFT JOIN chunks c ON c.id = v.rowid WHERE c.id IS NULL"
    ).fetchone()[0]
    assert missing == 0


@corpus
def test_stored_vectors_are_unit_length(db) -> None:
    for (blob,) in db.execute("SELECT embedding FROM vec_chunks LIMIT 20"):
        assert abs(np.linalg.norm(np.frombuffer(blob, dtype=np.float32)) - 1) < 1e-3


@corpus
def test_meta_guard_refuses_other_embedder(db, monkeypatch) -> None:
    assert check_meta(db)["embed_model"] == settings.embed_model
    monkeypatch.setattr(settings, "embed_model", "BAAI/bge-base-en-v1.5")
    with pytest.raises(RuntimeError, match="rebuild"):
        check_meta(db)


@corpus
def test_smoke_lotka_volterra_finds_p12_both_ways(db) -> None:
    q = "Lotka-Volterra Adams-Bashforth"
    for fn in (bm25, dense):
        papers = [pid for _, pid, _, _ in describe(db, [r for r, _ in fn(db, q, 5)])]
        assert "p12" in papers, fn.__name__

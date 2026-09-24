"""SQLite index: chunk rows + FTS5 (BM25) + sqlite-vec (dense) in one file.

One file because ~650 chunks don't need a vector database, and one artifact is
easier to ship to the server (Phase 11) and to sync to the HF dataset.

Build is idempotent: it writes a fresh file next to the target and swaps it in,
so a failed build never leaves a half-written index behind.
"""

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from rag.config import settings
from rag.manifest import Paper
from rag.schemas import Chunk

SCHEMA = """
CREATE TABLE papers (
  paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INT,
  venue TEXT, short_cite TEXT, doi TEXT,
  -- 'core' | 'user'. In from the start (PLAN_ADDENDUM 13.1 adds it later via
  -- ALTER); uploads must never mix into the gold-set corpus.
  collection TEXT NOT NULL DEFAULT 'core'
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  chunk_id TEXT UNIQUE, paper_id TEXT, seq INT, section TEXT, heading TEXT,
  page_start INT, page_end INT, kind TEXT,
  text TEXT, embed_text TEXT, n_tokens INT, content_hash TEXT,
  collection TEXT NOT NULL DEFAULT 'core'
);
CREATE INDEX idx_chunks_paper ON chunks(paper_id, seq);
CREATE INDEX idx_chunks_collection ON chunks(collection);

-- External-content FTS5: the text lives once, in chunks. Porter stemming so
-- "forecasts" matches "forecasting"; unicode61 folds diacritics (Lyócsa).
CREATE VIRTUAL TABLE fts_chunks USING fts5(
  embed_text, content='chunks', content_rowid='id',
  tokenize='porter unicode61'
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


PAPER_COLS = ("paper_id", "title", "authors", "year", "venue", "short_cite", "doi")
CHUNK_COLS = (
    "chunk_id", "paper_id", "seq", "section", "heading", "page_start", "page_end", "kind",
    "text", "embed_text", "n_tokens", "content_hash",
)  # fmt: skip


def _insert(table: str, cols: tuple[str, ...]) -> str:
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"


def connect(path: Path | None = None, readonly: bool = False) -> sqlite3.Connection:
    path = path or settings.index_path
    uri = f"file:{path.as_posix()}{'?mode=ro' if readonly else ''}"
    db = sqlite3.connect(uri, uri=True, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def build(chunks: list[Chunk], papers: list[Paper], out: Path | None = None) -> dict:
    from rag import embed

    out = out or settings.index_path
    tmp = out.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    db = connect(tmp)
    dim = embed.dim()
    db.executescript(SCHEMA)
    # Cosine on unit vectors; vec0 rowid = chunks.id.
    db.execute(
        f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{dim}] distance_metric=cosine)"
    )

    used = {c.paper_id for c in chunks}
    paper_rows = [
        (
            p.id,
            p.title,
            json.dumps(p.authors, ensure_ascii=False),
            p.year,
            p.venue,
            p.short_cite,
            p.doi,
        )
        for p in papers
        if p.id in used
    ]
    db.executemany(_insert("papers", PAPER_COLS), paper_rows)
    db.executemany(
        _insert("chunks", CHUNK_COLS), [tuple(getattr(c, k) for k in CHUNK_COLS) for c in chunks]
    )
    db.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")

    rows = db.execute("SELECT id, embed_text FROM chunks ORDER BY id").fetchall()
    vectors = embed.embed_passages([t for _, t in rows])
    db.executemany(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        [(rid, v.tobytes()) for (rid, _), v in zip(rows, vectors, strict=True)],
    )

    corpus_sha = hashlib.sha256("".join(c.content_hash for c in chunks).encode()).hexdigest()
    meta = {
        "embed_model": settings.embed_model,
        "embed_dim": str(dim),
        "query_instruction": settings.query_instruction,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_sha": corpus_sha,
        "n_chunks": str(len(chunks)),
        "parse_backend": settings.parse_backend,
    }
    db.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    db.commit()
    db.close()
    os.replace(tmp, out)
    return meta


def check_meta(db: sqlite3.Connection) -> dict:
    """Refuse to serve an index built with a different embedder (silent drift)."""
    meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    if meta.get("embed_model") != settings.embed_model:
        raise RuntimeError(
            f"index built with {meta.get('embed_model')!r} but EMBED_MODEL is "
            f"{settings.embed_model!r}; rebuild with `make index`"
        )
    return meta

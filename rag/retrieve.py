"""Retrieval primitives. Phase 4: BM25 and dense search. Phase 5 adds RRF fusion,
the diversity cap and `search()`."""

import re
import sqlite3

from rag import embed

# Kept deliberately small: FTS5 BM25 already down-weights common terms; this
# just stops "what/which/the" from contributing noise matches.
STOPWORDS = frozenset(
    """a an and are as at be by did do does for from how in is it of on or that the
    their them these this those to was were what when where which who why with
    paper papers study studies corpus""".split()
)


def fts_query(q: str) -> str:
    """User text -> safe FTS5 MATCH expression.

    Raw input breaks FTS5 syntax: "ARDL(4,4,1)", "Adams-Bashforth", quotes, "NOT".
    Tokenise to \\w+, drop stopwords, quote each token, OR them together.
    """
    tokens = [t for t in re.findall(r"\w+", q.lower()) if t not in STOPWORDS]
    tokens = list(dict.fromkeys(tokens))  # dedupe, keep order
    return " OR ".join(f'"{t}"' for t in tokens)


def bm25(db: sqlite3.Connection, q: str, k: int) -> list[tuple[int, float]]:
    """(chunk rowid, bm25 score). FTS5 bm25() is lower-is-better, so ascending."""
    match = fts_query(q)
    if not match:
        return []
    return db.execute(
        "SELECT rowid, bm25(fts_chunks) AS s FROM fts_chunks WHERE fts_chunks MATCH ? "
        "ORDER BY s LIMIT ?",
        (match, k),
    ).fetchall()


def dense(db: sqlite3.Connection, q: str, k: int) -> list[tuple[int, float]]:
    """(chunk rowid, cosine distance), nearest first."""
    v = embed.embed_query(q)
    return db.execute(
        "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance",
        (v.tobytes(), k),
    ).fetchall()


def describe(db: sqlite3.Connection, rowids: list[int]) -> list[tuple]:
    """rowid -> (chunk_id, paper_id, section, page_start) in the given order."""
    out = []
    for rid in rowids:
        out.append(
            db.execute(
                "SELECT chunk_id, paper_id, section, page_start FROM chunks WHERE id = ?", (rid,)
            ).fetchone()
        )
    return out

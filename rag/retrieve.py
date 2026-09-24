"""Hybrid retrieval: BM25 (FTS5) + dense (sqlite-vec), fused with Reciprocal Rank
Fusion, then a per-paper diversity cap.

    uv run python -m rag.retrieve "which papers use the geopolitical risk index"
"""

import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from rag import embed
from rag.config import settings

# Kept deliberately small: FTS5 BM25 already down-weights common terms; this
# just stops "what/which/the" from contributing noise matches.
STOPWORDS = frozenset(
    """a an and are as at be by did do does for from how in is it of on or that the
    their them these this those to was were what when where which who why with
    paper papers study studies corpus""".split()
)
MAX_PER_PAPER = 3  # in the final top_k; cross-paper questions need breadth


@dataclass
class Hit:
    rowid: int
    chunk_id: str
    paper_id: str
    short_cite: str
    seq: int
    section: str
    heading: str
    page_start: int
    page_end: int
    kind: str
    text: str
    score: float  # fused RRF score, higher is better
    bm25_rank: int | None  # 1-based rank in each list, None if absent
    dense_rank: int | None


# --- primitives -----------------------------------------------------------------


def fts_query(q: str) -> str:
    """User text -> safe FTS5 MATCH expression.

    Raw input breaks FTS5 syntax: "ARDL(4,4,1)", "Adams-Bashforth", quotes, "NOT".
    Tokenise to \\w+, drop stopwords, quote each token, OR them together.
    """
    tokens = [t for t in re.findall(r"\w+", q.lower()) if t not in STOPWORDS]
    tokens = list(dict.fromkeys(tokens))  # dedupe, keep order
    return " OR ".join(f'"{t}"' for t in tokens)


def _filter_sql(collections: list[str] | None, paper_ids: list[str] | None) -> tuple[str, list]:
    clauses, params = [], []
    if collections:
        clauses.append(f"c.collection IN ({','.join('?' * len(collections))})")
        params += collections
    if paper_ids:
        clauses.append(f"c.paper_id IN ({','.join('?' * len(paper_ids))})")
        params += paper_ids
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def bm25(
    db: sqlite3.Connection,
    q: str,
    k: int,
    collections: list[str] | None = None,
    paper_ids: list[str] | None = None,
) -> list[tuple[int, float]]:
    """(chunk rowid, bm25 score). FTS5 bm25() is lower-is-better, so ascending.
    Filters go straight into SQL."""
    match = fts_query(q)
    if not match:
        return []
    where, params = _filter_sql(collections, paper_ids)
    return db.execute(
        "SELECT f.rowid, bm25(fts_chunks) AS s FROM fts_chunks f JOIN chunks c ON c.id = f.rowid "
        f"WHERE fts_chunks MATCH ?{where} ORDER BY s LIMIT ?",
        (match, *params, k),
    ).fetchall()


def dense(
    db: sqlite3.Connection,
    q: str,
    k: int,
    collections: list[str] | None = None,
    paper_ids: list[str] | None = None,
) -> list[tuple[int, float]]:
    """(chunk rowid, cosine distance), nearest first.

    vec0 has no metadata filter, so with filters the KNN over-fetches (4x, at
    least 60) and filters after the join (PLAN_ADDENDUM 13.1). Cheap at this size.
    """
    v = embed.embed_query(q).tobytes()
    if not (collections or paper_ids):
        return db.execute(
            "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (v, k),
        ).fetchall()
    where, params = _filter_sql(collections, paper_ids)
    return db.execute(
        "SELECT v.rowid, v.distance FROM (SELECT rowid, distance FROM vec_chunks "
        "WHERE embedding MATCH ? AND k = ?) v JOIN chunks c ON c.id = v.rowid "
        f"WHERE 1=1{where} ORDER BY v.distance LIMIT ?",
        (v, max(60, 4 * k), *params, k),
    ).fetchall()


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank).

    Uses ranks only, so BM25 scores and cosine distances never need calibrating
    against each other. k=60 (Cormack et al. 2009) flattens the head so one
    list's rank-1 can't dominate.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# --- named-paper detection --------------------------------------------------------


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c)
    )


@lru_cache(maxsize=1)
def _paper_names() -> list[tuple[str, set[str], int]]:
    from rag.manifest import load_manifest

    out = []
    for p in load_manifest():
        names = {_fold(a.split()[-1]) for a in p.authors + p.alt_authors if len(a.split()[-1]) > 2}
        out.append((p.id, names, p.year))
    return out


def named_papers(q: str) -> set[str]:
    """Papers the question names: "p12", or an author surname that identifies one
    paper (optionally with its year: "Noviantri 2023"). Ambiguous names don't count."""
    fq = _fold(q)
    ids = set(re.findall(r"\bp(0[1-9]|1[0-9])\b", fq))
    if ids:
        return {f"p{i}" for i in ids}
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", fq)}
    hits = set()
    for pid, names, year in _paper_names():
        if any(re.search(rf"\b{re.escape(n)}\b", fq) for n in names):
            if not years or year in years:
                hits.add(pid)
    return hits


# --- search -----------------------------------------------------------------------


def search(
    db: sqlite3.Connection,
    q: str,
    top_k: int | None = None,
    paper_ids: list[str] | None = None,
    collections: list[str] | None = None,
    diversity: bool = True,
    use_bm25: bool = True,
    use_dense: bool = True,
) -> list[Hit]:
    """Hybrid search. The use_* flags exist for the Phase 6 ablations."""
    top_k = top_k or settings.top_k
    lists: list[list[int]] = []
    b = bm25(db, q, settings.bm25_k, collections, paper_ids) if use_bm25 else []
    d = dense(db, q, settings.dense_k, collections, paper_ids) if use_dense else []
    if use_bm25:
        lists.append([r for r, _ in b])
    if use_dense:
        lists.append([r for r, _ in d])
    fused = rrf(lists, settings.rrf_k)

    b_rank = {r: i for i, (r, _) in enumerate(b, start=1)}
    d_rank = {r: i for i, (r, _) in enumerate(d, start=1)}
    # The cap is lifted when the question names exactly one paper ("what did
    # Smales find ...") - then depth within that paper is what's wanted.
    cap = MAX_PER_PAPER if diversity and len(named_papers(q)) != 1 else top_k

    rows = {r[0]: r for r in _rows(db, [rid for rid, _ in fused])}
    picked, spill, per_paper = [], [], {}
    for rid, score in fused:
        pid = rows[rid][2]
        if per_paper.get(pid, 0) < cap:
            picked.append((rid, score))
            per_paper[pid] = per_paper.get(pid, 0) + 1
        else:
            spill.append((rid, score))
        if len(picked) == top_k:
            break
    # Fewer papers than the cap needs (tiny filtered corpus): top up in rank order.
    picked += spill[: top_k - len(picked)]

    return [
        Hit(
            *rows[rid],
            score=score,
            bm25_rank=b_rank.get(rid),
            dense_rank=d_rank.get(rid),
        )
        for rid, score in picked
    ]


def _rows(db: sqlite3.Connection, rowids: list[int]) -> list[tuple]:
    if not rowids:
        return []
    return db.execute(
        "SELECT c.id, c.chunk_id, c.paper_id, p.short_cite, c.seq, c.section, c.heading, "
        "c.page_start, c.page_end, c.kind, c.text FROM chunks c JOIN papers p USING (paper_id) "
        f"WHERE c.id IN ({','.join('?' * len(rowids))})",
        rowids,
    ).fetchall()


def neighbours(db: sqlite3.Connection, hit: Hit, span: int = 1) -> list[tuple[int, str]]:
    """Adjacent chunks from the same section ("small-to-big" context for the
    generator): (seq, text) for seq within +/- span, same paper and heading."""
    return db.execute(
        "SELECT seq, text FROM chunks WHERE paper_id = ? AND heading = ? AND kind = 'prose' "
        "AND seq BETWEEN ? AND ? AND seq != ? ORDER BY seq",
        (hit.paper_id, hit.heading, hit.seq - span, hit.seq + span, hit.seq),
    ).fetchall()


def describe(db: sqlite3.Connection, rowids: list[int]) -> list[tuple]:
    """rowid -> (chunk_id, paper_id, section, page_start) in the given order."""
    by_id = {r[0]: (r[1], r[2], r[5], r[7]) for r in _rows(db, rowids)}
    return [by_id[r] for r in rowids]


def main() -> None:
    from rag.index import check_meta, connect

    q = " ".join(sys.argv[1:]) or "which papers use the geopolitical risk index"
    db = connect(readonly=True)
    check_meta(db)
    named = named_papers(q)
    print(f"query: {q!r}" + (f"   (names: {', '.join(sorted(named))})" if named else ""))
    cols = f"{'#':>2}  {'fused':>7}  {'bm25':>4} {'dense':>5}  {'paper':<24} {'section':<18}"
    print(f"{cols} page  heading")
    for i, h in enumerate(search(db, q, collections=["core"]), start=1):
        br = h.bm25_rank or "-"
        dr = h.dense_rank or "-"
        print(
            f"{i:>2}  {h.score:.5f}  {br!s:>4} {dr!s:>5}  {h.paper_id} {h.short_cite[:19]:<20} "
            f"{h.section:<18} p.{h.page_start:<3} {h.heading[:40]}"
        )


if __name__ == "__main__":
    main()

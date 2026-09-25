"""User PDF uploads (PLAN_ADDENDUM Phase 13).

Uploads go into the 'user' collection, never 'core', so the gold-set numbers
can't move (the eval scripts are locked to core; tests/test_eval_guard.py).

    validate(bytes)  cheap checks, in the request: %PDF magic, size, password,
                     page count, text layer, duplicate
    create(...)      save the PDF, add a `documents` row, return a job
    ingest(job)      in a background thread: the same pymupdf parser and chunker
                     as the core papers (no second code path) -> embed -> insert
                     in one transaction, so a failure leaves nothing behind
    delete(id)       chunks, FTS and vector rows, paper row, PDF

Parser: pymupdf (seconds per paper). MinerU is ~8 s/page on CPU: fine offline for
the core corpus, too slow for an upload someone is waiting on.

Jobs are kept in memory (one server process); the documents table is in the
index, so uploaded documents survive restarts.
"""

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag.config import settings

log = logging.getLogger("thesis_rag")

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 60
MAX_DOCS = 20
MIN_ALPHA_PER_PAGE = 200
COLLECTION = "user"

TABLES = """
CREATE TABLE IF NOT EXISTS documents (
  paper_id TEXT PRIMARY KEY, collection TEXT NOT NULL,
  filename TEXT, title TEXT, n_pages INT, n_chunks INT,
  sha256 TEXT UNIQUE, uploaded_at TEXT, status TEXT  -- parsing | ready | failed
);
"""


class Rejected(Exception):
    """A user-safe reason the upload can't be used."""


@dataclass
class Job:
    job_id: str
    paper_id: str
    status: str = "queued"  # queued | parsing | embedding | ready | failed
    stage_detail: str = ""
    error: str = ""


_jobs: dict[str, Job] = {}
_pending: dict[str, "Checked"] = {}  # job_id -> validated metadata, until ingested
_ingest_lock = threading.Lock()  # one ingest at a time: SQLite has one writer


def uploads_dir() -> Path:
    return settings.data_dir / "uploads"


def ensure_tables(db: sqlite3.Connection) -> None:
    db.executescript(TABLES)
    db.commit()


# --- validation ------------------------------------------------------------------


@dataclass
class Checked:
    sha256: str
    n_pages: int
    title: str
    authors: list[str]
    year: int


def validate(data: bytes, filename: str) -> Checked:
    import pymupdf

    if not data.startswith(b"%PDF"):
        raise Rejected("That file isn't a PDF.")
    if len(data) > MAX_BYTES:
        raise Rejected(f"PDFs up to {MAX_BYTES // (1024 * 1024)} MB only.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise Rejected("That PDF couldn't be opened; it may be damaged.") from e
    with doc:
        if doc.needs_pass:
            raise Rejected("This PDF is password-protected.")
        if doc.page_count > MAX_PAGES:
            raise Rejected(f"PDFs up to {MAX_PAGES} pages only.")
        # Text layer: a scanned PDF indexes to nothing, and the bot would then say
        # the document doesn't discuss things it plainly does. Page 1 can be a
        # sparse title page, so it fails only if the middle page is empty too.
        probe = sorted({0, doc.page_count // 2})
        alpha = [sum(ch.isalpha() for ch in doc[i].get_text()) for i in probe]
        if max(alpha) < MIN_ALPHA_PER_PAGE:
            raise Rejected(
                "This looks like a scanned PDF. The system reads text, not images, "
                "so it can't index this."
            )
        meta = doc.metadata or {}
        return Checked(
            sha256=hashlib.sha256(data).hexdigest(),
            n_pages=doc.page_count,
            title=_title(meta.get("title", ""), doc[0], filename),
            authors=_authors(meta.get("author", "")),
            year=_year(meta.get("creationDate", "")),
        )


def _title(meta_title: str, page, filename: str) -> str:
    t = " ".join(meta_title.split())
    if len(t) >= 8 and not re.match(r"(microsoft word|untitled|\S+\.(docx?|pdf|tex))", t, re.I):
        return t[:200]
    # Largest text in the top half of page 1 is usually the title.
    spans = [
        (round(s["size"], 1), s["text"].strip())
        for b in page.get_text("dict")["blocks"]
        for line in b.get("lines", [])
        for s in line["spans"]
        if s["text"].strip() and s["bbox"][1] < page.rect.height / 2
    ]
    if spans:
        top = max(size for size, _ in spans)
        t = " ".join(text for size, text in spans if size == top)
        if len(t) >= 8:
            return " ".join(t.split())[:200]
    return Path(filename).stem.replace("_", " ")[:200] or "Uploaded document"


def _authors(meta_author: str) -> list[str]:
    names = [a.strip() for a in re.split(r";|,|\band\b|&", meta_author) if a.strip()]
    return names[:10] or ["Unknown author"]


def _year(creation: str) -> int:
    m = re.match(r"D:(\d{4})", creation or "")
    return int(m.group(1)) if m else datetime.now(UTC).year


# --- create / ingest -------------------------------------------------------------


def documents(db: sqlite3.Connection) -> list[dict]:
    cols = ("paper_id", "filename", "title", "n_pages", "n_chunks", "uploaded_at", "status")
    rows = db.execute(f"SELECT {', '.join(cols)} FROM documents ORDER BY paper_id").fetchall()
    return [dict(zip(cols, r, strict=True)) for r in rows]


def _next_id(db: sqlite3.Connection) -> str:
    used = {r[0] for r in db.execute("SELECT paper_id FROM documents")}
    return next(f"u{i:02d}" for i in range(1, 100) if f"u{i:02d}" not in used)


def create(db: sqlite3.Connection, data: bytes, filename: str) -> tuple[Job | None, str]:
    """Validate and register an upload. Returns (job, paper_id); job is None when
    the same file was uploaded before (its existing id is returned instead)."""
    checked = validate(data, filename)
    dup = db.execute(
        "SELECT paper_id FROM documents WHERE sha256 = ? AND status != 'failed'",
        (checked.sha256,),
    ).fetchone()
    if dup:
        return None, dup[0]
    db.execute("DELETE FROM documents WHERE sha256 = ? AND status = 'failed'", (checked.sha256,))
    n = db.execute("SELECT count(*) FROM documents").fetchone()[0]
    if n >= MAX_DOCS:
        raise Rejected(f"There are already {MAX_DOCS} uploads. Remove one first.")
    paper_id = _next_id(db)
    uploads_dir().mkdir(parents=True, exist_ok=True)
    (uploads_dir() / f"{paper_id}.pdf").write_bytes(data)
    db.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, 0, ?, ?, 'parsing')",
        (
            paper_id,
            COLLECTION,
            Path(filename).name[:200],
            checked.title,
            checked.n_pages,
            checked.sha256,
            datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    job = Job(uuid.uuid4().hex[:12], paper_id)
    _jobs[job.job_id] = job
    _pending[job.job_id] = checked
    return job, paper_id


def get_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    return asdict(job) if job else None


def _paper(paper_id: str, c: Checked, filename: str):
    from rag.manifest import Paper

    surname = c.authors[0].split()[-1] if c.authors[0] != "Unknown author" else ""
    if surname:
        cite = surname + (" et al." if len(c.authors) > 2 else "") + f" {c.year}"
    else:
        cite = c.title if len(c.title) <= 40 else c.title[:39].rsplit(" ", 1)[0] + "…"
    return Paper(
        id=paper_id,
        file=filename,
        sha256=c.sha256,
        title=c.title,
        authors=c.authors,
        year=c.year,
        venue="Uploaded document",
        short_cite=cite,
        layout="one_column",  # column order is detected per page anyway
        open_access=False,
    )


def ingest(job_id: str, index_path: Path | None = None) -> None:
    """Background task. Everything is written in one transaction at the end."""
    from rag import embed, storage
    from rag.chunk import chunk_doc
    from rag.index import CHUNK_COLS, _insert, connect
    from rag.parse.pipeline import parse_pdf

    job, checked = _jobs[job_id], _pending.pop(job_id)
    pdf = uploads_dir() / f"{job.paper_id}.pdf"
    with _ingest_lock:
        db = connect(index_path)
        try:
            filename = db.execute(
                "SELECT filename FROM documents WHERE paper_id = ?", (job.paper_id,)
            ).fetchone()[0]
            paper = _paper(job.paper_id, checked, filename)
            job.status, job.stage_detail = "parsing", "parsing…"
            chunks = chunk_doc(paper, parse_pdf(paper, pdf))
            if not chunks:
                raise Rejected("No text could be extracted from this PDF.")

            job.status = "embedding"
            vectors = []
            for i in range(0, len(chunks), 16):
                job.stage_detail = f"embedding {i}/{len(chunks)}"
                vectors.extend(embed.embed_passages([c.embed_text for c in chunks[i : i + 16]]))
            job.stage_detail = f"embedding {len(chunks)}/{len(chunks)}"

            with db:  # one transaction: all or nothing
                db.execute(
                    "INSERT INTO papers (paper_id, title, authors, year, venue, short_cite, doi, "
                    "collection) VALUES (?, ?, ?, ?, ?, ?, '', ?)",
                    (
                        paper.id,
                        paper.title,
                        json.dumps(paper.authors, ensure_ascii=False),
                        paper.year,
                        paper.venue,
                        paper.short_cite,
                        COLLECTION,
                    ),
                )
                for c, v in zip(chunks, vectors, strict=True):
                    cur = db.execute(
                        _insert("chunks", (*CHUNK_COLS, "collection")),
                        (*(getattr(c, k) for k in CHUNK_COLS), COLLECTION),
                    )
                    rid = cur.lastrowid
                    db.execute(
                        "INSERT INTO fts_chunks (rowid, embed_text) VALUES (?, ?)",
                        (rid, c.embed_text),
                    )
                    db.execute(
                        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                        (rid, v.tobytes()),
                    )
                db.execute(
                    "UPDATE documents SET status = 'ready', n_chunks = ? WHERE paper_id = ?",
                    (len(chunks), paper.id),
                )
            job.status, job.stage_detail = "ready", f"ready · {len(chunks)} passages"
            log.info("upload %s ready: %d chunks", paper.id, len(chunks))
        except Exception as e:
            job.status = "failed"
            job.error = str(e) if isinstance(e, Rejected) else "Couldn't read this PDF."
            log.exception("upload %s failed", job.paper_id)
            db.execute("UPDATE documents SET status = 'failed' WHERE paper_id = ?", (job.paper_id,))
            db.commit()
            pdf.unlink(missing_ok=True)
            return
        finally:
            db.close()
    storage.push_index()
    storage.push_pdf(pdf, f"uploads/{job.paper_id}.pdf")


def delete(db: sqlite3.Connection, paper_id: str) -> bool:
    """Remove an upload everywhere. Core papers can't be deleted this way."""
    row = db.execute(
        "SELECT paper_id FROM documents WHERE paper_id = ? AND collection = ?",
        (paper_id, COLLECTION),
    ).fetchone()
    if not row:
        return False
    with _ingest_lock, db:
        rows = db.execute(
            "SELECT id, embed_text FROM chunks WHERE paper_id = ? AND collection = ?",
            (paper_id, COLLECTION),
        ).fetchall()
        for rid, text in rows:
            # External-content FTS5 needs the old text to remove its terms.
            db.execute(
                "INSERT INTO fts_chunks (fts_chunks, rowid, embed_text) VALUES ('delete', ?, ?)",
                (rid, text),
            )
            db.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rid,))
        db.execute(
            "DELETE FROM chunks WHERE paper_id = ? AND collection = ?", (paper_id, COLLECTION)
        )
        db.execute(
            "DELETE FROM papers WHERE paper_id = ? AND collection = ?", (paper_id, COLLECTION)
        )
        db.execute("DELETE FROM documents WHERE paper_id = ?", (paper_id,))
    (uploads_dir() / f"{paper_id}.pdf").unlink(missing_ok=True)
    from rag import storage

    storage.push_index()
    return True

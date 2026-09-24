"""Daily question cap + question log, in their own SQLite file.

Separate from index.sqlite, which the API opens read-only. The daily counter is
the spend guard for a public demo that costs money per call (PLAN.md Phase 9,
PLAN_ADDENDUM 14.1): it survives restarts, unlike in-memory rate limits.

Privacy: the log keeps the question, latency, citations and token counts for
30 days. No IP addresses are stored.
"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag.config import settings

RETENTION_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily (day TEXT PRIMARY KEY, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS log (
  ts TEXT, question TEXT, model TEXT, latency_ms INTEGER, cited TEXT,
  input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, refused INTEGER, error TEXT
);
"""


def _path() -> Path:
    return settings.usage_db_path


def _connect() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10, isolation_level=None)  # explicit transactions
    db.executescript(SCHEMA)
    return db


def today() -> str:
    return datetime.now(UTC).date().isoformat()


def reserve() -> bool:
    """Count one question against today's cap. False if the cap is reached.
    BEGIN IMMEDIATE makes check-and-increment atomic across threads/processes."""
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT n FROM daily WHERE day = ?", (today(),)).fetchone()
        n = row[0] if row else 0
        if n >= settings.daily_question_cap:
            db.execute("ROLLBACK")
            return False
        db.execute(
            "INSERT INTO daily (day, n) VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET n = n + 1",
            (today(),),
        )
        db.execute("COMMIT")
        return True
    finally:
        db.close()


def used_today() -> int:
    db = _connect()
    try:
        row = db.execute("SELECT n FROM daily WHERE day = ?", (today(),)).fetchone()
        return row[0] if row else 0
    finally:
        db.close()


def log(
    question: str,
    model: str,
    latency_ms: int,
    cited: list[int],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    refused: bool = False,
    error: str = "",
) -> None:
    db = _connect()
    try:
        now = datetime.now(UTC)
        db.execute(
            "INSERT INTO log VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), question, model, latency_ms, json.dumps(cited),
             input_tokens, output_tokens, cost_usd, int(refused), error),
        )  # fmt: skip
        cutoff = (now - timedelta(days=RETENTION_DAYS)).isoformat()
        db.execute("DELETE FROM log WHERE ts < ?", (cutoff,))
    finally:
        db.close()

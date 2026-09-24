"""HTTP API (PLAN.md Phase 9; access code from PLAN_ADDENDUM 14.1).

    uv run uvicorn app.main:app --port 8000

GET  /health       {ok, chunks, chunks_core, chunks_user, embed_model, built_at}
GET  /api/papers   manifest list (id, short_cite, title, year, venue, doi)
POST /api/ask      {"question": str}  ->  SSE: sources, token*, done | error

Guards on /api/ask, in order: access code (401), question length (400), rate
limit per access code (429), daily question cap (429). Same-origin only: no
CORS middleware, so browsers on other sites can't call it.
"""

import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from rag import embed, usage
from rag.config import settings
from rag.generate import client as anthropic_client
from rag.generate import stream_answer
from rag.index import check_meta, connect
from rag.manifest import load_manifest

log = logging.getLogger("thesis_rag")
STATIC = Path(__file__).resolve().parent / "static"
CORE = ["core"]  # Phase 13 adds the "user" collection and a scope toggle
BUDGET_USED = "The demo's question budget for today is used up. Please try again tomorrow."


def rate_key(request: Request) -> str:
    """Limit per access code (PLAN_ADDENDUM 14.1), not per IP: the demo link is
    shared, so the code is what identifies a budget. Falls back to IP when no
    code is configured (local development)."""
    code = request.headers.get("x-access-code")
    return f"code:{code}" if settings.access_code and code else get_remote_address(request)


limiter = Limiter(key_func=rate_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Refuse to start on an index built with a different embedder; then warm the
    # ONNX session so the first real question isn't slow (PLAN_ADDENDUM 11.5).
    db = connect(readonly=True)
    app.state.meta = check_meta(db)
    counts = dict(
        db.execute("SELECT collection, count(*) FROM chunks GROUP BY collection").fetchall()
    )
    app.state.counts = counts
    db.close()
    embed.embed_query("warm-up")
    app.state.papers = [
        {"id": p.id, "short_cite": p.short_cite, "title": p.title, "year": p.year,
         "venue": p.venue, "doi": p.doi}
        for p in load_manifest()
    ]  # fmt: skip
    log.info("ready: chunks core=%s user=%s", counts.get("core", 0), counts.get("user", 0))
    yield


app = FastAPI(
    title="thesis-rag", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def too_fast(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        {"error": "Too many questions in a short time. Please wait a minute and try again."},
        status_code=429,
    )


def get_llm() -> anthropic.Anthropic:
    """Dependency so tests can swap in a fake client (no API calls, no cost).
    FAKE_LLM=1 does the same for UI work and key-less demos."""
    if settings.fake_llm:
        from rag.fake_llm import FakeAnthropic

        return FakeAnthropic()  # type: ignore[return-value]
    return anthropic_client()


class AskBody(BaseModel):
    question: str = ""


@app.get("/health")
def health() -> dict:
    counts = app.state.counts
    return {
        "ok": True,
        "chunks": sum(counts.values()),
        "chunks_core": counts.get("core", 0),
        "chunks_user": counts.get("user", 0),
        "embed_model": app.state.meta["embed_model"],
        "built_at": app.state.meta.get("built_at"),
    }


@app.get("/api/papers")
def papers() -> list[dict]:
    return app.state.papers


@app.post("/api/ask")
@limiter.limit(settings.rate_limit)
def ask(
    request: Request,
    body: AskBody,
    x_access_code: Annotated[str | None, Header()] = None,
    llm: Annotated[anthropic.Anthropic, Depends(get_llm)] = None,
):
    if settings.access_code and not secrets.compare_digest(
        (x_access_code or "").encode(), settings.access_code.encode()
    ):
        return JSONResponse({"error": "That access code isn't right."}, status_code=401)
    question = body.question.strip()
    if not question:
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    if len(question) > settings.max_question_chars:
        msg = f"Please keep questions under {settings.max_question_chars} characters."
        return JSONResponse({"error": msg}, status_code=400)
    if not usage.reserve():
        return JSONResponse({"error": BUDGET_USED}, status_code=429)
    return EventSourceResponse(_events(question, llm))


def _events(question: str, llm: anthropic.Anthropic):
    """Generator run by sse-starlette in a worker thread. A connection per request:
    sqlite3 connections aren't meant to be shared across threads."""
    db = connect(readonly=True)
    result = None
    error = ""
    try:
        for kind, payload in stream_answer(db, question, collections=CORE, llm=llm):
            if kind == "sources":
                yield {"event": "sources", "data": json.dumps(payload, ensure_ascii=False)}
            elif kind == "token":
                yield {"event": "token", "data": json.dumps(payload, ensure_ascii=False)}
            else:
                result = payload
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "cited": result.cited,
                            "latency_ms": result.latency_ms,
                            "refused": result.refused,
                            "text": result.text,  # citation-validated final text
                        },
                        ensure_ascii=False,
                    ),
                }
    except anthropic.RateLimitError:
        error = "rate_limited"
        yield _error("The model is busy right now. Please try again in a minute.")
    except anthropic.APIConnectionError:
        error = "connection"
        yield _error("Couldn't reach the model. Please try again.")
    except anthropic.APIStatusError as e:
        error = f"api_{e.status_code}"
        log.warning("anthropic error %s: %s", e.status_code, e.message)
        yield _error("Something went wrong answering that. Please try again.")
    except Exception:
        error = "internal"
        log.exception("answer failed")
        yield _error("Something went wrong answering that. Please try again.")
    finally:
        db.close()
        usage.log(
            question=question,
            model=result.model if result else settings.llm_model,
            latency_ms=result.latency_ms if result else 0,
            cited=result.cited if result else [],
            input_tokens=result.input_tokens if result else 0,
            output_tokens=result.output_tokens if result else 0,
            cost_usd=result.cost_usd if result else 0.0,
            refused=result.refused if result else False,
            error=error,
        )


def _error(message: str) -> dict:
    # User-safe message only; details go to the server log.
    return {"event": "error", "data": json.dumps({"message": message})}


# Frontend (Phase 10). Mounted last so the API routes take precedence.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

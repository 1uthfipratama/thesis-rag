"""HTTP API (PLAN.md Phase 9; access code from PLAN_ADDENDUM 14.1).

    uv run uvicorn app.main:app --port 8000

GET  /health       {ok, chunks, chunks_core, chunks_user, embed_model, built_at}
GET  /api/papers   manifest list (id, short_cite, title, year, venue, doi)
POST /api/ask      {"question": str}  ->  SSE: sources, token*, done | error
POST /api/chat     {"message": str, "history": [{role, content}], "reuse_ids": [int]}
                   ->  same events; done adds passage_ids, search_query

Guards on both, in order: access code (401), question length (400), rate
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
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from rag import embed, usage
from rag.chat import stream_chat
from rag.config import settings
from rag.generate import client as anthropic_client
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


class Turn(BaseModel):
    role: str
    content: str = ""


class ChatBody(BaseModel):
    message: str = ""
    # The browser keeps the conversation (tab-scoped sessionStorage) and sends it
    # back; the server stays stateless. Bounded here, trimmed further in generate.
    history: list[Turn] = Field(default_factory=list, max_length=40)
    reuse_ids: list[int] = Field(default_factory=list, max_length=12)


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
    denied = _guard(x_access_code, body.question)
    if denied:
        return denied
    question = body.question.strip()
    return EventSourceResponse(_events(question, llm))


@app.post("/api/chat")
@limiter.limit(settings.rate_limit)
def chat(
    request: Request,
    body: ChatBody,
    x_access_code: Annotated[str | None, Header()] = None,
    llm: Annotated[anthropic.Anthropic, Depends(get_llm)] = None,
):
    """A conversation turn. Same guards and SSE events as /api/ask; `done` also
    carries passage_ids (for "rephrase that" follow-ups) and the search used."""
    denied = _guard(x_access_code, body.message)
    if denied:
        return denied
    history = [
        {"role": t.role, "content": t.content[: settings.max_history_chars]}
        for t in body.history
        if t.role in ("user", "assistant")
    ]
    return EventSourceResponse(
        _events(body.message.strip(), llm, history=history, reuse_ids=body.reuse_ids)
    )


def _guard(code: str | None, message: str) -> JSONResponse | None:
    if settings.access_code and not secrets.compare_digest(
        (code or "").encode(), settings.access_code.encode()
    ):
        return JSONResponse({"error": "That access code isn't right."}, status_code=401)
    message = message.strip()
    if not message:
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    if len(message) > settings.max_question_chars:
        msg = f"Please keep questions under {settings.max_question_chars} characters."
        return JSONResponse({"error": msg}, status_code=400)
    if not usage.reserve():
        return JSONResponse({"error": BUDGET_USED}, status_code=429)
    return None


def _events(
    question: str,
    llm: anthropic.Anthropic,
    history: list[dict] | None = None,
    reuse_ids: list[int] | None = None,
):
    """Generator run by sse-starlette in a worker thread. A connection per request:
    sqlite3 connections aren't meant to be shared across threads."""
    db = connect(readonly=True)
    result = None
    error = ""
    try:
        turns = stream_chat(
            db, question, history=history, reuse_ids=reuse_ids, collections=CORE, llm=llm
        )
        for kind, payload in turns:
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
                            "passage_ids": result.passage_ids,
                            "search_query": result.search_query,
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

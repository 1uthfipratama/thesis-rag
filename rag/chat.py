"""Chat turns: decide what to search for, then answer with the conversation in view.

Follow-ups like "explain that like I'm 12" or "what about the second paper?" carry
nothing searchable on their own. Before retrieval, a small Haiku call rewrites the
latest message into a standalone query ("SEARCH: ...") or says the user only wants
the previous answer rephrased ("SAME"), in which case the previous turn's passages
are rebuilt from their chunk ids instead of searching again.

The first message of a conversation skips this call (nothing to resolve), so a
plain question costs exactly what it did in single-turn mode.
"""

import re
import sqlite3
from collections.abc import Iterator

import anthropic

from rag.config import settings
from rag.generate import client, cost_usd, stream_answer, strip_markers

REWRITE_MODEL = "claude-haiku-4-5"  # cheapest; this is a one-line rewrite

REWRITE_PROMPT = """You prepare searches for a question-answering assistant over 19 research papers
(stock-price models with differential equations, volatility, geopolitical risk,
numerical methods).

Given the recent conversation and the user's latest message, reply with exactly
one line:
- "SEARCH: <query>" - a standalone search query for the latest message, with
  pronouns and references ("it", "that model", "the second paper") replaced by
  the specific paper, author, model or topic from the conversation. Keep the
  user's key terms.
- "SAME" - only if the latest message just asks to rephrase, simplify, shorten,
  expand, translate or reformat the previous answer, with no new information needed."""

_LINE = re.compile(r"^\s*(SEARCH:\s*(?P<q>.+?)|(?P<same>SAME))\s*$", re.I | re.M)


def _transcript(history: list[dict], message: str) -> str:
    lines = []
    for h in history[-6:]:
        text = " ".join(strip_markers(h.get("content", "")).split())[:600]
        lines.append(f"{h.get('role', 'user').upper()}: {text}")
    return "Conversation:\n" + "\n".join(lines) + f"\n\nLatest message: {message}"


def plan_retrieval(
    llm: anthropic.Anthropic, history: list[dict], message: str, can_reuse: bool
) -> tuple[str | None, bool, dict]:
    """(search query, reuse previous passages?, usage). Falls back to searching
    the message plus the previous question if the rewrite fails or is unusable."""
    last_q = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
    fallback = f"{last_q} {message}".strip()
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        resp = llm.messages.create(
            model=REWRITE_MODEL,
            max_tokens=120,
            system=REWRITE_PROMPT,
            messages=[{"role": "user", "content": _transcript(history, message)}],
            extra_body={"temperature": 0.0},  # not a 1.x SDK kwarg; see generate.request_params
        )
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        text = next((b.text for b in resp.content if b.type == "text"), "")
    except anthropic.APIError:
        return fallback, False, usage
    m = _LINE.search(text)
    if not m:
        return fallback, False, usage
    if m.group("same"):
        return (None, True, usage) if can_reuse else (fallback, False, usage)
    return m.group("q").strip()[: settings.max_question_chars], False, usage


def stream_chat(
    db: sqlite3.Connection,
    message: str,
    history: list[dict] | None = None,
    reuse_ids: list[int] | None = None,
    collections: list[str] | None = None,
    llm: anthropic.Anthropic | None = None,
    model: str | None = None,
) -> Iterator[tuple[str, object]]:
    """Same events as generate.stream_answer; the final Answer also carries the
    search query used (or None when passages were reused) and the chunk ids."""
    history = history or []
    llm = llm or client()
    query, reuse, rewrite_usage = message, False, {}
    if history:
        query, reuse, rewrite_usage = plan_retrieval(llm, history, message, bool(reuse_ids))
    for kind, payload in stream_answer(
        db,
        message,
        model=model,
        collections=collections,
        llm=llm,
        history=history,
        search_query=query,
        reuse_ids=reuse_ids if reuse else None,
    ):
        if kind == "done" and rewrite_usage:
            payload.input_tokens += rewrite_usage["input_tokens"]
            payload.output_tokens += rewrite_usage["output_tokens"]
            payload.cost_usd += cost_usd(REWRITE_MODEL, **rewrite_usage)
        yield kind, payload

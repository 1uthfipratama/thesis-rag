"""Answer generation (PLAN.md Phase 7): retrieved passages -> Claude -> cited answer.

    uv run python -m rag.generate "What was the fit MAPE for the IDX Composite logistic model?"

Flow: hybrid search -> each hit widened with its neighbouring chunks from the
same section ("small-to-big": chunks are small for precise matching, the LLM
gets more context) -> numbered passages -> streamed answer -> citation markers
validated against the passages actually provided.

`stream_answer()` yields the same events the API streams over SSE (Phase 9):
("sources", [...]) first, then ("token", text)*, then ("done", {...}).
"""

import re
import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import anthropic

from rag.config import settings
from rag.retrieve import Hit, neighbours, search

REFUSAL = "The corpus doesn't cover this."

# PLAN.md Phase 7 prompt, with changes (DECISIONS.md):
# - equations are indexed as LaTeX now, so they may be quoted verbatim, never derived;
# - after the first Phase 8 run: the plan's all-or-nothing refusal rule made Haiku
#   refuse when the passages held part of the answer (q26, q38, q64), and phrase
#   "the paper reports no MAPE" (q40) as a refusal. Partial answers are now
#   required; refusals may not quote figures (q53); lists get a larger word budget.
# Chat mode (DECISIONS.md): the plan's "passages only, no outside knowledge" became
# two tiers, so "explain like I'm 12" works without letting general knowledge pass
# as a paper's finding.
SYSTEM_PROMPT = f"""You are a friendly research assistant for a fixed corpus of 19 research
papers on stock-price modelling with differential equations, volatility, geopolitical risk,
and numerical methods. You are in a conversation: earlier turns are context, and
each new message comes with freshly numbered passages from the papers.

Two kinds of statements, kept strictly apart:
1. About the papers (what a paper did, used, found, reported, or what its numbers
   and equations are): only from the numbered passages of the current message,
   with citation markers at the end of the sentence, like [2] or [1][3]. Report
   numbers exactly as written, with units and the paper they come from.
2. General background that helps explain (what a derivative, volatility or MAPE
   is; an everyday analogy): you may use your own knowledge. Signal it ("In
   general, ...", "Think of it like ...") and do not cite it. Never present
   general knowledge as something a paper says.

Follow the user's requested style: simpler ("explain like I'm 12"), shorter,
longer, bullet points, step by step, or another language. Simplify the wording,
never the facts: keep the papers' numbers and findings accurate and cited.

When the papers don't cover it:
- If the passages contain nothing relevant to a question about the papers, reply
  exactly: "{REFUSAL}" Then, in one sentence and without quoting any figures, say
  what related topic the corpus does cover, if any.
- If they answer only part, answer that part with citations and say briefly what
  they don't include. Do not refuse.
- If the user asks whether a paper reports something and it doesn't, say so
  plainly with citations. That is an answer, not a refusal.
- A general concept question ("what is a differential equation?") is answered from
  background knowledge, then linked to the papers with citations if the passages
  are relevant.

Also:
- If passages from different papers disagree, say so and attribute each view. If
  a paper's abstract and its tables disagree, trust the table and say so.
- Equations appear as LaTeX between $$ or $ signs. Quote a paper's equation exactly
  as given and explain it in words. Never derive, simplify or invent an equation
  and attribute it to a paper.
- For "which papers" questions, cover every relevant paper in the passages with
  its key finding.
- Keep it conversational and concise: about 200 words unless the user asks for
  more or the question needs a list. Short **bold** and "- " bullet lists are
  fine; no headings."""

# USD per million tokens (input, output); for cost logging only.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
}

_MARKER = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
SNIPPET_CHARS = 300


@dataclass
class Passage:
    n: int
    hit: Hit
    text: str  # chunk text, widened with neighbours
    attached: bool = False  # a referenced table added after retrieval


@dataclass
class Answer:
    question: str
    text: str  # with invalid citation markers removed
    cited: list[int]
    sources: list[dict]
    passages: list[Passage]
    refused: bool
    uncited: bool  # answered (not a refusal) but no valid citation: logged as a failure
    stop_reason: str | None
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    dropped_markers: list[int] = field(default_factory=list)
    passage_ids: list[int] = field(default_factory=list)  # for "rephrase that" follow-ups
    search_query: str | None = None  # what was searched (None = previous passages reused)


# --- context -----------------------------------------------------------------------


def build_passages(
    db: sqlite3.Connection, hits: list[Hit], span: int | None = None
) -> list[Passage]:
    """Number the hits and widen each with same-section neighbours, never showing
    the same chunk twice (a neighbour that is itself a hit stays its own passage)."""
    span = settings.context_neighbours if span is None else span
    shown = {(h.paper_id, h.seq) for h in hits}
    passages = []
    for n, h in enumerate(hits, start=1):
        before, after = [], []
        if span and h.kind == "prose":
            for seq, text in neighbours(db, h, span):
                if (h.paper_id, seq) in shown:
                    continue
                shown.add((h.paper_id, seq))
                (before if seq < h.seq else after).append(text)
        passages.append(Passage(n, h, "\n\n".join([*before, h.text, *after])))
    if settings.attach_tables:
        passages += referenced_tables(db, passages, shown)
    return passages


_TABLE_REF = re.compile(r"\bTable\s+(\d+|[IVX]+)\b")


def referenced_tables(
    db: sqlite3.Connection, passages: list[Passage], shown: set[tuple[str, int]]
) -> list[Passage]:
    """Tables that retrieved passages refer to, from the same paper, as extra passages.

    Tables are mostly digits, so neither BM25 nor the embedder finds them well:
    9 of 16 evidence misses were table cells (q27's 0.1438 sits in Table 5, which
    the retrieved prose says "is reported in Table 5"). Following the reference is
    the table equivalent of neighbour expansion. Bounded by max_attached_tables.
    """
    from rag.retrieve import Hit, _rows

    extra: list[Passage] = []
    for p in passages:
        for num in dict.fromkeys(_TABLE_REF.findall(p.text)):
            caption = re.compile(rf"^(Table|TABLE)\s+{re.escape(num)}\b")
            rows = db.execute(
                "SELECT id, seq, heading FROM chunks WHERE paper_id = ? AND kind = 'table' "
                "ORDER BY seq",
                (p.hit.paper_id,),
            ).fetchall()
            for rid, seq, heading in rows:
                if not caption.match(heading) or (p.hit.paper_id, seq) in shown:
                    continue
                if len(extra) >= settings.max_attached_tables:
                    return extra
                shown.add((p.hit.paper_id, seq))
                hit = Hit(*_rows(db, [rid])[0], score=0.0, bm25_rank=None, dense_rank=None)
                extra.append(Passage(len(passages) + len(extra) + 1, hit, hit.text, attached=True))
    return extra


def format_context(passages: list[Passage]) -> str:
    blocks = []
    for p in passages:
        h = p.hit
        pages = (
            f"p.{h.page_start}" if h.page_start == h.page_end else f"pp.{h.page_start}-{h.page_end}"
        )
        blocks.append(f"[{p.n}] {h.paper_id} · {h.short_cite} · {h.heading} · {pages}\n{p.text}")
    return "\n\n".join(blocks)


# Uploads (Phase 13) aren't among the 19 papers the system prompt describes.
SCOPE_NOTES = {
    ("user",): "Scope: documents the user uploaded (ids u01, u02, ...), not the 19 core papers.",
    ("core", "user"): "Scope: the 19 core papers plus documents the user uploaded (ids u..).",
}


def user_message(
    question: str, passages: list[Passage], collections: list[str] | None = None
) -> str:
    note = SCOPE_NOTES.get(tuple(sorted(collections or ["core"])), "")
    head = f"{note}\n\n" if note else ""
    return f"{head}Passages:\n\n{format_context(passages)}\n\nQuestion: {question}"


# --- request -----------------------------------------------------------------------


def request_params(model: str) -> dict:
    """Per-model request settings.

    - Haiku 4.5 takes sampling params: temperature 0.2 per the plan. anthropic 1.x
      removed `temperature` from the Python signatures (a TypeError) but not from
      the API, so it travels in extra_body, which is merged into the request JSON.
    - Sonnet 5 / Opus 5 reject temperature (400) and think by default; thinking is
      off unless settings.llm_thinking == "adaptive", because thinking tokens count
      against max_tokens and add latency to a streamed answer.
    """
    params: dict = {"model": model, "max_tokens": settings.answer_max_tokens}
    if model.startswith("claude-haiku-4-5"):
        params["extra_body"] = {"temperature": 0.2}
    elif settings.llm_thinking == "adaptive":
        params["thinking"] = {"type": "adaptive"}
        params["max_tokens"] = max(settings.answer_max_tokens, 8000)
    else:
        params["thinking"] = {"type": "disabled"}
    return params


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = next((v for k, v in PRICES.items() if model.startswith(k)), None)
    return 0.0 if price is None else (input_tokens * price[0] + output_tokens * price[1]) / 1e6


# --- post-processing -----------------------------------------------------------------


def validate_citations(text: str, n_passages: int) -> tuple[str, list[int], list[int]]:
    """Keep markers that point at provided passages; drop the rest.
    Returns (clean text, cited passage numbers in first-use order, dropped numbers)."""
    cited: list[int] = []
    dropped: list[int] = []

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*[,;]\s*", m.group(1))]
        ok = [n for n in nums if 1 <= n <= n_passages]
        dropped.extend(n for n in nums if n not in ok)
        for n in ok:
            if n not in cited:
                cited.append(n)
        return "".join(f"[{n}]" for n in ok) or "\x00"  # sentinel: marker removed

    clean = _MARKER.sub(fix, text)
    clean = re.sub(r"[ \t]*\x00", "", clean)  # and the space that preceded it
    return clean, cited, dropped


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """At most `limit` characters, cut at a word boundary (UI shows short snippets only)."""
    t = " ".join(text.split())
    if len(t) <= limit:
        return t
    return t[: limit - 1].rsplit(" ", 1)[0] + "…"


def source_card(p: Passage, db: sqlite3.Connection) -> dict:
    """What the UI shows for a passage: citation line, <= 300-char snippet, DOI."""
    h = p.hit
    row = db.execute(
        "SELECT doi, collection FROM papers WHERE paper_id = ?", (h.paper_id,)
    ).fetchone()
    return {
        "n": p.n,
        "paper_id": h.paper_id,
        "short_cite": h.short_cite,
        "section": h.section,
        "heading": h.heading,
        "page_start": h.page_start,
        "snippet": snippet(h.text),
        "doi": (row[0] if row else "") or "",
        "collection": row[1] if row else "core",
    }


def sources_for(passages: list[Passage], cited: list[int], db: sqlite3.Connection) -> list[dict]:
    by_n = {p.n: p for p in passages}
    return [source_card(by_n[n], db) for n in cited]


# --- generation --------------------------------------------------------------------


def client() -> anthropic.Anthropic:
    # Credentials resolve from the environment (ANTHROPIC_API_KEY or an `ant auth`
    # profile); .env is read by pydantic-settings.
    if settings.anthropic_api_key:
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return anthropic.Anthropic()


HISTORY_MESSAGES = 8  # last 4 exchanges are sent back to the model
HISTORY_CHARS = 2000  # per message; old answers are context, not the source of truth


def strip_markers(text: str) -> str:
    return re.sub(r"[ \t]*" + _MARKER.pattern, "", text).strip()


def history_messages(history: list[dict] | None) -> list[dict]:
    """Recent turns as alternating user/assistant messages. Old answers lose their
    [n] markers: those numbers pointed at passages that are no longer in view, and
    the model must cite only the passages of the current turn."""
    turns = [
        {"role": h["role"], "content": strip_markers(h["content"])[:HISTORY_CHARS]}
        for h in (history or [])
        if h.get("role") in ("user", "assistant") and h.get("content", "").strip()
    ][-HISTORY_MESSAGES:]
    while turns and turns[0]["role"] != "user":  # the API wants a user message first
        turns.pop(0)
    out: list[dict] = []
    for t in turns:  # merge accidental repeats so roles alternate
        if out and out[-1]["role"] == t["role"]:
            out[-1]["content"] += "\n\n" + t["content"]
        else:
            out.append(t)
    if out and out[-1]["role"] == "user":  # a user turn left without an answer
        out.pop()
    return out


def passages_for(
    db: sqlite3.Connection,
    query: str,
    collections: list[str],
    reuse_ids: list[int] | None = None,
) -> list[Passage]:
    """Search for `query`, or rebuild the previous turn's passages from their chunk
    ids when a follow-up only asks to rephrase ("explain it more simply")."""
    from rag.retrieve import _rows

    if reuse_ids:
        rows = {r[0]: r for r in _rows(db, reuse_ids)}
        hits = [
            Hit(*rows[i], score=0.0, bm25_rank=None, dense_rank=None)
            for i in reuse_ids
            if i in rows
        ]
        if hits:
            return build_passages(db, hits)
    return build_passages(db, search(db, query, collections=collections))


def stream_answer(
    db: sqlite3.Connection,
    question: str,
    model: str | None = None,
    collections: list[str] | None = None,
    llm: anthropic.Anthropic | None = None,
    history: list[dict] | None = None,
    search_query: str | None = None,
    reuse_ids: list[int] | None = None,
) -> Iterator[tuple[str, object]]:
    """Yield ("sources", passages-as-dicts), ("token", str)..., ("done", Answer).

    Single-turn (the eval path) when history/search_query/reuse_ids are unset.
    Chat turns pass the conversation, a standalone search query from rag.chat,
    or the previous turn's chunk ids to reuse."""
    t0 = time.perf_counter()
    model = model or settings.llm_model
    passages = passages_for(db, search_query or question, collections or ["core"], reuse_ids)
    # Every retrieved passage, so the UI can render sources while the answer streams.
    yield "sources", [source_card(p, db) for p in passages]

    llm = llm or client()
    parts: list[str] = []
    messages = history_messages(history) + [
        {"role": "user", "content": user_message(question, passages, collections)}
    ]
    with llm.messages.stream(
        **request_params(model),
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
            yield "token", text
        final = stream.get_final_message()

    raw = "".join(parts)
    if final.stop_reason == "refusal":  # safety decline: never show partial output as an answer
        raw = "I can't help with that request."
    text, cited, dropped = validate_citations(raw, len(passages))
    refused = text.strip().startswith(REFUSAL)
    usage = final.usage
    yield (
        "done",
        Answer(
            question=question,
            text=text,
            cited=cited,
            sources=sources_for(passages, cited, db),
            passages=passages,
            refused=refused,
            uncited=not refused and not cited,
            stop_reason=final.stop_reason,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(model, usage.input_tokens, usage.output_tokens),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            dropped_markers=dropped,
            # Retrieved chunks only: attached tables are re-attached when rebuilt.
            passage_ids=[p.hit.rowid for p in passages if not p.attached],
            search_query=None if reuse_ids else (search_query or question),
        ),
    )


def answer(db: sqlite3.Connection, question: str, **kwargs) -> Answer:
    """Non-streaming convenience: run the stream to completion, return the Answer."""
    for kind, payload in stream_answer(db, question, **kwargs):
        if kind == "done":
            return payload  # type: ignore[return-value]
    raise RuntimeError("stream ended without a result")


def main() -> None:
    from rag.index import check_meta, connect

    q = " ".join(sys.argv[1:]) or "What was the fit MAPE for the IDX Composite logistic model?"
    db = connect(readonly=True)
    check_meta(db)
    print(f"Q: {q}\n")
    result: Answer | None = None
    for kind, payload in stream_answer(db, q):
        if kind == "token":
            print(payload, end="", flush=True)
        elif kind == "done":
            result = payload
    assert result is not None
    print("\n\nSources:")
    for s in result.sources:
        where = f"{s['heading'][:50]} · p.{s['page_start']}"
        print(f"  [{s['n']}] {s['paper_id']} {s['short_cite']} · {where}")
    flags = [f for f, on in (("REFUSED", result.refused), ("UNCITED", result.uncited)) if on]
    print(
        f"\n{result.model}  in={result.input_tokens} out={result.output_tokens} "
        f"${result.cost_usd:.4f}  {result.latency_ms} ms  stop={result.stop_reason}"
        + (f"  dropped markers={result.dropped_markers}" if result.dropped_markers else "")
        + (f"  [{', '.join(flags)}]" if flags else "")
    )


if __name__ == "__main__":
    main()

"""Chat mode: history handling, the retrieval rewrite, reuse, and /api/chat.
All with fake clients: no network, no cost."""

from types import SimpleNamespace

import anthropic
import pytest

from rag.chat import plan_retrieval, stream_chat
from rag.config import settings
from rag.fake_llm import FakeAnthropic
from rag.generate import history_messages


def test_history_messages_strips_markers_and_alternates() -> None:
    h = [
        {"role": "assistant", "content": "orphan greeting"},  # dropped: must start with user
        {"role": "user", "content": "What MAPE?"},
        {"role": "assistant", "content": "It was 5.067% [1][2]."},
        {"role": "assistant", "content": "More [3]."},  # merged into the previous turn
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "unanswered"},  # dropped: no answer followed
    ]
    out = history_messages(h)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[1]["content"] == "It was 5.067%.\n\nMore."
    assert history_messages(None) == []


def test_history_is_bounded() -> None:
    h = [{"role": r, "content": "x" * 5000} for _ in range(10) for r in ("user", "assistant")]
    out = history_messages(h)
    assert len(out) <= 8 and all(len(m["content"]) <= 2000 for m in out)


class RewriteClient:
    def __init__(self, reply: str | Exception) -> None:
        self.reply, self.calls = reply, []
        outer = self

        class Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                if isinstance(outer.reply, Exception):
                    raise outer.reply
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=outer.reply)],
                    usage=SimpleNamespace(input_tokens=100, output_tokens=10),
                )

        self.messages = Messages()


HIST = [
    {"role": "user", "content": "What model did Noviantri use for the IDX Composite?"},
    {"role": "assistant", "content": "A logistic growth model [1]."},
]


def test_plan_retrieval_search_same_and_fallbacks() -> None:
    q, reuse, use = plan_retrieval(
        RewriteClient("SEARCH: logistic model IDX Composite MAPE"), HIST, "its MAPE?", True
    )
    assert q == "logistic model IDX Composite MAPE" and not reuse and use["input_tokens"] == 100
    assert plan_retrieval(RewriteClient("SAME"), HIST, "explain like I'm 12", True)[:2] == (
        None,
        True,
    )
    fallback = "What model did Noviantri use for the IDX Composite? explain like I'm 12"
    # SAME with nothing to reuse, garbage replies and API errors all search instead
    assert plan_retrieval(RewriteClient("SAME"), HIST, "explain like I'm 12", False)[0] == fallback
    assert plan_retrieval(RewriteClient("hmm"), HIST, "explain like I'm 12", True)[0] == fallback
    err = anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
    assert plan_retrieval(RewriteClient(err), HIST, "explain like I'm 12", True)[0] == fallback


def test_rewrite_request_matches_real_sdk_signature() -> None:
    import inspect

    from anthropic.resources.messages import Messages

    fake = RewriteClient("SAME")
    plan_retrieval(fake, HIST, "simpler please", True)
    inspect.signature(Messages.create).bind(None, **fake.calls[0])


corpus = pytest.mark.skipif(not settings.index_path.exists(), reason="run `make index` first")


@pytest.fixture(scope="module")
def db():
    from rag.index import connect

    conn = connect(readonly=True)
    yield conn
    conn.close()


def run(db, message, history=None, reuse_ids=None, llm=None):
    llm = llm or FakeAnthropic(delay=0)
    return list(stream_chat(db, message, history=history, reuse_ids=reuse_ids, llm=llm))[-1][1]


@corpus
def test_first_turn_searches_the_message_without_a_rewrite(db) -> None:
    llm = FakeAnthropic(delay=0)
    llm.messages.create = None  # type: ignore[assignment]  # would raise if called
    ans = run(db, "What was the IDX Composite fit MAPE?", llm=llm)
    assert ans.search_query == "What was the IDX Composite fit MAPE?" and ans.passage_ids


@corpus
def test_style_follow_up_reuses_previous_passages(db) -> None:
    first = run(db, "What was the IDX Composite fit MAPE?")
    history = [
        {"role": "user", "content": first.question},
        {"role": "assistant", "content": first.text},
    ]
    again = run(db, "explain like I'm 12", history=history, reuse_ids=first.passage_ids)
    assert again.search_query is None and again.passage_ids == first.passage_ids
    new = run(db, "Which papers use GARCH?", history=history, reuse_ids=first.passage_ids)
    assert new.search_query == "Which papers use GARCH?"


@pytest.fixture()
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import main
    from rag import usage

    monkeypatch.setattr(usage, "_path", lambda: tmp_path / "usage.sqlite")
    monkeypatch.setattr(settings, "access_code", "c")
    main.limiter.reset()
    main.app.dependency_overrides[main.get_llm] = lambda: FakeAnthropic(delay=0)
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()


@corpus
def test_api_chat_round_trip(api) -> None:
    from tests.test_api import sse_events

    def chat(body, code="c"):
        return api.post("/api/chat", json=body, headers={"x-access-code": code})

    r = chat({"message": "What was the IDX Composite fit MAPE?"})
    done = sse_events(r.text)[-1]
    assert done[0] == "done" and done[1]["passage_ids"] and done[1]["search_query"]
    body = {
        "message": "simpler please",
        "history": [
            {"role": "user", "content": "What was the IDX Composite fit MAPE?"},
            {"role": "assistant", "content": done[1]["text"]},
        ],
        "reuse_ids": done[1]["passage_ids"],
    }
    done2 = sse_events(chat(body).text)[-1][1]
    assert done2["search_query"] is None and done2["passage_ids"] == done[1]["passage_ids"]
    assert chat(body, code="wrong").status_code == 401
    assert chat({**body, "history": body["history"] * 21}).status_code == 422  # > 40 turns

"""Phase 9: HTTP API with a fake Anthropic client (no network, no cost)."""

import json

import pytest

from rag import usage
from rag.config import settings

pytestmark = pytest.mark.skipif(not settings.index_path.exists(), reason="run `make index` first")

CODE = "test-code"


@pytest.fixture()
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import main
    from tests.test_generate import FakeClient

    monkeypatch.setattr(usage, "_path", lambda: tmp_path / "usage.sqlite")
    monkeypatch.setattr(settings, "access_code", CODE)
    monkeypatch.setattr(settings, "daily_question_cap", 100)
    main.limiter.reset()
    fake = FakeClient(["The fit MAPE was 5.067% ", "[1]."])
    main.app.dependency_overrides[main.get_llm] = lambda: fake
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()


def ask(client, question: str, code: str | None = CODE):
    headers = {"x-access-code": code} if code is not None else {}
    return client.post("/api/ask", json={"question": question}, headers=headers)


def sse_events(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health_and_papers(api) -> None:
    h = api.get("/health").json()
    assert h["ok"] and h["chunks_core"] > 0 and h["chunks_user"] == 0
    assert h["embed_model"] == settings.embed_model
    papers = api.get("/api/papers").json()
    assert len(papers) == 19 and {"id", "short_cite", "title", "doi"} <= set(papers[0])


def test_ask_streams_sources_tokens_done(api) -> None:
    r = ask(api, "What was the IDX Composite fit MAPE?")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    sources = events[0][1]
    assert sources[0]["n"] == 1 and len(sources[0]["snippet"]) <= 300
    done = events[-1][1]
    assert done["cited"] == [1] and "5.067%" in done["text"] and done["latency_ms"] >= 0
    assert usage.used_today() == 1


def test_guards(api) -> None:
    assert ask(api, "q", code="wrong").status_code == 401
    assert ask(api, "q", code=None).status_code == 401
    assert ask(api, "   ").status_code == 400
    assert ask(api, "x" * (settings.max_question_chars + 1)).status_code == 400
    assert usage.used_today() == 0  # rejected requests don't use the budget


def test_daily_cap(api, monkeypatch) -> None:
    monkeypatch.setattr(settings, "daily_question_cap", 1)
    assert ask(api, "first").status_code == 200
    r = ask(api, "second")
    assert r.status_code == 429 and "budget" in r.json()["error"]


def test_rate_limit_per_minute(api) -> None:
    limit = int(settings.rate_limit.split("/")[0])
    codes = [ask(api, f"question {i}").status_code for i in range(limit + 1)]
    assert codes[:limit] == [200] * limit and codes[-1] == 429

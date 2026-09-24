"""Phase 7: generation without network calls (a fake Anthropic client)."""

from types import SimpleNamespace

import pytest

from rag.config import settings
from rag.generate import (
    REFUSAL,
    SYSTEM_PROMPT,
    build_passages,
    format_context,
    request_params,
    snippet,
    stream_answer,
    validate_citations,
)


def test_request_params_per_model(monkeypatch) -> None:
    haiku = request_params("claude-haiku-4-5")
    assert haiku["extra_body"] == {"temperature": 0.2} and "thinking" not in haiku
    sonnet = request_params("claude-sonnet-5")
    assert "extra_body" not in sonnet  # Sonnet 5 rejects sampling params
    assert sonnet["thinking"] == {"type": "disabled"}
    monkeypatch.setattr(settings, "llm_thinking", "adaptive")
    s2 = request_params("claude-sonnet-5")
    assert s2["thinking"] == {"type": "adaptive"} and s2["max_tokens"] >= 8000


def test_validate_citations_drops_markers_to_missing_passages() -> None:
    text, cited, dropped = validate_citations("MAPE was 5.067% [2][9]. RMSPE 6.754% [1, 2].", 6)
    assert text == "MAPE was 5.067% [2]. RMSPE 6.754% [1][2]."
    assert cited == [2, 1] and dropped == [9]
    text, cited, dropped = validate_citations("Only a fake one [7].", 6)
    assert text == "Only a fake one." and cited == [] and dropped == [7]


def test_snippet_is_short_and_cut_at_a_word() -> None:
    s = snippet("word " * 200)
    assert len(s) <= 300 and s.endswith("…") and not s.endswith(" …")


def test_system_prompt_keeps_plan_rules_and_equation_change() -> None:
    assert f'"{REFUSAL}"' in SYSTEM_PROMPT
    assert "Use only the numbered passages" in SYSTEM_PROMPT
    assert "Never derive, simplify or invent an equation" in SYSTEM_PROMPT


class FakeStream:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn") -> None:
        self.chunks, self.stop_reason = chunks, stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from self.chunks

    def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=4000, output_tokens=120),
        )


class FakeClient:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        outer = self

        class Messages:
            def stream(self, **kw):
                outer.calls.append(kw)
                return FakeStream(chunks, stop_reason)

        self.messages = Messages()


corpus = pytest.mark.skipif(not settings.index_path.exists(), reason="run `make index` first")


@pytest.fixture(scope="module")
def db():
    from rag.index import connect

    conn = connect(readonly=True)
    yield conn
    conn.close()


@corpus
def test_passages_widen_with_neighbours_without_repeats(db) -> None:
    from rag.retrieve import search

    hits = search(db, "Heston model variance process Feller condition")
    ps = build_passages(db, hits)
    assert [p.n for p in ps] == list(range(1, len(hits) + 1))
    assert all(p.hit.text in p.text for p in ps)
    ctx = format_context(ps)
    assert ctx.startswith("[1] p") and " · p." in ctx.split("\n", 1)[0]


@corpus
def test_stream_answer_event_order_and_flags(db) -> None:
    fake = FakeClient(["The fit MAPE was 5.067% ", "and RMSPE 6.754% [1][12]."])
    events = list(stream_answer(db, "IDX Composite fit MAPE", model="claude-haiku-4-5", llm=fake))
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and set(kinds[1:-1]) == {"token"}
    ans = events[-1][1]
    assert ans.cited == [1] and ans.dropped_markers == [12] and not ans.uncited
    assert ans.sources[0]["n"] == 1 and len(ans.sources[0]["snippet"]) <= 300
    assert ans.cost_usd == pytest.approx((4000 * 1 + 120 * 5) / 1e6)
    call = fake.calls[0]
    assert call["system"] == SYSTEM_PROMPT and call["extra_body"] == {"temperature": 0.2}
    assert "Question: IDX Composite fit MAPE" in call["messages"][0]["content"]


@corpus
def test_refusal_and_uncited_are_flagged(db) -> None:
    ans = list(
        stream_answer(
            db, "BBCA price today", llm=FakeClient([REFUSAL + " It covers BBCA in 2020."])
        )
    )[-1][1]
    assert ans.refused and not ans.uncited
    ans = list(stream_answer(db, "BBCA price today", llm=FakeClient(["BBCA is up."])))[-1][1]
    assert ans.uncited and not ans.refused
    ans = list(stream_answer(db, "q", llm=FakeClient(["partial"], stop_reason="refusal")))[-1][1]
    assert "partial" not in ans.text and ans.stop_reason == "refusal"


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-5"])
@pytest.mark.parametrize("thinking", ["off", "adaptive"])
def test_request_params_match_real_sdk_signature(model, thinking, monkeypatch) -> None:
    """The fake client accepts anything; bind against the installed SDK instead, so a
    parameter the SDK dropped (anthropic 1.x removed `temperature`) fails here."""
    import inspect

    from anthropic.resources.messages import Messages

    monkeypatch.setattr(settings, "llm_thinking", thinking)
    params = request_params(model)
    sig = inspect.signature(Messages.stream)
    sig.bind(None, **params, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": "q"}])

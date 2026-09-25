"""A stand-in for anthropic.Anthropic that costs nothing.

Used by FAKE_LLM=1 (UI development, demos without a key) and by tests. It
streams a short canned answer built from the real passages it is given, so
retrieval, citations and sources all behave as in production; only the prose
is fake.
"""

import re
import time
from types import SimpleNamespace


class _Stream:
    def __init__(self, text: str, delay: float) -> None:
        self._text, self._delay = text, delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for word in re.findall(r"\S+\s*", self._text):
            if self._delay:
                time.sleep(self._delay)  # make streaming visible in the UI
            yield word

    def get_final_message(self):
        usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        return SimpleNamespace(stop_reason="end_turn", usage=usage)


def canned_answer(user_message: str) -> str:
    """Quote the first sentence of the first two passages, cited."""
    blocks = re.findall(
        r"^\[(\d+)\] ([pu]\d\d) ·([^·]+) · .*?\n(.+?)(?=\n\n\[\d+\] |\Z)", user_message, re.S | re.M
    )
    if not blocks:
        return "The corpus doesn't cover this. (Demo mode: no model was called.)"
    parts = ["Demo mode, no model was called; this answer just quotes the top passages."]
    for n, _pid, cite, text in blocks[:2]:
        first = re.split(r"(?<=[.!?])\s", " ".join(text.split()), maxsplit=1)[0][:280]
        parts.append(f"{cite.strip()} writes: “{first}” [{n}]")
    return " ".join(parts)


class FakeAnthropic:
    def __init__(self, delay: float = 0.02) -> None:
        outer = self
        self.delay = delay

        class _Messages:
            def stream(self, **kw):
                return _Stream(canned_answer(kw["messages"][-1]["content"]), outer.delay)

            def create(self, **kw):
                """The chat query-rewrite call: SAME for style-only follow-ups."""
                latest = kw["messages"][-1]["content"].rsplit("Latest message:", 1)[-1].strip()
                style = (
                    r"\b(simpl\w*|like i'?m|eli5|shorter|rephrase|bullet"
                    r"|in (english|indonesian))\b"
                )
                style_only = re.search(style, latest, re.I)
                text = "SAME" if style_only else f"SEARCH: {latest}"
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=text)],
                    usage=SimpleNamespace(input_tokens=0, output_tokens=0),
                    stop_reason="end_turn",
                )

        self.messages = _Messages()

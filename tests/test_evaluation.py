"""Phase 8: automatic answer checks and the judge request (no API calls)."""

from rag.evaluation import auto_checks, corpus_numbers_in, judge_request
from rag.generate import REFUSAL, Answer


def _answer(text: str, cited=(1,), refused=False, dropped=()) -> Answer:
    return Answer(
        question="q", text=text, cited=list(cited), sources=[], passages=[], refused=refused,
        uncited=not refused and not cited, stop_reason="end_turn", model="m",
        dropped_markers=list(dropped),
    )  # fmt: skip


def test_must_include_normalises_case_and_thousands() -> None:
    q = {"id": "q31", "type": "numeric", "must_include": ["1,025", "2003"]}
    assert auto_checks(q, _answer("The index peaked at 1025 in March 2003 [1]."))["auto_pass"]
    bad = auto_checks(q, _answer("It peaked in 2003 [1]."))
    assert not bad["auto_pass"] and bad["missing"] == ["1,025"]


def test_answerable_needs_citation_and_valid_markers() -> None:
    q = {"id": "q01", "type": "numeric", "must_include": ["5.067"]}
    assert not auto_checks(q, _answer("MAPE 5.067%.", cited=()))["auto_pass"]
    assert not auto_checks(q, _answer("MAPE 5.067% [1].", dropped=(9,)))["auto_pass"]


def test_unanswerable_must_refuse_without_corpus_numbers() -> None:
    q = {"id": "q55", "type": "unanswerable"}
    ok = _answer(REFUSAL + " It covers BBCA prices in 2020.", cited=(), refused=True)
    assert auto_checks(q, ok)["auto_pass"]  # a year is fine
    assert corpus_numbers_in("It fell to 5.067 then", "MAPE was 5.067%") == ["5.067"]


def test_q40_trap_rejects_invented_metrics() -> None:
    q = {"id": "q40", "type": "discrepancy", "must_include": []}
    assert not auto_checks(q, _answer("They report a MAPE of 3.2 [1]."))["auto_pass"]
    assert not auto_checks(q, _answer("Error was 2.1% [1]."))["auto_pass"]
    assert auto_checks(q, _answer("No MAPE or RMSE is reported [1]."))["auto_pass"]


def test_judge_request_shapes() -> None:
    q = {"question": "Q?", "answer": "gold"}
    a = _answer("sys [1]")
    opus = judge_request(q, a, "claude-opus-5")
    assert opus["betas"] == ["server-side-fallback-2026-07-01"]
    assert opus["extra_body"] == {"fallbacks": "default"} and "temperature" not in opus
    assert opus["output_config"]["format"]["type"] == "json_schema"
    sonnet = judge_request(q, a, "claude-sonnet-5")
    assert "betas" not in sonnet and "temperature" not in sonnet

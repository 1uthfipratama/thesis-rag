"""Answer evaluation (PLAN.md Phase 8): automatic checks + LLM judge.

Automatic checks are deterministic string rules from the gold set; the judge is a
separate Claude call that scores correctness and faithfulness 0-2. Both are kept
per question so judge/check disagreements can be read side by side.
"""

import json
import re

import anthropic

from rag.generate import REFUSAL, Answer, format_context

# --- automatic checks --------------------------------------------------------------


def _norm(s: str) -> str:
    """Case-folded; thousands separators and unicode minus normalised."""
    s = s.lower().replace("−", "-").replace("–", "-").replace("\u00a0", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,025 -> 1025
    return re.sub(r"\s+", " ", s)


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_YEAR = re.compile(r"^(19|20)\d\d$")


def corpus_numbers_in(answer_text: str, passages_text: str) -> list[str]:
    """Numbers in the answer that also occur in the passages (years excluded:
    a refusal may legitimately say what period the corpus covers)."""
    passage_nums = set(_NUMBER.findall(_norm(passages_text)))
    return [
        n
        for n in _NUMBER.findall(_norm(answer_text))
        if n in passage_nums and not _YEAR.match(n) and n not in {"0", "1", "2"}
    ]


def auto_checks(q: dict, a: Answer) -> dict:
    """Plan section 8 rules. Returns {check: bool} plus detail fields."""
    text = _norm(a.text)
    out: dict = {}
    if q["type"] == "unanswerable":
        out["refused"] = a.text.strip().startswith(REFUSAL)
        leaked = corpus_numbers_in(a.text.replace(REFUSAL, ""), format_context(a.passages))
        out["no_corpus_numbers"] = not leaked
        out["leaked_numbers"] = leaked
    else:
        missing = [s for s in q.get("must_include", []) if _norm(s) not in text]
        out["must_include"] = not missing
        out["missing"] = missing
        out["has_citation"] = bool(a.cited)
        out["not_refused"] = not a.refused
    out["citations_valid"] = not a.dropped_markers
    if q["id"] == "q40":
        # Trap: p06 reports no MAPE/RMSE. Any percentage, or MAPE/RMSE next to a
        # number, means the model invented a metric.
        # Citation markers are digits too ("... reported [1]"), so strip them first.
        bare = re.sub(r"\[\d+\]", "", text)
        out["q40_no_invented_metric"] = not (
            re.search(r"\d(?:\.\d+)?\s*%", bare)
            or re.search(r"\b(mape|rmse)\b[^.\n]{0,25}\d", bare)
        )
    flags = [k for k, v in out.items() if isinstance(v, bool)]
    out["auto_pass"] = all(out[k] for k in flags)
    return out


# --- LLM judge ----------------------------------------------------------------------

JUDGE_PROMPT = """You grade answers from a question-answering system over research papers.

You get: the question, a gold answer written from the papers, the system's answer,
and the numbered passages the system was given.

Score 0-2 on each:
correctness: 2 = matches the gold answer on all key facts, 1 = partly, 0 = wrong or missing
faithfulness: 2 = every claim is supported by the passages, 1 = minor unsupported detail,
0 = fabricated

For a question the corpus cannot answer, the gold answer is a refusal: a correct
system answer declines ("The corpus doesn't cover this.") and invents nothing.
Judge facts, not wording or style. Keep the reason to one or two sentences."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "faithfulness": {"type": "integer", "enum": [0, 1, 2]},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "faithfulness", "reason"],
    "additionalProperties": False,
}


def judge_request(q: dict, a: Answer, judge_model: str) -> dict:
    content = (
        f"Question:\n{q['question']}\n\nGold answer:\n{q['answer']}\n\n"
        f"System answer:\n{a.text}\n\nPassages given to the system:\n\n{format_context(a.passages)}"
    )
    req: dict = {
        "model": judge_model,
        "max_tokens": 2000,
        "system": JUDGE_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    }
    if judge_model.startswith("claude-opus-5"):
        # Opus 5 thinks by default; low effort is plenty for 0-2 grading and keeps
        # cost down. Server-side fallback re-runs a safety decline on the model
        # Anthropic recommends instead of returning a refusal.
        req["output_config"]["effort"] = "low"
        req["betas"] = ["server-side-fallback-2026-07-01"]
        req["extra_body"] = {"fallbacks": "default"}
    elif judge_model.startswith("claude-haiku-4-5"):
        req["extra_body"] = {"temperature": 0.0}  # not a 1.x SDK kwarg; see generate.request_params
    return req


def judge(client: anthropic.Anthropic, q: dict, a: Answer, judge_model: str) -> dict:
    """{correctness, faithfulness, reason, input_tokens, output_tokens} or {error}."""
    req = judge_request(q, a, judge_model)
    api = client.beta.messages if "betas" in req else client.messages
    try:
        resp = api.create(**req)
    except anthropic.APIStatusError as e:
        return {"error": f"{e.status_code}: {e.message}"}
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return {"error": "judge refused", **usage}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return {**json.loads(text), **usage}
    except json.JSONDecodeError:
        return {"error": f"unparseable judge output: {text[:120]!r}", **usage}

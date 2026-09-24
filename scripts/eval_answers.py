"""Answer evaluation (PLAN.md Phase 8): gold questions -> answers -> checks + judge.

Every run that calls the API costs money, so:
- the cost is estimated from the real prompt sizes BEFORE any call is made,
- the run aborts if the estimate exceeds --max-usd (default $1),
- --dry-run exercises the whole pipeline with a fake model and spends nothing.

    uv run python scripts/eval_answers.py --dry-run
    uv run python scripts/eval_answers.py --models claude-haiku-4-5 --limit 5 --max-usd 0.20
    uv run python scripts/eval_answers.py --models claude-haiku-4-5 claude-sonnet-5 --max-usd 6

Writes, per model, eval/results/answers_{model}_{ts}.jsonl and one summary
eval/results/answers_{ts}.md, plus a manual-review sheet (plan: read every answer
the judge scored < 2 correct, plus 10 random others, and record disagreements).
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from rag.config import settings
from rag.evaluation import auto_checks, judge
from rag.generate import PRICES, build_passages, client, stream_answer, user_message
from rag.index import check_meta, connect
from rag.retrieve import search

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold_questions.jsonl"
RESULTS = ROOT / "eval" / "results"
COLLECTIONS = ["core"]  # never evaluate against uploads (PLAN_ADDENDUM 13.1)
# Sonnet 5 is stronger than the default answer model (Haiku 4.5) and keeps a full
# run near $1.40 within the project's $10 budget. When Sonnet itself is being
# graded, pass --judge-model claude-opus-5 (a warning is printed otherwise).
DEFAULT_JUDGE = "claude-sonnet-5"

CHARS_PER_TOKEN = 3.5  # conservative for English academic text with numbers
ANSWER_TOKENS = 350  # answers are < 200 words; lists run longer
JUDGE_OUT_TOKENS = 500  # JSON verdict + low-effort thinking


def price(model: str) -> tuple[float, float]:
    return next((v for k, v in PRICES.items() if model.startswith(k)), (5.0, 25.0))


def estimate(db, questions: list[dict], models: list[str], judge_model: str | None) -> dict:
    """USD estimate from the actual retrieved context for each question."""
    in_tokens = []
    for q in questions:
        ps = build_passages(db, search(db, q["question"], collections=COLLECTIONS))
        in_tokens.append(len(user_message(q["question"], ps)) / CHARS_PER_TOKEN + 350)
    total_in = sum(in_tokens)
    per_model = {}
    for m in models:
        pi, po = price(m)
        answers = (total_in * pi + len(questions) * ANSWER_TOKENS * po) / 1e6
        judged = 0.0
        if judge_model:
            ji, jo = price(judge_model)
            j_in = total_in + len(questions) * (ANSWER_TOKENS + 400)
            judged = (j_in * ji + len(questions) * JUDGE_OUT_TOKENS * jo) / 1e6
        per_model[m] = {"answers": answers, "judge": judged}
    return {
        "per_model": per_model,
        "total": sum(v["answers"] + v["judge"] for v in per_model.values()),
        "input_tokens_per_question": statistics.median(in_tokens),
    }


# --- dry run --------------------------------------------------------------------------


class FakeClient:
    """Stands in for anthropic.Anthropic: answers with the gold answer plus [1], and
    a judge verdict of 2/2. Exercises checks, reporting and file output for free."""

    def __init__(self, gold_by_question: dict[str, dict]) -> None:
        self._gold = gold_by_question
        outer = self

        class _Stream:
            def __init__(self, text: str) -> None:
                self._text = text

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            @property
            def text_stream(self):
                yield self._text

            def get_final_message(self):
                usage = SimpleNamespace(input_tokens=4500, output_tokens=ANSWER_TOKENS)
                return SimpleNamespace(stop_reason="end_turn", usage=usage)

        class _Messages:
            def stream(self, **kw):
                content = kw["messages"][0]["content"]
                q = outer._gold[content.rsplit("Question: ", 1)[-1]]
                if q["type"] == "unanswerable":
                    return _Stream("The corpus doesn't cover this. It covers related topics.")
                return _Stream(q["answer"] + " [1]")

            def create(self, **kw):
                verdict = '{"correctness": 2, "faithfulness": 2, "reason": "dry run"}'
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=verdict)],
                    usage=SimpleNamespace(input_tokens=5000, output_tokens=JUDGE_OUT_TOKENS),
                )

        self.messages = _Messages()
        self.beta = SimpleNamespace(messages=self.messages)


# --- run ------------------------------------------------------------------------------


def run_model(db, llm, model: str, questions: list[dict], judge_model: str | None) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        ans = None
        for kind, payload in stream_answer(
            db, q["question"], model=model, collections=COLLECTIONS, llm=llm
        ):
            if kind == "done":
                ans = payload
        checks = auto_checks(q, ans)
        verdict = judge(llm, q, ans, judge_model) if judge_model else {}
        judge_cost = 0.0
        if "input_tokens" in verdict:
            ji, jo = price(judge_model)
            judge_cost = (verdict["input_tokens"] * ji + verdict["output_tokens"] * jo) / 1e6
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "gold": q["answer"],
                "answer": ans.text,
                "cited": ans.cited,
                "sources": ans.sources,
                "refused": ans.refused,
                "uncited": ans.uncited,
                "stop_reason": ans.stop_reason,
                "checks": checks,
                "judge": verdict,
                "cost_usd": ans.cost_usd,
                "judge_cost_usd": judge_cost,
                "latency_ms": ans.latency_ms,
                "input_tokens": ans.input_tokens,
                "output_tokens": ans.output_tokens,
            }
        )
        c = verdict.get("correctness", "-")
        print(
            f"  {model} {i:>2}/{len(questions)} {q['id']:<5} auto={'ok' if checks['auto_pass'] else 'FAIL'} judge={c}"
        )
    return rows


def summarise(rows: list[dict]) -> dict:
    judged = [r for r in rows if "correctness" in r["judge"]]
    unans = [r for r in rows if r["type"] == "unanswerable"]
    lat = [r["latency_ms"] for r in rows]
    by_type: dict[str, list] = {}
    for r in judged:
        by_type.setdefault(r["type"], []).append(r["judge"]["correctness"] == 2)
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "accuracy": sum(r["judge"]["correctness"] == 2 for r in judged) / max(1, len(judged)),
        "mean_correctness": statistics.mean([r["judge"]["correctness"] for r in judged])
        if judged
        else float("nan"),
        "faithfulness": sum(r["judge"]["faithfulness"] == 2 for r in judged) / max(1, len(judged)),
        "auto_pass": sum(r["checks"]["auto_pass"] for r in rows) / n,
        "refusal_accuracy": sum(
            r["checks"].get("refused", False) and r["checks"].get("no_corpus_numbers", False)
            for r in unans
        )
        / max(1, len(unans)),
        "uncited": sum(r["uncited"] for r in rows),
        "judge_errors": sum("error" in r["judge"] for r in rows),
        "cost_per_100": 100 * sum(r["cost_usd"] for r in rows) / n,
        "judge_cost_total": sum(r["judge_cost_usd"] for r in rows),
        "answer_cost_total": sum(r["cost_usd"] for r in rows),
        "median_latency_ms": statistics.median(lat) if lat else 0,
        "by_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }


def report(
    all_rows: dict[str, list[dict]], judge_model: str | None, ts: datetime, est: dict, dry: bool
) -> Path:
    S = {m: summarise(rows) for m, rows in all_rows.items()}
    models = list(S)
    L = [f"# Answer evaluation — {ts:%Y-%m-%d %H:%M}" + (" (DRY RUN: fake model, numbers meaningless)" if dry else ""), "",
         f"Judge: `{judge_model or 'none'}`. Estimated cost before run: ${est['total']:.2f}.", "",
         "| metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]  # fmt: skip
    rows = [
        ("questions", "n", "{:d}"),
        ("accuracy (judge correctness = 2)", "accuracy", "{:.1%}"),
        ("mean correctness (0-2)", "mean_correctness", "{:.2f}"),
        ("faithfulness (judge = 2)", "faithfulness", "{:.1%}"),
        ("automatic checks passed", "auto_pass", "{:.1%}"),
        ("refusal accuracy (unanswerable)", "refusal_accuracy", "{:.1%}"),
        ("uncited answers", "uncited", "{:d}"),
        ("judge errors", "judge_errors", "{:d}"),
        ("answer cost per 100 questions", "cost_per_100", "${:.2f}"),
        ("answer cost, this run", "answer_cost_total", "${:.3f}"),
        ("judge cost, this run", "judge_cost_total", "${:.3f}"),
        ("median latency", "median_latency_ms", "{:.0f} ms"),
    ]
    L += [
        f"| {label} | " + " | ".join(fmt.format(S[m][key]) for m in models) + " |"
        for label, key, fmt in rows
    ]
    types = sorted({t for s in S.values() for t in s["by_type"]})
    L += [
        "",
        "## Accuracy by question type",
        "",
        "| type | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    L += [
        f"| {t} | "
        + " | ".join(f"{S[m]['by_type'].get(t, float('nan')):.0%}" for m in models)
        + " |"
        for t in types
    ]
    L += ["", "## Failed automatic checks", ""]
    for m, rs in all_rows.items():
        for r in rs:
            if not r["checks"]["auto_pass"]:
                bad = [k for k, v in r["checks"].items() if v is False]
                extra = r["checks"].get("missing") or r["checks"].get("leaked_numbers") or ""
                L.append(f"- `{m}` {r['id']} ({r['type']}): {', '.join(bad)} {extra}")
    out = RESULTS / f"answers_{'dryrun_' if dry else ''}{ts:%Y%m%d_%H%M}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def review_sheet(all_rows: dict[str, list[dict]], ts: datetime, dry: bool = False) -> Path:
    """Plan: read every answer the judge scored < 2, plus 10 random others; record
    where you disagree with the judge (judge-human agreement is a reported number)."""
    rng = random.Random(42)
    L = [f"# Manual review — {ts:%Y-%m-%d %H:%M}", "",
         "Fill `human` (0-2) and `agree` (y/n). Judge-human agreement = share of `y`.", ""]  # fmt: skip
    for m, rows in all_rows.items():
        low = [r for r in rows if r["judge"].get("correctness", 2) < 2 or "error" in r["judge"]]
        rest = [r for r in rows if r not in low]
        picked = low + rng.sample(rest, min(10, len(rest)))
        L.append(f"## {m} ({len(low)} judged < 2, {len(picked) - len(low)} random)\n")
        for r in picked:
            j = r["judge"]
            L += [
                f"### {r['id']} · {r['type']}", f"**Q:** {r['question']}", f"**Gold:** {r['gold']}",
                f"**Answer:** {r['answer']}",
                f"**Judge:** correctness {j.get('correctness', '-')}, faithfulness {j.get('faithfulness', '-')} — {j.get('reason', j.get('error', ''))}",
                "**human:** _ **agree:** _", "",
            ]  # fmt: skip
    out = RESULTS / f"manual_review_{'dryrun_' if dry else ''}{ts:%Y%m%d_%H%M}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[settings.llm_model])
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE, help='"none" to skip the judge')
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--limit", type=int, help="first N questions only")
    ap.add_argument("--max-usd", type=float, default=1.0, help="abort if the estimate is higher")
    ap.add_argument("--dry-run", action="store_true", help="fake model, no API calls, $0")
    args = ap.parse_args()

    judge_model = None if args.judge_model == "none" else args.judge_model
    gold = [json.loads(x) for x in args.gold.read_text(encoding="utf-8").splitlines() if x.strip()]
    questions = [q for q in gold if not args.ids or q["id"] in args.ids][: args.limit]
    db = connect(readonly=True)
    check_meta(db)

    if judge_model and judge_model in args.models:
        print(
            f"WARNING: {judge_model} would grade its own answers; consider --judge-model claude-opus-5"
        )
    est = estimate(db, questions, args.models, judge_model)
    print(
        f"{len(questions)} questions x {len(args.models)} model(s), judge {judge_model or 'none'}"
    )
    print(f"median input ~{est['input_tokens_per_question']:.0f} tokens/question")
    for m, v in est["per_model"].items():
        print(f"  {m}: answers ~${v['answers']:.2f}, judge ~${v['judge']:.2f}")
    print(
        f"ESTIMATED TOTAL: ~${est['total']:.2f}"
        + ("  (dry run: $0 will be spent)" if args.dry_run else "")
    )
    if not args.dry_run and est["total"] > args.max_usd:
        print(f"Aborting: estimate exceeds --max-usd {args.max_usd:.2f}. Raise it to proceed.")
        return 2

    llm = FakeClient({q["question"]: q for q in questions}) if args.dry_run else client()
    ts = datetime.now()
    all_rows = {}
    for m in args.models:
        t0 = time.perf_counter()
        all_rows[m] = run_model(db, llm, m, questions, judge_model)
        tag = "dryrun_" if args.dry_run else ""
        path = RESULTS / f"answers_{tag}{m}_{ts:%Y%m%d_%H%M}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows[m]), encoding="utf-8"
        )
        print(f"{m}: {len(questions)} answers in {time.perf_counter() - t0:.0f}s -> {path.name}")

    summary = report(all_rows, judge_model, ts, est, args.dry_run)
    sheet = review_sheet(all_rows, ts, args.dry_run)
    spent = sum(r["cost_usd"] + r["judge_cost_usd"] for rows in all_rows.values() for r in rows)
    print(
        f"\nsummary: {summary}\nreview:  {sheet}\nactual cost: ${0 if args.dry_run else spent:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

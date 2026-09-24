"""Retrieval evaluation (PLAN.md Phase 6), before any LLM code.

Over eval/gold_questions.jsonl, skipping `unanswerable`, at paper level:
- paper_hit@k (k = 1, 3, 6, 10): an expected paper is in the top k
- paper_coverage@6: fraction of expected papers in the top 6 (multi-paper questions)
- MRR: 1 / rank of the first expected paper (0 if absent from the top 10)
Four configurations, a per-type breakdown and every miss, written to
eval/results/retrieval_{timestamp}.md.

Always restricted to the core collection (PLAN_ADDENDUM 13.1): uploads must
never change these numbers.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rag.config import settings
from rag.index import check_meta, connect
from rag.retrieve import search

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold_questions.jsonl"
COLLECTIONS = ["core"]
KS = (1, 3, 6, 10)

CONFIGS = {
    "bm25": dict(use_dense=False, diversity=False),
    "dense": dict(use_bm25=False, diversity=False),
    "hybrid": dict(diversity=False),
    "hybrid+cap": dict(diversity=True),
}


def ranked_papers(db, q: str, cfg: dict) -> list[str]:
    """Distinct papers in rank order over the top 10 chunks."""
    assert COLLECTIONS == ["core"], "eval must only ever see the core corpus"
    hits = search(db, q, top_k=max(KS), collections=COLLECTIONS, **cfg)
    return [h.paper_id for h in hits]


def score_question(q: dict, papers: list[str]) -> dict:
    exp = set(q["expected_papers"])
    first = next((i for i, p in enumerate(papers, 1) if p in exp), None)
    r = {f"hit@{k}": float(bool(exp & set(papers[:k]))) for k in KS}
    r["mrr"] = 1 / first if first else 0.0
    r["cov@6"] = len(exp & set(papers[:6])) / len(exp)
    return r


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    gold = [
        json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    questions = [q for q in gold if q["type"] != "unanswerable" and q["expected_papers"]]
    db = connect(readonly=True)
    meta = check_meta(db)

    results: dict[str, dict[str, dict]] = {}
    returned: dict[str, dict[str, list[str]]] = {}
    for name, cfg in CONFIGS.items():
        results[name], returned[name] = {}, {}
        for q in questions:
            papers = ranked_papers(db, q["question"], cfg)
            returned[name][q["id"]] = papers
            results[name][q["id"]] = score_question(q, papers)
        print(
            name,
            {
                m: round(mean([r[m] for r in results[name].values()]), 3)
                for m in ("hit@1", "hit@6", "mrr")
            },
        )

    single = [q for q in questions if len(q["expected_papers"]) == 1]
    multi = [q for q in questions if len(q["expected_papers"]) > 1]
    agg = [q for q in questions if q["type"] == "aggregation"]

    def row(name: str, qs: list[dict], metrics: tuple[str, ...]) -> str:
        return " | ".join(f"{mean([results[name][q['id']][m] for q in qs]):.3f}" for m in metrics)

    ts = datetime.now()
    L = [
        f"# Retrieval evaluation — {ts:%Y-%m-%d %H:%M}",
        "",
        f"Index: {meta['n_chunks']} chunks, parser `{meta['parse_backend']}`, embedder "
        f"`{meta['embed_model']}`, query instruction {'on' if meta.get('query_instruction') else 'off'}. "
        f"Collections: {COLLECTIONS}. top_k={settings.top_k}, BM25 k={settings.bm25_k}, "
        f"dense k={settings.dense_k}, RRF k={settings.rrf_k}.",
        "",
        f"Questions: {len(questions)} answerable ({len(single)} single-paper, {len(multi)} multi-paper).",
        "",
        "## All questions",
        "",
        "| config | hit@1 | hit@3 | hit@6 | hit@10 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    all_m = ("hit@1", "hit@3", "hit@6", "hit@10", "mrr")
    L += [f"| {n} | {row(n, questions, all_m)} |" for n in CONFIGS]
    L += [
        "",
        "## Targets (PLAN.md Phase 6)",
        "",
        "| config | single-paper hit@6 (target ≥ 0.90) | aggregation cov@6 (target ≥ 0.70) | all multi-paper cov@6 |",
        "|---|---|---|---|",
    ]
    L += [
        f"| {n} | {row(n, single, ('hit@6',))} | {row(n, agg, ('cov@6',))} | {row(n, multi, ('cov@6',))} |"
        for n in CONFIGS
    ]

    by_type = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q)
    L += ["", "## hit@6 by question type", "", "| type | n | " + " | ".join(CONFIGS) + " |",
          "|---|---|" + "---|" * len(CONFIGS)]  # fmt: skip
    for t, qs in sorted(by_type.items()):
        L.append(
            f"| {t} | {len(qs)} | " + " | ".join(row(n, qs, ("hit@6",)) for n in CONFIGS) + " |"
        )

    best = "hybrid+cap"
    L += ["", f"## Misses and partial coverage ({best})", "",
          "| id | type | expected | top-6 papers (rank order) | hit@6 | cov@6 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for q in questions:
        r = results[best][q["id"]]
        if r["hit@6"] < 1 or r["cov@6"] < 1:
            top6 = list(dict.fromkeys(returned[best][q["id"]][:6]))
            L.append(
                f"| {q['id']} | {q['type']} | {', '.join(q['expected_papers'])} | {', '.join(top6)} "
                f"| {r['hit@6']:.0f} | {r['cov@6']:.2f} |"
            )
    L += ["", "Per-question detail for every config: the `.json` file next to this report."]

    out = ROOT / "eval" / "results" / f"retrieval_{ts:%Y%m%d_%H%M}"
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": results, "returned": returned}, indent=1),
        encoding="utf-8",
    )
    print(f"\nreport: {out.with_suffix('.md')}")


if __name__ == "__main__":
    sys.exit(main())

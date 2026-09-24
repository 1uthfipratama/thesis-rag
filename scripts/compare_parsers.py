"""Parser bake-off: run every backend over the corpus and score them the same way.

Writes each backend's output to data/parsed_alt/{backend}/ and a report to
eval/results/parsers_{timestamp}.md. Parse-level only: the real verdict is
retrieval recall in Phase 6, but these checks catch parsing damage early.

    uv run python scripts/compare_parsers.py            # all backends
    uv run python scripts/compare_parsers.py pymupdf hybrid
"""

import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag.config import settings
from rag.manifest import load_manifest
from rag.parse.backends import BACKENDS
from rag.schemas import ParsedDoc

ALT = settings.data_dir / "parsed_alt"
BACKMATTER_LEADS = [
    "author contributions:",
    "conflicts of interest:",
    "data availability statement",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace(",", "").replace("−", "-"))


def doc_text(d: ParsedDoc) -> str:
    return " ".join(b.text for s in d.sections for b in s.blocks)


def score(docs: dict[str, ParsedDoc], gold: list[dict]) -> dict:
    m: dict = {}
    m["papers"] = len(docs)
    m["abstract+conclusion"] = sum(
        {"abstract", "conclusion"} <= {s.label for s in d.sections} for d in docs.values()
    )
    m["references>0"] = sum(bool(d.references) for d in docs.values())
    m["tables"] = sum(len(d.tables) for d in docs.values())

    eq = Counter()
    blocks = junk = 0
    for d in docs.values():
        for s in d.sections:
            for b in s.blocks:
                if b.kind == "equation":
                    eq["latex" if b.latex else "placeholder"] += 1
                elif b.kind == "text":
                    blocks += 1
                    ch = b.text.replace(" ", "")
                    alpha = sum(c.isalpha() for c in ch) / max(1, len(ch))
                    junk += alpha < 0.55 or len(b.text.split()) < 4
    m["equations_latex"] = eq["latex"]
    m["equations_placeholder"] = eq["placeholder"]
    m["inline_math_spans"] = sum(
        b.text.count("$") // 2
        for d in docs.values()
        for s in d.sections
        for b in s.blocks
        if b.kind != "equation"
    )
    m["junk_block_rate"] = f"{junk / max(1, blocks):.1%}"

    # plan section 2.9 checks, where the paper was parsed
    checks = {}
    if "p01" in docs:
        checks["p01 no Wiley stamp"] = "onlinelibrary.wiley.com" not in doc_text(docs["p01"])
    if "p10" in docs:
        t = " ".join(doc_text(docs["p10"]).split())
        a = t.find("less for volatility forecasting purposes.")
        b = t.find("This study focuses on the 15 individual S&P 500")
        checks["p10 column order"] = a != -1 and b != -1 and a < b
    if "p12" in docs:
        t = doc_text(docs["p12"])
        checks["p12 cover/ids gone"] = "Articles You May Be" not in t and "030005-" not in t
    if "p08" in docs:
        checks["p08 'financial' >= 5"] = (
            len(re.findall(r"\bfinancial\b", doc_text(docs["p08"]), re.I)) >= 5
        )
    leaks = [
        pid for pid, d in docs.items() if any(x in doc_text(d).lower() for x in BACKMATTER_LEADS)
    ]
    checks["no back-matter leaks"] = not leaks
    m["checks"] = checks
    m["leaks"] = leaks

    hit = tot = 0
    misses = []
    corpus = {
        pid: norm(doc_text(d) + " " + " ".join(t.caption + " " + t.markdown for t in d.tables))
        for pid, d in docs.items()
    }
    for q in gold:
        if q["type"] == "unanswerable" or not all(p in docs for p in q["expected_papers"]):
            continue
        text = " ".join(corpus[p] for p in q["expected_papers"])
        for s in q.get("must_include", []):
            tot += 1
            if norm(s) in text:
                hit += 1
            else:
                misses.append(f"{q['id']}:{s}")
    m["gold_strings"] = f"{hit}/{tot}"
    m["gold_misses"] = misses
    return m


def has_output(backend: str, pid: str) -> bool:
    """ML backends need their tool's cached output (the model run is done separately)."""
    raw = settings.data_dir / "parsed_raw"
    if backend in ("mineru", "hybrid"):
        return (raw / "mineru" / pid / "structured_content.json").exists()
    if backend == "marker":
        return (raw / "marker" / pid / f"{pid}.json").exists()
    return True


def run(backend: str, ids: set[str]) -> tuple[dict[str, ParsedDoc], float]:
    out = ALT / backend
    out.mkdir(parents=True, exist_ok=True)
    docs: dict[str, ParsedDoc] = {}
    t0 = time.perf_counter()
    for paper in load_manifest():
        if paper.id not in ids:
            continue
        d = BACKENDS[backend](paper, settings.raw_dir / paper.file)
        (out / f"{paper.id}.json").write_text(d.model_dump_json(indent=1), encoding="utf-8")
        docs[paper.id] = d
    return docs, time.perf_counter() - t0


def main() -> None:
    names = sys.argv[1:] or list(BACKENDS)
    gold = [json.loads(line) for line in open(ROOT_GOLD, encoding="utf-8")]
    # Score every backend on the same papers: those every backend has output for.
    ids = {p.id for p in load_manifest() if all(has_output(n, p.id) for n in names)}
    print(f"papers compared: {len(ids)} {sorted(ids)}")
    results = {}
    for name in names:
        docs, secs = run(name, ids)
        results[name] = score(docs, gold) | {"assemble_seconds": round(secs, 1)}
        print(
            name,
            json.dumps(
                {k: v for k, v in results[name].items() if k != "gold_misses"}, ensure_ascii=False
            ),
        )

    rows = [k for k in next(iter(results.values())) if k not in ("checks", "gold_misses", "leaks")]
    check_names = sorted({c for r in results.values() for c in r["checks"]})
    lines = [
        f"# Parser comparison — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Papers compared ({len(ids)}): {', '.join(sorted(ids))}",
        "",
        "| metric | " + " | ".join(results) + " |",
        "|---|" + "---|" * len(results),
    ]
    lines += [f"| {k} | " + " | ".join(str(r[k]) for r in results.values()) + " |" for k in rows]
    lines += [
        f"| {c} | "
        + " | ".join(
            "pass" if r["checks"].get(c) else ("FAIL" if c in r["checks"] else "n/a")
            for r in results.values()
        )
        + " |"
        for c in check_names
    ]
    lines += [
        "",
        "`assemble_seconds` excludes MinerU's own model run (cached; ~8 s/page on CPU).",
        "",
    ]
    for name, r in results.items():
        lines.append(
            f"**{name}** gold misses: {', '.join(r['gold_misses']) or 'none'}; "
            f"back-matter leaks: {', '.join(r['leaks']) or 'none'}  "
        )
    report = (
        settings.data_dir.parent / "eval" / "results" / f"parsers_{datetime.now():%Y%m%d_%H%M}.md"
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {report}")


ROOT_GOLD = Path(__file__).resolve().parent.parent / "eval" / "gold_questions.jsonl"

if __name__ == "__main__":
    main()

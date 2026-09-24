# Decisions log

Deviations from `PLAN.md` / `PLAN_ADDENDUM.md`, with the reason. Newest first.

## 2026-09-24 — Adaptive diversity cap; paraphrased question set

**Paraphrase set.** `eval/gold_paraphrased.jsonl`: 20 plain-language rewrites of gold
questions (no model names, acronyms or author names), linked by `paraphrase_of`.
The original gold set shares the papers' wording, so every config scored hit@6 = 1.0
and it couldn't tell BM25 from hybrid. On paraphrases it can:

| | hit@1 | hit@6 | MRR |
|---|---|---|---|
| bm25 | 0.45 | 0.85 | 0.64 |
| dense | 0.75 | 0.90 | 0.82 |
| hybrid | 0.75 | **1.00** | **0.88** |

Hybrid earns its place on how real users ask. Written by Claude; worth adding
some of your own phrasings.

**Cap.** The plan's cap (3 of 6 per paper) let a list question see only two papers
(q46). A flat cap of 2 fixes coverage but halves what the LLM reads on single-paper
questions. Adaptive (`settings.cap_mode`, default): 2 for list-style questions
(`BREADTH` regex: "which papers", "across", "compare", "consensus"...), 3 otherwise,
none when one paper is named.

| | agg. cov@6 (orig / para) | single-paper depth@6 (orig / para) |
|---|---|---|
| cap 3 (plan) | 0.72 / 0.49 | 0.64 / 0.48 |
| cap 2 | 0.80 / 0.60 | 0.53 / 0.36 |
| **adaptive** | **0.80 / 0.60** | **0.62 / 0.48** |

For uploads (usually one document) the cap costs nothing: a single-document scope
or a named paper lifts it. Lost depth is recovered in Phase 7 by neighbour expansion.

## 2026-09-24 — Chunks sized in the embedder's tokens (replaces 450/700 cl100k)

**Problem.** Phase 3 as planned (target 450, max 700 cl100k tokens) produced 483
chunks, and 214 of them (44%) were longer than bge-small-en-v1.5's 512-token input:
their tails were silently cut before embedding. BM25 still saw them; dense search
didn't. No plan test would have caught it.

**Decision.** Count tokens with the embedder's own tokenizer (`rag/tokens.py`). Each
chunk's budget is 512 minus [CLS]/[SEP] minus its context header; target is 70% of
that (~340), overlap 50 tokens. `Chunk.n_tokens` is now the full `embed_text` as the
embedder sees it, and a test re-tokenises every chunk to prove none is truncated.
Result: 647 chunks, median 383, max 512, 0 truncated.

**Small-to-big.** Chunks are kept small for precise matching; each carries `seq` so
the generator (Phase 7) can give the LLM neighbouring chunks from the same section.
This is also how large uploads scale: more chunks, not bigger ones.

**Not chosen.** A long-context embedder (nomic-embed-text, jina-v2: 8k tokens) avoids
truncation but blurs long chunks into one vector and is 3-10x heavier on CPU. Kept as
a Phase 6b experiment if retrieval eval asks for it. `tiktoken` stays in deps for
LLM token/cost accounting later.

## 2026-09-24 — Pluggable parser backends; equations indexed as LaTeX

**Decision.** `rag/parse` now has four backends that all produce the same `ParsedDoc`
(`rag/parse/backends.py`), selected with `PARSE_BACKEND` / `make parse BACKEND=...`:

| backend | what it is | speed (CPU) | where it runs |
|---|---|---|---|
| `pymupdf` | the rule-based parser from Phase 2 | ~5 s / paper | everywhere, incl. the free HF Space (uploads) |
| `mineru` | MinerU 4 "basic" tier (ONNX layout, table and formula models) + our assembly | ~8 s / page | offline, core corpus |
| `marker` | Marker (surya via local llama.cpp) + our assembly | ~10–36 s / page | offline, evaluation only |
| `hybrid` | `pymupdf` structure, equation placeholders filled with MinerU LaTeX | as pymupdf + MinerU | offline |

Everything after the tool's raw output is shared: boilerplate/licence cleanup,
section labels, abstract/back-matter rules, reference splitting
(`rag/parse/external.py`, `pipeline.assemble`). So every backend is held to the same
tests; `PARSED_DIR_OVERRIDE=data/parsed_alt/<backend> make test` runs them.

**Why not "framework as architecture".** MinerU and Marker run as CLIs in isolated
venvs (`.venvs/`), and their output is cached in `data/parsed_raw/`. Nothing in the
serving path imports them, the deploy image doesn't carry their weights, and their
licences (MinerU: Apache-2.0-based with extra conditions; Marker weights: modified
OpenRAIL-M) stay out of the core code.

**Equations (changes a PLAN.md non-goal).** Equations are now indexed as tidied
LaTeX (`rag/parse/latex.py`) when a math-aware backend recognised them; raw LaTeX is
kept in `Block.latex`. Placeholders remain only where no LaTeX exists. Knock-on
changes still to make:
- PLAN.md non-goal "No equation reproduction" and the Phase 7 system-prompt rule
  "Do not reproduce equations" need rewording.
- The gold set has no equation questions yet; add some (e.g. "What SDE does p06
  use for the stock price?") so the change is measured, not assumed.

**How the backend is chosen.** `scripts/compare_parsers.py` scores backends on the
same papers (Phase 2 checks, gold `must_include` coverage, tables, LaTeX coverage,
junk-block rate) and writes `eval/results/parsers_*.md`. Final choice waits for
Phase 6 retrieval recall; parse-level metrics only rule out damage.

**Results so far** (`eval/results/parsers_20260924_1635.md`, all 19 papers;
`..._1636.md` adds Marker on p05, p06, p10, p12, p13):

| | pymupdf | mineru | hybrid | marker (5 papers) |
|---|---|---|---|---|
| tables | 80 | **92** | 80 | 16 (MinerU 16, pymupdf 11) |
| equations as LaTeX / placeholders | 0 / 411 | **325 / 0** | 239 / 151 | 150 / 1 (MinerU 152 / 0) |
| inline-math spans | 0 | **876** | 0 | 117 (MinerU 391) |
| junk-block rate | 6.6% | **5.4%** | 6.6% | 8.0% (MinerU 12.3%) |
| gold `must_include` found | 94/96 | 94/96 | 94/96 | 30/30 |

All backends pass the Phase 2 checks (91 tests, run per backend). Provisional
choice for the core corpus: **mineru**; `pymupdf` stays for uploads and as the
tested fallback. Marker matches MinerU on display equations and tables but finds
far less inline math and needs a llama.cpp server, so it isn't worth the extra
dependency here.

**Correction.** Phase 2 flagged gold q19 ("Adams") as a gold-set problem. It wasn't:
p12 writes "Adams-Moulton" once, in a PDF block shared with equation lines, and
the pymupdf parser dropped the whole block as math. Fixed by splitting mixed
blocks line by line (`math.split_mixed`). Remaining misses q43 ("Runge" only in
p15's references) and q45 ("68.7" not in p11's text layer, likely a chart) stand.

**Privacy.** MinerU is run with local parsing only (never `--remote`); its telemetry
lives in the `mineru` doc-library server, which isn't used (`mineru-kit` only).

## 2026-09-24 — Phase 2 parsing deviations

- Tables are found from their captions, not `pdfplumber.find_tables()` alone:
  most tables here have horizontal rules only, which pdfplumber's default misses.
- p18's "duplicated text layer" came from the planning extractor, not PyMuPDF
  (0 near-duplicate lines); the filter still runs everywhere.
- p08 ligatures are fine with PyMuPDF + NFKC; no `pdftotext` fallback.
- Heading rules extended beyond the plan's regexes (unnumbered size-only headings,
  letter-spaced "A B S T R A C T", IEEE small caps); labels: an explicit "Results"
  beats "numerical", subsections keep their own label only for explicit section words.
- Extra drop rules: footnotes to page end, MDPI back-matter lead-in paragraphs,
  sidebars, licence notices, drop caps, repeated chart labels, hidden 1 pt text.

## 2026-09-24 — Manifest conventions

- `short_cite` follows APA 7 ("A", "A & B", "A et al."); collisions get extra
  surnames (p02 vs p13). `year` = year of the official citation.
- IAENG (p03, p05) issues no DOIs; `doi` is empty.

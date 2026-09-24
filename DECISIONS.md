# Decisions log

Deviations from `PLAN.md` / `PLAN_ADDENDUM.md`, with the reason. Newest first.

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

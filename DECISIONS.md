# Decisions log

Deviations from `PLAN.md` / `PLAN_ADDENDUM.md`, with the reason. Newest first.

## 2026-10-02 — Terminal theme; local launchers

- The Phase 10 design (light, serif answers, near-monochrome) is replaced at the
  owner's request by a terminal look: monospace throughout (JetBrains Mono), dark
  only, a neofetch-style header (ASCII chart logo + live corpus/model info from
  `/health`), framed panels with titles in the border, `[ button ]` controls, and
  Claude Code-style turn markers (`>` question, `⏺` answer, `⎿` status and source
  tree, `✻` spinner). Behaviour and element ids are unchanged; citations render
  as `[n]` instead of superscripts.
- `/health` also reports `llm_model` and `demo`, so the header shows whether real
  answers are on.
- `start.bat` / `start-demo.bat` start the server on port 8010 and open the browser
  (the demo needs no key); `make serve` / `make demo` moved to 8010 to match.

## 2026-09-25 — Phase 13 uploads: deviations

- **Parser:** uploads use pymupdf, not the core corpus's MinerU (~8 s/page on CPU is
  too slow for someone waiting). Same `parse_pdf` + `chunk_doc` code as the core
  papers otherwise, as the plan asks. No equation recognition for uploads.
- **Jobs in memory, not a `jobs` table.** One server process; progress is only
  interesting while it runs. The `documents` table is in the index, so uploaded
  documents themselves survive restarts. Embedding happens before the write, then
  everything goes in one transaction, so a failed ingest leaves no rows behind.
- **Text-layer check:** rejects only if page 1 *and* the middle page each have
  < 200 letters (a sparse title page alone shouldn't reject a real paper).
- **Title/author** come from PDF metadata, else the largest text on page 1, else
  the file name. Journal mastheads sometimes win (p19 uploaded as a test got
  "Research in Business & Social Science"). Good enough to identify the document.
- **UI:** the upload panel (drop zone, warning, progress, list with `remove`,
  "Remove all") opens from "Add a PDF" under the message box, instead of a list in
  the About panel: that's where a first-time user looks. Dragging a file anywhere
  opens it. The scope switch ("The 19 papers / My uploads / Both") appears only
  when an upload is ready; a successful upload switches to "My uploads".
- **Scope reaches the model:** for user or mixed scope, the passages message starts
  with a one-line scope note (the system prompt describes the 19 papers). Core-only
  requests are unchanged, so the eval path is identical.
- `make index` rebuilds from scratch and drops uploads (their PDFs stay in
  `data/uploads/`). Acceptable for a demo; noted in the README.
- Acceptance run offline (tests/test_upload.py, and in the browser in demo mode):
  p19 uploaded as `u01`, "My uploads" answers cite only `u01`, the upload survived a
  server restart, and deletion removed chunks, FTS terms, vectors and the PDF (FTS
  integrity check passes; 647 core chunks untouched).

## 2026-09-25 — Hosting: local + screen share for now

HF blocked the Space: free accounts can no longer run Docker Spaces on CPU Basic
(402, "requires a PRO subscription"; policy changed mid-2026). Rather than pay or
move to Render's free tier (15-min sleep, 512 MB), the demo runs on my machine
(`uv run uvicorn app.main:app --port 8010`) and is shown over screen share. The
deploy files below stay: the index is already in the private dataset
`1vecs/thesis-rag-data`, and `rag/storage.py` works on any host (Render, a VPS,
or HF PRO) if public hosting is wanted later.

## 2026-09-25 — Phase 11 deploy mechanics

- The Space is filled by `scripts/deploy_space.py` (an HF commit of Dockerfile,
  requirements, `rag/`, `app/`, manifest, and `deploy/SPACE_README.md`), not by
  mirroring the GitHub repo as a git remote. The Space is public; this keeps the
  plan, eval results and gold set out of it and keeps GitHub's README free of HF
  frontmatter.
- `rag/storage.py` only pulls `index.sqlite` from the private dataset for now;
  `push_index`/`push_pdf` come with uploads (Phase 13). The script refuses to
  upload the index if the dataset repo is public.
- `requirements.txt` is `uv export --no-dev` of the lockfile, so the Space runs the
  exact versions tested locally.

## 2026-09-25 — Chat mode (replaces the single-question non-goal)

The plan made the demo single-turn and "passages only". In use that felt rigid:
"explain that like I'm 12" or "what about the second paper?" failed. Changed, at
the owner's request:

- **Conversation.** `POST /api/chat {message, history, reuse_ids}`; the browser
  keeps the thread in `sessionStorage` (this tab only, gone on close) and sends the
  last 4 exchanges back. The server stays stateless. Old answers lose their `[n]`
  markers before reaching the model: those numbers pointed at passages no longer
  in view. `/api/ask` stays for the eval path.
- **Follow-up retrieval** (`rag/chat.py`). From the second message on, a small
  Haiku 4.5 call (temperature 0, ~120 output tokens max) rewrites the message into
  a standalone query, or answers `SAME` for style-only requests, which rebuild the
  previous turn's passages from their chunk ids instead of searching. Falls back to
  "previous question + message" if the call fails. First messages skip it, so a
  plain question costs what it did (~$0.006); a follow-up adds ~$0.001 plus the
  history tokens.
- **Two-tier prompt.** Statements about the papers: passages only, cited (as
  before). General background (what MAPE is, an analogy) is now allowed if signalled
  ("In general...") and uncited, and must never be presented as a paper's claim.
  The judge prompt accepts labelled background as faithful. Refusal, partial-answer
  and equation rules are unchanged.
- Rate limit and daily cap count each chat message as one question.
- Phase 8 numbers were measured before this change and need a re-run.

## 2026-09-24 — Phase 8 first run; prompt changed for partial answers

First full answer eval (`eval/results/answers_20260924_1920.md`; Haiku 4.5 answering,
Sonnet 5 judging; actual cost $1.29): accuracy 76.9% (judge correctness = 2), mean
1.72/2, faithfulness 89.2%, 0 uncited answers, numeric 100%, unanswerable refused
100%, comparison 50% / aggregation 60%.

**Main failure: over-refusal.** The plan's rule "if the passages do not contain the
answer, reply exactly <refusal>" is all-or-nothing, so Haiku refused when the
passages held part of the answer (q26, q38, q64) and phrased "the paper reports no
MAPE" (q40) as a refusal. Changes to the system prompt:
- refuse only when nothing relevant is there; otherwise answer the covered part
  and say what's missing;
- "the paper does not report X" is an answer with citations;
- the refusal's follow-up sentence may not quote figures (q53 quoted MAPE values
  from an unrelated paper);
- list / "which papers" answers cover every relevant paper with its finding, up to
  350 words (q48-q51 named papers but dropped their findings).

**Judge fix.** Sonnet 5 thinks by default and thinking counts against max_tokens;
on q30 it used all 2000 and truncated the JSON verdict. Thinking is off for the
Sonnet judge (`rag/evaluation.judge_request`).

## 2026-09-24 — Chunk-level evidence recall; three retrieval fixes (reranker rejected)

**Found by a live question.** "What SDE defines the SP-SPDE model?" got a faithful
refusal: the chunk with equation (3) was dense rank 1 but BM25 rank 20, and RRF put
it 7th, one outside the top 6. Phase 6 scored it a hit because it measured papers,
not passages. New metric `evidence@6` (in `scripts/eval_retrieval.py`): do the
question's reachable `must_include` strings appear in the passages the LLM receives?

**Plan 6b tried first: cross-encoder rerankers** (fastembed MiniLM-L-6, jina-tiny on
the fused top 30): no gain (83-88% vs 87%) and +3.2-3.7 s per question on CPU. They
are trained on web search, not LaTeX or regression tables. Rejected.

**Diagnosis of the 16 misses:** 9 were table cells; several were chunks ranked 3rd-4th
dropped by the per-paper cap; paraphrase misses were dense rank 4-6 but absent from
BM25, and fusion buried them at 17th-44th. Fixes, each measured:

1. Adaptive cap caps only list-style questions ("across" needs a collection noun);
   two-letter surnames count in citation form ("Li et al.").
2. `guaranteed_per_retriever = 2`: each retriever's top 2 always reach the top 6.
3. `attach_tables`: tables a retrieved passage names ("reported in Table 5") are
   attached from the same paper, at most 3 (`rag/generate.referenced_tables`).

| hybrid + adaptive | evidence@6 orig / para | depth@6 orig | context sent |
|---|---|---|---|
| before | 87% / 60% | 0.62 | ~4,100 tokens |
| after | **92% / 70%** | **0.85** | ~3,600 tokens |

Paper hit@6 and multi-paper coverage unchanged. **Open:** on paraphrases, dense alone
reaches 90% evidence vs hybrid 70% (BM25 adds noise when wording differs). Not tuned
further on 20 questions; next step is a larger paraphrase set, then weighting.

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
- Done: equation gold questions q56-q66 added after review (retrieval hit@6 9/9).

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

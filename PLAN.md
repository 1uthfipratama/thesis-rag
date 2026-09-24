# thesis-rag — Build Plan

A small, deployed retrieval-augmented Q&A system over the 19 papers behind my skripsi (adaptive-parameter ODE models for stock prices, volatility, geopolitical risk, numerical methods). Portfolio project and a rehearsal for the Upskill TB LMS chatbot at work.

This document is written to be handed to Claude Code. Read the whole thing before writing code.

---

## 0. Working agreement (for Claude Code)

1. **Build phase by phase.** Each phase ends with an acceptance check. Run it, show me the output, and stop. Do not start the next phase until I say so.
2. **Every stage writes its output to disk** (`data/parsed/`, `data/chunks/`, `data/index.sqlite`, `eval/results/`). When retrieval misbehaves, I need to open the intermediate file and see what was actually indexed.
3. **Do not edit `eval/gold_questions.jsonl`.** If a question looks wrong, flag it; I decide.
4. **Retrieval is evaluated before any LLM call is written** (Phase 6 before Phase 7). No exceptions.
5. **No framework-as-architecture.** No LangChain/LlamaIndex pipelines. Individual utilities are fine if they save real time; say why.
6. Explain non-obvious decisions in short code comments. I need to be able to defend every choice in an interview.
7. Python 3.11, type hints, `ruff` for lint/format, `pytest` for tests.

---

## 1. Goals and non-goals

**Goals**

- Answer questions about the 19 papers: methods, data, findings, numbers, comparisons across papers.
- Every answer cites paper, section and page. Every citation is checkable.
- Say "the corpus doesn't cover this" instead of guessing.
- Measured quality: retrieval recall and answer correctness on a 55-question gold set, reported in the README.
- Deployed at a public URL with HTTPS, rate-limited, cheap to run.

**Non-goals (write these in the README)**

- No equation reproduction or explanation from rendered math. Equations are detected and excluded from the index; the prose around them is kept.
- No figures or charts.
- English only.
- Single-turn Q&A. No chat memory, no accounts.
- No live market data.

---

## 2. Corpus manifest

Paper IDs follow the upload filename prefix. Layout and hazards are what I observed during planning; verify them in Phase 2.

| ID | Short cite | Venue | Layout | Licence (check) | Known parsing hazards |
|---|---|---|---|---|---|
| p01 | Li, Chiang-Lin & Lee 2018 | Int. J. Differential Equations (Hindawi) | 2-col | CC BY | Wiley download stamp repeated on every page ("…Downloaded from https://onlinelibrary.wiley.com…"); heavy math |
| p02 | Noviantri, Chandra & Yusof 2022 | EBEE '22 (ACM) | 2-col | ACM copyright | Flowchart figure; tables |
| p03 | Chandra, Noviantri & Iswanto 2025 | IAENG Int. J. Applied Mathematics 55(6) | 2-col | unclear | Running header/footer "Volume 55, Issue 6, June 2025, Pages 1502-1511"; GUI screenshots |
| p04 | Workineh, Mekonnen & Belew 2024 | Front. Appl. Math. Stat. | 2-col + left sidebar | CC BY | Left-margin metadata sidebar on p.1; wide numeric tables |
| p05 | Noviantri, Nariswari, Saputra & Yolandito 2026 | IAENG Int. J. Computer Science 53(6) | 2-col | unclear | IAENG running header/footer; many equations; flowcharts |
| p06 | Eissa & Elsayed 2022 | Symmetry (MDPI) | 1-col + left sidebar | CC BY | MDPI left sidebar; heavy stochastic-calculus math |
| p07 | Truong, Doan & Nguyen 2024 | Int. J. Energy Economics & Policy | 2-col | CC BY | Journal footer on every page; regression tables |
| p08 | Budiharto 2021 | Journal of Big Data | 1-col | CC BY | Ligature glyphs: one extractor produced "fnancial"/"efcient"; `pdftotext` recovers "financial". Verify PyMuPDF output + NFKC |
| p09 | Xie, Xia & Gao 2021 | PLOS ONE | 1-col + left sidebar | CC BY | Noise lines "a1111111111" from margin art; left sidebar |
| p10 | Lyócsa & Todorova 2021 | Energy Economics (Elsevier) | 2-col | Elsevier, not open | Best two-column test document; long footnotes; appendix tables |
| p11 | Koç, Erdoğan, Barjakly & Peker 2021 | Proceedings (MDPI) | 1-col + left sidebar | CC BY | Off-topic for finance; tests out-of-domain retrieval |
| p12 | Noviantri, Kusnadi, Suhendar, Yusof & Aryuni 2025 | AIP Conf. Proc. 3272 | 1-col | AIP exclusive licence | **Page 1 is an AIP cover page** ("Articles You May Be Interested In"); vertical timestamp "30 September 2025 12:24:34" on every page; page ids "030005-3" |
| p13 | Noviantri/Chandra, Komsiyah & Suhendar 2023 | AIP Conf. Proc. 2975 | 1-col | AIP | Author order differs between the PDF header and how p03/p05 cite it; keep both in metadata |
| p14 | Smales (c. 2019) | SSRN working paper 3414134 | 1-col | author copyright | Footer "Electronic copy available at: https://ssrn.com/abstract=3414134" on every page; "<Insert Figure 1>" placeholders |
| p15 | Zhang, Tian & Chabani 2021 | Applied Mathematics and Nonlinear Sciences | 1-col | CC BY | Algorithm pseudo-code block; parameter table |
| p16 | Mihova, Georgiev, Raeva, Georgiev & Pavlov 2024 | Mathematics (MDPI) | 1-col + left sidebar | CC BY | MDPI sidebar; MATLAB code appendix (drop) |
| p17 | Oyuna & Liu 2021 | SAGE Open | 2-col | CC BY | Equations extract badly (symbol fonts); duplicated intro paragraph in the source itself |
| p18 | Özdemir, Vurur, Ozen, Świecka & Grima 2025 | Economies (MDPI) | 1-col + left sidebar | CC BY | **Duplicated text layer**: many lines appear twice in extraction. Needs near-duplicate line removal |
| p19 | Enow 2023 | Int. J. Research in Business & Social Science | 1-col | CC BY | Hurst tables repeated per market |

My own skripsi is **not** in the corpus. If I add it later, it becomes `p00`, and gold question q53 gets rewritten as answerable.

### Licensing and the public repo

- `data/raw/` is **gitignored**. Never commit the PDFs. Several are not open access (p02, p10, p12, p13, p14, and the IAENG papers are unclear).
- Commit `data/manifest.yaml` (metadata + DOI/URL) instead, so anyone can fetch the papers themselves.
- `data/index.sqlite` contains extracted text, so it is also **not committed**. It is built locally and shipped to the server as a deploy artifact (Phase 11).
- The UI shows short snippets (≤ 300 chars) with a citation, never full passages or PDFs.

---

## 3. Stack and rationale

| Concern | Choice | Why |
|---|---|---|
| PDF text + layout | **PyMuPDF** (`fitz`), `page.get_text("dict")` | Gives spans with bbox, font size and flags, which is what column detection and heading detection need. Fast. |
| Tables | **pdfplumber** `page.find_tables()` | Better table geometry than PyMuPDF text; only used for table regions. |
| Near-duplicate lines | **rapidfuzz** | For p18's duplicated text layer. |
| Token counting | **tiktoken** `cl100k_base` | Approximate but consistent; only used for chunk sizing. |
| Embeddings | **fastembed**, `BAAI/bge-small-en-v1.5` (384-d) | ONNX on CPU, no PyTorch, small container, runs query-time embedding on a cheap VPS. Corpus is English-only. Swap to `bge-base-en-v1.5` (768-d) only if eval says so. |
| Keyword search | **SQLite FTS5**, `bm25()` | Built into SQLite. Catches exact terms embeddings miss: MAPE, ARDL, HAR, Lotka-Volterra, Adams-Bashforth, Hurst, author surnames, tickers. |
| Vector search | **sqlite-vec** (`vec0` virtual table) | Same file as FTS5. ~1k chunks does not need a vector database. |
| Fusion | **Reciprocal Rank Fusion**, k = 60 | No score calibration needed between BM25 and cosine. |
| Reranker | Optional, Phase 6b only if needed | fastembed cross-encoder. Don't add unless the eval shows a ranking problem. |
| LLM | **Anthropic Messages API**, model from env `LLM_MODEL` | Default `claude-haiku-4-5-20251001` for cost; try `claude-sonnet-5` in answer eval and keep whichever wins on the gold set. |
| API | **FastAPI** + uvicorn, SSE via `sse-starlette` | Streaming answers; simple. |
| Rate limiting | **slowapi** | Public endpoint that costs money per call. |
| Frontend | Static `index.html` + `app.js` + `style.css`, no framework | Minimal by design; served by FastAPI `StaticFiles`. |
| Deploy | **Docker** + **Caddy** (auto-TLS) on a small VPS | This is the deployment rehearsal. Fallback: Render or Hugging Face Spaces (Docker). |

Python deps: `pymupdf pdfplumber rapidfuzz tiktoken fastembed sqlite-vec anthropic fastapi "uvicorn[standard]" sse-starlette slowapi pydantic-settings pyyaml`
Dev deps: `pytest ruff`

---

## 4. Repository layout

```
thesis-rag/
├── PLAN.md
├── README.md
├── pyproject.toml
├── Makefile
├── .env.example            # ANTHROPIC_API_KEY=, LLM_MODEL=, EMBED_MODEL=, RATE_LIMIT=
├── .gitignore              # data/raw/, data/parsed/, data/chunks/, data/index.sqlite, .env
├── data/
│   ├── manifest.yaml       # committed
│   ├── raw/                # PDFs, gitignored
│   ├── parsed/             # {id}.json + {id}.md, gitignored
│   ├── chunks/             # chunks.jsonl, gitignored
│   └── index.sqlite        # gitignored
├── rag/
│   ├── __init__.py
│   ├── config.py           # pydantic-settings
│   ├── parse/
│   │   ├── extract.py      # PyMuPDF → blocks with bbox/font
│   │   ├── layout.py       # sidebar + column detection, reading order
│   │   ├── clean.py        # boilerplate, dedupe, NFKC, hyphenation
│   │   ├── sections.py     # heading detection, canonical labels, references cut
│   │   ├── tables.py       # pdfplumber tables → markdown
│   │   ├── math.py         # math-density scoring
│   │   └── pipeline.py     # orchestrates one PDF → ParsedDoc
│   ├── chunk.py
│   ├── index.py            # build SQLite (FTS5 + vec0)
│   ├── retrieve.py         # bm25, dense, rrf
│   ├── generate.py         # prompt, Anthropic call, citation validation
│   └── schemas.py          # pydantic models
├── app/
│   ├── main.py             # FastAPI routes
│   └── static/             # index.html, app.js, style.css
├── scripts/
│   ├── parse_all.py
│   ├── inspect_parse.py    # human review of one paper
│   ├── build_chunks.py
│   ├── build_index.py
│   ├── eval_retrieval.py
│   └── eval_answers.py
├── eval/
│   ├── gold_questions.jsonl   # provided, do not edit
│   ├── gold_questions.md      # readable rendering
│   └── results/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── Caddyfile
```

`Makefile` targets: `setup`, `parse`, `chunks`, `index`, `eval-retrieval`, `eval-answers`, `serve`, `test`, `docker`, `deploy`.

---

## Phase 0 — Scaffold (≈30 min)

- Create the layout above, `pyproject.toml`, `Makefile`, `.gitignore`, `.env.example`.
- `rag/config.py` with pydantic-settings: paths, `EMBED_MODEL`, `LLM_MODEL`, `TOP_K=6`, `BM25_K=30`, `DENSE_K=30`, `RRF_K=60`, `MAX_QUESTION_CHARS=500`.
- Copy the 19 PDFs into `data/raw/` as `p01.pdf` … `p19.pdf`.

**Accept:** `make setup && make test` passes on an empty test suite; `git status` shows no PDFs.

---

## Phase 1 — Manifest (≈30 min)

`data/manifest.yaml`, one entry per paper:

```yaml
- id: p10
  file: p10.pdf
  sha256: <computed>
  title: "What drives volatility of the U.S. oil and gas firms?"
  authors: ["Štefan Lyócsa", "Neda Todorova"]
  year: 2021
  venue: "Energy Economics 100"
  doi: "10.1016/j.eneco.2021.105367"
  short_cite: "Lyócsa & Todorova 2021"
  layout: two_column          # expected; verified in Phase 2
  open_access: false
  drop_pages: []              # e.g. p12 → [1]
  notes: ""
```

Fill title/authors/year/venue/DOI from the first page of each PDF (script can propose, I confirm). `short_cite` is what the UI shows.

**Accept:** a script validates every `file` exists and every `sha256` matches.

---

## Phase 2 — Parsing (the main work, ≈3–4 h)

Output per paper: `data/parsed/{id}.json` plus a human-readable `{id}.md`.

```python
class Block(BaseModel):
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    is_bold: bool
    kind: Literal["text", "heading", "table", "equation", "caption", "sidebar"]

class Section(BaseModel):
    label: str            # canonical: abstract | introduction | literature_review | methodology | data | results | discussion | conclusion | other
    heading: str          # as printed, e.g. "3.2. Methodology"
    page_start: int
    blocks: list[Block]

class ParsedDoc(BaseModel):
    paper_id: str
    title: str
    sections: list[Section]
    tables: list[TableBlock]      # markdown + caption + page
    references: list[str]         # kept as metadata, never chunked
    dropped: dict[str, int]       # counts per drop reason, for the report
```

### 2.1 Extraction

`page.get_text("dict")` → spans with bbox, size, flags (bold = `flags & 16`). Merge spans into lines, lines into blocks (PyMuPDF blocks are a fine starting unit). Skip pages listed in `drop_pages`.

### 2.2 Unicode and ligatures

Apply `unicodedata.normalize("NFKC", text)` to every span (turns `ﬁ`, `ﬀ` into `fi`, `ff`). Then run a check: count dictionary-word hits for `financial`, `efficient`, `the` on p08. If PyMuPDF still drops ligatures on p08, fall back to `pdftotext -layout` for that paper only and record it in the manifest `notes`.

### 2.3 Sidebar and column detection

Per page:

1. **Sidebar:** blocks with `x1 < 0.30 * page_width` whose column contains citation/received/copyright keywords, on the first 1–2 pages of MDPI/Frontiers/PLOS papers (p04, p06, p09, p11, p16, p18). Mark `kind="sidebar"` and drop.
2. **Full-width blocks:** width > 0.6 × text-area width (titles, abstracts, wide tables, figures).
3. **Columns:** remaining blocks, split by whether the block centre-x is left or right of the text-area midpoint. If fewer than ~15% of a page's text blocks fall in one side, treat the page as single-column.
4. **Reading order within a page:** full-width blocks above the column region (top-down), then left column top-down, then right column top-down, then full-width blocks below the column region.

Store the detected layout per page; compare with `manifest.layout` and report mismatches.

### 2.4 Boilerplate removal

- **Repeated-line detector:** normalize each line (lowercase, digits → `#`, collapse whitespace). Any normalized line appearing on ≥ 50% of a document's pages is boilerplate. Drop it.
- **Explicit patterns** (belt and braces), each with a test:
  - `Downloaded from https://onlinelibrary\.wiley\.com` (p01)
  - `Electronic copy available at: https://ssrn\.com` (p14)
  - `Volume \d+, Issue \d+, .* Pages \d+-\d+` and `IAENG International Journal of` (p03, p05)
  - `^\d{2} \w+ \d{4} \d{2}:\d{2}:\d{2}$` and `^030005-\d+$` (p12)
  - `^a1{10}$` (p09)
  - `<Insert (Figure|Table) \d+>` (p14)
  - Publisher boilerplate: `Publisher’s Note`, `Creative Commons`, `Copyright ©`, `This article is licensed under`
- **Near-duplicate lines** (p18): drop a line if `rapidfuzz.fuzz.ratio(line, previous_kept_line) > 95`. Run on every paper, report counts per paper; p18 should dominate.
- **Hyphenation:** join `word-\nword` when the joined form is alphabetic and the line break came from wrapping.

### 2.5 Headings and sections

Heading candidates: font size ≥ 1.1 × the document's median body size, or bold, **and** ≤ 12 words, **and** matching one of:
`^\d+(\.\d+)*\.?\s+\S`, `^[IVX]+\.\s+\S`, or ALL CAPS.

Map to canonical labels with a keyword table (first match wins):

| Canonical | Keywords |
|---|---|
| abstract | abstract |
| introduction | introduction |
| literature_review | literature, related work, review |
| data | data, sample, dataset |
| methodology | method, methodology, model, approach, algorithm, framework, solution, numerical |
| results | result, simulation, empirical, experiment, forecast, fit, analysis |
| discussion | discussion |
| conclusion | conclusion, concluding |
| references | references, bibliography, daftar pustaka |
| backmatter | acknowledg, author contribution, funding, conflict, data availability, appendix, publisher |

Abstract often has no heading (p01, p10, p13): treat the text block(s) between the title and the first detected heading as `abstract` if it contains ≥ 60 words.

### 2.6 References and back matter

Everything from the `references` heading onward goes to `ParsedDoc.references` (split on `^\[\d+\]`, `^\d+\.\s`, or author-year line starts). `backmatter` sections are dropped from the index, but keep "Data availability" text in metadata (p03 and p05 link GitHub repos).

### 2.7 Tables

For each page, `pdfplumber` `page.find_tables()`. Keep tables with ≥ 2 rows and ≥ 2 columns and at least one numeric cell. Serialize to Markdown. Caption: nearest block above (or below) matching `^(Table|TABLE)\s+[\dIVX]+`. Remove the text blocks inside the table's bbox from the prose stream so numbers aren't indexed twice.

Priority tables to verify by eye (these carry gold answers): p02 Tables 2–3, p03 Tables II and IV, p13 Tables 3–4, p05 forecast MAPE figures (may be charts, not tables — note it), p07 Tables 4–6, p10 Table 3, p17 Table 6, p19 Table 1, p01 Tables 2–3, p11 Table 4.

### 2.8 Math detection

For each text block, compute

```
math_density = (Greek letters + math operators + Unicode sub/superscripts + math-italic codepoints U+1D400–U+1D7FF) / max(1, alphabetic chars)
```

If `math_density > 0.25` (tune on p05 and p06), mark `kind="equation"`. Replace with a placeholder in the prose stream: `[equation (13) omitted]` if an equation number `\((\d+)\)` is present, else `[equation omitted]`. Keep the surrounding sentences; they carry the meaning.

### 2.9 Review tooling

`scripts/inspect_parse.py p10` prints: layout per page, section tree with word counts, first 300 chars per section, table captions, and `dropped` counts.

**Tests (`tests/test_parse.py`):**

- p01: no line contains `onlinelibrary.wiley.com`.
- p10 page 2: the sentence ending "…less for volatility forecasting purposes." appears **before** "This study focuses on the 15 individual S&P 500" (left column before right column).
- p12: "Articles You May Be Interested In" is absent; no `030005-` page ids remain.
- p08: the word `financial` appears ≥ 5 times.
- p18: no two consecutive lines with ratio > 95.
- Every paper: `references` section exists and is non-empty, and the word "References" does not appear inside any indexed section.
- Every paper: has an `abstract` and a `conclusion` section (p14 may use "Conclusion" differently; allow override in manifest).

**Accept:** all tests pass; I review `inspect_parse.py` output for p01, p05, p08, p10, p12, p18.

---

## Phase 3 — Chunking (≈1 h)

Output: `data/chunks/chunks.jsonl`.

```python
class Chunk(BaseModel):
    chunk_id: str        # f"{paper_id}:{section_label}:{n:03d}"
    paper_id: str
    section: str
    heading: str
    page_start: int
    page_end: int
    kind: Literal["paper_card", "prose", "table"]
    text: str            # shown to the user / LLM
    embed_text: str      # context header + text; what gets embedded and FTS-indexed
    n_tokens: int
    content_hash: str    # sha256(text)
```

Rules:

- **Paper card** (one per paper): title, authors, year, venue, abstract. Helps "which paper…" questions.
- **Prose:** split within a section only, never across sections. Target 450 tokens, max 700, overlap 60. Split on paragraph boundaries first, then sentences.
- **Tables:** one chunk per table, caption + Markdown, never split. If > 700 tokens, split by rows and repeat the header row.
- **Abstract** is its own chunk (inside the paper card).
- **Context header** prepended to `embed_text` only:
  `"{short_cite} — {title} > {heading}\n\n"`
  This is contextual retrieval; it makes an isolated chunk like "reduce the step size…" findable by paper and section.

**Accept:** print chunk count per paper and token histogram. Expect roughly 600–1,200 chunks total. No chunk > 700 tokens; no chunk with `section == "references"`.

---

## Phase 4 — Index (≈1 h)

Single SQLite file, idempotent build (`make index` rebuilds from scratch).

```sql
CREATE TABLE papers (
  paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INT,
  venue TEXT, short_cite TEXT, doi TEXT
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  chunk_id TEXT UNIQUE, paper_id TEXT, section TEXT, heading TEXT,
  page_start INT, page_end INT, kind TEXT,
  text TEXT, embed_text TEXT, n_tokens INT, content_hash TEXT
);

CREATE VIRTUAL TABLE fts_chunks USING fts5(
  embed_text, content='chunks', content_rowid='id',
  tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[384]);
-- vec_chunks.rowid = chunks.id

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- embed_model, embed_dim, built_at, corpus_sha (hash of all content_hashes)
```

- Embed with fastembed `passage_embed` for chunks and `query_embed` for queries. Check whether the library applies BGE's query instruction automatically; if not, prepend `"Represent this sentence for searching relevant passages: "` to queries only.
- Normalize vectors; use cosine distance.
- On startup, the API refuses to run if `meta.embed_model` ≠ configured `EMBED_MODEL` (prevents silent embedding drift).

**Accept:** row counts match `chunks.jsonl`; a smoke query "Lotka-Volterra Adams-Bashforth" returns p12 in both BM25 and dense top 5.

---

## Phase 5 — Retrieval (≈1 h)

`rag/retrieve.py`:

```python
def bm25(q: str, k: int) -> list[tuple[int, float]]
def dense(q: str, k: int) -> list[tuple[int, float]]
def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]
def search(q: str, top_k: int = 6, paper_ids: list[str] | None = None) -> list[Hit]
```

- **FTS5 query sanitization (gotcha):** raw input like `ARDL(4,4,1)` or `Adams-Bashforth` breaks FTS5 syntax. Tokenize to `\w+`, drop stopwords, quote each token and OR them: `"ardl" OR "4" OR …`. Never pass user text straight to `MATCH`.
- `bm25()` returns lower-is-better; order ascending.
- sqlite-vec KNN: `SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?`.
- Fuse BM25 top 30 and dense top 30 with RRF, return top 6.
- **Diversity cap:** at most 3 chunks from the same paper in the final 6, unless the question names one paper. Cross-paper questions need breadth.

**Accept:** `python -m rag.retrieve "which papers use the geopolitical risk index"` prints ranked hits with paper, section, page, fused score.

---

## Phase 6 — Retrieval evaluation (≈1 h) — before any LLM code

`scripts/eval_retrieval.py` over `eval/gold_questions.jsonl` (skip `type == "unanswerable"`):

- **paper_hit@k** (k = 1, 3, 6, 10): at least one expected paper in the top k.
- **paper_coverage@6**: for multi-paper questions, fraction of expected papers present in the top 6.
- **MRR** at paper level.
- Breakdown by question `type`.

Run four configurations and write a table to `eval/results/retrieval_{timestamp}.md`:

1. BM25 only
2. Dense only
3. Hybrid RRF
4. Hybrid RRF + diversity cap

Also log every miss: question id, expected papers, what came back. Misses are where the learning is.

**Targets:** hybrid paper_hit@6 ≥ 0.90 on single-paper questions; coverage@6 ≥ 0.70 on aggregation questions (q46–q52 are hard; q50 has ten expected papers and cannot reach full coverage at k = 6, which is fine).

**6b (only if targets are missed):** in order, try: larger `bge-base` model; chunk size 300 vs 600; adding a cross-encoder reranker on the top 30. Change one thing at a time and re-run.

**Accept:** results table committed; I review misses.

---

## Phase 7 — Generation (≈1.5 h)

`rag/generate.py`.

**Context format** given to the model:

```
[1] p13 · Noviantri et al. 2023 · Simulation Results · p.3
<chunk text>

[2] p02 · Noviantri, Chandra & Yusof 2022 · Results · p.4
<chunk text>
```

**System prompt** (start here; tune only via the answer eval):

```
You answer questions about a fixed corpus of 19 research papers on stock-price
modelling with differential equations, volatility, geopolitical risk, and numerical
methods.

Rules:
- Use only the numbered passages provided. Do not use outside knowledge.
- End every factual sentence with citation markers for the passages that support
  it, like [2] or [1][3].
- If the passages do not contain the answer, reply exactly:
  "The corpus doesn't cover this." Then, in one sentence, say what related topic
  the corpus does cover, if any.
- Report numbers exactly as written, with units and the paper they come from.
- If passages from different papers disagree, say so and attribute each view.
- If a paper's abstract and its tables disagree, trust the table and mention the
  discrepancy.
- Do not reproduce equations. Describe them in words.
- Plain prose, no headings, under 200 words unless the question asks for a list.
```

**Call:** Anthropic Messages API, streaming, `temperature=0.2`, `max_tokens=600`.

**Post-processing:**

- Extract `\[(\d+)\]`; drop markers that point to passages not provided; if an answer has zero valid citations and is not the refusal string, log it as `uncited`.
- Return `sources`: for each cited passage, `{n, paper_id, short_cite, section, page_start, snippet}` with `snippet` ≤ 300 chars.

**Accept:** `python -m rag.generate "What was the fit MAPE for the IDX Composite logistic model?"` prints an answer containing 5.067% with a citation to p13.

---

## Phase 8 — Answer evaluation (≈1.5 h)

`scripts/eval_answers.py`, all 55 questions, writes `eval/results/answers_{model}_{timestamp}.jsonl` and a summary `.md`.

**Automatic checks:**

- `must_include`: every string appears in the answer (case-insensitive; normalize `,` in numbers).
- Unanswerable (q53–q55): answer starts with the refusal string and contains no numbers taken from the corpus.
- Trap q40: answer must not contain any `%` value or the words MAPE/RMSE attached to a number.
- Citations valid; at least one citation per answerable question.

**LLM judge** (separate call, use the stronger model), per question, given question + gold answer + system answer + retrieved passages:

```
Score 0-2 on each:
correctness: 2 = matches gold on all key facts, 1 = partly, 0 = wrong or missing
faithfulness: 2 = every claim supported by the passages, 1 = minor unsupported detail, 0 = fabricated
Return JSON: {"correctness": n, "faithfulness": n, "reason": "..."}
```

**Manual:** I read every answer where judge correctness < 2, plus 10 random others. Record disagreements with the judge; judge-human agreement is itself a number worth reporting.

**Report:** accuracy by type, faithfulness rate, refusal accuracy on unanswerables, cost per 100 questions, median latency. Run for Haiku and Sonnet; pick one.

---

## Phase 9 — API (≈1 h)

`app/main.py`:

| Route | Behaviour |
|---|---|
| `GET /health` | `{"ok": true, "chunks": n, "embed_model": "..."}` |
| `GET /api/papers` | manifest list: id, short_cite, title, year, venue, doi |
| `POST /api/ask` | body `{"question": str}`; SSE stream |

SSE events, in order:

1. `sources` — JSON list of retrieved passages (so the UI can render them while the answer streams)
2. `token` — text deltas
3. `done` — `{ "cited": [1,3], "latency_ms": n }`
4. `error` — on failure, a user-safe message

Guards:

- Question length ≤ 500 chars; empty → 400.
- slowapi: 10 requests/minute per IP, plus a global daily cap (e.g. 300) stored in SQLite; over the cap → friendly 429.
- `ANTHROPIC_API_KEY` from env only. Set a monthly spend limit in the Anthropic Console as well.
- Log question, latency, cited ids, token counts. Retention 30 days. No IP storage beyond rate limiting.
- Same-origin only; no CORS.

**Accept:** `curl -N -X POST localhost:8000/api/ask -d '{"question":"..."}' -H 'content-type: application/json'` streams events.

---

## Phase 10 — Frontend (≈2 h)

### Design intent

Very minimal, text-first, closer to reading a paper than using an app. Near-monochrome. The only visual elements are the text, the input, and the citation list.

Avoid the generic AI-tool look: no monospace everywhere, no glowing gradients, no bracket tags, no HUD styling. Monospace is reserved for real metadata (paper IDs, page numbers).

### Tokens

```css
:root {
  --bg: #fafaf7;  --fg: #141414;  --muted: #6b6b6b;  --rule: #e4e2dc;
  --measure: 680px;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;   /* answers */
  --sans: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif; /* UI */
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;       /* ids only */
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #111110; --fg: #ececea; --muted: #9a9a96; --rule: #2a2a28; }
}
```

Body 17px, line-height 1.6. Links underlined, no colour accent.

### Layout and behaviour

- **Initial state:** lowercase wordmark `thesis-rag`, one line of description ("Ask about 19 papers on ODE stock models, volatility and geopolitical risk."), the input, and three example questions as plain text links (pick from gold set: q36, q27, q52).
- **After asking:** the input moves to the top; the answer streams below in the serif face.
- Citation markers `[n]` render as small superscript links that scroll to source *n*.
- **Sources list** under the answer, separated by a hairline rule: `[1] Lyócsa & Todorova 2021 · Results · p.7` in sans, paper id in mono and muted; clicking expands the ≤ 300-char snippet and the DOI link.
- Refusals render in muted italic.
- A small "About" link at the bottom opens the paper list (from `/api/papers`) and a one-paragraph method + eval summary.
- Keyboard: Enter submits, Shift+Enter newline, Esc clears. Visible focus states. Works at 360px width.
- No local storage, no history. One question at a time.

Fonts via Google Fonts `<link>` with the fallbacks above.

**Accept:** Lighthouse accessibility ≥ 95; works on mobile Safari and Chrome; streaming visible token by token.

---

## Phase 11 — Deploy (≈2 h)

### Dockerfile

- `python:3.11-slim`, non-root user.
- Install deps; set `FASTEMBED_CACHE_PATH=/models`; **pre-download the embedding model at build time** so the first request isn't a 100 MB download:
  `RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"`
- Copy `rag/`, `app/`, and `data/index.sqlite` (built locally; present in the build context but gitignored).
- `CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`
- `HEALTHCHECK` on `/health`.

### VPS (primary)

- 2 vCPU / 4 GB is plenty. Singapore or Jakarta region for latency.
- SSH key auth only, password login disabled, `ufw` allowing 22/80/443.
- `docker-compose.yml`: `app` + `caddy`. Caddy terminates TLS automatically.
- `Caddyfile`:

  ```
  rag.<your-domain> {
    reverse_proxy app:8000 {
      flush_interval -1   # needed so SSE tokens aren't buffered
    }
  }
  ```

- Ship the index: build the image locally and push to a **private** registry (GHCR private) or `scp data/index.sqlite` to the server and mount it read-only.
- `.env` lives on the server only.

### Fallback

Render or Hugging Face Spaces (Docker). Same image; set secrets in the dashboard; the index must still come from a private source, never the public repo.

**Accept:** HTTPS URL answers q27 correctly with citations; `/health` green; rate limit returns 429 after 10 quick requests.

---

## Phase 12 — README (≈1 h)

- One-paragraph what and why.
- Architecture diagram (Mermaid): PDFs → parse → chunks → SQLite (FTS5 + vec) → hybrid retrieve → Claude → SSE → UI.
- **Evaluation table** from Phases 6 and 8 (retrieval by config, answer accuracy by type, faithfulness, refusal accuracy). This is the part that makes the project stand out.
- Parsing decisions and what they fixed, with before/after snippets (p10 columns, p18 duplicates, p12 cover page).
- Known limitations (equations, figures, English only, small corpus).
- How to run locally with your own PDFs; note that the corpus is not redistributed.
- Link to the live demo.

---

## Gold evaluation set

`eval/gold_questions.jsonl` — 55 questions, one JSON object per line:

```json
{"id": "q27", "type": "numeric",
 "question": "...", "answer": "...",
 "expected_papers": ["p07"], "expected_sections": ["results"],
 "must_include": ["0.1438", "12.9467", "2.81"],
 "notes": "..."}
```

| Type | Count | What it tests |
|---|---|---|
| numeric | 14 | exact figures from tables and text |
| factoid | 11 | data, periods, entities |
| method | 10 | how a model or algorithm works |
| comparison | 9 | within- or cross-paper contrasts |
| aggregation | 5 | lists across many papers (hardest for retrieval) |
| discrepancy | 3 | abstract vs table (q12), inverted Hurst interpretation (q35), no metric reported (q40) |
| unanswerable | 3 | refusal behaviour (q53 is my own skripsi, not in corpus) |

The answers were extracted from the paper text by Claude during planning. **Spot-check at least 10 against the PDFs before trusting any score.** Several notes record inconsistencies inside the papers themselves (p08 abstract 94.57% vs table 94.59%; p13 simulation counts that don't sum to 40; p03 abstract figures that are HMSP's, not averages).

---

## Realistic timeline

- **Day 1:** Phases 0–6. Parsing will take longer than planned; that is normal, and it's the most valuable part.
- **Day 2:** Phases 7–9 and the answer eval.
- **Day 3:** Frontend, deploy, README.

If day 1 ends with parsing done and retrieval evaluated, that's a good day.

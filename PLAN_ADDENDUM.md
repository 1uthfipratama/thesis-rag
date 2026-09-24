# thesis-rag — Addendum: free hosting, sharing, and PDF upload

Read after `PLAN.md`. This **replaces Phase 11** and **adds Phases 13 and 14**. Everything else in `PLAN.md` stands.

Goal: a URL I can send to my supervisor. They open it, ask questions, optionally drop in a PDF of their own, and it works. No install, no account, no cost to them.

---

## Decision summary

| Concern | Choice | Why |
|---|---|---|
| Host | **Hugging Face Spaces, Docker SDK, CPU Basic (free)** | 2 vCPU / 16 GB RAM, no payment method, HTTPS included. Sleeps only after ~48 h idle, so a shared link is usually warm. |
| Keep-awake | GitHub Actions cron pinging `/health` daily | Free for public repos; 48 h threshold means one ping a day is enough. |
| Durable storage | **Private HF Dataset repo**, synced via `huggingface_hub` | Space disk is ephemeral; persistent storage is a paid add-on. Dataset repos are free, versioned, and private. Also keeps non-open-access PDFs out of the public GitHub repo. |
| Access control | Shared access code in an env var | The link stays shareable; the LLM spend stays gated. |
| Spend safety | Monthly cap in the Anthropic Console + per-code rate limit + daily question cap in SQLite | Public endpoint that costs money per call. |

Rejected: Render free (spins down after 15 min with ~1 min cold start on 512 MB / 0.1 vCPU — the supervisor would hit a loading screen almost every time); Fly.io (no real free tier now); Vercel/Cloudflare (no persistent SQLite without a rewrite); Oracle Cloud Always Free (best hardware, but painful signup, frequent ARM capacity refusals, and I'd own all the ops — save it for the home lab).

---

## Phase 11 (replaces original) — Deploy to Hugging Face Spaces

### 11.1 Repos

Three, and keep them straight:

1. **GitHub `thesis-rag`** (public) — source, `PLAN.md`, `eval/`, README with results. No PDFs, no index.
2. **HF Space `<user>/thesis-rag`** (public) — Dockerfile + app code. Mirrors GitHub; can be a `git remote` push target.
3. **HF Dataset `<user>/thesis-rag-data`** (**private**) — `index.sqlite`, `data/raw/*.pdf`, `data/uploads/*.pdf`. Never public: several papers are not open access.

### 11.2 Space configuration

`README.md` at the Space root needs YAML frontmatter:

```yaml
---
title: thesis-rag
emoji: 📄
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
```

`app_port: 7860` matters — Spaces routes to 7860 by default.

Secrets (Space settings → Repository secrets, exposed as env vars):

- `ANTHROPIC_API_KEY`
- `HF_TOKEN` — write-scoped, for dataset sync
- `HF_DATASET_REPO` — `<user>/thesis-rag-data`
- `ACCESS_CODE` — the passphrase sent to my supervisor
- `LLM_MODEL`, `EMBED_MODEL`

### 11.3 Dockerfile (HF-specific gotchas)

Spaces run the container as a **non-root user with UID 1000**. Writing to `/app` or `/data` fails with a permission error, and the Space just shows "Runtime error" with little detail. Three rules:

1. Create and switch to `user` with UID 1000; put everything under `/home/user`.
2. Write all runtime state to `/home/user/data`, never to the image root.
3. Pre-download the embedding model **at build time** so the first request isn't a 100 MB download.

```dockerfile
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    HF_HOME=/home/user/.cache/huggingface \
    DATA_DIR=/home/user/data \
    PORT=7860

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# bake the embedding model into the image
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY --chown=user rag/ ./rag/
COPY --chown=user app/ ./app/
COPY --chown=user data/manifest.yaml ./data/manifest.yaml

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

`--workers 1`: the SQLite index is written by the upload flow, and a single worker avoids write contention. Free Spaces run one replica, so this costs nothing.

### 11.4 `rag/storage.py` — dataset sync

```python
def pull() -> None:
    """On boot: snapshot the private dataset into DATA_DIR.
    If the dataset is empty (first deploy), fall back to the index baked
    into the image, then push it up."""

def push_index() -> None:
    """Upload index.sqlite after a successful ingest."""

def push_pdf(local_path: Path, name: str) -> None:
    """Upload one PDF under uploads/."""
```

- Use `snapshot_download(repo_id, repo_type="dataset", token=HF_TOKEN, local_dir=DATA_DIR)`.
- Wrap writes in a module-level `asyncio.Lock`; push is called from the background ingest task, never from a request handler.
- `push_index` uploads a ~30–80 MB file; do it once per ingest job, not per chunk.
- On boot, log `chunks=N core=N user=N` so I can see in the Space logs whether the pull worked.

### 11.5 Startup sequence

```
pull() → open SQLite → verify meta.embed_model == EMBED_MODEL → warm the
embedding model with one dummy encode → serve
```

The dummy encode matters: ONNX session init takes a few seconds, and without it the first real question is noticeably slow.

`/health` returns `{ok, chunks_core, chunks_user, embed_model, built_at}`.

### 11.6 Keep-awake

`.github/workflows/ping.yml`:

```yaml
on:
  schedule:
    - cron: "0 2 * * *"   # daily, well inside the 48 h window
  workflow_dispatch:
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS https://<user>-thesis-rag.hf.space/health
```

Scheduled workflows on public repos are free. GitHub disables cron on repos with no activity for 60 days — `workflow_dispatch` lets me re-arm it manually.

**Accept:** the public URL answers gold question q27 correctly with citations; `/health` shows the expected chunk count; a Space restart preserves uploaded documents (proves the dataset sync works).

---

## Phase 13 — PDF upload

### 13.1 The rule that protects the evaluation

Uploaded documents live in a **separate collection** from the 19 core papers. Without this, the first upload silently changes retrieval for the gold set and the README numbers stop meaning anything.

Schema changes:

```sql
ALTER TABLE papers ADD COLUMN collection TEXT NOT NULL DEFAULT 'core';  -- 'core' | 'user'
ALTER TABLE chunks ADD COLUMN collection TEXT NOT NULL DEFAULT 'core';
CREATE INDEX idx_chunks_collection ON chunks(collection);

CREATE TABLE documents (
  paper_id TEXT PRIMARY KEY,      -- 'u01', 'u02', …
  collection TEXT NOT NULL,
  filename TEXT, title TEXT, n_pages INT, n_chunks INT,
  sha256 TEXT UNIQUE,             -- reject duplicate uploads
  uploaded_at TEXT, status TEXT   -- ready | failed | deleted
);

CREATE TABLE jobs (
  job_id TEXT PRIMARY KEY, paper_id TEXT, status TEXT,  -- queued|parsing|embedding|ready|failed
  stage_detail TEXT, error TEXT, created_at TEXT, finished_at TEXT
);
```

`sqlite-vec` has no metadata filter, so: run the vector KNN with a larger `k` (say 60), then filter by collection in SQL after the join. At this corpus size the cost is negligible. BM25 filters directly in the SQL `WHERE`.

`scripts/eval_retrieval.py` and `eval_answers.py` **must** pass `collections=["core"]` and assert it. Put that assertion in a test.

### 13.2 Validation (reject early, reject loudly)

In order, before anything expensive:

1. **Magic bytes** — file starts with `%PDF`. Do not trust the extension or the client-supplied MIME type.
2. **Size** ≤ 20 MB.
3. **Encrypted?** `doc.needs_pass` → reject: "This PDF is password-protected."
4. **Pages** ≤ 60.
5. **Text layer** — extract page 1 (and the midpoint page). If `len(fonts) == 0` or extracted alphabetic chars < 200 per page, reject: "This looks like a scanned PDF. The system reads text, not images, so it can't index this."
6. **Duplicate** — sha256 already in `documents` → return the existing `paper_id` instead of re-ingesting.

Rule 5 is the important one. A scanned PDF indexes to nothing, and the bot then confidently tells the user their own document doesn't discuss things it plainly does. That failure looks like a broken product and is invisible without the check.

### 13.3 Ingest, in the background

Parsing plus embedding takes 10–60 s. Never do it inside the request.

```
POST /api/upload  (multipart, header X-Access-Code)
  → validate → write PDF to DATA_DIR/uploads/<paper_id>.pdf
  → insert documents row (status=parsing), jobs row (status=queued)
  → schedule asyncio background task
  → 202 {job_id, paper_id}

background task:
  parse (rag.parse.pipeline, same code as core — no separate path)
  → chunk (collection='user')
  → embed → insert chunks + vec rows
  → update documents(status=ready, n_chunks), jobs(status=ready)
  → storage.push_index() + storage.push_pdf()
  on exception: jobs(status=failed, error=<short, user-safe message>)
                and roll back any chunks already inserted

GET  /api/jobs/{job_id}   → {status, stage_detail, error}
GET  /api/documents       → uploaded docs with status and chunk counts
DELETE /api/documents/{paper_id}  (access code required)
  → delete chunks, vec rows (by rowid), papers row, PDF; push_index()
```

Reuse the core parser unchanged. An uploaded paper that happens to be two-column should get the same treatment — and if the parser mangles it, I want to see that, not hide it behind a second code path.

### 13.4 Retrieval scope

`search(..., collections: list[str])`. UI toggle, three states:

- **Core corpus** (default) — the 19 thesis papers
- **My uploads**
- **Both**

Show the active scope above the answer, and include the collection in each source line so it's never ambiguous which corpus a citation came from.

### 13.5 Limits and honesty

- Max 20 uploaded documents total; oldest-first deletion prompt when full.
- Upload rate limit: 5/hour per access code.
- A persistent line near the upload control: **"Anything uploaded is visible to everyone with this link. Don't upload confidential documents, patient data, or work that isn't yours to share."**
- A "clear all uploads" button for me.

### 13.6 UI additions (keep the minimalism)

- A single hairline-bordered drop zone below the input, collapsed to one line of text until a file is dragged over.
- Job progress as plain text that updates in place: `parsing… → embedding 34/78 → ready`. No spinner, no progress bar.
- Uploaded documents listed in the About panel with a small `remove` link, same visual treatment as the paper list.

**Accept:** upload one of the 19 PDFs the system already has as a *user* document, ask a question scoped to "My uploads", and confirm it answers from the uploaded copy with `u01` citations; then restart the Space and confirm the document survives; then delete it and confirm the chunks are gone.

---

## Phase 14 — Sharing and access

### 14.1 Access code

- `ACCESS_CODE` env var, single shared value.
- Required on `POST /api/ask`, `POST /api/upload`, `DELETE /api/documents/*`. Not required on `/health` or `/api/papers`.
- Frontend: on first ask, show a single input ("Access code") inline above the question box; store in `sessionStorage`; send as `X-Access-Code`. Wrong code → 401 and a one-line message.
- Compare with `secrets.compare_digest`.
- Rate limits keyed on the code, not the IP: 10 questions/minute, 200/day. Over the daily cap → a friendly message saying the demo budget for the day is used up.

This keeps the link shareable while keeping spend bounded. A private Space would also work but requires the supervisor to have an HF account and be added as a collaborator — more friction than it's worth.

### 14.2 What to actually send

> **thesis-rag** — https://<user>-thesis-rag.hf.space
> Access code: `<code>`
>
> Ask questions about the 19 papers behind my thesis. It cites the paper, section and page for every claim, and says so when the corpus doesn't cover something. You can also drop in your own PDF and ask about that.
>
> Three questions to start with are on the page. Retrieval and answer accuracy are measured against a 55-question set — results are in the README.

### 14.3 Fallbacks, in order, if the link won't do

1. **A 60-second screen recording** plus the README with the evaluation table. This is the right fallback for a non-technical reader, and worth making regardless — it survives the Space being down and works in a WhatsApp message.
2. **Run it locally:** `docker run -p 7860:7860 -e ANTHROPIC_API_KEY=... ghcr.io/<user>/thesis-rag:latest`. One command, but assumes Docker and a key. Only for a technical reader.
3. **Not viable:** a self-contained file with no backend. Retrieval needs the embedding model and the index, and answering needs an API key. Anything that ships the key to the user's machine is worse than the hosted version, not better.

**Accept:** I open the link in a private browser window on my phone, enter the code, ask a question, upload a PDF, and ask about it — without touching a terminal.

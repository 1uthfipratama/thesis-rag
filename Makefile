# All targets run inside the uv-managed Python 3.11 venv.
PY := uv run python
# Windows consoles default to cp1252; author names (Lyócsa, Özdemir) need UTF-8.
export PYTHONUTF8 := 1

.PHONY: demo setup manifest inspect parse compare-parsers setup-mineru setup-marker chunks index eval-retrieval eval-answers serve test lint docker deploy

setup:
	uv sync

manifest:
	$(PY) scripts/validate_manifest.py

parse:  ## BACKEND=pymupdf|mineru|marker|hybrid (default from PARSE_BACKEND, else pymupdf)
	$(PY) scripts/parse_all.py $(if $(BACKEND),--backend $(BACKEND))

compare-parsers:
	$(PY) scripts/compare_parsers.py

# Optional ML parser backends live in isolated venvs (heavy deps, separate licences).
setup-mineru:
	uv venv .venvs/mineru --python 3.11
	VIRTUAL_ENV=.venvs/mineru uv pip install "mineru>=4.0,<5"
	.venvs/mineru/Scripts/mineru-kit models download --tier basic --small-backend onnx

setup-marker:  ## also needs llama.cpp's llama-server (winget install ggml.llamacpp)
	uv venv .venvs/marker --python 3.11
	VIRTUAL_ENV=.venvs/marker uv pip install marker-pdf

inspect:  ## make inspect ID=p10
	$(PY) scripts/inspect_parse.py $(ID)

chunks:
	$(PY) scripts/build_chunks.py

index:
	$(PY) scripts/build_index.py

eval-retrieval:  ## GOLD=eval/gold_paraphrased.jsonl for the paraphrase set
	$(PY) scripts/eval_retrieval.py $(if $(GOLD),--gold $(GOLD))

eval-answers:  ## ARGS="--dry-run" or ARGS="--models claude-haiku-4-5 --max-usd 1"
	$(PY) scripts/eval_answers.py $(ARGS)

serve:
	uv run uvicorn app.main:app --reload --port 8000

demo:  ## $0: canned answers from real passages, no Claude calls
	uv run --env-file .env.demo uvicorn app.main:app --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

docker:
	docker build -t thesis-rag .

deploy:
	@echo "Phase 11: not implemented yet"

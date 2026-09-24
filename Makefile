# All targets run inside the uv-managed Python 3.11 venv.
PY := uv run python
# Windows consoles default to cp1252; author names (Lyócsa, Özdemir) need UTF-8.
export PYTHONUTF8 := 1

.PHONY: setup manifest inspect parse chunks index eval-retrieval eval-answers serve test lint docker deploy

setup:
	uv sync

manifest:
	$(PY) scripts/validate_manifest.py

parse:
	$(PY) scripts/parse_all.py

inspect:  ## make inspect ID=p10
	$(PY) scripts/inspect_parse.py $(ID)

chunks:
	$(PY) scripts/build_chunks.py

index:
	$(PY) scripts/build_index.py

eval-retrieval:
	$(PY) scripts/eval_retrieval.py

eval-answers:
	$(PY) scripts/eval_answers.py

serve:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

docker:
	docker build -t thesis-rag .

deploy:
	@echo "Phase 11: not implemented yet"

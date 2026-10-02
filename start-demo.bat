@echo off
rem thesis-rag demo mode: real search, canned answers, no Claude calls, $0.
rem Double-click, then open http://localhost:8010 . Close this window to stop.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist data\index.sqlite (
  echo data\index.sqlite is missing. Build it first: make parse, make chunks, make index.
  pause
  exit /b 1
)
start "" http://localhost:8010
uv run --env-file .env.demo uvicorn app.main:app --port 8010
pause

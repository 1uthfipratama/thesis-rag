@echo off
rem thesis-rag, real answers from Claude Haiku (needs ANTHROPIC_API_KEY in .env).
rem Double-click, then open http://localhost:8010 . Close this window to stop.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .env (
  echo No .env file. Copy .env.example to .env and put your ANTHROPIC_API_KEY in it,
  echo or run start-demo.bat for the free demo mode.
  pause
  exit /b 1
)
if not exist data\index.sqlite (
  echo data\index.sqlite is missing. Build it first: make parse, make chunks, make index.
  pause
  exit /b 1
)
start "" http://localhost:8010
uv run uvicorn app.main:app --port 8010
pause

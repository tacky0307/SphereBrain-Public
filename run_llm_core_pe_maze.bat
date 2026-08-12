@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain P-to-E maze puzzle
start "" http://127.0.0.1:5084
python experiments\run_llm_core_pe_maze.py

if errorlevel 1 (
  echo.
  echo Puzzle failed. Check the message above.
)

echo.
pause

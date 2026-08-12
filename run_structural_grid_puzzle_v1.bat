@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  echo Python was not found.
  echo Install Python or create .venv, then run this file again.
  pause
  exit /b 1
)

echo Starting Structural Grid Puzzle v2...
%PYTHON_CMD% experiments\run_structural_grid_puzzle_v2.py

if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  echo Copy the error text shown above and send it to ChatGPT.
  pause
)

endlocal

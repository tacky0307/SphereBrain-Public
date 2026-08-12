@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if not defined PYTHON_CMD (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  echo Python was not found.
  pause
  exit /b 1
)

echo Starting Core Growth Binding v12...
%PYTHON_CMD% experiments\run_core_growth_binding_v12.py
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

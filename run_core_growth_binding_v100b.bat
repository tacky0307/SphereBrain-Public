@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>nul && set "PYTHON_CMD=py"
if not defined PYTHON_CMD where python >nul 2>nul && set "PYTHON_CMD=python"
if not defined PYTHON_CMD where python3 >nul 2>nul && set "PYTHON_CMD=python3"

if not defined PYTHON_CMD (
  echo Python was not found. Install Python or add it to PATH.
  pause
  exit /b 1
)

echo Starting Core Growth Binding v100B...
%PYTHON_CMD% experiments\run_core_growth_binding_v100b.py

if errorlevel 1 (
  echo.
  echo v100B stopped with an error.
  pause
)

endlocal

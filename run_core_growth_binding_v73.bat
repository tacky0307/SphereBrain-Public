@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v73...

set "PYTHON_CMD="
where py >nul 2>nul && set "PYTHON_CMD=py"
if not defined PYTHON_CMD (
  where python >nul 2>nul && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  where python3 >nul 2>nul && set "PYTHON_CMD=python3"
)

if not defined PYTHON_CMD (
  echo Python was not found in PATH.
  echo Please install Python or add it to PATH.
  echo.
  pause
  exit /b 1
)

%PYTHON_CMD% experiments\run_core_growth_binding_v73.py
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

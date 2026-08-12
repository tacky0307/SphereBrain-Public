@echo off
setlocal
cd /d "%~dp0"
echo Starting Core Growth Binding v80...

set "PYTHON_CMD="
where py >nul 2>&1
if %errorlevel%==0 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
  where python >nul 2>&1
  if %errorlevel%==0 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  where python3 >nul 2>&1
  if %errorlevel%==0 set "PYTHON_CMD=python3"
)
if not defined PYTHON_CMD (
  echo Python was not found.
  pause
  exit /b 1
)

%PYTHON_CMD% experiments\run_core_growth_binding_v80.py
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

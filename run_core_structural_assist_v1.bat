@echo off
setlocal
cd /d %~dp0
python experiments\run_core_structural_assist_v1.py
if errorlevel 1 (
  echo.
  echo Core Structural Assist v1 failed.
  pause
  exit /b 1
)
echo.
pause

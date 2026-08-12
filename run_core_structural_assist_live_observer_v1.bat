@echo off
setlocal
cd /d "%~dp0"
python experiments\run_core_structural_assist_live_observer_v1.py
if errorlevel 1 (
  echo.
  echo Live Observer failed.
  pause
  exit /b 1
)
echo.
echo Live Observer finished. The HTML report should open in your browser.
pause

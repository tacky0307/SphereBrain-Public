@echo off
setlocal
cd /d %~dp0
python experiments\run_structural_working_state_v2.py
if errorlevel 1 (
  echo.
  echo Experiment failed.
  pause
  exit /b 1
)
echo.
echo Experiment completed.
pause

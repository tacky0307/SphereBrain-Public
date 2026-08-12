@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain baseline route change experiment
python experiments\run_llm_core_baseline_route_change.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

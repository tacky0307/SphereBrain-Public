@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain causal edge importance experiment
python experiments\run_llm_core_causal_edge_importance.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

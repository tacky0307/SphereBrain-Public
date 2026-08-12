@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain LLM-Core route stability experiment
python experiments\run_llm_core_route_stability.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

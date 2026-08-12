@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain LLM-Core route lifecycle experiment
python experiments\run_llm_core_route_lifecycle.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain LLM-Core staged comparison
python experiments\run_llm_core_stage_comparison.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

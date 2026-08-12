@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain LLM-Core repeated trials
python experiments\run_llm_core_repeated_trials.py --trials 5

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain functional pattern reproducibility
python experiments\run_llm_core_function_pattern_reproducibility.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

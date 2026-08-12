@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain sequential concept reorganization
python experiments\run_llm_core_sequential_concept_reorganization.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

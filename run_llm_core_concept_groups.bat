@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain multi-experience concept group experiment
python experiments\run_llm_core_concept_groups.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

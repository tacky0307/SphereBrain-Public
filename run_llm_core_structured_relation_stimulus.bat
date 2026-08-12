@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain structured relation stimulus experiment
python experiments\run_llm_core_structured_relation_stimulus.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
)

echo.
pause

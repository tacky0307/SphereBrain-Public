@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain minimal transitive relation experiment
python experiments\run_llm_core_transitive_relation.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
)

echo.
pause

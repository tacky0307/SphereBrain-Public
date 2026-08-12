@echo off
setlocal
cd /d "%~dp0"
echo SphereBrain minimal two-stage chain propagation
python experiments\run_llm_core_two_stage_chain_propagation.py
if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)
pause

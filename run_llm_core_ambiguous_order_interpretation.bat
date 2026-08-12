@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain ambiguous interpretation order-effect experiment
python experiments\run_llm_core_ambiguous_order_interpretation.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

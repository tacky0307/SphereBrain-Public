@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain experience order effect experiment
python experiments\run_llm_core_experience_order_effect.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

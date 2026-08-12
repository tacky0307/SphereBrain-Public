@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain graded ablation and recovery experiment
python experiments\run_llm_core_graded_ablation_recovery.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

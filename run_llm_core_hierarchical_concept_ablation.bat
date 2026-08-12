@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain hierarchical concept ablation
python experiments\run_llm_core_hierarchical_concept_ablation.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

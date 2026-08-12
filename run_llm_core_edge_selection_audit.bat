@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain edge selection and ablation audit
python experiments\run_llm_core_edge_selection_audit.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

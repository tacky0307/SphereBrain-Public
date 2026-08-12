@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain adaptive soft cluster discovery
python experiments\run_llm_core_adaptive_soft_clusters.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain cluster hierarchy and stability
python experiments\run_llm_core_cluster_hierarchy_stability.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

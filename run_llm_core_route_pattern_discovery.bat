@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain route pattern discovery
python experiments\run_llm_core_route_pattern_discovery.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain LLM-Core route role differentiation
python experiments\run_llm_core_route_roles.py

if errorlevel 1 (
  echo.
  echo Experiment failed. Check the message above.
) else (
  echo.
  echo Experiment completed successfully.
)

echo.
pause

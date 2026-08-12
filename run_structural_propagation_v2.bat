@echo off
setlocal
cd /d %~dp0
python experiments\run_structural_propagation_v2.py
if errorlevel 1 (
  echo.
  echo Structural Propagation v2 failed.
  pause
  exit /b 1
)
echo.
echo Structural Propagation v2 completed.
pause

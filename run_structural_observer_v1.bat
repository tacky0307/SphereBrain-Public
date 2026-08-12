@echo off
setlocal
cd /d %~dp0
python experiments\run_structural_observer_v1.py
if errorlevel 1 (
  echo.
  echo Structural Observer v1 failed.
  pause
  exit /b 1
)
echo.
echo Structural Observer v1 completed.
pause

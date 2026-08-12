@echo off
setlocal
cd /d %~dp0
python experiments\run_core_integration_shadow_v1.py
if errorlevel 1 (
  echo.
  echo Core Integration Shadow v1 failed.
  pause
  exit /b 1
)
echo.
echo Core Integration Shadow v1 completed.
pause

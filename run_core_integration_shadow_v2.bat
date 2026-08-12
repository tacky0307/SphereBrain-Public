@echo off
setlocal
cd /d %~dp0
python experiments\run_core_integration_shadow_v2.py
if errorlevel 1 (
  echo.
  echo Core Integration Shadow v2 failed.
  pause
  exit /b 1
)
echo.
echo Core Integration Shadow v2 completed.
pause

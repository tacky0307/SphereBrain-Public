@echo off
setlocal
cd /d %~dp0
python experiments\run_core_integration_shadow_v3.py
if errorlevel 1 (
  echo.
  echo Core Integration Shadow v3 failed.
  pause
  exit /b 1
)
echo.
echo Core Integration Shadow v3 completed.
pause

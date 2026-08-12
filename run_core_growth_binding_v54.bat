@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v54...
where python >nul 2>nul
if %errorlevel%==0 (
  python experiments\run_core_growth_binding_v54.py
  goto :after
)
where python3.12 >nul 2>nul
if %errorlevel%==0 (
  python3.12 experiments\run_core_growth_binding_v54.py
  goto :after
)

echo Python executable was not found.
exit /b 1

:after
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)

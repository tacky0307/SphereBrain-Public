@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v77...
where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 experiments\run_core_growth_binding_v77.py
) else (
  python experiments\run_core_growth_binding_v77.py
)

if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

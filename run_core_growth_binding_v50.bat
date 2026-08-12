@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v50...
py experiments\run_core_growth_binding_v50.py
if errorlevel 1 (
  python experiments\run_core_growth_binding_v50.py
)

if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)

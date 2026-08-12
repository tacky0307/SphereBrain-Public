@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v54B...

where python >nul 2>&1
if not errorlevel 1 (
  python experiments\run_core_growth_binding_v54b.py
  goto :done
)

where python3.12 >nul 2>&1
if not errorlevel 1 (
  python3.12 experiments\run_core_growth_binding_v54b.py
  goto :done
)

echo Python was not found. Tried: python, python3.12
exit /b 1

:done
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)

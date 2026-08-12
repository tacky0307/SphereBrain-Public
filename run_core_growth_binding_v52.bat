@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v52...
where python >nul 2>nul
if %errorlevel%==0 (
  python experiments\run_core_growth_binding_v52.py
  goto :done
)
where python3.12 >nul 2>nul
if %errorlevel%==0 (
  python3.12 experiments\run_core_growth_binding_v52.py
  goto :done
)
echo Python was not found.
exit /b 1

:done
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)

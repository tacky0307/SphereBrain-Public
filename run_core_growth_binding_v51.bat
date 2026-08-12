@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v51...

where python >nul 2>&1
if not errorlevel 1 (
  python experiments\run_core_growth_binding_v51.py
  goto :after_run
)

where python3.12 >nul 2>&1
if not errorlevel 1 (
  python3.12 experiments\run_core_growth_binding_v51.py
  goto :after_run
)

echo.
echo Python executable was not found.
echo Tried: python, python3.12
goto :error

:after_run
if errorlevel 1 goto :error
goto :end

:error
echo.
echo The program stopped with an error.
pause

:end
endlocal

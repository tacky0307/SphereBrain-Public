@echo off
setlocal
cd /d "%~dp0"
echo Starting Core Growth Binding v27...

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "experiments\run_core_growth_binding_v27.py"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Python was not found.
    echo Please create .venv or install Python and add it to PATH.
    pause
    exit /b 1
  )
  python "experiments\run_core_growth_binding_v27.py"
)

if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

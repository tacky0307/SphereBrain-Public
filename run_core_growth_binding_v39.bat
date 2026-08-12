@echo off
setlocal
cd /d "%~dp0"
echo Starting Core Growth Binding v39...
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe experiments\run_core_growth_binding_v39.py
) else (
  python experiments\run_core_growth_binding_v39.py
)
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

@echo off
setlocal
cd /d "%~dp0"
echo Starting Core Growth Binding v29...
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe experiments\run_core_growth_binding_v29.py
) else (
  python experiments\run_core_growth_binding_v29.py
)
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

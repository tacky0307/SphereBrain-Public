@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo Starting Core Growth Binding v25...
%PYTHON% experiments\run_core_growth_binding_v25.py
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  echo Copy the error text shown above and send it to ChatGPT.
  pause
)
endlocal

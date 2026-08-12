@echo off
setlocal
cd /d "%~dp0"
set "PY=py"
where py >nul 2>nul || set "PY=python"
echo Starting Core Growth Binding v67...
%PY% experiments\run_core_growth_binding_v67.py
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
endlocal

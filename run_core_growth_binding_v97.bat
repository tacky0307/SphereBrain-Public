@echo off
setlocal
cd /d "%~dp0"

set PYEXE=
where py >nul 2>nul && set PYEXE=py
if not defined PYEXE where python >nul 2>nul && set PYEXE=python
if not defined PYEXE where python3 >nul 2>nul && set PYEXE=python3

if not defined PYEXE (
  echo Python が見つかりません。
  pause
  exit /b 1
)

echo Starting Core Growth Binding v97...
%PYEXE% experiments\run_core_growth_binding_v97.py

if errorlevel 1 (
  echo.
  echo v97 ended with an error.
  pause
)
endlocal

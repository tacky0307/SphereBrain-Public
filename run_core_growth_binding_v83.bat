@echo off
setlocal
cd /d "%~dp0"
echo Starting Core Growth Binding v83...
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE where python >nul 2>nul && set "PYEXE=python"
if not defined PYEXE where python3 >nul 2>nul && set "PYEXE=python3"
if not defined PYEXE (
 echo Python was not found.
 pause
 exit /b 1
)
%PYEXE% experiments\run_core_growth_binding_v83.py
if not %errorlevel%==0 (
 echo.
 echo The program stopped with an error.
 pause
)
endlocal

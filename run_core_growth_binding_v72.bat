@echo off
setlocal
cd /d %~dp0

echo Starting Core Growth Binding v72...
where py >nul 2>nul
if %errorlevel%==0 (
    py experiments\run_core_growth_binding_v72.py
) else (
    python experiments\run_core_growth_binding_v72.py
)

if errorlevel 1 (
    echo.
    echo The program stopped with an error.
    pause
)
endlocal

@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v23...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" experiments\run_core_growth_binding_v23.py
) else (
    python experiments\run_core_growth_binding_v23.py
)

if errorlevel 1 (
    echo.
    echo The program stopped with an error.
    echo Copy the error text shown above and send it to ChatGPT.
)

pause

@echo off
setlocal
cd /d "%~dp0"
python experiments\run_core_state_observer_v1.py
echo.
pause

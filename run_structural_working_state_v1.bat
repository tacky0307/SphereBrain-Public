@echo off
setlocal
cd /d %~dp0
python experiments\run_structural_working_state_v1.py
echo.
pause

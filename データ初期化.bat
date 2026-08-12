@echo off
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 reset_project_data.py
) else (
    python reset_project_data.py
)

set EXIT_CODE=%errorlevel%
echo.
pause
exit /b %EXIT_CODE%

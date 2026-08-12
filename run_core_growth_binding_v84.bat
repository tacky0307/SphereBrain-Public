@echo off
setlocal
cd /d %~dp0

echo Starting Core Growth Binding v84...

where py >nul 2>nul
if %errorlevel%==0 (
    py experiments\run_core_growth_binding_v84.py
    goto :done
)

where python >nul 2>nul
if %errorlevel%==0 (
    python experiments\run_core_growth_binding_v84.py
    goto :done
)

where python3 >nul 2>nul
if %errorlevel%==0 (
    python3 experiments\run_core_growth_binding_v84.py
    goto :done
)

echo Python was not found.
exit /b 1

:done
if not %errorlevel%==0 (
    echo.
    echo The program stopped with an error.
    pause
)
endlocal

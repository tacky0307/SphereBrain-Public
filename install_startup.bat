@echo off
cd /d "%~dp0"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "TARGET=%STARTUP%\SphereBrain_v0_3.cmd"

(
echo @echo off
echo cd /d "%~dp0"
echo start "" /min "%~dp0run_windows.bat"
) > "%TARGET%"

echo Startup shortcut script created:
echo %TARGET%
echo.
echo Sphere Brain will start after your next Windows sign-in.
pause

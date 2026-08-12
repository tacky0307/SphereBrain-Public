@echo off
set "TARGET=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SphereBrain_v0_3.cmd"
if exist "%TARGET%" del "%TARGET%"
echo Startup entry removed.
pause

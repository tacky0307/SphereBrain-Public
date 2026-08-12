@echo off
cd /d "%~dp0"
title Sphere Brain v0.3

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  python -m venv .venv
  if errorlevel 1 goto ERROR
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 goto ERROR
) else (
  call ".venv\Scripts\activate.bat"
)

echo Starting Sphere Brain...
python app.py
if errorlevel 1 goto ERROR
exit /b 0

:ERROR
echo.
echo An error occurred. Please copy or photograph the message above.
pause
exit /b 1

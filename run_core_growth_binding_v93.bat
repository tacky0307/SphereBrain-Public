@echo off
setlocal
cd /d %~dp0

where py >nul 2>nul
if %errorlevel%==0 (
  py experiments\run_core_growth_binding_v93.py
  goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
  python experiments\run_core_growth_binding_v93.py
  goto :eof
)

where python3 >nul 2>nul
if %errorlevel%==0 (
  python3 experiments\run_core_growth_binding_v93.py
  goto :eof
)

echo Python が見つかりません。
pause

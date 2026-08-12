@echo off
setlocal
cd /d %~dp0
where py >nul 2>nul && (py experiments\run_core_growth_binding_v92b.py & goto :eof)
where python >nul 2>nul && (python experiments\run_core_growth_binding_v92b.py & goto :eof)
where python3 >nul 2>nul && (python3 experiments\run_core_growth_binding_v92b.py & goto :eof)
echo Python was not found.
pause

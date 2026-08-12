@echo off
cd /d "%~dp0"
py experiments\run_core_growth_binding_v95b.py
if errorlevel 1 python experiments\run_core_growth_binding_v95b.py

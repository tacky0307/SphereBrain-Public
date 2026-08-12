@echo off
cd /d %~dp0
python branch_observer.py
if errorlevel 1 pause

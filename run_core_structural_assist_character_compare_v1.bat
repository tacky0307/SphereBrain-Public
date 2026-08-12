@echo off
setlocal
cd /d "%~dp0"
python experiments\run_core_structural_assist_character_compare_v1.py
if errorlevel 1 (
  echo.
  echo 実行中にエラーが発生しました。
)
pause

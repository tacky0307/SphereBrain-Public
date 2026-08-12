@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Sphere Brain の現在状態をCSVへ出力します。
echo.

python export_experiment_snapshot.py

echo.
echo Enterキーを押すと閉じます。
pause > nul

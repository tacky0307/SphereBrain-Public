@echo off
setlocal
cd /d "%~dp0"

echo SphereBrain Core literal vs LLM expressive OUT demo
start "" http://127.0.0.1:5082
python experiments\run_llm_core_in_out_dual_demo.py

if errorlevel 1 (
  echo.
  echo Demo failed. Check the message above.
)

echo.
pause

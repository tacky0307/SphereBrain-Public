@echo off
setlocal
cd /d %~dp0

if "%OPENAI_API_KEY%"=="" (
  echo.
  echo [ERROR] OPENAI_API_KEY is not set.
  echo Set the environment variable before starting this experiment.
  echo Existing SphereBrain experiments are not affected.
  echo.
  pause
  exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [ERROR] Dependency installation failed.
  pause
  exit /b 1
)

start "SphereBrain LLM-Core-LLM" http://127.0.0.1:5078
python llm_core_lab.py

pause

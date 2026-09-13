@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=..\dollar_euro_16tile\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo error: no usable project .venv found. Create one with:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%PYTHON_EXE%" -u "scripts\debug\run_diagnostic_experiment.py" %*

exit /b %ERRORLEVEL%

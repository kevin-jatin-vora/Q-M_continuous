@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=..\dollar_euro_16tile\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo error: no usable project .venv found.
  exit /b 1
)

"%PYTHON_EXE%" -u "scripts\run_experiment_qbounds.py" %*

exit /b %ERRORLEVEL%

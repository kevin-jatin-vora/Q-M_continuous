@echo off
setlocal
cd /d "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
  echo error: project .venv not found.
  exit /b 1
)
".venv\Scripts\python.exe" -u "scripts\run_q_bound_experiment.py" %*
exit /b %ERRORLEVEL%

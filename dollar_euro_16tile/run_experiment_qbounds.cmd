@echo off
setlocal
cd /d "%~dp0"

".venv\Scripts\python.exe" -u "scripts\run_experiment_qbounds.py" %*

exit /b %ERRORLEVEL%
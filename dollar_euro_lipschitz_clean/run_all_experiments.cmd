@echo off
setlocal EnableExtensions
cd /d "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo error: project .venv not found.
  exit /b 1
)

echo Running the full sigma sweep with empirical RA-DQN enabled.
echo Base sigma=0.00008, scaled by 1,2,3,4,5:
echo   0.00008  0.00016  0.00024  0.00032  0.00040
echo Each scale regenerates R1/R2, radii, center Q, heatmaps/CSVs,
echo then DQN, RA-DQN, and plots.
echo.

rem Hardcoded 0.00008 * {1,2,3,4,5}. cmd.exe cannot reliably quote
rem python -c inside for /f.
for %%S in (0.00008 0.00016 0.00024 0.00032 0.00040) do (
  echo.
  echo ===== sigma=%%S =====
  call run_experiment.cmd configs\default.json --sigma %%S --profile full
  if errorlevel 1 (
    echo error: experiment failed at sigma=%%S
    exit /b 1
  )
)

echo.
echo All experiments complete.
exit /b 0

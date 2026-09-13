@echo off
setlocal EnableExtensions
cd /d "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo error: project .venv not found.
  exit /b 1
)

rem Three full experiments in separate terminals:
rem   det 0 / 0.20 / 0.40, sigma=0.002, radial_match, no empirical RA.
rem Each terminal still parallelizes independent seed runs via --parallel-runs.
rem Workers per terminal ~= logical CPUs / 3 so the three jobs do not thrash.

set "CFG=configs\radial_match.json"
set "SIGMA=0.002"
set "PROFILE=full"
set "SEED=0"

set "WORKERS=1"
for /f %%N in ('powershell -NoProfile -Command "[Math]::Max(1, [int]([Environment]::ProcessorCount / 3))"') do set "WORKERS=%%N"

echo Launching 3 experiment terminals (det 0, 0.20, 0.40^)
echo   config:  %CFG%
echo   sigma:   %SIGMA%
echo   profile: %PROFILE%  seed=%SEED%  --no-empirical
echo   parallel seed workers per terminal: %WORKERS%  (CPU/3^)
echo.

for %%D in (0 0.2 0.4) do (
  echo   start  det=%%D
  start "DE sigma=%SIGMA% det=%%D" cmd /v:on /k "cd /d "%~dp0" && title DE sigma=%SIGMA% det=%%D && call run_experiment.cmd %CFG% --sigma %SIGMA% --determinism %%D --profile %PROFILE% --seed %SEED% --no-empirical --parallel-runs %WORKERS% & echo. & echo ===== Finished det=%%D  exitcode=^!ERRORLEVEL^! ===== & pause"
)

echo.
echo All three terminals started. Close each window when its run finishes.
exit /b 0

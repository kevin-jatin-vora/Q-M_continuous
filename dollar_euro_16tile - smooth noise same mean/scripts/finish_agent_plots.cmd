@echo off
setlocal
cd /d "%~dp0\.." || exit /b 1

set EXP40=outputs\experiments\sigma_2e-03_det40_cfg_radial_match_8337d5775344_full_seed0
set EXP00=outputs\experiments\sigma_2e-03_det00_cfg_radial_match_1190df34e5c2_full_seed0
set PY=.venv\Scripts\python.exe

call :finish_one %EXP40% det40
call :finish_one %EXP00% det00
echo.
echo Done: returns plots + policy videos for both experiments.
exit /b 0

:finish_one
set EXP=%~1
set TAG=%~2
if not exist "%EXP%\models\dqn_seed0.npy" (
  echo error: missing %EXP%\models\dqn_seed0.npy — finish DQN training first.
  exit /b 1
)
if not exist "%EXP%\models\ra_dqn_theoretical_seed0.npy" (
  echo error: missing %EXP%\models\ra_dqn_theoretical_seed0.npy — finish RA-DQN training first.
  exit /b 1
)
echo === %TAG%: returns plot ===
"%PY%" scripts\plot_returns.py "%EXP%\models\dqn_seed0.npy" "%EXP%\models\ra_dqn_theoretical_seed0.npy" --labels DQN RA-DQN-Theoretical --step 20000 --out "%EXP%\plots\returns_plot.png"
echo === %TAG%: policy videos ===
"%PY%" scripts\record_agent_policy_videos.py --experiment-dir "%EXP%" --seed 0 --skip-empirical
exit /b 0

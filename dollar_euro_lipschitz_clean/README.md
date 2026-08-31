# Dollar-Euro Lipschitz Pruning

Cleaned project structure for the continuous Dollar-Euro reinforcement learning experiments.

## What This Project Does

The environment is a continuous 2D Gymnasium task with state `(x, y)` in `[0, 1]^2`.
The square is split into four regions. Each region has its own terrain dynamics.
The agent receives a two-component reward vector, then most experiments train on `R1 + R2`.

The pruning method trains a center-model Q function, builds Lipschitz intervals around each action value, and removes actions whose optimistic upper bound is below another action's pessimistic lower bound.

## Original theoretical recipe, generic in sigma

The source folder `D:\Region based Lq from sources` is a single-sigma snapshot: the env default is `0.00008`, bounds and Lipschitz constants are frozen JSON, and `QM_using json_LQ_theoritical.py` trains five independent RA-DQN seeds against those files. This project keeps that theoretical procedure and makes every data-derived input a function of the runtime sigma:

| Source step | What we generalize |
|---|---|
| `ContinuousDollarEuroEnv(sigma=0.00008)` | `--sigma` / `configs/default.json` for every stage |
| `two_reward_q_learning_fixed_action_lipschitz.py` | Retrain R1/R2 at that sigma (`max_t=200`, `eps_decay=0.998`) |
| `reformatting json.py` mean-CI `radius_scalar` | Recompute the pooled Student-t mean CI; that is `radius_scalar` |
| Unique pairwise trimmed `Lr`, `Lf`, empirical `Lq` | Recompute unique unordered pairs at that sigma |
| Theoretical `Lq = Lr / (1 - γ Lf)` | Recompute per region-action; the heatmap uses these values, not a global `Lq=70` |
| Frozen `Q_single_region.pth` | Retrain center-Q on the new mean deltas |
| `runs = 5` in theoretical `main` | `n_runs=5` (smoke: 2) for DQN and both RA-DQN variants; plot mean with a shaded 95% Student-t CI |
| Margin `(Lr + γ Lq) r / (1-γ)` | Unchanged |
| Clip Q intervals to `[-100, 100]` | **Not used** |

The original heatmap script (`new pruning heatmap visualization .py`) is not reproduced: it used global `Lr=1.399`, `Lq=70`. Here the heatmap is the theoretical RA-DQN map with the sigma-specific `Lr`, `Lf`, and Bellman `Lq`, and `vmin=0`.

Predictive residual quantiles are still written to `transition_bounds.json` for diagnostics. They are not the pruning radius.

## Structure

| Path | Purpose |
|---|---|
| `src/dollar_euro_lipschitz/` | Shared reusable code |
| `scripts/` | Runnable experiment entry points |
| `data/` | Provided bounds/constants |
| `outputs/` | Generated models, return arrays, and plots |
| `docs/` | Project writeup |
| `original_uploads/` | Untouched uploaded source files |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Windows, run the training scripts only after activating `.venv`. Using
the base Anaconda interpreter may load conflicting NumPy and PyTorch Intel
OpenMP DLLs and stop with `OMP: Error #15`. Setting
`KMP_DUPLICATE_LIB_OK=TRUE` is not a safe fix.

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Quick Check

Preview the resolved config without training:

```bat
run_experiment.cmd configs\default.json --sigma 0.00004 --profile smoke --dry-run
```

## Main Workflow

Prefer the config-driven runner. `--sigma` overrides `configs/default.json`
for every stage. Training hyperparameters (`gamma`, `learning_rate`,
`batch_size`, `buffer_size`, `target_tau`, `update_every`, epsilon schedule,
and `max_steps_per_episode`) come from the same JSON unless a script flag is
passed explicitly.

```bash
python scripts/train_q_single.py
python scripts/train_baseline_dqn.py
python scripts/train_ra_dqn.py --lq-source theoretical
python scripts/plot_returns.py outputs/dqn.npy outputs/ra_dqn_theoretical.npy --labels DQN RA-DQN-Theoretical
```

Those standalone commands use `configs/default.json`. They will refuse the
bundled `data/*.json` unless you pass `--sigma 0.00008` or regenerate bounds
at the runtime sigma. The runner below regenerates bounds first.

## Complete experiment from CMD

From `cmd.exe`, this now calls the project `.venv` Python directly. `--sigma`
overrides `configs/default.json` for data collection, radii, training, heatmaps,
and plots. Empirical RA-DQN is included by default.

```bat
cd /d D:\dollar_euro_lipschitz_clean\dollar_euro_lipschitz_clean
run_experiment.cmd configs\default.json --sigma 0.00004 --profile full
```

Preview the resolved sigma without training:

```bat
run_experiment.cmd configs\default.json --sigma 0.00004 --profile smoke --dry-run
```

Skip empirical RA-DQN only if you explicitly want the theoretical comparison alone:

```bat
run_experiment.cmd configs\default.json --sigma 0.00004 --profile full --no-empirical
```


Changing `sigma` requires more than retraining the final agents. The R1 and R2
names refer to two separately trained component-reward DQNs. In the supplied
source, their separate epsilon-greedy trajectories also supplied the labeled
transition samples used for region-action statistics and empirical pairwise
constants. Therefore all of those data products must be regenerated.

Run a short end-to-end check first:

```bat
run_experiment.cmd configs\default.json --sigma 0.00004 --profile smoke
```

Then run the research profile, or the full sigma sweep (`0.00008 × 1..5`):

```bat
run_experiment.cmd configs\default.json --sigma 0.00008 --profile full
run_all_experiments.cmd
```

Each run writes under `outputs/experiments/sigma_<tag>_cfg_<stem>_<hash>_<profile>_seed<seed>/`.
Pruning follows the source folder: theoretical RA-DQN uses Bellman `Lq`, empirical RA-DQN uses
empirical `Lq` with the same Bellman margin \((L_r+\gamma L_q)r/(1-\gamma)\), and both clamp Q intervals to `[-100, 100]`.
Heatmaps are that same `allowed()` mask, written after center Q.

To reproduce the original `sigma=0.00008` JSON snapshot (skip R1/R2 collection):

```bat
run_experiment.cmd configs\default.json --sigma 0.00008 --profile full --from-source-json --runs 1
```
The full profile trains `n_runs` (default 5) independent DQN / RA-DQN seeds and writes
`*_runs.npy` arrays of shape `(n_runs, n_evals)`. `scripts/plot_returns.py` plots
the mean with a shaded 95% Student-t CI (`stats.sem` × `t.ppf`, same formula and
`fill_between(..., alpha=0.25)` style as `CI_DE_RT_plots5.py`). Bounds and Lipschitz inputs
with mismatched sigma fail before pruning.

The generated transition table reports the Student-t confidence interval for the
pooled cell-mean Δs. Its L2 norm is `radius_scalar`, the radius used by original
theoretical RA-DQN pruning.

**Note (boundary handling vs Δ estimates):** Mean Δ, radius, and pairwise
`Lr` / `Lf` / empirical `Lq` are estimated only from transitions that were **not**
altered by environment boundary cancel or post-noise clip. That keeps `delta_s`
as the free region–action step (so deterministic regions can have std ≈ 0).
Boundary constraints are still applied at **use** time: center-Q steps with
`clip(s + μ)`, and RA-DQN / live rollouts use the environment’s normal clipping.
RL behavior training for R1/R2 still learns from all transitions, including
boundary-affected ones.

`Lr`, `Lf`, and empirical `Lq` are recomputed from unique same-action pairs
with a 10% trimmed mean on those free transitions.
Theoretical RA-DQN uses `Lr_sum / (1 - gamma * Lf_sum)` from those
sigma-specific values. If `gamma * Lf >= 1`, Bellman `Lq` is undefined and the
pipeline aborts (typically a sign that sigma / estimated `Lf` is too large).
Because `Lr` and `Lf` are data-derived, this is not an analytic theorem for the
environment. Terrain drift vectors remain stored but unused in `step()`, matching
the original environment.

Generate the publication heatmap and Tables 1-2 without retraining:

```bash
python scripts/generate_analysis.py --lq-source theoretical --sigma 0.00008
```

This writes PNG/PDF heatmaps and both CSVs under `outputs/`. Use
`--lq-source empirical` for the empirical-Lq pruning map, and use
`--grid-size`, `--dpi`, and `--formats png pdf svg` to control rendering.

## Dynamics stochasticity

`configs/default.json` is the shared source for environment and training
settings. The constructor `ContinuousDollarEuroEnv()` without arguments now
reads that file. The current default is `sigma = 0.00004`. Regional Gaussian
noise covariances scale with `v = sigma^2`:

- region 1: `[[1.0 v, 0], [0, 1.0 v]]`
- region 2: `[[1.4 v, 0], [0, 1.1 v]]`
- region 3: `[[1.2 v, 0.2 v], [0.2 v, 1.5 v]]`
- region 4: `[[0.9 v, 0], [0, 1.6 v]]`

The original uploaded environment used `sigma = 0.00008`; a later cleaned copy
hard-coded `0.00016`. Neither value is the runtime default anymore.

Top-level `tau` / `target_tau` is the DQN Polyak coefficient (`0.0005`).
`environment.tau` is the living-penalty scale (`2.4`). Agent episodes use
`max_steps_per_episode` (`100`), matching original theoretical RA-DQN.
R1/R2 collection uses `component_max_steps_per_episode` (`200`) and
`component_epsilon_decay` (`0.998`), matching the original two-reward
script that produced the source JSON. `environment.horizon` (`200`) remains
the environment truncation limit. Full experiments average `n_runs` (`5`)
independent agent seeds.

All training, smoke, and analysis entry points accept either `--sigma VALUE`
or `--stochasticity-scale VALUE`. A scale `c` multiplies standard deviations
by `c` and covariances by `c^2`.

The supplied transition radii and Lipschitz tables were estimated from runs
whose environment default was `sigma = 0.00008`; this provenance is recorded
in the data JSON. RA-DQN, center-Q training, and heatmap analysis reject a
different runtime sigma by default because fixed JSON radii do not update when
runtime noise changes. `bounds_sigma_policy` in the config (`error` by default)
and `--allow-radius-sigma-mismatch` are explicit escape hatches, not a
recalibration.

Increasing sigma increases conditional transition variance and generally
increases uncertainty in estimated mean transitions. With newly collected
data, confidence radii depend on both this variability and sample count.
Reusing the current fixed radii at higher sigma can make intervals too narrow,
pruning less conservative than warranted and invalidating the intended
coverage/safety interpretation. Re-estimate the bounds JSON at the new sigma
before drawing scientific conclusions.

To compare both RA-DQN variants, additionally train with
`--lq-source empirical` and include `outputs/ra_dqn_empirical.npy` in the
plot command.

The defaults are moderate sanity-run settings. For longer research runs, increase `--steps` and `--iters`, for example:

```bash
python scripts/train_q_single.py --iters 80000
python scripts/train_baseline_dqn.py --steps 150000
python scripts/train_ra_dqn.py --lq-source empirical --steps 150000
```

## Notes

- The original files are preserved in `original_uploads/`.
- Hard-coded Windows paths from the original experiments are not used by the cleaned scripts.
- `data/lipschitz_constants_from_doc.json` was transcribed from the uploaded project writeup so the RA-DQN scripts can run without a missing `two_reward_dqn_summary.json`.
- `outputs/q_single_region.pth` is generated by `scripts/train_q_single.py` and is required before running RA-DQN.
- Table 1 component constants are the unrounded values from the supplied
  `two_reward_dqn_summary.json` generated by the original fixed-action
  Lipschitz script; no missing components are reconstructed from rounded sums.

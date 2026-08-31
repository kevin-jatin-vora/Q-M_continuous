# Artifacts

Typical layout after `scripts/run_experiment.py`:

```text
outputs/experiments/cartpole_full_seed0/
  run_config.json
  behaviors/
    transitions_b1.npz
    transitions_b2.npz
    q_b1.pth
    q_b2.pth
    manifest.json
  data/
    regions.json
    transition_bounds.json
    lipschitz_constants.json
  models/
    q_single.pth
    dqn_seed*.pth / .npy
    dqn_runs.npy
    ra_dqn_theoretical_seed*.pth / .npy
    ra_dqn_empirical_seed*.pth / .npy
  plots/
    returns_plot.png
```

## File roles

| File | Meaning |
|------|---------|
| `transitions_*.npz` | Behavior rollouts: states, actions, rewards, next_states, dones |
| `regions.json` | Frozen state partition (tree or grid) + normalization |
| `transition_bounds.json` | Per `(region, action)`: `delta_mean`, `delta_std`, Student-t hw, `pruning_radius` |
| `lipschitz_constants.json` | Per `(region, action)`: `Lr_sum`, `Lf_sum`, `Lq_empirical_sum`, `LQ_bellman_bound` |
| `q_single.pth` | Center Q trained on mean dynamics |
| `*_runs.npy` | Shape `(n_seeds, n_evals)` evaluation curves |

Prefer regenerating via `run_experiment` / the stage scripts rather than hand-editing JSON.

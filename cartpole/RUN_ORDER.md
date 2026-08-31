# Run order

From `D:\dollar_euro_lipschitz_clean\cartpole` with the project venv active.

## Full pipeline (recommended)

```bat
run_experiment.cmd --profile full --runs 5 --seed 0
```

Smoke (fast wiring check):

```bat
run_experiment.cmd --profile smoke
```

## Manual stages

```bat
python scripts\collect_behaviors.py --config configs\default.json --output-dir outputs\behaviors --seed 0

python scripts\discover_regions.py --config configs\default.json --behavior-dir outputs\behaviors --out data\regions.json

python scripts\compute_stats.py --config configs\default.json --behavior-dir outputs\behaviors --regions data\regions.json --bounds-out data\transition_bounds.json --lipschitz-out data\lipschitz_constants.json

python scripts\train_q_single.py --config configs\default.json --bounds data\transition_bounds.json --regions data\regions.json --out outputs\q_single.pth

python scripts\compute_stats.py --config configs\default.json --behavior-dir outputs\behaviors --regions data\regions.json --bounds-out data\transition_bounds.json --lipschitz-out data\lipschitz_constants.json --q-single outputs\q_single.pth

python scripts\train_baseline_dqn.py --config configs\default.json --out-prefix outputs\dqn --seed 0

python scripts\train_ra_dqn.py --config configs\default.json --lq-source theoretical --out-prefix outputs\ra_dqn_theoretical --seed 0

python scripts\train_ra_dqn.py --config configs\default.json --lq-source empirical --out-prefix outputs\ra_dqn_empirical --seed 0
```

## Region knobs (`configs/default.json`)

- `region_method`: `adaptive_tree` (default) or `grid`
- `region_min_leaf`, `region_max_depth`, `region_max_regions`, `region_var_eps`
- `grid_bins`: used only for `grid` (e.g. `[2,2,4,2]`)

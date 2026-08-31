# Run Order

From project root `dollar_euro_16tile`.

## 1. Install

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Full experiment (`runs=1`)

```bat
cd /d D:\dollar_euro_lipschitz_clean\dollar_euro_16tile

run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.5 --profile full --runs 1 --seed 0
run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.4 --profile full --runs 1 --seed 0
run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.3 --profile full --runs 1 --seed 0
```

Dry run:

```bat
run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.5 --profile smoke --dry-run
```

Config key for the next-state radius:

```json
"confidence_level": 0.95
```

Do **not** `--resume` older runs that used mean-CI / process-noise schemas.

## 3. Manual stages (optional)

```bat
python scripts\train_reward_components.py --config configs\radial_match.json --sigma 0.0005 --determinism 0.5 --confidence-level 0.95 --output-dir outputs\experiments\my_run --steps 150000

python scripts\train_q_single.py --config configs\radial_match.json --sigma 0.0005 --determinism 0.5 --bounds outputs\experiments\my_run\data\transition_bounds.json --out outputs\experiments\my_run\models\q_single_region.pth --iters 80000

python scripts\generate_analysis.py --config configs\radial_match.json --sigma 0.0005 --determinism 0.5 --bounds ... --lipschitz ... --q-single ... --lq-source theoretical --output-dir ...\plots
```

## 4. Notes

- **Q_single:** `s'=clip(s+delta_mean)`
- **RA:** `r = ‖ t·δ_std·√(1+1/n) ‖₂`
- Theoretical RA fails if `γ·Lf ≥ 1`
- `determinism=1`: σ unused for dynamics

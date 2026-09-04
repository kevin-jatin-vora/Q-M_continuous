# Run Order

From project root `dollar_euro_16tile`.

## 1. Install

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Full learned-bound experiment (`runs=1`)

The production entry point is the learned-Q-bound runner. It runs all seven
stages (R1/R2 data → transition bounds → ordinary + cross-category Lipschitz +
noise artifacts → learned Q_UB/Q_LB with overlap-aware constants → analysis →
baseline DQN → learned-bound RA-DQN → plots/videos).

```bat
cd /d D:\dollar_euro_lipschitz_clean\dollar_euro_16tile

run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.0 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.2 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.4 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
```

Dry run:

```bat
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.0 --deterministic-sigma-scale 0.001 --profile smoke --dry-run
```

The legacy `run_experiment.cmd` analytical-margin pipeline is retained for
backward compatibility but is no longer the primary path.

Config key for the next-state radius:

```json
"confidence_level": 0.95
```

Do **not** `--resume` older runs that used mean-CI / process-noise schemas or
v1/v2 Lipschitz artifacts. When resuming, the runner validates the v3 Lipschitz
artifacts (`lipschitz_constants.json`, `cross_tile_reward_lipschitz.json`,
`cross_category_dynamics_lipschitz.json`) including `sigma`, `gamma`,
`determinism`, `deterministic_sigma_scale`, `tile_map`, and neighbor sets, and
fails clearly on an incompatible experiment:

> Old Lipschitz artifacts use incompatible reward grouping/cross-boundary semantics; regenerate from scratch.

## 3. Manual stages (optional)

```bat
python scripts\train_reward_components.py --config configs\radial_match.json --sigma 0.0005 --determinism 0.5 --confidence-level 0.95 --output-dir outputs\experiments\my_run --steps 150000

python scripts\train_q_single.py --config configs\radial_match.json --sigma 0.0005 --determinism 0.5 --bounds outputs\experiments\my_run\data\transition_bounds.json --out outputs\experiments\my_run\models\q_single_region.pth --iters 80000

python scripts\generate_analysis.py --config configs\radial_match.json --sigma 0.0005 --determinism 0.5 --bounds ... --lipschitz ... --q-single ... --lq-source theoretical --output-dir ...\plots
```

## 4. Notes

- **Q_single:** `s'=clip(s+delta_mean)`
- **RA (learned):** prune with `Q_UB(s,a) ≥ max_b Q_LB(s,b) − ε` using
  overlap-aware effective constants. The CURRENT action `a` sets
  `s̄'=clip(s+delta_mean(ρ,a))` and the Student-t radius `r=pruning_radius(ρ,a)`.
  The next-state ball mask yields `Lr_eff(mask)` (max over intersected tiles +
  cross-tile Lr) and, for EVERY next action `b`, `Lf_eff(mask,b)` (max over
  represented categories + cross-category Lf); then
  `Lq_future_eff(mask) = max_b Lr_eff·Lf_eff(mask,b)/(1−γ·Lf_eff(mask,b))`.
  `delta_r = Lr_eff·r`, `delta_q = Lq_future_eff·r` (current-action radius).
  Same mask ⇒ same future Lq regardless of current action.
- **RA (legacy analytical):** `r = ‖ t·δ_std·√(1+1/n) ‖₂`
- **Lr** is grouped by NEXT-state tile (action-pooled); **Lf** by source
  category/action. Cross-boundary constants split into two v3 artifacts:
  `cross_tile_reward_lipschitz.json` (Lr per neighbor tile-pair, including
  same-category pairs) and `cross_category_dynamics_lipschitz.json` (Lf per
  neighbor category-pair/action). Auto-discovered, unordered-deduped, with
  `tile_edges` provenance.
- Theoretical RA fails if `γ·Lf ≥ 1` (raises a clear error, no substitution /
  never returns 0).
- `determinism=1`: σ unused for dynamics

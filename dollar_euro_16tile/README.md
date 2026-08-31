# Dollar–Euro 16-tile Lipschitz RA-DQN

Sibling project to `dollar_euro_lipschitz_clean` with a **4×4 tile** map and **5 dynamics categories**.

## Environment

- **16 tiles** on `[0,1]²`.
- Categories **1–4** = stochastic radial templates; **5** = deterministic (`Σ=0`).
- `--determinism` = fraction of tiles that are category 5.
- At `determinism=0`: radial quadrant layout. At `determinism>0`: deterministic shuffle seeded by `determinism`.

## One story for next-state bounds

From free R1/R2 data per **(category, action)**:

```text
δ = s' − s
δ̄ = mean(δ)                         → delta_mean   (Q_single)
σ̂ = sample_std(δ)                   → delta_std
hw = t · σ̂ · √(1 + 1/n)             → student_t_half_width
r  = ‖hw‖₂                          → pruning_radius         (RA-DQN)
```

Reward peaks (config `environment.reward_alpha` / `reward_beta`, defaults **2.0 / 1.2**) with living cost `tau` **4.4** so `alpha < tau/2` (non-terminal reward stays strictly negative; no zero plateau). R1→Dollar, R2→Euro, R1+R2→Both.

| Component | Uses |
|-----------|------|
| **Q_single** | `s' = clip(s + delta_mean)` |
| **RA-DQN** | margin with `r = pruning_radius` |

Config key: `"confidence_level": 0.95` (coverage for the Student-t interval).

## Lipschitz

| Quantity | Granularity |
|----------|-------------|
| `Lr`, `Lq_emp`, `Lq_th` | **tile × action** |
| `Lf` | **category × action** (`max(Lf1,Lf2)`) |
| `pruning_radius` | **category × action** |

`Lq_th = Lr_tile / (1 − γ · Lf_cat)`. Margins are looked up by **tile id**.

## Run

```bat
cd /d D:\dollar_euro_lipschitz_clean\dollar_euro_16tile
run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.5 --profile full --runs 1 --seed 0
```

More examples:

```bat
run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.4 --profile full --runs 1 --seed 0
run_experiment.cmd configs\radial_match.json --sigma 0.0005 --determinism 0.3 --profile full --runs 1 --seed 0
```

Use a **new** output directory (do not resume old runs with the previous radius schema).

## Output layout

```text
data/     transition_bounds.json, lipschitz_constants.json, tile_layout.json
models/   q_single, dqn*, ra_dqn*
plots/    heatmaps, table1 (tile Lip + both Lq), table2 (bounds)
videos/   policy mp4s
```

See `algorithm.md`, `RUN_ORDER.md`, `ARTIFACTS.md`.

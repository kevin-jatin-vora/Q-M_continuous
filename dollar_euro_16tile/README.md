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
| `Lr_cross`, `Lf_cross`, `Lq_cross_th` | **neighbor-pair × action** |
| `pruning_radius` | **category × action** |

`Lq_th = Lr_tile * Lf_cat / (1 − γ · Lf_cat)`. Margins are looked up by **tile id**.

Learned Q_UB/Q_LB training uses **overlap-aware effective constants** (see
`algorithm.md`). The CURRENT action `a` fixes the transition uncertainty:
`delta_mean(ρ,a)`, the nominal next state `s̄'`, and the Student-t radius
`r = pruning_radius(ρ,a)`. The next-state ball `B(s̄',r)` determines an intersected
tile mask from which `Lr_eff(mask)` and — over **every** possible next action
`b ∈ {0,1,2,3}` — `Lf_eff(mask,b)` are formed; the future Q Lipschitz constant is
`Lq_future_eff(mask) = max_b Lr_eff·Lf_eff(mask,b)/(1−γ·Lf_eff(mask,b))`, because
the Bellman continuation contains `max_b Q(s̄',b)`. Then `delta_r = Lr_eff·r` and
`delta_q = Lq_future_eff·r` (both scaled by the current-action radius). The same
mask returns the same future `Lq` regardless of which current action produced it.
An invalid `Lq` (`γ·Lf_eff(b) ≥ 1` for any future action `b`) raises an error
identifying the offending future action instead of silently substituting.

## Run

```bat
cd /d D:\dollar_euro_lipschitz_clean\dollar_euro_16tile
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.0 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
```

Recommended experiment set (same flags, varying determinism):

```bat
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.2 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.98 --determinism 0.4 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
```

Use a **new** output directory (do not resume old runs with the previous radius
schema). When resuming, `run_experiment_qbounds.py` validates the v3 Lipschitz
artifacts (`cross_tile_reward_lipschitz.json`,
`cross_category_dynamics_lipschitz.json`, `lipschitz_constants.json`) and refuses
incompatible/version-1/version-2 artifacts with:

> Old Lipschitz artifacts use incompatible reward grouping/cross-boundary semantics; regenerate from scratch.

## Output layout

```text
data/     transition_bounds.json, lipschitz_constants.json,
          cross_tile_reward_lipschitz.json,
          cross_category_dynamics_lipschitz.json,
          category_noise_action_ratio.json, tile_layout.json
models/   q_single, q_ub_*/q_lb_* (+ manifest), dqn*, ra_dqn*
plots/    heatmaps, table1 (tile Lip + both Lq), table2 (bounds),
          learned_q_bounds_* heatmaps
videos/   policy mp4s
```

See `algorithm.md`, `RUN_ORDER.md`, `ARTIFACTS.md`.

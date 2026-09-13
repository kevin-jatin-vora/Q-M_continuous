# Dollar–Euro 16-tile Lipschitz RA-DQN

Sibling project to `dollar_euro_lipschitz_clean` with a **4×4 tile** map and **5 dynamics categories**.

## Environment

- **16 tiles** on `[0,1]²`.
- Every category has the same four mean moves (`step_size × action_vector`).
- Terrain differs through isotropic noise multipliers **1.0, 1.1, 1.2, 1.3**;
  category 5 uses `deterministic_sigma_scale`.
- Scalar noise is bilinearly interpolated between tile centers, so the transition
  kernel is continuous across category boundaries.
- `--determinism` = fraction of tiles that are category 5.
- At `determinism=0`: radial quadrant layout. At `determinism>0`: deterministic shuffle seeded by `determinism`.

The live transition is

```text
s' = clip(s + step_size * action_vector[a] + sigma(s) * Z, 0, 1),  Z ~ N(0,I₂)
```

The nearest tile-center value is extended to the outer edge. Direct clipping
replaces boundary cancellation and is non-expansive.

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

The deterministic mean map has `Lf = 1`. The analytic global reference is
`K_P = sqrt(1 + 2*K_sigma^2)`, where
`K_sigma = sup_s ||grad sigma(s)||₂` is computed from the known environment.
It is a contractivity check and verification value, not the single constant used
for every row. Collected transitions fit the common action means and five
category-center noise scales. The normal experiment uses the **maximum** fitted
Wasserstein ratio over sampled pairs within each source category/action and
across each neighboring-category/action pair. Production therefore uses local-
max dynamics `Lf`; theoretical global `K_P` remains verification and preflight.
Production reward `Lr` also uses the maximum sampled reward ratio within each
next-state tile and across each neighboring next-state tile pair.
Both production constants reject pairs closer than the smallest represented
category/action pruning radius derived from `transition_bounds.json`. This
threshold is automatic, not a command-line parameter.

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
cd /d "D:\dollar_euro_lipschitz_clean\dollar_euro_16tile - smooth noise same mean"
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.01 --gamma 0.96 --determinism 0.0 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
```

Requested comparison (same flags, 0% versus 20% category-5 tiles):

```bat
run_experiment_qbounds.cmd configs\radial_match.json --sigma 0.01 --gamma 0.96 --determinism 0.2 --deterministic-sigma-scale 0.001 --profile full --runs 1 --seed 0
```

Use a **new** output directory (do not resume old runs with the previous radius
schema). When resuming, `run_experiment_qbounds.py` validates the v3 Lipschitz
artifacts (`cross_tile_reward_lipschitz.json`,
`cross_category_dynamics_lipschitz.json`, `lipschitz_constants.json`) and refuses
incompatible/version-1/version-2 artifacts with:

> Old Lipschitz artifacts use incompatible reward grouping/cross-boundary semantics; regenerate from scratch.

The experiment identity includes `reward_and_wasserstein_local_max_min_radius_v3`,
so a run cannot resume artifacts made with the former estimator or pair-distance
rule.

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

## Lipschitz diagnostics

The diagnostic runner keeps the original four experiment names and runs them in
conservative-first order: `local_max`, `global_max`, `local_mean`, then
`global_mean`. It reuses collected R1/R2
transitions only when smooth-noise provenance matches. Mean/max and local/global
control both reward aggregation and data-fitted Wasserstein aggregation. Local
variants produce category/action and neighboring-category/action values; global
variants pool all source states and produce one value per action, replicated
across tiles for schema compatibility. The minimum pair distance is derived
automatically as the smallest pruning radius among represented category/action
cells in the shared transition bounds.

`shared/wasserstein_sample_verification.json` records the fitted common action
means, fitted category sigmas, normalized residual moments, and fitted global
`K_P` beside theoretical global `K_P`. Diagnostic training uses the appropriate
data-derived local/cross/global values.

Each gap directory also contains `wasserstein_variant_comparison.json`, and the
runner prints ordinary and cross-category min/mean/max values for all four
variants before training.

```bat
run_diagnostics.cmd configs\radial_match.json --sigma 0.0005 --gamma 0.96 --determinism 0.0 --deterministic-sigma-scale 0.001 --steps 150000 --seed 0 --reuse-data
run_diagnostics.cmd configs\radial_match.json --sigma 0 --gamma 0.96 --determinism 1.0 --deterministic-sigma-scale 0.001 --steps 150000 --seed 0 --reuse-data
```

The derived threshold is printed and stored in every LC artifact. Results are
stored in a threshold-specific `gap_*` directory, while raw transitions and four
combined R1+R2 source-state count heatmaps (one per action) remain in `shared/`.
The first run after this noise-model change collects fresh data because the
model version, interpolation, multipliers, `K_sigma`, and `K_P` are hashed into
provenance. Later identical runs can reuse it.

# Artifacts

## Bundled inputs (`data/`)

| File | Meaning |
|------|---------|
| `single_q_bounds_from_json.json` | Category-level `delta_mean`, `delta_std`, `student_t_half_width`, `pruning_radius` (σ=8e-5 snapshot) |
| `lipschitz_constants_from_doc.json` | Legacy category-only Lip snapshot; full runs replace with per-tile JSON |
| `original_uploads/` | Untouched reference scripts |

Prefer regenerating via `run_experiment` / `train_reward_components.py`.

---

## Per-experiment `data/`

### `transition_bounds.json` (category × action)

| Field | Meaning |
|-------|---------|
| `delta_mean` | Mean free `δ = s'−s` → **Q_single** |
| `delta_std` | Sample std of `δ` |
| `student_t_half_width` | `t · delta_std · √(1+1/n)` per axis |
| `pruning_radius` | `‖student_t_half_width‖₂` → **RA** |
| `config.confidence_level` | Coverage for the Student-t interval (default 0.95) |

### `lipschitz_constants.json` (tile × action, v3)

| Field | Meaning |
|-------|---------|
| `tile`, `region`, `action` | Location |
| `Lr_*` | **NEXT-state-tile local-max** reward Lip (`Lr_grouping: next_state_tile`), replicated across the tile's 4 actions, `Lr_action_dependence: false` |
| `Lf_*` | Operational local-max data-fitted Wasserstein constants at the artifact's grouping |
| `mean_map_Lf` | Deterministic mean-map constant, always `1` for represented terrain |
| `Lq_empirical_sum`, `LQ_bellman_bound` | Per-tile Lq (`LQ_bellman_bound = Lr*Lf/(1−γ·Lf)`) |
| `lr_source` | `next_tile` |
| `lipschitz_method_version` | `3` |
| `min_pair_distance`, `min_pair_distance_source` | Automatic minimum represented pruning radius used to filter LC pairs |

Also: `tile_layout.json`, `run_config.json`, manifests.

### `cross_tile_reward_lipschitz.json` (neighbor tile-pair, v3)

Cross-boundary **reward** Lipschitz `Lr` across every physically neighboring
tile pair (NEXT-state tiles), including same-category pairs. Action-independent.

| Field | Meaning |
|-------|---------|
| `sigma`, `gamma`, `determinism`, `deterministic_sigma_scale`, `deterministic_sigma` | Provenance of the generating run |
| `tile_map`, `Lr_estimator` (`local_max`), `min_pair_distance`, `max_pairs` | Provenance + estimation controls |
| `neighbor_tile_pairs` | Discovered `[tile_a, tile_b]` list (4-neighbor pairs) |
| `Lr_definition`, `Lr_grouping` (`next_state_tile_pair`), `Lr_action_dependence` (false) | Semantics |
| `constants[]` | One entry per tile pair |

Per `constants[]` entry: `tile_i`, `tile_j`, `region_i`, `region_j`,
`n_i_r1`, `n_j_r1`, `n_i_r2`, `n_j_r2`, `pairs_r1`, `pairs_r2`, `Lr1`, `Lr2`,
`Lr_sum` (= Lr1+Lr2).

### `cross_category_dynamics_lipschitz.json` (neighbor category-pair × action, v3)

Cross-boundary **dynamics** Lipschitz `Lf` across neighbor category pairs
(different categories) for each action. Auto-discovered neighbor pairs are
canonicalized to `(min, max)` and deduplicated.

In the smooth-noise pipeline, `Lf_sum` is computed from the fitted conditional
Gaussian transition model. Each row uses source pairs from its neighboring
category pair; theoretical global `K_P` is verification metadata only.

| Field | Meaning |
|-------|---------|
| `sigma`, `gamma`, `determinism`, `deterministic_sigma_scale`, `deterministic_sigma` | Provenance of the generating run |
| `tile_map`, `Lf_estimator` (`local_max`), `min_pair_distance`, `max_pairs` | Provenance + estimation controls |
| `neighbor_category_pairs` | Discovered `{category_i, category_j, tile_edges}` list |
| `mean_map_Lf`, `Lf_sum` | Mean-map and operational data-fitted stochastic constants |
| `theoretical_global_K_P_verification` | Known-environment verification value |
| `constants[]` | One entry per pair × action |

Per `constants[]` entry: `category_i`, `category_j`, `action`, `tile_edges`,
sample counts, data-derived `Lf1`, `Lf2`, and `Lf_sum`, plus `mean_map_Lf=1`
and theoretical-global verification metadata.

`Lf1` and `Lf2` are maxima over sampled fitted-Gaussian Wasserstein ratios for
their respective behaviors, and `Lf_sum = max(Lf1, Lf2)`. `Lr1` and `Lr2` are
also per-behavior maxima, while `Lr_sum = Lr1 + Lr2`.

**Separation guarantee:** a same-category neighboring tile pair yields a
cross-tile Lr entry in `cross_tile_reward_lipschitz.json` but **no** cross-category
Lf entry. All three Lipschitz artifacts declare `lipschitz_method_version: 3`.

### `category_noise_action_ratio.json`

Noise magnitude relative to the common deterministic movement at category tile
centers. Categories 1–4 use scalar multipliers 1.0, 1.1, 1.2, and 1.3;
category 5 uses `deterministic_sigma_scale`. The live field interpolates these
values continuously between centers.

Diagnostic provenance additionally records `noise_model_version`, interpolation
method, category multipliers, tile multiplier grid, `noise_multiplier_lipschitz`,
`sigma_lipschitz`, `wasserstein_kernel_lipschitz`, and
`gamma_times_wasserstein_kernel_lipschitz`.

The diagnostics also write `shared/wasserstein_sample_verification.json`. It
contains per-action fitted mean displacement, fitted category-center sigmas,
normalized residual moments, and fitted global `K_P` next to the theoretical
value. The four variants use data-derived Wasserstein ratios operationally.
`gap_*/wasserstein_variant_comparison.json` provides a compact ordinary and
cross-category min/mean/max comparison across all four variants.

---

## Plots

| File | Content |
|------|---------|
| `table1_lipschitz_constants_sigma_*_det*.csv` | Tile Lip + `Lq_empirical` + `Lq_theoretical` |
| `table2_transition_bounds_sigma_*_det*.csv` | `delta_mean`, `delta_std`, `student_t_half_width`, `pruning_radius` |
| `pruning_heatmap_{empirical,theoretical}_*.png` | Surviving actions |

---

## Schema note

Older fields (`mean_ci_half_width`, `process_std`, `process_noise_confidence`, `radius_scalar`, …) are **not** supported. Regenerate bounds.

---

## Learned Q-bound manifest (`q_bounds_theoretical_manifest.json`)

The overlap-aware learned Q_UB/Q_LB manifest records
`q_bound_runtime_method_version: 2` and the corrected current-vs-future role
separation:

- `current_action_role`: the current action determines `delta_mean`, the nominal
  next state `s̄'`, and the Student-t radius.
- `future_action_role`: the future Bellman max requires
  `Lq_future_effective = max_b Lq_effective(b)` over all next actions.
- `delta_r` = `Lr_effective(mask) * radius(source_category,current_action)`.
- `delta_q` = `max_b[Lr_effective(mask)*Lf_effective(mask,b)/
  (1-gamma*Lf_effective(mask,b))] * radius(source_category,current_action)`.

The `overlap_diagnostics` block reports tile/category boundary crossings, per-
future-action ordinary-vs-cross `Lf` selection counts, the future-action argmax
histogram, and `max_lr_effective`, `max_lf_effective_over_all_next_actions`, and
`max_lq_future_effective`.

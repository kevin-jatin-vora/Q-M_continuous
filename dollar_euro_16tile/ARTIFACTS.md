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

### `lipschitz_constants.json` (tile × action)

| Field | Meaning |
|-------|---------|
| `tile`, `region`, `action` | Location |
| `Lr_*` | Per-tile reward Lip |
| `Lf_*` | Category-level dynamics Lip |
| `Lq_empirical_sum`, `LQ_bellman_bound` | Per-tile Lq |
| `lr_source` | `tile` or `category_fallback` |

Also: `tile_layout.json`, `run_config.json`, manifests.

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

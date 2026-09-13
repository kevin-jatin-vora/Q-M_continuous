# Dollar–Euro Lipschitz RA-DQN — Algorithm Guide

4×4 tiles, 5 terrain categories with common mean motion and smoothly varying
isotropic noise.

## Smooth terrain transition kernel

Every terrain uses the same mean action displacement. Category identity changes
the target scalar noise multiplier at tile centers:

```text
category:       1    2    3    4    5
multiplier:    1.0  1.1  1.2  1.3  deterministic_sigma_scale
```

The multiplier is bilinearly interpolated between tile centers with constant
extension to the domain edge. The environment samples

```text
s' = clip(s + step_size * action_vector[a] + sigma(s) * Z, 0, 1), Z ~ N(0,I₂)
```

There is no boundary cancellation. Under common-noise coupling and
non-expansive clipping:

```text
mean-map Lf = 1
K_sigma     = sup_s ||grad sigma(s)||₂
K_P         = sqrt(1 + 2*K_sigma²)
```

`K_sigma` is the exact maximum over bilinear-cell corner gradients. This known-
environment global `K_P` is used as a pre-collection contractivity check and as
a verification reference. Operational artifacts are estimated from data at the
local, cross-category, or global granularity requested by the experiment.

---

## Big picture

1. Collect free `(s,a,s')` from R1/R2 behavior.
2. Per **(category, action)** estimate mean move `δ̄` and Student-t radius `r`.
3. Train `Q_single` on `s' = clip(s + δ̄)`.
4. Prune with margins that use per-tile `Lr`/`Lq` and category `r`.
5. Train DQN and RA-DQN on the live env.

---

## Next-state model (the only radius story)

Free steps in cell `(ρ, a)`:

```text
δᵢ = s'ᵢ − sᵢ
n  = number of free samples (R1+R2 pooled)

δ̄  = (1/n) Σ δᵢ
σ̂² = (1/(n−1)) Σ (δᵢ − δ̄)²          # per axis
σ̂  = √σ̂²

hw = t_{1−α/2, n−1} · σ̂ · √(1 + 1/n)  # student_t_half_width
r  = ‖hw‖₂                              # pruning_radius
```

`α` from config `confidence_level` (default **0.95** → `t` at 0.975).

| Field | Role |
|-------|------|
| `delta_mean` (`δ̄`) | Mean free move → **Q_single** |
| `delta_std` (`σ̂`) | Scatter around the mean |
| `student_t_half_width` | Per-axis Student-t interval half-width |
| `pruning_radius` (`r`) | L2 radius → **RA margins** |

**Env comparison:** free dynamics ≈ `s' = s + Δ(ρ,a) + ε`. Then `δ̄ ≈ Δ` and `σ̂ ≈` env noise scale. Deterministic category 5 → `σ̂ ≈ 0` → `r ≈ 0`.

The `√(1+1/n)` factor widens the interval beyond a mean confidence interval — appropriate for where the next `s'` lands, not just uncertainty in `δ̄`.

---

## Lipschitz (method version 3)

Reward depends on the next state `R(s')`, while dynamics are a function of the
source state.  Therefore the ordinary constants use two **different** grouping
domains:

- **`Lr`** — per **NEXT-state tile** (`Lr_grouping: next_state_tile`),
  **action-pooled** (`Lr_action_dependence = false`).  All free next-states that
  land in tile `Ti`, regardless of source category or action, contribute

  ```text
  Lr(Ti) = max( |R(s1') − R(s2')| / ‖s1' − s2'‖ )       # distance >= r_min
  Lr_sum = Lr1 + Lr2
  ```
- **Data-fitted Wasserstein `Lf`** — per **source (category, action)** for local
  variants, or pooled by action for global variants:

  ```text
  f_hat(s,a) = s + mu_hat_a
  P_hat(.|s,a) = N(f_hat(s,a), sigma_hat(s)^2 I)
  ratio(s,t,a) = W2(P_hat_s,a, P_hat_t,a) / ‖s−t‖
  Lf_sum = max(Lf1, Lf2)
  ```
- **`Lq_th`** (derived, per tile×action row, shared across the tile's actions):
  ```text
  Lq = Lr(Ti) · Lf(Ci,a) / (1 − γ · Lf(Ci,a))           # needs γ·Lf < 1
  ```

The noise field is fitted from successor residuals using the same bilinear model
class as the environment. This avoids comparing independent realized successors.
The resulting data-derived Wasserstein value is placed in `Lf_sum`; theoretical
global `K_P` is saved only for verification.

For the normal experiment, `aggregate` is the maximum over sampled source-state
pairs. Ordinary `Lf` is therefore local-max within each source category/action;
cross `Lf` is local-max across each neighboring category pair/action. Production
reward `Lr` likewise uses the maximum ordinary next-tile ratio and maximum
neighboring-tile ratio.

Here `r_min` is automatic. After R1/R2 transition bounds are built, it is set
to the smallest `pruning_radius` among represented category/action cells. The
same `r_min` filters source distances for Wasserstein `Lf` and next-state
distances for reward `Lr`, in production and the standard diagnostics. If
`r_min` is zero, only numerical zero-distance pairs are rejected.

Diagnostics retain the four original labels and run max before mean
(`local_max`, `global_max`, `local_mean`, `global_mean`). Those choices still change the aggregation and
grouping of `Lr`, empirical `Lq`, and Wasserstein ratios. Local variants estimate
per-category/action and neighboring-category/action constants. Global variants
pool all categories and estimate one constant per action. The fitted model and
the theoretical global verification value are saved in
`shared/wasserstein_sample_verification.json`.

---

## Cross-boundary Lipschitz (separation of reward vs dynamics)

Cross-boundary constants are split into **two independent artifacts** (v3):

### Cross-tile reward Lr — `data/cross_tile_reward_lipschitz.json`

For every physically neighboring **tile pair** `(Ti, Tj)`, reward Lr is estimated
across the shared edge by pairing one next-state sample in `Ti` with one in `Tj`
(action-independent, since reward is `R(s')`):

```text
  Lr_cross = Lr1 + Lr2   # maximum of |R(s1')−R(s2')|/‖s1'−s2'‖ per behavior
```

This applies to ALL neighboring tile pairs, **including same-category pairs**
(e.g. two adjacent tiles in one region) — a cross-tile reward gradient can exist
even when the dynamics category does not change.

### Cross-category dynamics Lf — `data/cross_category_dynamics_lipschitz.json`

For the smooth-noise diagnostic, each neighboring category pair is evaluated
using source-state pairs from that pair and the data-fitted stochastic model.

For every neighbor **category pair** `(Ci, Cj)` and action, the operational
dynamics constant is:

```text
Lf_cross = aggregate W2(P_hat(.|s,a), P_hat(.|t,a)) / ||s-t||
```

- Each category pair is canonicalized to `(min, max)` and deduplicated across
  the multiple `tile_edges` realizing the adjacency.
- Pairs are **auto-discovered** from the 4×4 tile map
  (`discover_neighbor_pairs`), with `tile_edges` provenance; tile pairs come from
  `discover_tile_neighbor_pairs`.
- **Key separation guarantee:** a same-category neighboring tile pair produces a
  cross-tile Lr entry but contributes **no** cross-category Lf.
- Both artifacts declare `lipschitz_method_version: 3` plus `sigma`, `gamma`,
  `determinism`, `deterministic_sigma_scale`, `tile_map`, their estimator
  metadata, `max_pairs`, neighbor sets, and per-pair/action `constants`.

---

## Overlap-aware effective constants (learned Q bounds)

The learned Q_UB/Q_LB training uses a one-stage, category/tile-agnostic bound with
**geometric next-state ball intersection** instead of a single tile lookup.

### Current action vs future action

The **current action `a`** controls the transition uncertainty:

- source category `ρ = tile_map[t]` of the pre-step state `s`;
- ball center `s̄' = clip(s + δ̄(ρ,a), 0, 1)` (the nominal next state);
- ball radius `r = pruning_radius(ρ,a)` (the Student-t L2 radius of the *current*
  transition — never maximized over actions).

The **future action `b`** appears only inside the Bellman continuation
`max_b Q(s̄',b)`. Because

```text
|max_b Q(s1',b) − max_b Q(s2',b)| ≤ max_b |Q(s1',b) − Q(s2',b)| ≤ (max_b Lq(b))·‖s1'−s2'‖
```

the future Q Lipschitz constant must be the maximum over **all** possible next
actions `b ∈ {0,1,2,3}` — it is **not** the `Lf`/`Lq` of the current action.

### Ball selection

1. Compute the exact minimum squared L2 distance from `s̄'` to each of the 16 tile
   rectangles (vectorized, no `sqrt`); a tile is in the ball iff that distance ≤
   radius². Collect the intersected tiles and the distinct represented categories.
2. **Lr_eff(mask)** = max over:
   - ordinary next-tile `Lr(Ti)` for every intersected tile;
   - cross-tile `Lr_cross` for every neighbor tile pair whose two tiles are both
     intersected.
3. For **each** future action `b` in `{0,1,2,3}`, **Lf_eff(mask,b)** = max over:
   - ordinary source-category `Lf(ρ,b)` for every represented category;
   - cross-category `Lf_cross` for every neighbor category pair among the
     represented categories.
4. **Lq_eff(mask,b) = Lr_eff(mask) · Lf_eff(mask,b) / (1 − γ · Lf_eff(mask,b))**.
5. **Lq_future_eff(mask) = max_b Lq_eff(mask,b)** (the controlling next action is
   recorded as `future_action_argmax`).
6. `delta_r = Lr_eff(mask) · r`, `delta_q = Lq_future_eff(mask) · r`, both scaled by
   the **current-action** radius `r`.

Invariants:

- Results are cached by the **16-bit integer tile mask** (dependent only on the
  set of intersected tiles), never on float center/radius and never on the current
  action. The same mask always yields the same `Lq_future_eff`.
- If **any** future action `b` has `γ·Lf_eff(mask,b) ≥ 1`, training **raises a
  clear error** reporting the mask, intersected tile IDs, represented categories,
  `Lr_eff` and its provenance, the offending future action `b`, `Lf_eff(b)` and its
  provenance, `γ·Lf_eff(b)`, and the denominator — it never returns 0, never
  skips/clamps, and never substitutes another action or a finite-horizon fallback.
- A ball that stays inside one category reproduces the ordinary per-tile lookup.
- Production reward and Wasserstein constants use **local maxima** after the
  automatic minimum-radius pair filter. Debug mean/max variants use the
  aggregation named by the variant.

Diagnostics (printed and saved in the Q-bound manifest): total bound lookups,
stays-in-one-tile / crossed-≥1-tile-boundary count + percentage, stays-in-one-
category / crossed-≥1-category-boundary count + percentage, max number of tiles and
categories intersected, counts of effective `Lr` selected from ordinary next-tile
vs cross-tile, per-future-action counts of `Lf_eff(b)` selected from ordinary vs
cross-category, max selected `Lr_eff`, `Lf_future_eff`, `Lq_future_eff`, and the
`future_action_argmax` histogram showing which next action controls the bound.

---

## Noise relative to deterministic movement

`data/category_noise_action_ratio.json` reports, per category and action:

```text
move_norm       = || action_vector · step_size ||₂
noise_std_norm  = sqrt(2) · sigma · category_center_multiplier
noise_percent   = 100 · noise_std_norm / move_norm
```

These rows describe category values at tile centers. Between centers, the live
noise magnitude is the bilinearly interpolated value.

---

## Q_single

```text
s' = clip(s + delta_mean_{ρ(s), a})
```

Frozen afterward; used only as the pruning center.

---

## RA margin and pruning

The active production path trains **learned** Q_UB/Q_LB (see "Overlap-aware
effective constants" above) and prunes with:

```text
keep a if Q_UB(s,a) ≥ max_b Q_LB(s,b) − ε
```

The older analytical closed-form margin (below) is retained only for reference:

```text
m = (Lr + γ·Lq) · r / (1 − γ)     # default Bellman
r = pruning_radius(ρ(s), a)
Lr, Lq from the tile of s
```

```text
U_a = Q_single(s,a) + m_a
L_a = Q_single(s,a) − m_a
keep a if U_a ≥ max L − ε
```

---

## Checklist

1. One radius: Student-t `r` for next `s'`.
2. Q_single uses **mean** only; RA uses **`r`**.
3. `Lr` per next-state tile (action-pooled); `Lf` per source category/action;
   `r` per category. `Lq = Lr·Lf/(1−γ·Lf)`.
4. `γ·Lf ≥ 1` → theoretical Lq undefined → abort (raises, never 0).
5. At `sigma=0`, `K_sigma=0`, `K_P=1`, and all transitions are deterministic.

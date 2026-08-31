# CartPole Lipschitz RA-DQN — Algorithm Guide

Same pruning story as Dollar–Euro 16-tile, adapted to **Gymnasium CartPole-v1**
where dynamics categories are **not given** and must be estimated from data.

---

## Big picture

1. Train **B1 / B2** behavior DQNs (angle-shaped rewards) and dump transitions.
2. Discover a **state-space partition** (regions) that keeps `δ = s'−s` homogeneous.
3. Per **(region, action)** estimate `delta_mean`, Student-t radius `r`, and Lipschitz.
4. Train **Q_single** on mean dynamics `s' = s + delta_mean`.
5. Recompute empirical `Lq` with Q_single; build RA margins.
6. Train **DQN** and **RA-DQN** on classic CartPole reward (+1 / step).

---

## Behaviors (B1 / B2)

CartPole has no R1/R2 peaks in the Dollar–Euro sense. We reuse the offline
angle-threshold shaping (threshold ≈ **0.174 rad ≈ 10°**):

| Behavior | Preference on pole angle `θ = s[2]` |
|----------|----------------------------------|
| **B1** | favor `θ < 0` (lean left) |
| **B2** | favor `θ > 0` (lean right) |

### Reward shape: `step` vs `tanh` (`reward_shape` in config)

The behavior *goal* is identical in both shapes; only the grading changes.

```text
step (original):  R = (20·sign·s(θ) + 0.5)/20,  s(θ) = +1 if θ<-thr, -1 if θ>+thr, else 0
tanh (default) :  R = (20·sign·(-tanh(θ/κ)) + 0.5)/20
```

- `step` is discontinuous at `θ = ±thr`, so **no finite reward Lipschitz constant
  `Lr`** exists near the boundary; empirical `Lr` blows up for cells straddling it.
- `tanh` is continuous with `|dR/dθ| ≤ 1/κ` (κ = `reward_kappa`, default 0.174),
  giving a **finite, well-conditioned `Lr`** and a smoother learning signal
  (partial credit for reducing |θ| instead of a flat 0 inside the band).
- Effect on behavior: the same sign structure and saturation, so B1/B2 still
  cover the left/right tilt regions; policies are typically slightly more
  aggressive because gradient information exists inside the band.

Script: `scripts/collect_behaviors.py`  
Artifacts: `outputs/.../behaviors/transitions_b1.npz`, `transitions_b2.npz`,
`outputs/.../behaviors/q_b{1,2}.pth`, and greedy rollout videos in
`outputs/.../behaviors/videos/behavior_b{1,2}-episode-*.mp4`
(`record_behavior_videos`, `video_episodes`; disable with `--no-videos`).

These policies are **coverage / diversity** tools for dynamics estimation.
The final agent optimizes **classic survival reward**, not B1+B2.

---

## How regions are identified

CartPole does **not** provide tile categories. Regions are a **frozen partition of
state `s ∈ ℝ⁴`** learned from pooled B1∪B2 transitions.

### Default: adaptive tree (`region_method: adaptive_tree`)

1. Normalize states with fixed scales `[2.4, 3.0, 0.21, 3.0]` (and data mean).
2. Grow a binary tree on normalized `s`.
3. At each node, try thresholds that **minimize weighted within-leaf variance of `δ`**.
4. Stop when leaf is small, depth hits `region_max_depth`, region budget hits
   `region_max_regions`, or δ-variance ≤ `region_var_eps`.
5. Each leaf gets an integer **region id**.

**Online lookup is exact:** `ρ = partition.region_of(s)` — no `s'` and no classifier.

### Optional: axis-aligned grid (`region_method: grid`)

Quantize normalized state with `grid_bins` (default `[2,2,4,2]`). Exact lookup;
coarser control of cell count.

### Why this helps RA

Good regions ⇒ within each `(ρ, a)`:

- `δ` is similar ⇒ small `delta_std` ⇒ small Student-t **`r`**
- local dynamics are smoother ⇒ usable **`Lf`** for theoretical `Lq`

CartPole is **deterministic**, so with fine enough leaves `r ≈ 0`. Remaining
`r > 0` is mostly **binning / coverage** uncertainty, not process noise.

Script: `scripts/discover_regions.py` → `data/regions.json`

---

## Transition stats (same fields as 16-tile)

For each cell `(ρ, a)` from free behavior transitions:

```text
δᵢ = s'ᵢ − sᵢ
n  = #samples in cell

δ̄  = mean(δ)
σ̂  = sample std(δ)          # per axis, ddof=1
hw = t_{1−α/2, n−1} · σ̂ · √(1 + 1/n)
r  = ‖hw‖₂                  # pruning_radius
```

| Field | Role |
|-------|------|
| `delta_mean` | Mean step → **Q_single** target dynamics |
| `delta_std` | Scatter around the mean |
| `student_t_half_width` | Per-axis predictive half-width |
| `pruning_radius` (`r`) | L2 radius → **RA margins** |

`confidence_level` defaults to **0.95**, `gamma` defaults to **0.97**.

Script: `scripts/compute_stats.py` → `transition_bounds.json`

---

## Lipschitz

Inside each `(ρ, a)` cell, 10% trimmed pairwise ratios:

```text
Lr  ≈ |r_i − r_j| / ‖s_i − s_j‖
Lf  ≈ ‖s'_i − s'_j‖ / ‖s_i − s_j‖
Lq_emp ≈ |Q(s_i,a) − Q(s_j,a)| / ‖s_i − s_j‖   # after Q_single exists
Lq_th = Lr / (1 − γ Lf)     # requires γ·Lf < 1
```

Pipeline runs stats once before Q_single (`Lq_emp=0`), trains Q_single, then
**recomputes** stats so empirical RA has a real `Lq_emp`.

---

## Q_single

```text
s'_model = s + delta_mean_{ρ(s), a}
```

Train a DQN on classic reward using this mean next-state in the Bellman backup
(environment still steps for real trajectories / rewards).

Frozen afterward; used only as the **pruning center**.

---

## RA margin and pruning

```text
m_a = (Lr + γ·Lq) · r / (1 − γ)     # default Bellman form
r   = pruning_radius(ρ(s), a)
```

```text
U_a = Q_single(s,a) + m_a
L_a = Q_single(s,a) − m_a
keep a if U_a ≥ max_b L_b − ε
```

RA-DQN ε-greedy / greedy only among **allowed** actions; Double DQN target also
masked to allowed actions.

**Note:** if regions are fine and `r≈0`, margins shrink and RA ≈ DQN. That is
expected for a nearly deterministic local model — regions still matter for the
mean dynamics used by Q_single.

---

## Checklist

1. Behaviors provide coverage; task reward is classic CartPole.
2. Regions = state partition with exact `ρ(s)` lookup.
3. One radius story: Student-t `r` for next `s'` around `s+δ̄`.
4. Q_single uses **mean** only; RA uses **`r`** + Lipschitz.
5. `γ·Lf ≥ 1` ⇒ theoretical Lq undefined for that cell (common on CartPole;
   prefer **empirical** Lq, or collect more data / coarser regions).
6. Recompute Lipschitz after Q_single for empirical RA.
7. Default `run_experiment` uses `--lq-source empirical` for this reason.

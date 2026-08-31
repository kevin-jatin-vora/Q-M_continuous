# Dollar–Euro Lipschitz RA-DQN — Algorithm Guide

4×4 tiles, 5 dynamics categories (1–4 stochastic; 5 deterministic).

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

## Lipschitz

- **`Lr`, `Lq_emp`:** per **(tile, action)** from free pairwise ratios (`Lr = Lr1+Lr2`).
- **`Lf`:** per **(category, action)** = `max(Lf1, Lf2)`.
- **`Lq_th`:** `Lr_tile / (1 − γ · Lf_cat)` (needs `γ·Lf < 1`).

Sparse tiles may fall back to category-pooled `Lr`/`Lq` (`lr_source: category_fallback`).

---

## Q_single

```text
s' = clip(s + delta_mean_{ρ(s), a})
```

Frozen afterward; used only as the pruning center.

---

## RA margin and pruning

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
3. `Lr`/`Lq` per tile; `Lf` and `r` per category.
4. `γ·Lf ≥ 1` → theoretical Lq undefined → abort.
5. At `determinism=1`, σ does not change dynamics.

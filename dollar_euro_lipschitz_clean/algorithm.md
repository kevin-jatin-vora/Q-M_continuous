# Dollar–Euro Lipschitz RA-DQN — Algorithm Guide

This document describes what the code does, in order, with every symbol defined before it is used.

---

## Big picture

We solve a continuous 2D navigation MDP (Dollar–Euro).

1. Learn how each action typically moves the state in each region (mean step + how sure we are about that mean).
2. Learn how sensitive reward / dynamics / Q are to state changes (Lipschitz numbers).
3. Build a frozen center Q-network (`Q_single`) on those mean steps.
4. At each state, drop actions that cannot be optimal according to a Q-interval test.
5. Train agents: plain DQN, and RA-DQN that only uses surviving actions.

---

## Glossary (read this first)

| Name in text | Code / math | Meaning |
|--------------|-------------|---------|
| state | `s` | Point in the unit square `[0,1]×[0,1]` |
| next state | `s'` | State after one environment step |
| action | `a` | `0=Down, 1=Up, 2=Left, 3=Right` |
| region | `ρ` | One of 4 quadrants about center `(0.5, 0.5)` |
| cell | `(ρ, a)` | All free transitions that started in region `ρ` and took action `a` |
| displacement | `δ = s' − s` | How much the state moved on that step (a 2D vector) |
| mean step | `μ̂` | Average displacement in a cell |
| sample std | `σ` | How much individual displacements vary around `μ̂` |
| radius | `r` | How uncertain we are about the **mean** `μ̂` (see §3). Not “how far one step can wander.” |
| margin | `m` | How wide the Q-interval is for pruning; grows with `r` and Lipschitz numbers |
| discount | `γ = 0.99` | Standard RL discount |
| rewards | `R1`, `R2`, `R=R1+R2` | Two RBF reward components; agents often optimize their sum |
| free transition | `boundary_affected = false` | Step was **not** changed by wall cancel or wall clip |
| Lipschitz `Lr` | sensitivity of reward to state | “If states differ by distance d, rewards differ by at most about `Lr·d`” |
| Lipschitz `Lf` | sensitivity of next-state to state | “If states differ by d, next-states differ by at most about `Lf·d`” |
| Lipschitz `Lq` | sensitivity of Q to state | Same idea for action-values |

**Cell rule:** region is computed from the **current** state `s` before the step.

---

## 1. Environment

### 1.1 One step

Given current state `s` and action `a`:

```text
1. Look up free move vector for this region and action:  Δ = Δ(a, ρ(s))
2. Propose:  s_tilde = s + Δ
3. If s_tilde is outside [0,1]²: cancel the move → s_tilde = s     (boundary cancel)
4. Add region noise:  noisy = s_tilde + ε
5. Clip to the square:  s' = clip(noisy, [0,1]²)                   (boundary clip)
```

- Top half (`y ≥ 0.5`, regions 1–2): **no noise** (`ε = 0`). Deterministic free move.
- Bottom half (`y < 0.5`, regions 3–4): Gaussian noise with size controlled by config `σ`.
- Reward: two RBF potentials (Dollar / Euro / Both) minus living penalty `τ/2`.
- Episodes last at most `H = 200` steps.

The environment also sets flags in `info`:

- `boundary_cancel` — step 3 fired
- `boundary_clip` — step 5 changed the point
- `boundary_affected` — either of the above

### 1.2 Why we care about “free” transitions

`μ̂` is meant to answer: **“In this region, what is the usual free move for this action?”**

Wall cancel/clip is a **separate** constraint. If we average cancelled moves (`δ = 0`) into `μ̂`, we corrupt that answer and create fake variance in deterministic regions.

So:

- **Learning R1/R2 DQNs:** use every transition (walls included).
- **Estimating `μ̂`, `r`, `Lr`, `Lf`, empirical `Lq`:** use only free transitions.
- **Using the estimates later:** walls still apply (`Q_single` does `clip(s+μ̂)`; RA-DQN steps in the real env).

---

## 2. Collect R1 and R2 behavior

For each component reward `Rk` (`k = 1` then `k = 2`):

1. Train an ε-greedy DQN on `Rk` for many steps (default 150k).
2. While training, record every transition `(s, a, s', Rk, ρ)`.
3. Keep a second copy of those transitions with boundary-affected rows removed.
4. Group the free copy by cell `(ρ, a)`.

**DQN learning rule** (also used later for baseline and RA-DQN’s TD loss):

```text
target y = reward + γ · Q_target(s', best_action) · (1 if not done else 0)
loss     = (Q(s,a) − y)²

soft target update:
  Q_target ← (1 − τ)·Q_target + τ·Q
  with τ = 5e-4
```

After both R1 and R2 finish, save greedy rollout videos `r1_policy.mp4` and `r2_policy.mp4`.

---

## 3. Mean step and radius (the part people confuse)

Fix one cell `(ρ, a)`. Look only at free transitions in that cell.

### 3.1 What we measure on each transition

```text
δ = s' − s     # 2D vector: “how far did we move, and in which direction?”
```

### 3.2 Mean step `μ̂`

Average all those `δ` values (R1 and R2 pooled by sample size):

```text
μ̂ = average of δ in this cell
```

**Intuition:** “Taking action `a` in region `ρ` usually moves you by about `μ̂`.”

`Q_single` will later use this as a fixed move: `s' = clip(s + μ̂)`.

### 3.3 Sample std `σ` vs radius `r` (different ideas)

Two different questions:

| Question | Quantity | Rough size |
|----------|----------|------------|
| How much do **single steps** scatter around `μ̂`? | sample std `σ` | does **not** shrink just because you collected more data |
| How sure are we about the **average** `μ̂` itself? | radius `r` | **shrinks** like `σ / √n` as you get more samples |

**Radius is the second one.**

Concrete picture:

```text
True unknown average move:     μ
Our estimate from n samples:   μ̂

We build a Student-t confidence interval for μ:

    μ is likely inside the box   μ̂ ± e

where, per coordinate (x and y):

    e = t · σ_pooled / √n

    t = Student-t critical value at confidence 95% (default), df = n−1
    σ_pooled = pooled sample std of δ from R1 and R2
    n = total free samples in the cell

Then:
    r_vec = (e_x, e_y)
    r     = length of r_vec = sqrt(e_x² + e_y²)
```

So if you say “the average change lies between mean−e and mean+e”, then **`r` is that e, summarized as one number**.

**It is not:**

- a ball around the average next-state E[s′]
- a guarantee that every single step lands in `μ̂ ± r` (that would be closer to using `σ`, or a prediction interval)

**It is:**

- uncertainty about the **mean displacement** `μ̂ = E[s′ − s]`

For a fixed current state `s`, that mean-displacement uncertainty shifts the next state like:

```text
best guess:     s' ≈ s + μ̂
mean-uncertain: s' ≈ s + (something within about r of μ̂)
```

### 3.4 Pooling R1 and R2 (detail)

```text
n1, μ1, σ1   from free R1 samples in the cell
n2, μ2, σ2   from free R2 samples in the cell

n  = n1 + n2
μ̂ = (n1·μ1 + n2·μ2) / n

σ_pooled² = [
    (n1−1)·σ1² + (n2−1)·σ2²
  + n1·(μ1 − μ̂)² + n2·(μ2 − μ̂)²
] / (n − 1)          # done separately for x and for y

then e and r as above
```

If the free dynamics are deterministic, `σ ≈ 0`, so `r ≈ 0`.

---

## 4. Lipschitz numbers

Still using **free** transitions in each cell.

### 4.1 Pairwise ratios

Pick many unique pairs of samples `(i, j)` with the same action. Let `d = distance(s_i, s_j)`.

```text
reward sensitivity sample:   |R(s_i) − R(s_j)| / d
dynamics sensitivity sample: distance(s'_i, s'_j) / d
Q sensitivity sample:        |Q(s_i, a) − Q(s_j, a)| / d
```

Take a 10% trimmed mean of each list → `Lr`, `Lf`, `Lq_emp` for that behavior.

### 4.2 Combine R1 and R2

```text
Lr_sum     = Lr_from_R1 + Lr_from_R2
Lf         = max(Lf_from_R1, Lf_from_R2)
Lq_emp_sum = Lq_emp_from_R1 + Lq_emp_from_R2
```

### 4.3 Theoretical Lq (Bellman)

```text
Lq_th = Lr_sum / (1 − γ · Lf)
```

This formula needs `γ · Lf < 1`. If not, theoretical Lq is undefined and the run **stops** (usually `σ` made estimated `Lf` too large).

**Meaning in one sentence:** `Lq_th` is a bound on how fast optimal Q can change with state, derived from reward and dynamics Lipschitz numbers.

---

## 5. Center model `Q_single`

Train a Q-network on the **combined** reward `R = R1 + R2`, but with **mean free dynamics**:

```text
s' = clip(s + μ̂_{ρ(s), a})
```

- No stochastic noise in this training loop.
- Walls are handled only by `clip(...)`.
- This network is frozen afterward.
- It is used only to score actions for pruning, not as the online controller.

---

## 6. Margin and action pruning

### 6.1 Margin

For each cell `(ρ, a)`:

```text
m = (Lr_sum + γ · Lq) · r / (1 − γ)
```

- Theoretical pruning: `Lq = Lq_th`
- Empirical pruning: `Lq = Lq_emp_sum`

Same formula in both cases. Only which `Lq` is plugged in changes.

**Intuition:** larger mean-step uncertainty `r`, or larger Lipschitz numbers → wider margin `m` → harder to certify that an action is dominated → less pruning.

### 6.2 Allowed-action test

At state `s`, region `ρ = ρ(s)`:

```text
For each action a:
  U_a = Q_single(s, a) + m_{ρ,a}     # upper end of interval
  L_a = Q_single(s, a) − m_{ρ,a}     # lower end
  (then clamp both to [-100, 100] by default)

Keep action a if:
  U_a ≥ (maximum over a' of L_{a'}) − ε

If that would keep nothing, keep all actions.
```

**Intuition:** if even the optimistic value of action `a` is worse than the pessimistic value of the best alternative, drop `a`.

Heatmaps plot how many actions survive on a grid of states.

---

## 7. Train the agents

### 7.1 Baseline DQN

ε-greedy DQN on `R = R1 + R2`. No pruning. Uses the live stochastic env (with walls and noise).

### 7.2 RA-DQN (theoretical or empirical)

Same TD learning as DQN, plus:

1. When acting: choose only among **allowed** actions (from §6).
2. When bootstrapping: `best next action` is also chosen only among allowed actions at `s'`.
3. Extra ranking loss that pushes pruned actions below the best allowed action.

Theoretical RA-DQN uses theoretical margins; empirical RA-DQN uses empirical margins.

Default training: several independent seeds (`n_runs`, often 5), then average curves.

---

## 8. Evaluation and plots

- Periodic eval with a fixed eval seed.
- Save mean return curves and multi-seed arrays.
- Plot mean ± 95% Student-t CI across seeds.
- Save heatmaps and CSV tables of Lipschitz / radius values.

---

## 9. End-to-end pipeline (one σ)

```text
1.  Train R1 DQN + collect transitions
2.  Train R2 DQN + collect transitions
3.  Drop boundary-affected rows for statistics
4.  For each cell: estimate μ̂ and r
5.  For each cell: estimate Lr, Lf, Lq_emp; compute Lq_th
6.  Save r1_policy.mp4, r2_policy.mp4
7.  Train Q_single on clip(s + μ̂)
8.  Build theoretical + empirical pruning heatmaps / tables
9.  Train baseline DQN (n_runs seeds)
10. Train theoretical RA-DQN (n_runs seeds)
11. Train empirical RA-DQN (n_runs seeds)
12. Plot returns
13. Save dqn_policy.mp4, ra_dqn_theoretical_policy.mp4, ra_dqn_empirical_policy.mp4
```

Useful flags:

```text
--stop-after videos   stop after steps 1–6
--resume              continue an existing output folder
--runs N              number of agent seeds for steps 9–11
--sigma VALUE         dynamics noise scale for bottom regions
```

---

## 10. Short checklist of common confusions

1. **`r` is not “step size.”** Step size is `|μ̂|`. `r` is uncertainty of the **average** step.
2. **`r` is not “noise σ.”** Noise affects how large sample std is; `r` also depends on how many samples you have (`/√n`).
3. **`r` is about `δ = s'−s`, not about a cloud around average `s'`.**
4. **Free-step stats ignore wall hits; the live env still has walls.**
5. **Empirical vs theoretical RA-DQN** share the same margin formula; they differ in which `Lq` they use.
6. **If `γ·Lf ≥ 1`, theoretical Lq cannot be formed and the run fails on purpose.**

"""Train separate R1/R2 behavior DQNs and rebuild sigma-specific pruning inputs.

From free (s, a, s') data per (category, action):
  delta_mean  = average free displacement δ = s' − s   → used by Q_single
  pruning_radius = ‖Student-t half-width‖₂ → used by RA-DQN

Q_single trains on s' = clip(s + delta_mean).
RA margins use pruning_radius around that mean next state.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_determinism,
    apply_resolved_deterministic_sigma_scale,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    format_training_args,
    load_config,
    resolve_determinism,
    resolve_deterministic_sigma_scale,
    resolve_sigma,
)
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
from dollar_euro_lipschitz.layout import (
    N_TILES,
    discover_neighbor_pairs,
    discover_tile_neighbor_pairs,
    layout_summary,
    tile_ids_from_states,
)
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.replay import ReplayBuffer
from dollar_euro_lipschitz.bounds import (
    DEFAULT_CONFIDENCE_LEVEL,
    require_finite_bellman_lq,
    student_t_half_width,
    student_t_radius_l2,
)


class ComponentAgent:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        for parameter in self.qt.parameters():
            parameter.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.mem = ReplayBuffer(args.buffer_size)
        self.update_clock = 0

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(4)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            return int(self.q(state_tensor).argmax(1).item())

    def observe(self, state, action, reward, next_state, done):
        self.mem.add(state, action, reward, next_state, done)
        self.update_clock = (self.update_clock + 1) % self.args.update_every
        if self.update_clock or len(self.mem) <= self.args.batch_size:
            return
        states, actions, rewards, next_states, dones = self.mem.sample(self.args.batch_size, self.device)
        with torch.no_grad():
            best = self.q(next_states).argmax(1, keepdim=True)
            target = rewards + self.args.gamma * self.qt(next_states).gather(1, best) * (1.0 - dones)
        loss = F.mse_loss(self.q(states).gather(1, actions), target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        for target_param, local_param in zip(self.qt.parameters(), self.q.parameters()):
            target_param.data.lerp_(local_param.data, self.args.tau)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_component(args, environment, reward_index, seed, device):
    set_seed(seed)
    env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False, seed=seed, **environment)
    if abs(float(env.sigma) - float(environment["sigma"])) > 1e-18:
        raise RuntimeError(
            f"Environment ignored requested sigma={environment['sigma']:.17g}; env.sigma={env.sigma:.17g}"
        )
    include_boundary = bool(getattr(args, "include_boundary_transitions", False))
    agent = ComponentAgent(args, device)
    state, _ = env.reset(seed=seed)
    epsilon = args.eps_start
    episode_step = 0
    episode = 0
    n_steps = int(args.steps)
    states = np.empty((n_steps, 2), dtype=np.float32)
    next_states = np.empty((n_steps, 2), dtype=np.float32)
    rewards = np.empty(n_steps, dtype=np.float32)
    regions = np.empty(n_steps, dtype=np.int16)
    actions = np.empty(n_steps, dtype=np.int8)
    free_mask = np.empty(n_steps, dtype=np.bool_)
    started = time.perf_counter()
    for step in range(n_steps):
        action = agent.act(state, epsilon)
        next_state, reward_vec, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        reward = float(reward_vec[reward_index])
        states[step] = state
        next_states[step] = next_state
        rewards[step] = reward
        regions[step] = int(info["region"])
        actions[step] = action
        # Keep wall cancel/clip out of delta_s / Lipschitz estimates unless radial parity is requested.
        free_mask[step] = include_boundary or not bool(info.get("boundary_affected", False))
        agent.observe(state, action, reward, next_state, done)
        state = next_state
        episode_step += 1
        if done or episode_step >= args.max_t:
            episode += 1
            state, _ = env.reset(seed=seed + episode)
            episode_step = 0
            epsilon = max(args.eps_end, epsilon * args.eps_decay)
        if (step + 1) % max(1, args.log_every) == 0:
            kept = int(free_mask[: step + 1].sum())
            print(
                f"R{reward_index + 1} step={step + 1} episodes={episode} "
                f"epsilon={epsilon:.4f} free_for_stats={kept}/{step + 1}"
            )
    env.close()
    n_free = int(free_mask.sum())
    print(
        f"R{reward_index + 1} boundary filter: kept {n_free}/{n_steps} "
        f"({100.0 * n_free / max(n_steps, 1):.2f}%) for delta_s/Lipschitz"
    )
    records = _group_transitions(states, next_states, rewards, regions, actions, free_mask)
    tile_records = _group_transitions_by_tile(states, next_states, rewards, actions, free_mask)
    next_tile_records = _group_transitions_by_next_tile(
        states, next_states, rewards, free_mask
    )
    return agent, records, tile_records, next_tile_records, time.perf_counter() - started


def _group_transitions(states, next_states, rewards, regions, actions, free_mask=None):
    """Group transitions for bounds/Lipschitz. Optionally keep only non-boundary-affected steps."""
    if free_mask is None:
        free_mask = np.ones(len(states), dtype=np.bool_)
    records = {}
    for region in range(1, 6):
        region_mask = (regions == region) & free_mask
        for action in range(4):
            mask = region_mask & (actions == action)
            records[(region, action)] = (
                states[mask],
                next_states[mask],
                rewards[mask],
            )
    return records


def _group_transitions_by_tile(states, next_states, rewards, actions, free_mask=None):
    """Group free transitions by (tile_id, action) using the pre-step state position.

    Kept for source-tile dynamics grouping / diagnostics.  Reward estimation
    does NOT use this; see ``_group_transitions_by_next_tile``.
    """
    if free_mask is None:
        free_mask = np.ones(len(states), dtype=np.bool_)
    tile_ids = tile_ids_from_states(states)
    records = {}
    for tile_id in range(N_TILES):
        tile_mask = (tile_ids == tile_id) & free_mask
        for action in range(4):
            mask = tile_mask & (actions == action)
            records[(tile_id, action)] = (
                states[mask],
                next_states[mask],
                rewards[mask],
            )
    return records


def _group_transitions_by_next_tile(states, next_states, rewards, free_mask=None):
    """Group free transitions by NEXT-state tile, pooled over action, for Lr.

    Library: reward R(s') lives in next-state space, so ordinary reward
    Lipschitz Lr(Ti) must pool every next state that lands in tile Ti,
    regardless of the source category or the action that produced it.
    """
    if free_mask is None:
        free_mask = np.ones(len(states), dtype=np.bool_)
    next_tile_ids = tile_ids_from_states(next_states)
    records = {}
    for tile_id in range(N_TILES):
        mask = (next_tile_ids == tile_id) & free_mask
        records[tile_id] = (
            states[mask],
            next_states[mask],
            rewards[mask],
        )
    return records


def trimmed_mean(values, ratio=0.1):
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    trim = int(values.size * ratio)
    core = values[trim : values.size - trim] if values.size > 2 * trim else values
    return float(np.mean(core)) if core.size else 0.0


def unique_unordered_pairs(n, max_pairs, rng):
    """Unique unordered pairs, matching two_reward_q_learning_fixed_action_lipschitz.sample_pairs."""
    n = int(n)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    total = n * (n - 1) // 2
    if max_pairs is None or total <= int(max_pairs):
        left, right = np.triu_indices(n, k=1)
        return left.astype(np.int64, copy=False), right.astype(np.int64, copy=False)
    pairs = set()
    while len(pairs) < int(max_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    left = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    right = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    return left, right


def reward_lipschitz_per_tile(next_tile_records, model, device, max_pairs, seed):
    """Compute ordinary NEXT-STATE-TILE reward Lipschitz Lr per behavior.

    ``next_tile_records`` maps tile_id -> (states, next_states, rewards),
    action-pooled.  For each next tile Ti:
      Lr ratio = |R(s1') - R(s2')| / ||s1' - s2'||
    with validity ||s1' - s2'|| > 1e-12 (NEXT distance, not source).
    Returns dict tile_id -> {"Lr": ..., "n": ..., "pairs_used": ...}.
    """
    report = {}
    for tile_id, (states, next_states, rewards) in sorted(next_tile_records.items()):
        n_rows = int(states.shape[0])
        if n_rows == 0:
            report[tile_id] = {"n": 0, "Lr": 0.0, "pairs_used": 0}
            continue
        next64 = next_states.astype(np.float64, copy=False)
        rewards64 = rewards.astype(np.float64, copy=False)
        rng = np.random.default_rng(seed + 17 * n_rows + 31 * tile_id)
        left, right = unique_unordered_pairs(n_rows, max_pairs, rng)
        if left.size == 0:
            report[tile_id] = {"n": n_rows, "Lr": 0.0, "pairs_used": 0}
            continue
        distances_nxt = np.linalg.norm(next64[left] - next64[right], axis=1)  # ||s1' - s2'||
        valid = distances_nxt > 1e-12
        distances_nxt = distances_nxt[valid]
        if distances_nxt.size == 0:
            report[tile_id] = {"n": n_rows, "Lr": 0.0, "pairs_used": 0}
            continue
        left, right = left[valid], right[valid]
        report[tile_id] = {
            "n": n_rows,
            "Lr": trimmed_mean(np.abs(rewards64[left] - rewards64[right]) / distances_nxt),
            "pairs_used": int(valid.sum()),
        }
    return report


def dynamics_lipschitz_per_category(records, model, device, max_pairs, seed):
    """Compute ordinary SOURCE-category/action dynamics Lipschitz Lf per behavior.

    ``records`` maps (category, action) -> (states, next_states, rewards).
    For each group:
      Lf ratio = ||s1' - s2'|| / ||s1 - s2||
    with validity ||s1 - s2|| > 1e-12 (SOURCE distance).
    Returns dict (category, action) -> {"Lf": ..., "n": ..., "pairs_used": ...}.
    """
    report = {}
    for (region, action), (states, next_states, rewards) in sorted(records.items()):
        n_rows = int(states.shape[0])
        if n_rows == 0:
            report[(region, action)] = {"n": 0, "Lf": 0.0, "pairs_used": 0}
            continue
        states64 = states.astype(np.float64, copy=False)
        next64 = next_states.astype(np.float64, copy=False)
        rng = np.random.default_rng(seed + 17 * n_rows + 31 * action)
        left, right = unique_unordered_pairs(n_rows, max_pairs, rng)
        if left.size == 0:
            report[(region, action)] = {"n": n_rows, "Lf": 0.0, "pairs_used": 0}
            continue
        distances_src = np.linalg.norm(states64[left] - states64[right], axis=1)  # ||s1 - s2||
        valid = distances_src > 1e-12
        if not valid.any():
            report[(region, action)] = {"n": n_rows, "Lf": 0.0, "pairs_used": 0}
            continue
        distances_src = distances_src[valid]
        left, right = left[valid], right[valid]
        report[(region, action)] = {
            "n": n_rows,
            "Lf": trimmed_mean(np.linalg.norm(next64[left] - next64[right], axis=1) / distances_src),
            "pairs_used": int(valid.sum()),
        }
    return report


def empirical_q_lipschitz_per_tile_action(source_tile_records, model, device, max_pairs, seed):
    """Diagnostic empirical Lq per (SOURCE tile, action) for one behavior.

    Lq ratio = |Q(s1,a) - Q(s2,a)| / ||s1 - s2||  (empirical, diagnostic only).
    Kept for artifact parity; production Lq is theoretical.
    """
    report = {}
    model.eval()
    for (tile_id, action), (states, next_states, rewards) in sorted(source_tile_records.items()):
        n_rows = int(states.shape[0])
        if n_rows == 0:
            report[(tile_id, action)] = {"n": 0, "Lq": 0.0, "pairs_used": 0}
            continue
        states64 = states.astype(np.float64, copy=False)
        with torch.inference_mode():
            q_values = (
                model(torch.as_tensor(states, dtype=torch.float32, device=device))[:, action]
                .detach().cpu().numpy().astype(np.float64, copy=False)
            )
        rng = np.random.default_rng(seed + 19 * n_rows + 37 * tile_id + 41 * action)
        left, right = unique_unordered_pairs(n_rows, max_pairs, rng)
        if left.size == 0:
            report[(tile_id, action)] = {"n": n_rows, "Lq": 0.0, "pairs_used": 0}
            continue
        distances_src = np.linalg.norm(states64[left] - states64[right], axis=1)
        valid = distances_src > 1e-12
        if not valid.any():
            report[(tile_id, action)] = {"n": n_rows, "Lq": 0.0, "pairs_used": 0}
            continue
        report[(tile_id, action)] = {
            "n": n_rows,
            "Lq": trimmed_mean(np.abs(q_values[left[valid]] - q_values[right[valid]]) / distances_src[valid]),
            "pairs_used": int(valid.sum()),
        }
    return report


def component_constants(records, model, device, max_pairs, seed):
    """Legacy combined per-(category, action) constants: Lr/Lf/Lq.

    Kept only for the diagnostic script and backwards compatibility of the
    per-category empirical tables.  Production reward Lr now comes from
    ``reward_lipschitz_per_tile`` (next-tile pooled); production Lf from
    ``dynamics_lipschitz_per_category``.
    """
    report = {}
    model.eval()
    for (region, action), (states, next_states, rewards) in sorted(records.items()):
        n_rows = int(states.shape[0])
        if n_rows == 0:
            report[(region, action)] = {"n": 0, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0, "pairs_used": 0}
            continue
        states64 = states.astype(np.float64, copy=False)
        next64 = next_states.astype(np.float64, copy=False)
        rewards64 = rewards.astype(np.float64, copy=False)
        with torch.inference_mode():
            q_values = model(
                torch.as_tensor(states, dtype=torch.float32, device=device)
            )[:, action].detach().cpu().numpy().astype(np.float64, copy=False)
        rng = np.random.default_rng(seed + 17 * n_rows + 31 * action)
        left, right = unique_unordered_pairs(n_rows, max_pairs, rng)
        if left.size == 0:
            report[(region, action)] = {"n": n_rows, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0, "pairs_used": 0}
            continue
        distances_src = np.linalg.norm(states64[left] - states64[right], axis=1)  # ||s1 - s2||
        distances_nxt = np.linalg.norm(next64[left] - next64[right], axis=1)     # ||s1' - s2'||
        valid = distances_src > 1e-12
        distances_src = distances_src[valid]
        distances_nxt = distances_nxt[valid]
        left, right = left[valid], right[valid]
        report[(region, action)] = {
            "n": n_rows,
            "Lr": trimmed_mean(np.abs(rewards64[left] - rewards64[right]) / distances_nxt),
            "Lf": trimmed_mean(np.linalg.norm(next64[left] - next64[right], axis=1) / distances_src),
            "Lq": trimmed_mean(np.abs(q_values[left] - q_values[right]) / distances_src),
            "pairs_used": int(valid.sum()),
        }
    return report


def _behavior_delta_stats(states, next_states):
    n = int(states.shape[0])
    if n == 0:
        return {"n": 0, "delta_mean": np.zeros(2, dtype=np.float64), "delta_std": np.zeros(2, dtype=np.float64)}
    deltas = (next_states - states).astype(np.float64, copy=False)
    mean = deltas.mean(axis=0)
    std = deltas.std(axis=0, ddof=1) if n >= 2 else np.zeros(2, dtype=np.float64)
    return {"n": n, "delta_mean": mean, "delta_std": std}


def combine_two_behavior_stats(stat_q1, stat_q2, confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """Pool R1/R2 free transitions: mean for Q_single, Student-t radius for RA."""
    n1 = int(stat_q1["n"])
    n2 = int(stat_q2["n"])
    mu1 = np.asarray(stat_q1["delta_mean"], dtype=np.float64).reshape(-1)
    mu2 = np.asarray(stat_q2["delta_mean"], dtype=np.float64).reshape(-1)
    s1 = np.asarray(stat_q1["delta_std"], dtype=np.float64).reshape(-1)
    s2 = np.asarray(stat_q2["delta_std"], dtype=np.float64).reshape(-1)
    if n1 < 2:
        s1 = np.zeros_like(mu1)
    if n2 < 2:
        s2 = np.zeros_like(mu2)
    n_total = n1 + n2
    if n_total == 0:
        delta_mean = np.zeros(2, dtype=np.float64)
        delta_std = np.zeros(2, dtype=np.float64)
    else:
        delta_mean = (n1 * mu1 + n2 * mu2) / n_total
        if n_total > 1:
            pooled_var = (
                (n1 - 1) * (s1 ** 2)
                + (n2 - 1) * (s2 ** 2)
                + n1 * ((mu1 - delta_mean) ** 2)
                + n2 * ((mu2 - delta_mean) ** 2)
            ) / (n_total - 1)
            delta_std = np.sqrt(np.maximum(pooled_var, 0.0))
        else:
            delta_std = np.zeros(2, dtype=np.float64)
    student_t_hw = student_t_half_width(
        delta_std, n_total, confidence_level
    )
    pruning_radius = student_t_radius_l2(student_t_hw)
    return {
        "n_total": int(n_total),
        "n_q1_behavior": n1,
        "n_q2_behavior": n2,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "student_t_half_width": student_t_hw,
        "pruning_radius": pruning_radius,
    }


def build_bounds(q1_records, q2_records, args, sigma, category_counts=None):
    output = {
        "source_json": "generated directly by scripts/train_reward_components.py",
        "metadata": {
            "dynamics_sigma": sigma,
            "sigma_source": "explicit runtime argument/config",
            "deterministic_sigma_scale": float(args.deterministic_sigma_scale),
            "deterministic_sigma": float(args.deterministic_sigma_scale) * float(sigma),
            "behavior_policies": ["R1 epsilon-greedy DQN", "R2 epsilon-greedy DQN"],
            "seed_r1": args.seed,
            "seed_r2": args.seed + args.r2_seed_offset,
            "delta_mean_semantics": (
                "Pooled mean free displacement delta = s' - s per (category, action). "
                "Q_single trains on s' = clip(s + delta_mean)."
            ),
            "delta_std_semantics": "Pooled per-axis sample std of delta around delta_mean.",
            "student_t_half_width_semantics": (
                "Student-t interval half-width per axis: "
                "t * delta_std * sqrt(1 + 1/n) at confidence_level."
            ),
            "pruning_radius_semantics": (
                "r = L2 of student_t_half_width; RA-DQN margins use this as the "
                "radius of next s' around s + delta_mean."
            ),
            "delta_estimation": (
                "Free region-action steps only (exclude boundary cancel/clip when configured)."
            ),
        },
        "config": {
            "confidence_level": args.confidence_level,
            "exclude_boundary_affected_transitions": not bool(
                getattr(args, "include_boundary_transitions", False)
            ),
        },
        "by_region_action": {},
    }
    counts = category_counts or {}
    for region in range(1, 6):
        output["by_region_action"][str(region)] = {}
        tiles_for_cat = int(counts.get(region, counts.get(str(region), 1)))
        for action in range(4):
            rows1 = q1_records.get((region, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            rows2 = q2_records.get((region, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            s1, ns1, _r1 = rows1
            s2, ns2, _r2 = rows2
            if tiles_for_cat <= 0:
                output["by_region_action"][str(region)][str(action)] = {
                    "n_total": 0,
                    "n_q1_behavior": 0,
                    "n_q2_behavior": 0,
                    "delta_mean": [0.0, 0.0],
                    "delta_std": [0.0, 0.0],
                    "student_t_half_width": [0.0, 0.0],
                    "pruning_radius": 0.0,
                    "unused_category": True,
                }
                continue
            if s1.shape[0] < args.min_cell_samples or s2.shape[0] < args.min_cell_samples:
                raise ValueError(
                    f"Insufficient free (non-boundary) cell ({region}, {action}): "
                    f"R1={s1.shape[0]}, R2={s2.shape[0]}; "
                    f"need at least {args.min_cell_samples} from each behavior after "
                    f"excluding boundary cancel/clip."
                )
            combined = combine_two_behavior_stats(
                _behavior_delta_stats(s1, ns1),
                _behavior_delta_stats(s2, ns2),
                args.confidence_level,
            )
            output["by_region_action"][str(region)][str(action)] = {
                "n_total": combined["n_total"],
                "n_q1_behavior": combined["n_q1_behavior"],
                "n_q2_behavior": combined["n_q2_behavior"],
                "delta_mean": combined["delta_mean"].tolist(),
                "delta_std": combined["delta_std"].tolist(),
                "student_t_half_width": combined["student_t_half_width"].tolist(),
                "pruning_radius": combined["pruning_radius"],
            }
    return output


def build_lipschitz(
    tile_reward1,
    tile_reward2,
    category_dynamics1,
    category_dynamics2,
    empirical_q1,
    empirical_q2,
    args,
    sigma,
    tile_map,
):
    """Ordinary Lipschitz constants (tile x action) from the corrected domains.

    - Lr:  next-state tile, independent of action.  Lr(Ti) is replicated across
      all four action rows of that tile (recorded as ``lr_action_dependence``
      false), because reward is R(s').
    - Lf:  source category + action, Lf(Ci,a) = max(Lf1, Lf2), shared by all
      tiles in the category.
    - LQ_bellman_bound = Lr(Ti) * Lf(Ci,a) / (1 - gamma * Lf(Ci,a)).
    - Lq_empirical_sum retained as a diagnostic only (source tile x action).

    Requires gamma * Lf_sum < 1; invalid rows (denominator <= 0) are preserved
    with a null value and rejected downstream by ``require_finite_bellman_lq``.
    """
    category_lf = {}
    for region in range(1, 6):
        for action in range(4):
            first = category_dynamics1.get((region, action), {"n": 0, "Lf": 0.0, "pairs_used": 0})
            second = category_dynamics2.get((region, action), {"n": 0, "Lf": 0.0, "pairs_used": 0})
            category_lf[(region, action)] = {
                "Lf1": float(first["Lf"]),
                "Lf2": float(second["Lf"]),
                "Lf_sum": max(float(first["Lf"]), float(second["Lf"])),
                "n_r1": int(first["n"]),
                "n_r2": int(second["n"]),
            }

    rows = []
    min_samples = int(args.min_cell_samples)
    for tile_id in range(N_TILES):
        region = int(tile_map[tile_id])
        tr1 = tile_reward1.get(tile_id, {"n": 0, "Lr": 0.0, "pairs_used": 0})
        tr2 = tile_reward2.get(tile_id, {"n": 0, "Lr": 0.0, "pairs_used": 0})
        lr1 = float(tr1["Lr"])
        lr2 = float(tr2["Lr"])
        lr_sum = lr1 + lr2
        lr_n1 = int(tr1["n"])
        lr_n2 = int(tr2["n"])
        for action in range(4):
            lf_row = category_lf[(region, action)]
            lf_sum = float(lf_row["Lf_sum"])
            # Diagnostic empirical Lq from the SOURCE tile/action grouping.
            eq1 = empirical_q1.get((tile_id, action), {"n": 0, "Lq": 0.0, "pairs_used": 0})
            eq2 = empirical_q2.get((tile_id, action), {"n": 0, "Lq": 0.0, "pairs_used": 0})
            lq_emp = float(eq1["Lq"]) + float(eq2["Lq"])
            x = args.gamma * lf_sum
            denominator = 1.0 - x
            if denominator > 0:
                bellman = lr_sum * lf_sum / denominator
            else:
                bellman = None
            rows.append(
                {
                    "tile": tile_id,
                    "region": region,
                    "action": action,
                    "lr_source": "next_tile",
                    "n_r1": lr_n1,
                    "n_r2": lr_n2,
                    "Lr1": lr1,
                    "Lr2": lr2,
                    "Lr_sum": lr_sum,
                    "Lr_action_dependence": False,
                    "Lf1": lf_row["Lf1"],
                    "Lf2": lf_row["Lf2"],
                    "Lf_sum": lf_sum,
                    "Lq1": float(eq1["Lq"]),
                    "Lq2": float(eq2["Lq"]),
                    "Lq_empirical_sum": lq_emp,
                    "LQ_bellman_bound": bellman,
                    "pairs_r1": int(tr1.get("pairs_used", 0)),
                    "pairs_r2": int(tr2.get("pairs_used", 0)),
                }
            )
    require_finite_bellman_lq(rows, args.gamma, path=f"sigma={sigma:.17g} (post R1/R2)")
    return {
        "source": "scripts/train_reward_components.py",
        "gamma": args.gamma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": float(args.deterministic_sigma_scale),
        "deterministic_sigma": float(args.deterministic_sigma_scale) * float(sigma),
        "Lr_definition": "|R(s1')-R(s2')| / ||s1'-s2'||",
        "Lr_grouping": "next_state_tile",
        "Lr_action_dependence": False,
        "Lf_definition": "||s1'-s2'|| / ||s1-s2||",
        "Lf_grouping": "source_category_action",
        "Lq_definition": "Lr*Lf/(1-gamma*Lf)",
        "method_notes": {
            "Lr": (
                "Per-NEXT-STATE-tile 10% trimmed pairwise ratios on free transitions "
                "|R(s1')-R(s2')|/||s1'-s2'||, pooled over action "
                "(reward is R(s')); Lr_sum = Lr1 + Lr2. Lr replicated across all "
                "action rows of the tile; Lr_action_dependence = false."
            ),
            "Lf": (
                "Per-source-category/action 10% trimmed pairwise dynamics ratios "
                "||s1'-s2'||/||s1-s2||; Lf_sum = max(Lf1, Lf2) from R1/R2 behavior."
            ),
            "LQ_bellman_bound": (
                "LQ_bellman_bound = Lr(Ti) * Lf(Ci,a) / (1 - gamma * Lf(Ci,a)); "
                "requires gamma * Lf_sum < 1."
            ),
            "Lq_empirical_sum": (
                "Diagnostic per-source tile/action empirical |Q-Q|/||s1-s2||; NOT used "
                "for production theoretical bounds."
            ),
            "pruning_radius": (
                "Student-t L2 radius of next free s' around s+delta_mean; "
                "RA-DQN uses the live env"
            ),
            "delta_estimation": (
                "Exclude boundary cancel/clip from delta and Lipschitz sample sets so "
                "deterministic regions can report std≈0 for the free step"
            ),
        },
        "lipschitz_method_version": 3,
        "constants": rows,
    }


def cross_category_constants(set_a, set_b, model, action, device, max_pairs, seed):
    """Cross-category 10% trimmed pairwise Lipschitz ratios for one behavior.

    ``set_a`` / ``set_b`` are ``(states, next_states, rewards)`` free-transition
    arrays for the two categories under ``action``. Pairs are formed with one
    sample from each category, mirroring ``component_constants`` (1e-12 distance
    filter, 10% trimmed mean, capped unique pair count).
    """
    states_a, next_a, rewards_a = set_a
    states_b, next_b, rewards_b = set_b
    n_a = int(states_a.shape[0])
    n_b = int(states_b.shape[0])
    empty = {"n_a": n_a, "n_b": n_b, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0, "pairs_used": 0}
    if n_a == 0 or n_b == 0:
        return empty
    states_a64 = states_a.astype(np.float64, copy=False)
    next_a64 = next_a.astype(np.float64, copy=False)
    rewards_a64 = rewards_a.astype(np.float64, copy=False)
    states_b64 = states_b.astype(np.float64, copy=False)
    next_b64 = next_b.astype(np.float64, copy=False)
    rewards_b64 = rewards_b.astype(np.float64, copy=False)
    with torch.inference_mode():
        qa = (
            model(torch.as_tensor(states_a, dtype=torch.float32, device=device))[:, action]
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
        qb = (
            model(torch.as_tensor(states_b, dtype=torch.float32, device=device))[:, action]
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 41 * action)
    total = n_a * n_b
    if max_pairs is None or total <= int(max_pairs):
        idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
    else:
        idx_a = np.empty(int(max_pairs), dtype=np.int64)
        idx_b = np.empty(int(max_pairs), dtype=np.int64)
        used = set()
        k = 0
        while k < int(max_pairs):
            i = int(rng.integers(0, n_a))
            j = int(rng.integers(0, n_b))
            key = i * n_b + j
            if key in used:
                continue
            used.add(key)
            idx_a[k] = i
            idx_b[k] = j
            k += 1
    distances_src = np.linalg.norm(states_a64[idx_a] - states_b64[idx_b], axis=1)  # ||s1 - s2||
    distances_nxt = np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1)     # ||s1' - s2'||
    valid = distances_src > 1e-12
    distances_src = distances_src[valid]
    distances_nxt = distances_nxt[valid]
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    if idx_a.size == 0:
        return empty
    return {
        "n_a": n_a,
        "n_b": n_b,
        "Lr": trimmed_mean(np.abs(rewards_a64[idx_a] - rewards_b64[idx_b]) / distances_nxt),
        "Lf": trimmed_mean(np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1) / distances_src),
        "Lq": trimmed_mean(np.abs(qa[idx_a] - qb[idx_b]) / distances_src),
        "pairs_used": int(valid.sum()),
    }


def cross_tile_reward_constants(set_a, set_b, max_pairs, seed):
    """Cross-tile reward Lipschitz Lr for one behavior, paired by NEXT-state tile.

    ``set_a`` / ``set_b`` are ``(states, next_states, rewards)`` arrays whose
    NEXT states lie in two physically neighboring spatial tiles Ti / Tj.  Source
    category and action are irrelevant for the reward function R(s').  Pairs are
    formed with one NEXT-state sample in Ti and one in Tj.

      Lr ratio = |R(s1') - R(s2')| / ||s1' - s2'||
      validity  ||s1' - s2'|| > 1e-12   (next-state distance)
    """
    states_a, next_a, rewards_a = set_a
    states_b, next_b, rewards_b = set_b
    n_a = int(states_a.shape[0])
    n_b = int(states_b.shape[0])
    empty = {"n_a": n_a, "n_b": n_b, "Lr": 0.0, "pairs_used": 0}
    if n_a == 0 or n_b == 0:
        return empty
    next_a64 = next_a.astype(np.float64, copy=False)
    next_b64 = next_b.astype(np.float64, copy=False)
    rewards_a64 = rewards_a.astype(np.float64, copy=False)
    rewards_b64 = rewards_b.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 43 * max(min(n_a, n_b), 0))
    total = n_a * n_b
    if max_pairs is None or total <= int(max_pairs):
        idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
    else:
        n_pairs = int(max_pairs)
        idx_a = np.empty(n_pairs, dtype=np.int64)
        idx_b = np.empty(n_pairs, dtype=np.int64)
        # Unique flat pair IDs without replacement (O(max_pairs) memory).
        flat = rng.choice(total, size=n_pairs, replace=False)
        idx_a = flat // n_b
        idx_b = flat % n_b
    distances_nxt = np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1)  # ||s1' - s2'||
    valid = distances_nxt > 1e-12
    distances_nxt = distances_nxt[valid]
    if distances_nxt.size == 0:
        return empty
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    return {
        "n_a": n_a,
        "n_b": n_b,
        "Lr": trimmed_mean(np.abs(rewards_a64[idx_a] - rewards_b64[idx_b]) / distances_nxt),
        "pairs_used": int(valid.sum()),
    }


def cross_category_dynamics_constants(set_a, set_b, max_pairs, seed):
    """Cross-category dynamics Lipschitz Lf for one behavior, by SOURCE category.

    ``set_a`` / ``set_b`` are ``(states, next_states, rewards)`` free-transition
    arrays for two neighboring dynamics categories Ci / Cj under the SAME action.
    Pairs are formed with one source sample in Ci and one in Cj.

      Lf ratio = ||s1' - s2'|| / ||s1 - s2||
      validity  ||s1 - s2|| > 1e-12   (source distance)
    """
    states_a, next_a, _rewards_a = set_a
    states_b, next_b, _rewards_b = set_b
    n_a = int(states_a.shape[0])
    n_b = int(states_b.shape[0])
    empty = {"n_a": n_a, "n_b": n_b, "Lf": 0.0, "pairs_used": 0}
    if n_a == 0 or n_b == 0:
        return empty
    states_a64 = states_a.astype(np.float64, copy=False)
    next_a64 = next_a.astype(np.float64, copy=False)
    states_b64 = states_b.astype(np.float64, copy=False)
    next_b64 = next_b.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 41 * max(min(n_a, n_b), 0))
    total = n_a * n_b
    if max_pairs is None or total <= int(max_pairs):
        idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
    else:
        n_pairs = int(max_pairs)
        flat = rng.choice(total, size=n_pairs, replace=False)
        idx_a = flat // n_b
        idx_b = flat % n_b
    distances_src = np.linalg.norm(states_a64[idx_a] - states_b64[idx_b], axis=1)  # ||s1 - s2||
    valid = distances_src > 1e-12
    distances_src = distances_src[valid]
    if distances_src.size == 0:
        return empty
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    return {
        "n_a": n_a,
        "n_b": n_b,
        "Lf": trimmed_mean(np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1) / distances_src),
        "pairs_used": int(valid.sum()),
    }


def compute_noise_action_ratio(env):
    """Compute noise magnitude relative to deterministic action displacement.

    Uses the ACTUAL environment configuration (base step size, action vectors,
    per-category action_scale, and per-category noise_cov). Category 5's
    noise_cov already encodes the effective cov = (sigma*det_sigma_scale)^2.

    For each category and action:
      deterministic_displacement = action_vector * step_size * action_scale[category]
      move_norm = ||deterministic_displacement||_2
      noise_std_vector = sqrt(diag(noise_cov[category]))
      noise_std_norm = ||noise_std_vector||_2
      noise_percent = 100 * noise_std_norm / move_norm
    """
    from dollar_euro_lipschitz.layout import CATEGORY_IDS

    regions = env.terrain_config.get("regions", {})
    dynamics = env.terrain_config.get("dynamics", {})
    step_size = float(dynamics.get("base_step_size", 0.04))
    action_vectors = np.asarray(dynamics.get("action_vectors", [
        [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]
    ]), dtype=np.float64)

    sigma = float(env.sigma)
    det_sigma_scale = float(env.deterministic_sigma_scale)
    determinism = float(env.determinism)

    categories = []
    all_percentages = []

    for cat in CATEGORY_IDS:
        spec = regions.get(str(cat)) or regions.get(cat)
        if spec is None:
            continue
        action_scale = np.asarray(spec["action_scale"], dtype=np.float64)
        noise_cov = np.asarray(spec["noise_cov"], dtype=np.float64)

        noise_std_vector = np.sqrt(np.diag(noise_cov))
        noise_std_norm = float(np.linalg.norm(noise_std_vector))

        cat_actions = []
        for action in range(4):
            det_disp = action_vectors[action] * step_size * action_scale
            move_norm = float(np.linalg.norm(det_disp))
            noise_pct = 100.0 * noise_std_norm / move_norm if move_norm > 1e-18 else 0.0
            cat_actions.append({
                "action": action,
                "deterministic_displacement": det_disp.tolist(),
                "move_norm": move_norm,
                "noise_percent": noise_pct,
            })
            all_percentages.append((noise_pct, cat, action))

        categories.append({
            "category": cat,
            "noise_std_vector": noise_std_vector.tolist(),
            "noise_std_norm": noise_std_norm,
            "actions": cat_actions,
        })

    if all_percentages:
        min_pct, min_cat, min_act = min(all_percentages, key=lambda t: t[0])
        max_pct, max_cat, max_act = max(all_percentages, key=lambda t: t[0])
    else:
        min_pct = max_pct = 0.0
        min_cat = max_cat = 1
        min_act = max_act = 0

    payload = {
        "sigma": float(sigma),
        "determinism": determinism,
        "deterministic_sigma_scale": det_sigma_scale,
        "deterministic_sigma": det_sigma_scale * sigma,
        "step_size": step_size,
        "definition": {
            "move_norm": "||action_vector * step_size * action_scale[category]||_2",
            "noise_norm": "||sqrt(diag(noise_cov[category]))||_2",
            "noise_percent": "100 * noise_norm / move_norm",
        },
        "categories": categories,
        "summary": {
            "minimum_noise_percent": min_pct,
            "minimum_location": {"category": min_cat, "action": min_act},
            "maximum_noise_percent": max_pct,
            "maximum_location": {"category": max_cat, "action": max_act},
        },
    }
    return payload


def json_dump(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser)
    parser.add_argument("--steps", type=int, default=150_000, help="Training/collection steps per reward behavior.")
    add_training_arguments(parser)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--min-cell-samples", type=int, default=2)
    parser.add_argument(
        "--include-boundary-transitions",
        action="store_true",
        help="Include boundary cancel/clip steps in delta and Lipschitz stats (matches radial two_reward collector).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.confidence_level < 1:
        parser.error("confidence-level must be in (0, 1)")
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    determinism = resolve_determinism(config, args.determinism)
    apply_resolved_determinism(config, determinism)
    det_sigma_scale = resolve_deterministic_sigma_scale(
        config, args.deterministic_sigma_scale
    )
    apply_resolved_deterministic_sigma_scale(config, det_sigma_scale)
    args.deterministic_sigma_scale = det_sigma_scale
    apply_training_defaults(args, config, role="component")
    environment = env_kwargs(config, sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    data_dir = output_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}; requested_sigma={sigma:.17g}; determinism={determinism:.4g}; "
        f"deterministic_sigma={det_sigma_scale * sigma:.17g} "
        f"(scale={det_sigma_scale:.4g}); "
        f"env_kwargs_sigma={environment['sigma']:.17g}; "
        f"16-tile layout (Lr/Lq per tile, Lf per category); "
        f"env_horizon={environment['horizon']}; {format_training_args(args)}"
    )
    print(f"tile layout: {layout_summary(determinism)['counts']}")
    layout = layout_summary(determinism)
    cat_counts = {int(k): int(v) for k, v in layout["counts"].items()}

    tile_map = np.asarray(layout["tile_map_row_major"], dtype=np.int32)

    q1, records1, tile_records1, next_tile_records1, seconds1 = train_component(
        args, environment, 0, args.seed, device
    )
    q2, records2, tile_records2, next_tile_records2, seconds2 = train_component(
        args, environment, 1, args.seed + args.r2_seed_offset, device
    )
    torch.save(q1.q.state_dict(), models_dir / "q1_dqn_r1.pth")
    torch.save(q2.q.state_dict(), models_dir / "q2_dqn_r2.pth")
    # Ordinary reward Lipschitz Lr: next-state tile, action-pooled.
    tile_reward1 = reward_lipschitz_per_tile(
        next_tile_records1, q1.q, device, args.max_pairs, args.seed
    )
    tile_reward2 = reward_lipschitz_per_tile(
        next_tile_records2, q2.q, device, args.max_pairs, args.seed + 7
    )
    # Ordinary dynamics Lipschitz Lf: source category/action.
    category_dynamics1 = dynamics_lipschitz_per_category(
        records1, q1.q, device, args.max_pairs, args.seed
    )
    category_dynamics2 = dynamics_lipschitz_per_category(
        records2, q2.q, device, args.max_pairs, args.seed + 7
    )
    # Diagnostic empirical Lq from source tile/action grouping.
    empirical_q1 = empirical_q_lipschitz_per_tile_action(
        tile_records1, q1.q, device, args.max_pairs, args.seed + 11
    )
    empirical_q2 = empirical_q_lipschitz_per_tile_action(
        tile_records2, q2.q, device, args.max_pairs, args.seed + 13
    )
    bounds = build_bounds(records1, records2, args, sigma, category_counts=cat_counts)
    constants = build_lipschitz(
        tile_reward1,
        tile_reward2,
        category_dynamics1,
        category_dynamics2,
        empirical_q1,
        empirical_q2,
        args,
        sigma,
        tile_map,
    )
    json_dump(bounds, data_dir / "transition_bounds.json")
    json_dump(constants, data_dir / "lipschitz_constants.json")
    json_dump(layout, data_dir / "tile_layout.json")

    # ---------------------------------------------------------------
    # Cross-tile reward Lipschitz Lr (neighboring NEXT-state spatial tiles)
    # ---------------------------------------------------------------
    neigh_tile_pairs = discover_tile_neighbor_pairs()
    cross_tile_rows = []
    for tile_i, tile_j in neigh_tile_pairs:
        set_a1 = next_tile_records1.get(
            tile_i, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))
        )
        set_b1 = next_tile_records1.get(
            tile_j, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))
        )
        set_a2 = next_tile_records2.get(
            tile_i, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))
        )
        set_b2 = next_tile_records2.get(
            tile_j, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))
        )
        x1 = cross_tile_reward_constants(set_a1, set_b1, args.max_pairs, args.seed + 21)
        x2 = cross_tile_reward_constants(set_a2, set_b2, args.max_pairs, args.seed + 23)
        cross_tile_rows.append({
            "tile_i": tile_i,
            "tile_j": tile_j,
            "region_i": int(tile_map[tile_i]),
            "region_j": int(tile_map[tile_j]),
            "n_i_r1": x1["n_a"],
            "n_j_r1": x1["n_b"],
            "n_i_r2": x2["n_a"],
            "n_j_r2": x2["n_b"],
            "pairs_r1": x1["pairs_used"],
            "pairs_r2": x2["pairs_used"],
            "Lr1": x1["Lr"],
            "Lr2": x2["Lr"],
            "Lr_sum": x1["Lr"] + x2["Lr"],
        })
    cross_tile_artifact = {
        "source": "scripts/train_reward_components.py",
        "sigma": sigma,
        "gamma": float(args.gamma),
        "determinism": determinism,
        "deterministic_sigma_scale": det_sigma_scale,
        "deterministic_sigma": det_sigma_scale * sigma,
        "tile_map": tile_map.tolist(),
        "trim_ratio": 0.1,
        "max_pairs": int(args.max_pairs),
        "neighbor_tile_pairs": [[int(ti), int(tj)] for ti, tj in neigh_tile_pairs],
        "Lr_definition": "|R(s1')-R(s2')| / ||s1'-s2'||",
        "Lr_grouping": "next_state_tile_pair",
        "Lr_action_dependence": False,
        "lipschitz_method_version": 3,
        "constants": cross_tile_rows,
    }
    json_dump(cross_tile_artifact, data_dir / "cross_tile_reward_lipschitz.json")

    # ---------------------------------------------------------------
    # Cross-category dynamics Lipschitz Lf (neighboring source categories)
    # ---------------------------------------------------------------
    neighbor_cat_pairs = discover_neighbor_pairs(tile_map)
    cross_cat_rows = []
    for pair_info in neighbor_cat_pairs:
        ci = pair_info["category_i"]
        cj = pair_info["category_j"]
        for action in range(4):
            set_a1 = records1.get(
                (ci, action),
                (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))),
            )
            set_b1 = records1.get(
                (cj, action),
                (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))),
            )
            set_a2 = records2.get(
                (ci, action),
                (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))),
            )
            set_b2 = records2.get(
                (cj, action),
                (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))),
            )
            x1 = cross_category_dynamics_constants(
                set_a1, set_b1, args.max_pairs, args.seed + 31
            )
            x2 = cross_category_dynamics_constants(
                set_a2, set_b2, args.max_pairs, args.seed + 33
            )
            cross_cat_rows.append({
                "category_i": ci,
                "category_j": cj,
                "action": action,
                "tile_edges": pair_info["tile_edges"],
                "n_i_r1": x1["n_a"],
                "n_j_r1": x1["n_b"],
                "n_i_r2": x2["n_a"],
                "n_j_r2": x2["n_b"],
                "pairs_r1": x1["pairs_used"],
                "pairs_r2": x2["pairs_used"],
                "Lf1": x1["Lf"],
                "Lf2": x2["Lf"],
                "Lf_sum": max(x1["Lf"], x2["Lf"]),
            })
    cross_cat_artifact = {
        "source": "scripts/train_reward_components.py",
        "sigma": sigma,
        "gamma": float(args.gamma),
        "determinism": determinism,
        "deterministic_sigma_scale": det_sigma_scale,
        "deterministic_sigma": det_sigma_scale * sigma,
        "tile_map": tile_map.tolist(),
        "trim_ratio": 0.1,
        "max_pairs": int(args.max_pairs),
        "neighbor_category_pairs": neighbor_cat_pairs,
        "Lf_definition": "||s1'-s2'|| / ||s1-s2||",
        "Lf_grouping": "source_category_action_pair",
        "lipschitz_method_version": 3,
        "constants": cross_cat_rows,
    }
    json_dump(cross_cat_artifact, data_dir / "cross_category_dynamics_lipschitz.json")

    # ---------------------------------------------------------------
    # Noise / action ratio (uses actual environment configuration)
    # ---------------------------------------------------------------
    from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
    _noise_env = ContinuousDollarEuroEnv(
        render_mode=None, auto_render=False, seed=0, **environment
    )
    noise_payload = compute_noise_action_ratio(_noise_env)
    _noise_env.close()
    json_dump(noise_payload, data_dir / "category_noise_action_ratio.json")

    # ---------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------
    manifest = {
        "sigma": sigma,
        "determinism": determinism,
        "deterministic_sigma_scale": det_sigma_scale,
        "deterministic_sigma": det_sigma_scale * sigma,
        "layout": layout,
        "steps_per_behavior": args.steps,
        "seed_r1": args.seed,
        "seed_r2": args.seed + args.r2_seed_offset,
        "seconds_r1": seconds1,
        "seconds_r2": seconds2,
        "device": str(device),
        "gamma": args.gamma,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "target_tau": args.tau,
        "update_every": args.update_every,
        "epsilon_start": args.eps_start,
        "epsilon_end": args.eps_end,
        "epsilon_decay": args.eps_decay,
        "max_steps_per_episode": args.max_t,
        "env_horizon": environment["horizon"],
        "cell_counts": {
            f"{region}:{action}": {
                "r1": int(records1[(region, action)][0].shape[0]),
                "r2": int(records2[(region, action)][0].shape[0]),
            }
            for region in range(1, 6)
            for action in range(4)
        },
        "tile_counts": {
            f"{tile}:{action}": {
                "r1": int(tile_records1[(tile, action)][0].shape[0]),
                "r2": int(tile_records2[(tile, action)][0].shape[0]),
            }
            for tile in range(N_TILES)
            for action in range(4)
        },
        "cross_tile_reward_lipschitz": {
            "artifact": "cross_tile_reward_lipschitz.json",
            "lipschitz_method_version": 3,
            "tile_neighbor_pairs": [list(p) for p in neigh_tile_pairs],
            "n_pairs": len(neigh_tile_pairs),
            "methodology": (
                "10% trimmed pairwise NEXT-state-tile reward ratios "
                "|R(s1')-R(s2')|/||s1'-s2'|| across physically neighboring tiles; "
                "Lr_sum=Lr1+Lr2"
            ),
        },
        "cross_category_dynamics_lipschitz": {
            "artifact": "cross_category_dynamics_lipschitz.json",
            "lipschitz_method_version": 3,
            "neighbor_pairs": [
                {"category_i": p["category_i"], "category_j": p["category_j"],
                 "n_edges": len(p["tile_edges"])}
                for p in neighbor_cat_pairs
            ],
            "n_pairs": len(neighbor_cat_pairs),
            "methodology": (
                "10% trimmed pairwise source-category/action dynamics ratios "
                "||s1'-s2'||/||s1-s2|| across physically neighboring categories; "
                "Lf_sum=max(Lf1,Lf2)"
            ),
        },
        "noise_action_ratio": {
            "artifact": "category_noise_action_ratio.json",
            "minimum_noise_percent": noise_payload["summary"]["minimum_noise_percent"],
            "maximum_noise_percent": noise_payload["summary"]["maximum_noise_percent"],
        },
    }
    json_dump(manifest, data_dir / "component_training_manifest.json")
    print(f"saved models under {models_dir.resolve()}")
    print(f"saved data under {data_dir.resolve()}")


if __name__ == "__main__":
    main()

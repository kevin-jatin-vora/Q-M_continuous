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
    return agent, records, tile_records, time.perf_counter() - started


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
    """Group free transitions by (tile_id, action) using the pre-step state position."""
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


def component_constants(records, model, device, max_pairs, seed):
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
        distances = np.linalg.norm(states64[left] - states64[right], axis=1)
        valid = distances > 1e-12
        distances = distances[valid]
        left, right = left[valid], right[valid]
        report[(region, action)] = {
            "n": n_rows,
            "Lr": trimmed_mean(np.abs(rewards64[left] - rewards64[right]) / distances),
            "Lf": trimmed_mean(np.linalg.norm(next64[left] - next64[right], axis=1) / distances),
            "Lq": trimmed_mean(np.abs(q_values[left] - q_values[right]) / distances),
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
    category_constants1,
    category_constants2,
    tile_constants1,
    tile_constants2,
    args,
    sigma,
    tile_map,
):
    """Per-tile Lr/Lq; category-level Lf (max over R1/R2) shared by all tiles in a category.

    Theoretical Lq is the infinite-horizon Bellman bound
      LQ_bellman_bound = Lr_sum / (1 - gamma * Lf_sum)
    for every tile/action, identical for terminal and non-terminal tiles.
    Requires gamma * Lf_sum < 1; invalid rows (denominator <= 0) are rejected
    downstream by ``require_finite_bellman_lq``.
    """
    category_lr_lq = {}
    category_lf = {}
    for region in range(1, 6):
        for action in range(4):
            first = category_constants1.get((region, action), {"n": 0, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0})
            second = category_constants2.get((region, action), {"n": 0, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0})
            category_lf[(region, action)] = {
                "Lf1": float(first["Lf"]),
                "Lf2": float(second["Lf"]),
                "Lf_sum": max(float(first["Lf"]), float(second["Lf"])),
                "n_r1": int(first["n"]),
                "n_r2": int(second["n"]),
            }
            category_lr_lq[(region, action)] = {
                "Lr1": float(first["Lr"]),
                "Lr2": float(second["Lr"]),
                "Lr_sum": float(first["Lr"]) + float(second["Lr"]),
                "Lq1": float(first["Lq"]),
                "Lq2": float(second["Lq"]),
                "Lq_empirical_sum": float(first["Lq"]) + float(second["Lq"]),
            }

    rows = []
    min_samples = int(args.min_cell_samples)
    for tile_id in range(N_TILES):
        region = int(tile_map[tile_id])
        for action in range(4):
            first = tile_constants1.get((tile_id, action), {"n": 0, "Lr": 0.0, "Lq": 0.0, "pairs_used": 0})
            second = tile_constants2.get((tile_id, action), {"n": 0, "Lr": 0.0, "Lq": 0.0, "pairs_used": 0})
            lf_row = category_lf[(region, action)]
            lf_sum = float(lf_row["Lf_sum"])
            use_category_fallback = int(first["n"]) < min_samples or int(second["n"]) < min_samples
            if use_category_fallback:
                pooled = category_lr_lq[(region, action)]
                lr1 = pooled["Lr1"]
                lr2 = pooled["Lr2"]
                lr_sum = pooled["Lr_sum"]
                lq1 = pooled["Lq1"]
                lq2 = pooled["Lq2"]
                lq_emp = pooled["Lq_empirical_sum"]
            else:
                lr1 = float(first["Lr"])
                lr2 = float(second["Lr"])
                lr_sum = lr1 + lr2
                lq1 = float(first["Lq"])
                lq2 = float(second["Lq"])
                lq_emp = lq1 + lq2
            x = args.gamma * lf_sum
            denominator = 1.0 - x
            # Infinite-horizon Bellman bound. Invalid rows (denominator <= 0) are
            # preserved with a NaN/invalid value and rejected downstream by
            # require_finite_bellman_lq (original behavior).
            if denominator > 0:
                bellman = lr_sum / denominator
            else:
                bellman = None
            rows.append(
                {
                    "tile": tile_id,
                    "region": region,
                    "action": action,
                    "lr_source": "category_fallback" if use_category_fallback else "tile",
                    "n_r1": int(first["n"]),
                    "n_r2": int(second["n"]),
                    "Lr1": lr1,
                    "Lr2": lr2,
                    "Lr_sum": lr_sum,
                    "Lf1": lf_row["Lf1"],
                    "Lf2": lf_row["Lf2"],
                    "Lf_sum": lf_sum,
                    "Lq1": lq1,
                    "Lq2": lq2,
                    "Lq_empirical_sum": lq_emp,
                    "LQ_bellman_bound": bellman,
                    "pairs_r1": int(first.get("pairs_used", 0)),
                    "pairs_r2": int(second.get("pairs_used", 0)),
                }
            )
    require_finite_bellman_lq(rows, args.gamma, path=f"sigma={sigma:.17g} (post R1/R2)")
    return {
        "source": "scripts/train_reward_components.py",
        "gamma": args.gamma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": float(args.deterministic_sigma_scale),
        "deterministic_sigma": float(args.deterministic_sigma_scale) * float(sigma),
        "method_notes": {
            "Lr_Lq": (
                "Per-tile 10% trimmed pairwise ratios on free transitions; "
                "Lr_sum=Lr1+Lr2 and Lq_emp=Lq1+Lq2. Falls back to category pool when "
                f"a tile has < {min_samples} samples from either behavior."
            ),
            "Lf": (
                "Category-level (pooled over all tiles in the category): "
                "Lf_sum=max(Lf1, Lf2) from R1/R2 behavior."
            ),
            "LQ_bellman_bound": (
                "LQ_bellman_bound = Lr_sum / (1 - gamma * Lf_sum); "
                "requires gamma * Lf_sum < 1."
            ),
            "pruning_radius": (
                "Student-t L2 radius of next free s' around s+delta_mean; "
                "Q_single uses delta_mean; RA-DQN uses the live env"
            ),
            "delta_estimation": (
                "Exclude boundary cancel/clip from delta and Lipschitz sample sets so "
                "deterministic regions can report std≈0 for the free step"
            ),
        },
        "constants": rows,
    }


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

    q1, records1, tile_records1, seconds1 = train_component(args, environment, 0, args.seed, device)
    q2, records2, tile_records2, seconds2 = train_component(
        args, environment, 1, args.seed + args.r2_seed_offset, device
    )
    torch.save(q1.q.state_dict(), models_dir / "q1_dqn_r1.pth")
    torch.save(q2.q.state_dict(), models_dir / "q2_dqn_r2.pth")
    category_constants1 = component_constants(records1, q1.q, device, args.max_pairs, args.seed)
    category_constants2 = component_constants(records2, q2.q, device, args.max_pairs, args.seed)
    tile_constants1 = component_constants(tile_records1, q1.q, device, args.max_pairs, args.seed + 7)
    tile_constants2 = component_constants(tile_records2, q2.q, device, args.max_pairs, args.seed + 8)
    bounds = build_bounds(records1, records2, args, sigma, category_counts=cat_counts)
    constants = build_lipschitz(
        category_constants1,
        category_constants2,
        tile_constants1,
        tile_constants2,
        args,
        sigma,
        tile_map,
    )
    json_dump(bounds, data_dir / "transition_bounds.json")
    json_dump(constants, data_dir / "lipschitz_constants.json")
    json_dump(layout, data_dir / "tile_layout.json")
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
    }
    json_dump(manifest, data_dir / "component_training_manifest.json")
    print(f"saved models under {models_dir.resolve()}")
    print(f"saved data under {data_dir.resolve()}")


if __name__ == "__main__":
    main()

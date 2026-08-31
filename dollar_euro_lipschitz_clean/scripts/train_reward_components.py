"""Train separate R1/R2 behavior DQNs and rebuild sigma-specific pruning inputs.

The original experiment used each reward-specific epsilon-greedy trajectory both
to train its DQN and to collect labeled transition samples.  This script
reproduces that design, but records provenance and separates uncertainty about
the cell mean from a split-calibrated predictive/process radius.
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
from scipy.stats import t as student_t

from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    format_training_args,
    load_config,
    resolve_sigma,
)
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.replay import ReplayBuffer
from dollar_euro_lipschitz.bounds import require_finite_bellman_lq


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
        # Keep wall cancel/clip out of delta_s / Lipschitz estimates; RL still uses all steps.
        free_mask[step] = not bool(info.get("boundary_affected", False))
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
    return agent, records, time.perf_counter() - started


def _group_transitions(states, next_states, rewards, regions, actions, free_mask=None):
    """Group transitions for bounds/Lipschitz. Optionally keep only non-boundary-affected steps."""
    if free_mask is None:
        free_mask = np.ones(len(states), dtype=np.bool_)
    records = {}
    for region in range(1, 5):
        region_mask = (regions == region) & free_mask
        for action in range(4):
            mask = region_mask & (actions == action)
            records[(region, action)] = (
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
        return {"n": 0, "mean_delta": np.zeros(2, dtype=np.float64), "sample_std_delta": np.zeros(2, dtype=np.float64)}
    deltas = (next_states - states).astype(np.float64, copy=False)
    mean = deltas.mean(axis=0)
    std = deltas.std(axis=0, ddof=1) if n >= 2 else np.zeros(2, dtype=np.float64)
    return {"n": n, "mean_delta": mean, "sample_std_delta": std}


def combine_two_behavior_stats(stat_q1, stat_q2, confidence, norm_ord=2):
    """Exact pooling from reformatting json.py."""
    n1 = int(stat_q1["n"])
    n2 = int(stat_q2["n"])
    mu1 = np.asarray(stat_q1["mean_delta"], dtype=np.float64).reshape(-1)
    mu2 = np.asarray(stat_q2["mean_delta"], dtype=np.float64).reshape(-1)
    s1 = np.asarray(stat_q1["sample_std_delta"], dtype=np.float64).reshape(-1)
    s2 = np.asarray(stat_q2["sample_std_delta"], dtype=np.float64).reshape(-1)
    if n1 < 2:
        s1 = np.zeros_like(mu1)
    if n2 < 2:
        s2 = np.zeros_like(mu2)
    n_total = n1 + n2
    if n_total == 0:
        mean_delta = np.zeros(2, dtype=np.float64)
        sample_std = np.zeros(2, dtype=np.float64)
        t_mult = 0.0
        radius_vec = np.zeros(2, dtype=np.float64)
    else:
        mean_delta = (n1 * mu1 + n2 * mu2) / n_total
        if n_total > 1:
            pooled_var = (
                (n1 - 1) * (s1 ** 2)
                + (n2 - 1) * (s2 ** 2)
                + n1 * ((mu1 - mean_delta) ** 2)
                + n2 * ((mu2 - mean_delta) ** 2)
            ) / (n_total - 1)
            sample_std = np.sqrt(np.maximum(pooled_var, 0.0))
            t_mult = float(student_t.ppf(0.5 + confidence / 2.0, n_total - 1))
            radius_vec = t_mult * sample_std / math.sqrt(n_total)
        else:
            sample_std = np.zeros(2, dtype=np.float64)
            t_mult = 0.0
            radius_vec = np.zeros(2, dtype=np.float64)
    return {
        "n_total": int(n_total),
        "n_q1_behavior": n1,
        "n_q2_behavior": n2,
        "mean_delta": mean_delta,
        "sample_std_delta": sample_std,
        "student_t_multiplier": float(t_mult),
        "student_t_conf_radius_mean_delta": radius_vec,
        "radius_scalar": float(np.linalg.norm(radius_vec, ord=norm_ord)),
    }


def quantile_higher_columns(values, probability):
    values = np.sort(np.asarray(values, dtype=np.float64), axis=0)
    index = min(values.shape[0] - 1, max(0, math.ceil(probability * values.shape[0]) - 1))
    return values[index]


def build_bounds(q1_records, q2_records, args, sigma):
    output = {
        "source_json": "generated directly by scripts/train_reward_components.py",
        "metadata": {
            "dynamics_sigma": sigma,
            "sigma_source": "explicit runtime argument/config",
            "behavior_policies": ["R1 epsilon-greedy DQN", "R2 epsilon-greedy DQN"],
            "seed_r1": args.seed,
            "seed_r2": args.seed + args.r2_seed_offset,
            "radius_scalar_semantics": (
                "L2 norm of the Student-t CI of the pooled R1/R2 mean delta; "
                "estimated only on transitions not affected by boundary cancel/clip"
            ),
            "delta_estimation": (
                "mean_delta / radius use free region-action steps only "
                "(exclude env boundary cancel and post-noise clip). "
                "Q_single applies clip(s+mu) at use time; RA-DQN uses the live env clip."
            ),
        },
        "config": {
            "confidence": args.confidence,
            "norm_ord": 2,
            "exclude_boundary_affected_transitions": True,
        },
        "by_region_action": {},
    }
    for region in range(1, 5):
        output["by_region_action"][str(region)] = {}
        for action in range(4):
            rows1 = q1_records.get((region, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            rows2 = q2_records.get((region, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            s1, ns1, _r1 = rows1
            s2, ns2, _r2 = rows2
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
                args.confidence,
                norm_ord=2,
            )
            output["by_region_action"][str(region)][str(action)] = {
                "n_total": combined["n_total"],
                "n_q1_behavior": combined["n_q1_behavior"],
                "n_q2_behavior": combined["n_q2_behavior"],
                "mean_delta": combined["mean_delta"].tolist(),
                "sample_std_delta": combined["sample_std_delta"].tolist(),
                "student_t_multiplier": combined["student_t_multiplier"],
                "student_t_conf_radius_mean_delta": combined["student_t_conf_radius_mean_delta"].tolist(),
                "radius_scalar": combined["radius_scalar"],
            }
    return output


def build_lipschitz(constants1, constants2, args, sigma):
    rows = []
    for region in range(1, 5):
        for action in range(4):
            first = constants1[(region, action)]
            second = constants2[(region, action)]
            lf_sum = max(first["Lf"], second["Lf"])
            lr_sum = first["Lr"] + second["Lr"]
            denominator = 1.0 - args.gamma * lf_sum
            bellman = lr_sum / denominator if denominator > 0 else None
            rows.append(
                {
                    "region": region,
                    "action": action,
                    "n_r1": first["n"],
                    "n_r2": second["n"],
                    "Lr1": first["Lr"],
                    "Lr2": second["Lr"],
                    "Lr_sum": lr_sum,
                    "Lf1": first["Lf"],
                    "Lf2": second["Lf"],
                    "Lf_sum": lf_sum,
                    "Lq1": first["Lq"],
                    "Lq2": second["Lq"],
                    "Lq_empirical_sum": first["Lq"] + second["Lq"],
                    "LQ_bellman_bound": bellman,
                    "pairs_r1": first["pairs_used"],
                    "pairs_r2": second["pairs_used"],
                }
            )
    require_finite_bellman_lq(rows, args.gamma, path=f"sigma={sigma:.17g} (post R1/R2)")
    return {
        "source": "scripts/train_reward_components.py",
        "gamma": args.gamma,
        "dynamics_sigma": sigma,
        "method_notes": {
            "Lr_Lf_Lq": (
                "data-derived 10% trimmed means of unique same-action pairwise ratios "
                "on non-boundary-affected transitions only"
            ),
            "LQ_bellman_bound": "Lr_sum / (1 - gamma*Lf_sum); requires gamma*Lf_sum < 1 or the run aborts",
            "pruning_radius": (
                "Student-t CI of the pooled cell-mean delta on free (non-clipped) transitions; "
                "boundary handling remains in Q_single clip(s+mu) and in the live RA env"
            ),
            "delta_estimation": (
                "Exclude boundary cancel/clip from delta_s and Lipschitz sample sets so "
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
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--predictive-coverage", type=float, default=0.95)
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--min-cell-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("confidence values must be in (0, 1)")
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    apply_training_defaults(args, config, role="component")
    environment = env_kwargs(config, sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}; requested_sigma={sigma:.17g}; "
        f"env_kwargs_sigma={environment['sigma']:.17g}; "
        f"top-half deterministic; bottom noise scales with sigma; "
        f"env_horizon={environment['horizon']}; {format_training_args(args)}"
    )

    q1, records1, seconds1 = train_component(args, environment, 0, args.seed, device)
    q2, records2, seconds2 = train_component(
        args, environment, 1, args.seed + args.r2_seed_offset, device
    )
    torch.save(q1.q.state_dict(), output_dir / "q1_dqn_r1.pth")
    torch.save(q2.q.state_dict(), output_dir / "q2_dqn_r2.pth")
    constants1 = component_constants(records1, q1.q, device, args.max_pairs, args.seed)
    constants2 = component_constants(records2, q2.q, device, args.max_pairs, args.seed)
    bounds = build_bounds(records1, records2, args, sigma)
    constants = build_lipschitz(constants1, constants2, args, sigma)
    json_dump(bounds, output_dir / "transition_bounds.json")
    json_dump(constants, output_dir / "lipschitz_constants.json")
    manifest = {
        "sigma": sigma,
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
            for region in range(1, 5)
            for action in range(4)
        },
    }
    json_dump(manifest, output_dir / "component_training_manifest.json")
    print(f"saved sigma-specific inputs under {output_dir.resolve()}")


if __name__ == "__main__":
    main()

# two_reward_q_learning_fixed_action_lipschitz.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ContinuousDollarEuroEnv_radial import ContinuousDollarEuroEnv


# ============================================================
# Device / constants
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

GLOBAL_REGION = "__global__"


# ============================================================
# Student-t critical value helper
# ============================================================

try:
    from scipy.stats import t as student_t_dist

    def student_t_critical(confidence: float, dof: int) -> float:
        if dof <= 0:
            return 0.0
        return float(student_t_dist.ppf(0.5 + confidence / 2.0, dof))
except Exception:
    def student_t_critical(confidence: float, dof: int) -> float:
        if dof <= 0:
            return 0.0
        return float(NormalDist().inv_cdf(0.5 + confidence / 2.0))


# ============================================================
# Config / data containers
# ============================================================

@dataclass
class DQNConfig:
    n_steps: int = 150_000
    max_steps_per_episode: int = 200

    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = int(1e5)
    batch_size: int = 256
    update_every: int = 4
    tau: float = 5e-4

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    seed: int = 0
    confidence: float = 0.95

    learn_lc_per_region: bool = True
    same_action_only: bool = True
    norm_ord: int = 2
    max_lipschitz_pairs: int = 50_000
    zero_distance_tol: float = 1e-12
    min_samples_for_region_report: int = 2

    train_log_every_steps: int = 10_000
    eval_episodes: int = 5

    output_dir: str = "two_reward_dqn_outputs"
    output_pickle_path: str = "two_reward_dqn_artifact.pkl"
    output_json_path: str = "two_reward_dqn_summary.json"
    output_q1_path: str = "q1_dqn_r1.pth"
    output_q2_path: str = "q2_dqn_r2.pth"
    output_transition_dataset_path: Optional[str] = "two_reward_dqn_raw_transitions.pkl"
    save_raw_transitions: bool = True


@dataclass
class TransitionRecord:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    region: int


# ============================================================
# Seed helper
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Environment hooks from your template
# ============================================================

def make_env(render: bool = False, seed: Optional[int] = None):
    render_mode = "human" if render else None
    return ContinuousDollarEuroEnv(
        render_mode=render_mode,
        auto_render=bool(render),
        seed=seed,
    )


def legal_actions(state):
    return [0, 1, 2, 3]


def r1(state, action, next_state, env_reward, done, info):
    reward_vec = np.asarray(env_reward, dtype=np.float32)
    return float(reward_vec[0])


def r2(state, action, next_state, env_reward, done, info):
    reward_vec = np.asarray(env_reward, dtype=np.float32)
    return float(reward_vec[1])


def region(state, action, next_state, info):
    return int(info["region"])


# ============================================================
# Networks / agent
# ============================================================

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.mem = deque(maxlen=buffer_size)

    def add(self, s, a, r, ns, d):
        self.mem.append((np.asarray(s, dtype=np.float32),
                         int(a),
                         float(r),
                         np.asarray(ns, dtype=np.float32),
                         float(d)))

    def sample(self, batch_size: int):
        batch = random.sample(self.mem, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.as_tensor(np.asarray(s), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(a), dtype=torch.long, device=device).unsqueeze(1),
            torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.asarray(ns), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(d), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.mem)


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.mem = ReplayBuffer(cfg.buffer_size)
        self.t = 0

    def act(self, state, eps: float, action_fn: Callable[[Any], Sequence[int]]):
        if random.random() < eps:
            return int(random.choice(list(action_fn(state))))

        s = torch.as_tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q(s)
        return int(torch.argmax(q_vals, dim=1).item())

    def step(self, s, a, r, ns, d):
        self.mem.add(s, a, r, ns, d)
        self.t = (self.t + 1) % self.cfg.update_every

        if self.t == 0 and len(self.mem) >= self.cfg.batch_size:
            experiences = self.mem.sample(self.cfg.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            q_local_next = self.q(next_states)
            best_next_actions = q_local_next.argmax(dim=1, keepdim=True)
            q_target_next = self.qt(next_states).gather(1, best_next_actions)
            q_targets = rewards + self.cfg.gamma * q_target_next * (1.0 - dones)

        q_expected = self.q(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        self.soft_update(self.q, self.qt, self.cfg.tau)

    @torch.no_grad()
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.lerp_(local_param.data, tau)


# ============================================================
# Utility helpers
# ============================================================

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    return obj


def group_transitions(transitions: List[TransitionRecord], learn_lc_per_region: bool):
    grouped = defaultdict(lambda: defaultdict(list))
    for tr in transitions:
        region_key = tr.region if learn_lc_per_region else GLOBAL_REGION
        grouped[region_key][tr.action].append(tr)
    return grouped


def compute_state_change_stats(group: List[TransitionRecord], confidence: float):
    n = len(group)
    if n == 0:
        return {
            "n": 0,
            "mean_delta": [0.0, 0.0],
            "sample_std_delta": [0.0, 0.0],
            "student_t_multiplier": 0.0,
            "student_t_conf_radius_mean_delta": [0.0, 0.0],
        }

    deltas = np.stack([tr.next_state - tr.state for tr in group], axis=0).astype(np.float64)
    mean_delta = np.mean(deltas, axis=0)
    if n > 1:
        sample_std = np.std(deltas, axis=0, ddof=1)
        t_mult = student_t_critical(confidence, n - 1)
        conf_radius = t_mult * (sample_std / math.sqrt(n))
    else:
        sample_std = np.zeros_like(mean_delta)
        t_mult = 0.0
        conf_radius = np.zeros_like(mean_delta)

    return {
        "n": int(n),
        "mean_delta": mean_delta.tolist(),
        "sample_std_delta": sample_std.tolist(),
        "student_t_multiplier": float(t_mult),
        "student_t_conf_radius_mean_delta": conf_radius.tolist(),
    }


def sample_pairs(n: int, max_pairs: int, rng: np.random.Generator):
    if n < 2:
        return []
    total = n * (n - 1) // 2
    if max_pairs is None or total <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    pairs = set()
    while len(pairs) < max_pairs:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return list(pairs)


def predict_q_values(model: QNet, states: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(states, dtype=torch.float32, device=device)
        q = model(x).detach().cpu().numpy()
    return q

def trimmed_mean(x, trim_ratio=0.1):
    x = np.sort(np.asarray(x))
    k = int(len(x) * trim_ratio)
    if len(x) <= 2 * k:
        return float(np.mean(x))
    return float(np.mean(x[k:-k]))

def compute_region_action_lipschitz(
    group: List[TransitionRecord],
    model: QNet,
    cfg: DQNConfig,
):
    n = len(group)
    if n < 2:
        return {
            "n": int(n),
            "Lr": 0.0,
            "Lf": 0.0,
            "Lq": 0.0,
            "pairs_used": 0,
        }

    states = np.stack([tr.state for tr in group], axis=0).astype(np.float64)
    next_states = np.stack([tr.next_state for tr in group], axis=0).astype(np.float64)
    rewards = np.asarray([tr.reward for tr in group], dtype=np.float64)

    action = group[0].action
    q_vals = predict_q_values(model, states)[:, action]

    rng = np.random.default_rng(cfg.seed + 17 * n + 31 * action)
    pairs = sample_pairs(n, cfg.max_lipschitz_pairs, rng)

    lr_vals = []
    lf_vals = []
    lq_vals = []

    for i, j in pairs:
        dist = float(np.linalg.norm(states[i] - states[j], ord=cfg.norm_ord))
        if dist <= cfg.zero_distance_tol:
            continue

        lr_vals.append(abs(rewards[i] - rewards[j]) / dist)
        lf_vals.append(np.linalg.norm(next_states[i] - next_states[j], ord=cfg.norm_ord) / dist)
        lq_vals.append(abs(q_vals[i] - q_vals[j]) / dist)

    if len(lr_vals) == 0:
        return {
            "n": int(n),
            "Lr": 0.0,
            "Lf": 0.0,
            "Lq": 0.0,
            "pairs_used": 0,
        }

    return {
        "n": int(n),
        "Lr": trimmed_mean(lr_vals, 0.1),
        "Lf": trimmed_mean(lf_vals, 0.1),
        "Lq": trimmed_mean(lq_vals, 0.1),
        "pairs_used": int(len(lr_vals)),
    }

def build_lipschitz_report(transitions: List[TransitionRecord], model: QNet, cfg: DQNConfig):
    grouped = group_transitions(transitions, cfg.learn_lc_per_region)
    report = {}

    for region_key, by_action in grouped.items():
        report[region_key] = {}
        for action, group in by_action.items():
            stats = compute_region_action_lipschitz(group, model, cfg)
            report[region_key][int(action)] = stats

    return report


def build_state_change_report(transitions: List[TransitionRecord], cfg: DQNConfig):
    grouped = group_transitions(transitions, cfg.learn_lc_per_region)
    report = {}
    for region_key, by_action in grouped.items():
        report[region_key] = {}
        for action, group in by_action.items():
            report[region_key][int(action)] = compute_state_change_stats(group, cfg.confidence)
    return report


def combine_reports(lc1: Dict, lc2: Dict, gamma: float):
    combined = {}
    regions = set(lc1.keys()) | set(lc2.keys())
    for region_key in regions:
        combined[region_key] = {}
        actions = set(lc1.get(region_key, {}).keys()) | set(lc2.get(region_key, {}).keys())
        for action in actions:
            b1 = lc1.get(region_key, {}).get(action, {"Lr": 0.0, "Lf": 0.0, "Lq": 0.0})
            b2 = lc2.get(region_key, {}).get(action, {"Lr": 0.0, "Lf": 0.0, "Lq": 0.0})

            Lr1 = float(b1["Lr"])
            Lf1 = float(b1["Lf"])
            Lq1 = float(b1["Lq"])

            Lr2 = float(b2["Lr"])
            Lf2 = float(b2["Lf"])
            Lq2 = float(b2["Lq"])

            Lr_sum = Lr1 + Lr2
            Lf_sum = max(Lf1, Lf2)
            Lq_sum_empirical = Lq1 + Lq2

            denom = 1.0 - gamma * Lf_sum
            if denom > 0.0:
                bellman_bound = Lr_sum / denom
            else:
                bellman_bound = float("inf")

            combined[region_key][int(action)] = {
                "Lr1": Lr1,
                "Lf1": Lf1,
                "Lq1": Lq1,
                "Lr2": Lr2,
                "Lf2": Lf2,
                "Lq2": Lq2,
                "Lr_sum": Lr_sum,
                "Lf_sum": Lf_sum,
                "Lq_empirical_sum": Lq_sum_empirical,
                "LQ_bellman_bound": bellman_bound,
            }
    return combined


# ============================================================
# Training / evaluation
# ============================================================

def train_one_reward_dqn(
    env_factory: Callable[[], ContinuousDollarEuroEnv],
    reward_fn: Callable[[Any, int, Any, Any, bool, Dict[str, Any]], float],
    region_fn: Callable[[Any, int, Any, Dict[str, Any]], int],
    action_fn: Callable[[Any], Sequence[int]],
    cfg: DQNConfig,
    reward_name: str,
    seed_offset: int,
):
    agent = DQNAgent(cfg)
    env = env_factory()

    transitions: List[TransitionRecord] = []
    eps = cfg.epsilon_start
    episode_step = 0
    episode_idx = 0
    running_return = 0.0
    running_done_count = 0

    obs, _ = env.reset(seed=cfg.seed + seed_offset)

    for step in range(1, cfg.n_steps + 1):
        episode_step += 1
        action = agent.act(obs, eps=eps, action_fn=action_fn)

        next_obs, env_reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        reward = float(reward_fn(obs, action, next_obs, env_reward, done, info))
        reg = int(region_fn(obs, action, next_obs, info))

        transitions.append(
            TransitionRecord(
                state=np.asarray(obs, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_obs, dtype=np.float32).copy(),
                done=bool(done),
                region=int(reg),
            )
        )

        agent.step(obs, action, reward, next_obs, done)

        obs = next_obs
        running_return += reward

        if done or episode_step >= cfg.max_steps_per_episode:
            episode_idx += 1
            running_done_count += int(done)
            obs, _ = env.reset(seed=cfg.seed + seed_offset + episode_idx)
            episode_step = 0
            running_return = 0.0
            eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)

        if step % cfg.train_log_every_steps == 0:
            print(
                f"[{reward_name}] step={step:>7} | episodes={episode_idx:>5} "
                f"| eps={eps:.4f} | buffer={len(agent.mem):>7}"
            )

    env.close()
    return agent, transitions


def evaluate_agent(
    agent: DQNAgent,
    env_factory: Callable[[], ContinuousDollarEuroEnv],
    reward_fn: Callable[[Any, int, Any, Any, bool, Dict[str, Any]], float],
    action_fn: Callable[[Any], Sequence[int]],
    cfg: DQNConfig,
    seed_offset: int,
):
    env = env_factory()
    returns = []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=cfg.seed + seed_offset + 10_000 + ep)
        total = 0.0
        for _ in range(cfg.max_steps_per_episode):
            action = agent.act(obs, eps=0.0, action_fn=action_fn)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total += float(reward_fn(obs, action, next_obs, env_reward, done, info))
            obs = next_obs
            if done:
                break
        returns.append(total)

    env.close()
    return float(np.mean(returns)) if returns else 0.0


# ============================================================
# Main collector
# ============================================================

def learn_q1_q2_collect_stats_and_constants(
    env_factory: Callable[[], ContinuousDollarEuroEnv],
    action_fn: Callable[[Any], Sequence[int]],
    reward_r1_fn: Callable[[Any, int, Any, Any, bool, Dict[str, Any]], float],
    reward_r2_fn: Callable[[Any, int, Any, Any, bool, Dict[str, Any]], float],
    region_fn: Callable[[Any, int, Any, Dict[str, Any]], int],
    config: DQNConfig,
):
    set_seed(config.seed)

    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Train Q1 on R1
    print("\n=== Training Q1 on R1 ===")
    q1_agent, q1_transitions = train_one_reward_dqn(
        env_factory=env_factory,
        reward_fn=reward_r1_fn,
        region_fn=region_fn,
        action_fn=action_fn,
        cfg=config,
        reward_name="R1",
        seed_offset=0,
    )

    # Train Q2 on R2
    print("\n=== Training Q2 on R2 ===")
    q2_agent, q2_transitions = train_one_reward_dqn(
        env_factory=env_factory,
        reward_fn=reward_r2_fn,
        region_fn=region_fn,
        action_fn=action_fn,
        cfg=config,
        reward_name="R2",
        seed_offset=1_000_000,
    )

    # Evaluate both agents
    q1_eval = evaluate_agent(
        agent=q1_agent,
        env_factory=env_factory,
        reward_fn=reward_r1_fn,
        action_fn=action_fn,
        cfg=config,
        seed_offset=0,
    )
    q2_eval = evaluate_agent(
        agent=q2_agent,
        env_factory=env_factory,
        reward_fn=reward_r2_fn,
        action_fn=action_fn,
        cfg=config,
        seed_offset=1_000_000,
    )

    # State-change stats by region/action
    print("\n=== Computing state-change stats ===")
    q1_state_change = build_state_change_report(q1_transitions, config)
    q2_state_change = build_state_change_report(q2_transitions, config)

    # Lipschitz constants by region/action
    print("\n=== Computing Lipschitz constants ===")
    q1_lipschitz = build_lipschitz_report(q1_transitions, q1_agent.q, config)
    q2_lipschitz = build_lipschitz_report(q2_transitions, q2_agent.q, config)
    combined = combine_reports(q1_lipschitz, q2_lipschitz, config.gamma)

    # Save model weights
    q1_path = outdir / config.output_q1_path
    q2_path = outdir / config.output_q2_path
    torch.save(q1_agent.q.state_dict(), q1_path)
    torch.save(q2_agent.q.state_dict(), q2_path)

    # Optional raw dataset save for later use
    raw_path = None
    if config.save_raw_transitions and config.output_transition_dataset_path:
        raw_path = outdir / config.output_transition_dataset_path
        with open(raw_path, "wb") as f:
            pickle.dump(
                {
                    "q1_transitions": q1_transitions,
                    "q2_transitions": q2_transitions,
                },
                f,
            )

    artifact = {
        "config": asdict(config),
        "paths": {
            "q1_weights": str(q1_path),
            "q2_weights": str(q2_path),
            "raw_transitions": str(raw_path) if raw_path is not None else None,
            "pickle": str(outdir / config.output_pickle_path),
            "json": str(outdir / config.output_json_path),
        },
        "q1": {
            "eval_return": q1_eval,
            "state_change_stats": q1_state_change,
            "lipschitz": q1_lipschitz,
        },
        "q2": {
            "eval_return": q2_eval,
            "state_change_stats": q2_state_change,
            "lipschitz": q2_lipschitz,
        },
        "combined": combined,
    }

    # Save compact pickle artifact
    pickle_path = outdir / config.output_pickle_path
    with open(pickle_path, "wb") as f:
        pickle.dump(artifact, f)

    # Save JSON summary
    json_path = outdir / config.output_json_path
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(artifact), f, indent=2)

    # Print summaries
    print("\n=== Final summary ===")
    print(f"Q1 eval return: {q1_eval:.6f}")
    print(f"Q2 eval return: {q2_eval:.6f}")
    print(f"Q1 weights saved to: {q1_path}")
    print(f"Q2 weights saved to: {q2_path}")
    if raw_path is not None:
        print(f"Raw transitions saved to: {raw_path}")
    print(f"Pickle artifact saved to: {pickle_path}")
    print(f"JSON summary saved to: {json_path}")

    def print_nested_report(title: str, report: Dict):
        print(f"\n--- {title} ---")
        for region_key in sorted(report.keys(), key=lambda x: str(x)):
            print(f"Region: {region_key}")
            for action in sorted(report[region_key].keys()):
                row = report[region_key][action]
                print(
                    f"  action {action}: "
                    f"n={row['n']}, "
                    f"Lr={row.get('Lr', 0.0):.6f}, "
                    f"Lf={row.get('Lf', 0.0):.6f}, "
                    f"Lq={row.get('Lq', 0.0):.6f}"
                )

    print_nested_report("Q1 state-change stats", q1_state_change)
    print_nested_report("Q2 state-change stats", q2_state_change)
    print_nested_report("Q1 Lipschitz", q1_lipschitz)
    print_nested_report("Q2 Lipschitz", q2_lipschitz)

    print("\n--- Combined bounds ---")
    for region_key in sorted(combined.keys(), key=lambda x: str(x)):
        print(f"Region: {region_key}")
        for action in sorted(combined[region_key].keys()):
            row = combined[region_key][action]
            print(
                f"  action {action}: "
                f"Lr1={row['Lr1']:.6f}, Lf1={row['Lf1']:.6f}, Lq1={row['Lq1']:.6f}, "
                f"Lr2={row['Lr2']:.6f}, Lf2={row['Lf2']:.6f}, Lq2={row['Lq2']:.6f}, "
                f"Lr_sum={row['Lr_sum']:.6f}, Lf_sum={row['Lf_sum']:.6f}, "
                f"Lq_empirical_sum={row['Lq_empirical_sum']:.6f}, "
                f"LQ_bellman_bound={row['LQ_bellman_bound']}"
            )

    return artifact


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = DQNConfig(
        n_steps=150_000,
        max_steps_per_episode=200,
        gamma=0.99,
        lr=1e-3,
        buffer_size=int(1e5),
        batch_size=256,
        update_every=4,
        tau=5e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        seed=0,
        confidence=0.95,
        learn_lc_per_region=True,
        same_action_only=True,
        norm_ord=2,
        max_lipschitz_pairs=50_000,
        zero_distance_tol=1e-12,
        min_samples_for_region_report=2,
        train_log_every_steps=10_000,
        eval_episodes=5,
        output_dir="two_reward_dqn_outputs",
        output_pickle_path="two_reward_dqn_artifact.pkl",
        output_json_path="two_reward_dqn_summary.json",
        output_q1_path="q1_dqn_r1.pth",
        output_q2_path="q2_dqn_r2.pth",
        output_transition_dataset_path="two_reward_dqn_raw_transitions.pkl",
        save_raw_transitions=True,
    )

    artifact = learn_q1_q2_collect_stats_and_constants(
        env_factory=lambda: make_env(render=False, seed=cfg.seed),
        action_fn=legal_actions,
        reward_r1_fn=r1,
        reward_r2_fn=r2,
        region_fn=region,
        config=cfg,
    )

    print("\nDone.")
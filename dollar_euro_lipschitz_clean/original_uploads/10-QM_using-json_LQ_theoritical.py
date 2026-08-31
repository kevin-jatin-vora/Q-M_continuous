# RA_DQN_pruned_from_json.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from ContinuousDollarEuroEnv_radial import ContinuousDollarEuroEnv, ScalarizeReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================
# Hyperparameters
# ==============================
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 5e-4
LR = 1e-3
UPDATE_EVERY = 4

N_STEP = 200_000
TEST_STEPS = 10_000
TEST_RUNS = 5

STEP_EPS_DECAY = 0.01 ** (1.0 / 200000.0)

RUNS = 5
SAVE_FILE = "radqn_de.npy"

env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)


# =============================
# Network
# =============================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


# =============================
# Replay Buffer
# =============================
class ReplayBuffer:
    def __init__(self):
        self.mem = deque(maxlen=BUFFER_SIZE)

    def add(self, s, a, r, ns, d):
        self.mem.append((s, a, r, ns, d))

    def sample(self):
        batch = random.sample(self.mem, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32).to(device),
            torch.tensor(a, dtype=torch.long).unsqueeze(1).to(device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(ns, dtype=torch.float32).to(device),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.mem)


# =============================
# JSON-backed pruning info
# =============================
class JsonResolvedBounds:
    """
    Loads resolved transition bounds from JSON.

    Expected structure:
        bounds["by_region_action"][region][action] = {
            "n_total": ...,
            "mean_delta": [...],
            "student_t_conf_radius_mean_delta": [...],
            "radius_scalar": ...
        }

    This class is only used here to provide per-(region, action) radius.
    """

    def __init__(self, min_samples=2):
        self.min_samples = min_samples
        self.cell_stats = {}
        self.region_stats = {}

    def fit(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            bounds = json.load(f)

        by_region_action = bounds["by_region_action"]

        for region_key, action_map in by_region_action.items():
            r = int(region_key)

            region_entries = []

            for action_key, entry in action_map.items():
                a = int(action_key)
                n = int(entry.get("n_total", 0))
                mean = np.asarray(entry.get("mean_delta", [0.0, 0.0]), dtype=np.float32)
                radius_vec = np.asarray(
                    entry.get("student_t_conf_radius_mean_delta", [0.05, 0.05]),
                    dtype=np.float32,
                )
                radius_scalar = float(entry.get("radius_scalar", np.linalg.norm(radius_vec)))

                self.cell_stats[(r, a)] = {
                    "mean": mean,
                    "radius_vec": radius_vec,
                    "radius_scalar": radius_scalar,
                    "n": n,
                }

                if n > 0:
                    region_entries.append((mean, radius_vec, radius_scalar, n))

            if region_entries:
                means = np.stack([x[0] for x in region_entries], axis=0)
                radius_vecs = np.stack([x[1] for x in region_entries], axis=0)
                radius_scalars = np.asarray([x[2] for x in region_entries], dtype=np.float64)
                counts = np.asarray([x[3] for x in region_entries], dtype=np.float64)

                weighted_mean = np.sum(means * counts[:, None], axis=0) / np.sum(counts)
                conservative_radius_vec = np.max(radius_vecs, axis=0)
                conservative_radius_scalar = float(np.max(radius_scalars))

                self.region_stats[r] = {
                    "mean": weighted_mean.astype(np.float32),
                    "radius_vec": conservative_radius_vec.astype(np.float32),
                    "radius_scalar": conservative_radius_scalar,
                    "n": int(np.sum(counts)),
                }

    def get(self, region, action):
        region = int(region)
        action = int(action)

        if (region, action) in self.cell_stats:
            cell = self.cell_stats[(region, action)]
            if cell["n"] >= self.min_samples:
                return cell

        if region in self.region_stats:
            return self.region_stats[region]

        return {
            "mean": np.zeros(2, dtype=np.float32),
            "radius_vec": np.ones(2, dtype=np.float32) * 0.05,
            "radius_scalar": float(np.linalg.norm(np.ones(2, dtype=np.float32) * 0.05)),
            "n": 0,
        }

class JsonCombinedLC:
    def __init__(self):
        self.cell_stats = {}
        self.region_stats = {}

    def fit(self, path):
        with open(path, "r") as f:
            summary = json.load(f)

        combined = summary["combined"]

        for region_key, action_map in combined.items():
            r = int(region_key)
            region_vals = []

            for action_key, entry in action_map.items():
                a = int(action_key)

                Lr = float(entry["Lr_sum"])
                Lq = float(entry["LQ_bellman_bound"])

                self.cell_stats[(r, a)] = {
                    "Lr_sum": Lr,
                    "LQ_bellman_bound": Lq
                }

                region_vals.append((Lr, Lq))

            if region_vals:
                # conservative fallback
                Lr_max = max(x[0] for x in region_vals)
                Lq_max = max(x[1] for x in region_vals)

                self.region_stats[r] = {
                    "Lr_sum": Lr_max,
                    "LQ_bellman_bound": Lq_max
                }

    def get(self, region, action):
        if (region, action) in self.cell_stats:
            return self.cell_stats[(region, action)]

        if region in self.region_stats:
            return self.region_stats[region]

        return {"Lr_sum": 0.0, "LQ_bellman_bound": 0.0}
    
# =============================
# Agent (RA-DQN with pruning)
# =============================
class Agent:
    def __init__(
        self,
        bounds_json_path="single_q_bounds_from_json.json",
        summary_json_path="two_reward_dqn_summary.json",
        q_margin_path="Q_single_region.pth",
        min_samples=2,
        tol=1e-5,
    ):
        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.mem = ReplayBuffer()
        self.t = 0

        # Frozen q-margin network trained on mean transition dynamics
        self.q_margin = QNet().to(device)
        self.q_margin.load_state_dict(torch.load(q_margin_path, map_location=device))
        self.q_margin.eval()

        # JSON-backed transition radius
        self.bounds = JsonResolvedBounds(min_samples=min_samples)
        self.bounds.fit(bounds_json_path)

        # JSON-backed empirical Lq
        # self.qmargin_info = JsonEmpiricalQMargin()
        self.qmargin_info = JsonCombinedLC()
        self.qmargin_info.fit(summary_json_path)

        self.tol = tol
        self.prune_cnt = 0

        print(f"Loaded bounds JSON: {bounds_json_path}")
        print(f"Loaded summary JSON: {summary_json_path}")
        print(f"Loaded q_margin network: {q_margin_path}")

    def region_from_state(self, s):
        x, y = s[0], s[1]
        if x >= 0.5 and y >= 0.5:
            return 1
        if x < 0.5 and y >= 0.5:
            return 2
        if x < 0.5 and y < 0.5:
            return 3
        return 4

    def allowed(self, s):
        with torch.no_grad():
            q = self.q_margin(s)   # (B, A)

        B, A = q.shape

        ub = torch.zeros_like(q)
        lb = torch.zeros_like(q)

        for i in range(B):
            state_np = s[i].detach().cpu().numpy()
            r = self.region_from_state(state_np)

            margins = []
            for a in range(A):
                b = self.bounds.get(r, a)
                lc = self.qmargin_info.get(r, a)
            
                rad = float(b["radius_scalar"])
                Lr = float(lc["Lr_sum"])
                Lq = float(lc["LQ_bellman_bound"])
            
                # margin = Lr * rad + (Lq * rad) / (1 - GAMMA)
                margin = (Lr + GAMMA * Lq) * rad / (1 - GAMMA)
                margins.append(margin)

            margins = torch.tensor(margins, dtype=torch.float32, device=device)

            ub[i] = q[i] + margins
            lb[i] = q[i] - margins

        ub = torch.clamp(ub, -100, 100)
        lb = torch.clamp(lb, -100, 100)

        best_lb = lb.max(dim=1, keepdim=True)[0]
        mask = ub >= (best_lb - self.tol)

        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty] = True

        return mask

    # =============================
    # Action selection
    # =============================
    def act(self, s, eps):
        s = torch.tensor(s).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals = self.q(s)
            mask = self.allowed(s)

        q_vals = q_vals.cpu().numpy()[0]
        mask = mask.cpu().numpy()[0]

        allowed = np.where(mask)[0]

        if len(allowed) < 4:
            self.prune_cnt += 4 - len(allowed)

        if random.random() > eps:
            return int(allowed[np.argmax(q_vals[allowed])])

        return int(random.choice(allowed))

    # =============================
    # Step
    # =============================
    def step(self, s, a, r, ns, d):
        self.mem.add(s, a, r, ns, d)
        self.t = (self.t + 1) % UPDATE_EVERY

        if self.t == 0 and len(self.mem) > BATCH_SIZE:
            experiences = self.mem.sample()
            self.learn(experiences)

    # =============================
    # Learn
    # =============================
    def learn(self, experiences, gamma=GAMMA, rank_coef=0.01, rank_margin=1.0):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            q_local_next = self.q(next_states)
            q_target_next = self.qt(next_states)

            allowed_next = self.allowed(next_states)
            masked_q_local_next = q_local_next.masked_fill(~allowed_next, -1e9)

            best_next_actions = masked_q_local_next.argmax(dim=1, keepdim=True)
            q_targets_next = q_target_next.gather(1, best_next_actions)
            q_targets = rewards + gamma * q_targets_next * (1 - dones)

        q_all = self.q(states)
        q_expected = q_all.gather(1, actions)

        td_loss = F.mse_loss(q_expected, q_targets)

        # ranking loss
        allowed_curr = self.allowed(states)
        pruned_curr = ~allowed_curr

        q_allowed = q_all.masked_fill(~allowed_curr, -1e9)
        best_allowed_idx = q_allowed.argmax(dim=1, keepdim=True)
        q_best_allowed = q_all.gather(1, best_allowed_idx)

        ranking_violations = F.relu(q_all - q_best_allowed + rank_margin)
        ranking_violations = ranking_violations * pruned_curr.float()

        num_pruned = pruned_curr.float().sum()
        if num_pruned > 0:
            rank_loss = ranking_violations.sum() / num_pruned
        else:
            rank_loss = torch.tensor(0.0, device=states.device)

        loss = td_loss + rank_coef * rank_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        self.soft_update(self.q, self.qt, TAU)

    # =============================
    # Target update
    # =============================
    @torch.no_grad()
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.lerp_(local_param.data, tau)


# =============================
# Env helpers
# =============================
def make_env(render=False):
    render_mode = "human" if render else None
    auto = bool(render)
    base_env = ContinuousDollarEuroEnv(render_mode=render_mode, auto_render=auto)
    env = ScalarizeReward(base_env, weights=(1.0, 1.0))
    return env


def dqn_fixed_steps(
    agent,
    env,
    n_steps=200_000,
    max_t=200,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.995,
    test_interval=2_000,
    test_runs=5,
):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    state = env.reset()[0]
    score = 0.0
    step_in_ep = 0

    for step in range(1, n_steps + 1):
        step_in_ep += 1
        action = agent.act(state, eps=eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        if done or step_in_ep % max_t == 0:
            scores_window.append(score)
            state = env.reset()[0]
            score = 0.0
            step_in_ep = 0
            eps = max(eps_end, eps_decay * eps)
            print(f"\r Pruned: {agent.prune_cnt} ", end="")
            agent.prune_cnt = 0

        if step % test_interval == 0:
            avg_test_score = test_agent(agent, test_runs=test_runs, max_t=max_t)
            scores.append(avg_test_score)
            print(
                f"Step {step:>7} | Avg(100) {np.mean(scores_window):6.3f} "
                f"| eps {eps:5.3f} | Test {avg_test_score:6.3f}"
            )

    return scores


def test_agent(agent, test_runs=5, max_t=200, render=False):
    env_test = make_env(render=render)
    test_scores = []

    for _ in range(test_runs):
        state = env_test.reset(seed=0)[0]
        score = 0.0
        done = False
        steps = 0

        while not done and steps < max_t:
            steps += 1
            action = agent.act(state, eps=0.0)
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            done = terminated or truncated
            state = next_state
            score += reward

        test_scores.append(score)

    env_test.close()
    return float(np.mean(test_scores))


# =============================
# Main
# =============================
if __name__ == "__main__":
    env = make_env(render=False)

    runs = 5
    N_STEPS = 150_000
    TEST_EVERY = 2000
    ts = np.zeros((runs, int(N_STEPS / TEST_EVERY)))

    for i in range(runs):
        agent = Agent(
            bounds_json_path="single_q_bounds_from_json.json",
            summary_json_path="two_reward_dqn_outputs//two_reward_dqn_summary.json",
            q_margin_path="Q_single_region.pth",
            min_samples=2,
            tol=1e-5,
        )

        test_scores = dqn_fixed_steps(
            agent, env,
            n_steps=N_STEPS,
            max_t=100,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=0.995,
            test_interval=TEST_EVERY,
            test_runs=5,
        )

        ts[i] = test_scores

        plt.figure()
        plt.plot(np.arange(len(test_scores)) * TEST_EVERY, test_scores)
        plt.xlabel("Env steps")
        plt.ylabel("Test return (R1+R2)")
        plt.title("DQN pruned using JSON bounds + empirical Lq")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # torch.save(agent.q.state_dict(), "radqn.pth")
        # np.save("radqn.npy", np.array(ts))
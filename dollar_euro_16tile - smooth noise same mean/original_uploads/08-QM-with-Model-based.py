import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ContinuousDollarEuroEnv_radial import ContinuousDollarEuroEnv, ScalarizeReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # DQN params matched to QM
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 5e-4
    batch_size: int = 256
    buffer_size: int = int(1e5)
    update_every: int = 4

    # Training on model only
    train_steps: int = 200_000

    # Epsilon schedule matched to QM
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

    # Evaluation
    eval_every: int = 20_00
    eval_episodes: int = 10
    max_t: int = 100

    # Dynamics model mode
    # "sa"  -> learn P(s'|s,a)
    # "zsa" -> learn P(s'|z,s,a)
    dynamics_mode: str = "sa"

    # Optional MOReL-style pessimism
    pessimistic: bool = False
    beta: float = 0.05
    penalty: float = -10.0

    # Save files
    q_path: str = "morel_q.pth"
    dyn_path: str = "morel_dynamics.npz"
    
    returns_path: str = "morel_returns.npy"
    returns_plot_path: str = "morel_returns.png"


# ============================================================
# Q Network
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


# ============================================================
# Offline dataset helper
# ============================================================
# class OfflineDataset:
#     def __init__(self, path: str = "offline_transitions.npz"):
#         data = np.load(path)
#         self.states = data["states"].astype(np.float64)
#         self.actions = data["actions"].astype(np.int64)
#         self.next_states = data["next_states"].astype(np.float64)
#         self.rewards = data["rewards"].astype(np.float64)
#         self.regions = data["region"].astype(np.int64)
#         self.terminated = data["terminated"].astype(bool)
#         self.truncated = data["truncated"].astype(bool)

import pickle
from dataclasses import dataclass

@dataclass
class TransitionRecord:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    region: int
class OfflineDataset:
    def __init__(
        self,
        path: str = r"D:\Region based Lq from sources\two_reward_dqn_outputs\two_reward_dqn_raw_transitions.pkl",
    ):
        with open(path, "rb") as f:
            data = pickle.load(f)

        # combine both behaviors
        transitions = data["q1_transitions"] + data["q2_transitions"]

        # extract arrays (vectorized)
        self.states = np.array([tr.state for tr in transitions], dtype=np.float64)
        self.actions = np.array([tr.action for tr in transitions], dtype=np.int64)
        self.next_states = np.array([tr.next_state for tr in transitions], dtype=np.float64)
        self.regions = np.array([tr.region for tr in transitions], dtype=np.int64)
        self.terminated = np.array([tr.done for tr in transitions], dtype=bool)
        self.truncated = np.zeros_like(self.terminated)

        # vectorized reward computation
        self.rewards = self._compute_rewards_vectorized(self.next_states)

    def _compute_rewards_vectorized(self, S):
        """
        S: (N, 2) next_states
        returns: (N,) scalar reward = R1 + R2
        """
        env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)

        # ---- centers ----
        def center(rect):
            x0, x1, y0, y1 = rect
            return np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float64)

        c_d = center(env.rect_dollar)
        c_b = center(env.rect_both)
        c_e = center(env.rect_euro)

        # ---- kernel function (vectorized) ----
        def kernel(c, X):
            diff = X - c  # (N,2)
            d2 = np.sum(diff * diff, axis=1)  # (N,)
            return np.exp(-d2 / (2 * env.rbf_sigma**2))

        kd = kernel(c_d, S)
        kb = kernel(c_b, S)
        ke = kernel(c_e, S)

        # ---- rewards ----
        r1 = env._a_coeffs[0]*kd + env._a_coeffs[1]*kb + env._a_coeffs[2]*ke
        r2 = env._b_coeffs[0]*kd + env._b_coeffs[1]*kb + env._b_coeffs[2]*ke

        # ---- subtract living penalty ----
        half_tau = 0.5 * env.tau
        r1 -= half_tau
        r2 -= half_tau

        # ---- terminal mask (vectorized) ----
        terminal = np.array([env._which_terminal(s) != "none" for s in S])

        # ---- clamp non-terminal positives ----
        mask = ~terminal
        r1[mask] = np.minimum(r1[mask], 0.0)
        r2[mask] = np.minimum(r2[mask], 0.0)

        return (r1 + r2).astype(np.float64)


# ============================================================
# Gaussian dynamics model
#   mode = "sa"  : P(s'|s,a)
#   mode = "zsa" : P(s'|z,s,a)
# ============================================================
class GaussianDynamics:
    def __init__(self, reg: float = 1e-4, mode: str = "sa"):
        if mode not in {"sa", "zsa"}:
            raise ValueError("mode must be 'sa' or 'zsa'")

        self.reg = reg
        self.mode = mode

        # model key -> affine weights and residual covariance
        # W is (3, 2): [s_x, s_y, 1] -> next_state
        self.models: Dict[Tuple, np.ndarray] = {}
        self.covs: Dict[Tuple, np.ndarray] = {}
        self.counts: Dict[Tuple, int] = {}

    @staticmethod
    def _fit_affine(S: np.ndarray, SN: np.ndarray, reg: float) -> np.ndarray:
        X = np.concatenate([S, np.ones((len(S), 1))], axis=1)  # (n, 3)
        XtX = X.T @ X + reg * np.eye(3)
        W = np.linalg.solve(XtX, X.T @ SN)  # (3, 2)
        return W

    @staticmethod
    def _residual_cov(S: np.ndarray, SN: np.ndarray, W: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        X = np.concatenate([S, np.ones((len(S), 1))], axis=1)
        pred = X @ W
        R = SN - pred
        if len(R) < 2:
            return np.eye(2, dtype=np.float64) * 0.01
        cov = np.cov(R.T, bias=False)
        cov = np.asarray(cov, dtype=np.float64)
        return cov + eps * np.eye(2, dtype=np.float64)

    def _key(self, region: int, action: int):
        if self.mode == "sa":
            return (int(action),)
        return (int(region), int(action))

    def fit(self, dataset: OfflineDataset):
        self.models.clear()
        self.covs.clear()
        self.counts.clear()

        if self.mode == "sa":
            # One model per action: P(s' | s, a)
            for a in [0, 1, 2, 3]:
                idx = np.where(dataset.actions == a)[0]
                if len(idx) == 0:
                    continue
                S = dataset.states[idx]
                SN = dataset.next_states[idx]
                key = self._key(region=0, action=a)
                W = self._fit_affine(S, SN, self.reg)
                self.models[key] = W
                self.covs[key] = self._residual_cov(S, SN, W)
                self.counts[key] = len(idx)
        else:
            # One model per region-action: P(s' | z, s, a)
            for z in [1, 2, 3, 4]:
                for a in [0, 1, 2, 3]:
                    idx = np.where((dataset.regions == z) & (dataset.actions == a))[0]
                    if len(idx) == 0:
                        continue
                    S = dataset.states[idx]
                    SN = dataset.next_states[idx]
                    key = self._key(region=z, action=a)
                    W = self._fit_affine(S, SN, self.reg)
                    self.models[key] = W
                    self.covs[key] = self._residual_cov(S, SN, W)
                    self.counts[key] = len(idx)

    def _lookup(self, region: int, action: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        key = self._key(region, action)
        if key in self.models:
            return self.models[key], self.covs[key], self.counts.get(key, 0)

        # Simple backup if a key is missing.
        # This keeps the code runnable even if a bucket has no data.
        if self.mode == "zsa":
            # fall back to action-only model if available
            akey = (int(action),)
            if akey in self.models:
                return self.models[akey], self.covs[akey], self.counts.get(akey, 0)

        return None, None, 0

    def predict_params(self, s: np.ndarray, region: int, action: int):
        W, cov, n = self._lookup(region, action)
        if W is None or cov is None:
            return None, None, 0
        x = np.append(s, 1.0)
        mu = x @ W
        return mu.astype(np.float64), cov.astype(np.float64), n

    def sample_next_state(self, s: np.ndarray, region: int, action: int, rng: np.random.Generator):
        mu, cov, n = self.predict_params(s, region, action)
        if mu is None:
            return None
        sn = rng.multivariate_normal(mean=mu, cov=cov)
        sn = np.clip(sn, 0.0, 1.0)
        return sn.astype(np.float32)

    def uncertainty(self, s: np.ndarray, region: int, action: int) -> float:
        _, cov, _ = self.predict_params(s, region, action)
        if cov is None:
            return float("inf")
        return float(np.trace(cov))

    def save(self, path: str):
        np.savez_compressed(
            path,
            mode=np.array(self.mode),
            reg=np.array(self.reg),
            models=np.array(self.models, dtype=object),
            covs=np.array(self.covs, dtype=object),
            counts=np.array(self.counts, dtype=object),
        )

    @staticmethod
    def load(path: str) -> "GaussianDynamics":
        data = np.load(path, allow_pickle=True)
        obj = GaussianDynamics(reg=float(data["reg"]), mode=str(data["mode"]))
        obj.models = data["models"].item()
        obj.covs = data["covs"].item()
        obj.counts = data["counts"].item()
        return obj


# ============================================================
# Model environment
# ============================================================
class ModelEnv:
    def __init__(self, env: ContinuousDollarEuroEnv, dynamics: GaussianDynamics, pessimistic: bool = False, beta: float = 0.05, penalty: float = -10.0):
        self.env = env
        self.dynamics = dynamics
        self.pessimistic = pessimistic
        self.beta = beta
        self.penalty = penalty
        self.rng = np.random.default_rng(0)

    def region(self, s):
        return self.env._region_id(s)

    def reward(self, s):
        def center(rect):
            x0, x1, y0, y1 = rect
            return np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float64)

        c_d = center(self.env.rect_dollar)
        c_b = center(self.env.rect_both)
        c_e = center(self.env.rect_euro)

        def k(c, x):
            return np.exp(-np.sum((x - c) ** 2) / (2 * self.env.rbf_sigma ** 2))

        kd = k(c_d, s)
        kb = k(c_b, s)
        ke = k(c_e, s)

        r1 = self.env._a_coeffs @ np.array([kd, kb, ke], dtype=np.float64)
        r2 = self.env._b_coeffs @ np.array([kd, kb, ke], dtype=np.float64)

        half_tau = 0.5 * self.env.tau
        r1 -= half_tau
        r2 -= half_tau

        if self.env._which_terminal(s) == "none":
            r1 = min(r1, 0.0)
            r2 = min(r2, 0.0)

        return float(r1 + r2)

    def step(self, s, a):
        z = self.region(s)
        unc = self.dynamics.uncertainty(s, z, a)

        if self.pessimistic and unc > self.beta:
            return None, self.penalty, True

        sn = self.dynamics.sample_next_state(s, z, a, self.rng)
        if sn is None:
            return None, self.penalty, True

        r = self.reward(sn)
        done = self.env._which_terminal(sn) != "none"
        return sn, r, done


# ============================================================
# Replay buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, size=100000):
        self.mem = deque(maxlen=size)

    def add(self, s, a, r, ns, d):
        self.mem.append((s, a, r, ns, d))

    def sample(self, batch=256):
        batch = random.sample(self.mem, batch)
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


# ============================================================
# Agent
# ============================================================
# ============================================================
# JSON-backed pruning info
# ============================================================
import json
from pathlib import Path


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

                mean = np.asarray(
                    entry.get("mean_delta", [0.0, 0.0]),
                    dtype=np.float32,
                )

                radius_vec = np.asarray(
                    entry.get(
                        "student_t_conf_radius_mean_delta",
                        [0.05, 0.05],
                    ),
                    dtype=np.float32,
                )

                radius_scalar = float(
                    entry.get(
                        "radius_scalar",
                        np.linalg.norm(radius_vec),
                    )
                )

                self.cell_stats[(r, a)] = {
                    "mean": mean,
                    "radius_vec": radius_vec,
                    "radius_scalar": radius_scalar,
                    "n": n,
                }

                if n > 0:
                    region_entries.append(
                        (mean, radius_vec, radius_scalar, n)
                    )

            if region_entries:
                means = np.stack([x[0] for x in region_entries], axis=0)
                radius_vecs = np.stack([x[1] for x in region_entries], axis=0)
                radius_scalars = np.asarray(
                    [x[2] for x in region_entries],
                    dtype=np.float64,
                )

                counts = np.asarray(
                    [x[3] for x in region_entries],
                    dtype=np.float64,
                )

                weighted_mean = (
                    np.sum(means * counts[:, None], axis=0)
                    / np.sum(counts)
                )

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
            "radius_scalar": float(
                np.linalg.norm(np.ones(2, dtype=np.float32) * 0.05)
            ),
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
                Lq = float(entry["Lq_empirical_sum"])

                self.cell_stats[(r, a)] = {
                    "Lr_sum": Lr,
                    "Lq_empirical_sum": Lq,
                }

                region_vals.append((Lr, Lq))

            if region_vals:
                Lr_max = max(x[0] for x in region_vals)
                Lq_max = max(x[1] for x in region_vals)

                self.region_stats[r] = {
                    "Lr_sum": Lr_max,
                    "Lq_empirical_sum": Lq_max,
                }

    def get(self, region, action):
        if (region, action) in self.cell_stats:
            return self.cell_stats[(region, action)]

        if region in self.region_stats:
            return self.region_stats[region]

        return {
            "Lr_sum": 0.0,
            "Lq_empirical_sum": 0.0,
        }


# ============================================================
# Agent (MOReL + RA-DQN pruning)
# ============================================================
class Agent:
    def __init__(
        self,
        model_env: ModelEnv,
        cfg: Config,
        bounds_json_path="single_q_bounds_from_json.json",
        summary_json_path="two_reward_dqn_summary.json",
        q_margin_path="Q_single_region.pth",
        min_samples=2,
        tol=1e-5,
    ):
        self.env = model_env
        self.cfg = cfg

        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

        self.mem = ReplayBuffer(cfg.buffer_size)

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.t = 0

        # ====================================================
        # Frozen pruning network
        # ====================================================
        self.q_margin = QNet().to(device)
        self.q_margin.load_state_dict(
            torch.load(q_margin_path, map_location=device)
        )
        self.q_margin.eval()

        # ====================================================
        # Bounds
        # ====================================================
        self.bounds = JsonResolvedBounds(min_samples=min_samples)
        self.bounds.fit(bounds_json_path)

        self.qmargin_info = JsonCombinedLC()
        self.qmargin_info.fit(summary_json_path)

        self.tol = tol
        self.prune_cnt = 0

        print(f"Loaded bounds JSON: {bounds_json_path}")
        print(f"Loaded summary JSON: {summary_json_path}")
        print(f"Loaded q_margin network: {q_margin_path}")

    # ========================================================
    # Region helper
    # ========================================================
    def region_from_state(self, s):
        x, y = s[0], s[1]

        if x >= 0.5 and y >= 0.5:
            return 1

        if x < 0.5 and y >= 0.5:
            return 2

        if x < 0.5 and y < 0.5:
            return 3

        return 4

    # ========================================================
    # Allowed actions
    # ========================================================
    def allowed(self, s):
        with torch.no_grad():
            q = self.q_margin(s)

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
                Lq = float(lc["Lq_empirical_sum"])

                margin = Lr * rad + (Lq * rad) / (1 - self.gamma)

                margins.append(margin)

            margins = torch.tensor(
                margins,
                dtype=torch.float32,
                device=device,
            )

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

    # ========================================================
    # Action selection
    # ========================================================
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

    # ========================================================
    # Step
    # ========================================================
    def step(self, s, a):
        ns, r, done = self.env.step(s, a)

        if ns is None:
            ns = np.zeros(2, dtype=np.float32)

        self.mem.add(s, a, r, ns, done)

        self.t = (self.t + 1) % self.cfg.update_every

        if self.t == 0 and len(self.mem) > self.cfg.batch_size:
            self.learn()

        return ns, r, done

    # ========================================================
    # Learn
    # ========================================================
    def learn(self, rank_coef=0.01, rank_margin=1.0):
        states, actions, rewards, next_states, dones = \
            self.mem.sample(self.cfg.batch_size)

        with torch.no_grad():
            q_local_next = self.q(next_states)
            q_target_next = self.qt(next_states)

            allowed_next = self.allowed(next_states)

            masked_q_local_next = q_local_next.masked_fill(
                ~allowed_next,
                -1e9,
            )

            best_next_actions = masked_q_local_next.argmax(
                dim=1,
                keepdim=True,
            )

            q_targets_next = q_target_next.gather(
                1,
                best_next_actions,
            )

            q_targets = (
                rewards
                + self.gamma * q_targets_next * (1 - dones)
            )

        q_all = self.q(states)

        q_expected = q_all.gather(1, actions)

        td_loss = F.mse_loss(q_expected, q_targets)

        # ====================================================
        # Ranking loss
        # ====================================================
        allowed_curr = self.allowed(states)

        pruned_curr = ~allowed_curr

        q_allowed = q_all.masked_fill(~allowed_curr, -1e9)

        best_allowed_idx = q_allowed.argmax(dim=1, keepdim=True)

        q_best_allowed = q_all.gather(1, best_allowed_idx)

        ranking_violations = F.relu(
            q_all - q_best_allowed + rank_margin
        )

        ranking_violations = (
            ranking_violations * pruned_curr.float()
        )

        num_pruned = pruned_curr.float().sum()

        if num_pruned > 0:
            rank_loss = ranking_violations.sum() / num_pruned
        else:
            rank_loss = torch.tensor(
                0.0,
                device=states.device,
            )

        loss = td_loss + rank_coef * rank_loss

        self.opt.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.q.parameters(),
            1.0,
        )

        self.opt.step()

        self.soft_update(self.q, self.qt, self.tau)

    # ========================================================
    # Soft update
    # ========================================================
    @torch.no_grad()
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(),
            local_model.parameters(),
        ):
            target_param.data.lerp_(local_param.data, tau)

    # ========================================================
    # Save
    # ========================================================
    def save(self, path: str):
        torch.save(
            {
                "q": self.q.state_dict(),
                "qt": self.qt.state_dict(),
                "opt": self.opt.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    # ========================================================
    # Load
    # ========================================================
    def load(self, path: str):
        ckpt = torch.load(path, map_location=device)

        self.q.load_state_dict(ckpt["q"])
        self.qt.load_state_dict(ckpt["qt"])
        self.opt.load_state_dict(ckpt["opt"])
        
# ============================================================
# Eval on real env
# ============================================================
# def eval_true_env(agent: Agent, episodes=10, horizon=200):
#     env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)
#     returns = []

#     for _ in range(episodes):
#         s, _ = env.reset()
#         total = 0.0
#         for _ in range(horizon):
#             a = agent.act(s, eps=0.0)
#             s, r_vec, done, trunc, _ = env.step(a)
#             total += float(r_vec[0] + r_vec[1])
#             if done or trunc:
#                 break
#         returns.append(total)

#     mean = float(np.mean(returns))
#     std = float(np.std(returns))
#     print(f"[TRUE ENV] Avg Return: {mean:.2f} ± {std:.2f}")
#     return mean
def make_env(render=False):
    render_mode = "human" if render else None
    auto = bool(render)
    base_env = ContinuousDollarEuroEnv(render_mode=render_mode, auto_render=auto)
    env = ScalarizeReward(base_env, weights=(1.0, 1.0))
    return env
def eval_true_env(agent: Agent, episodes=10, horizon=100, render=False):
    """
    Evaluate exactly like QM:
      - wrapped env with ScalarizeReward(weights=(1.0, 1.0))
      - reset(seed=0) at episode start
      - done = terminated or truncated
      - greedy action selection
      - stop at horizon=max_t
    """
    env = make_env(render=render)
    returns = []

    for _ in range(episodes):
        s = env.reset(seed=0)[0]
        total = 0.0
        done = False
        steps = 0

        while not done and steps < horizon:
            steps += 1
            a = agent.act(s, eps=0.0)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += float(r)

        returns.append(total)

    env.close()

    mean = float(np.mean(returns))
    std = float(np.std(returns))
    print(f"[TRUE ENV] Avg Return: {mean:.2f} ± {std:.2f}")
    return mean


# ============================================================
# Train on model only, eval on real world, no real learning
# ============================================================
# def train_on_model_only(cfg: Config, offline_path=r"D:\Region based Lq from sources\two_reward_dqn_outputs\two_reward_dqn_raw_transitions.pkl"):
#     data = OfflineDataset(offline_path)
#     env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)

#     dynamics = GaussianDynamics(reg=1e-4, mode=cfg.dynamics_mode)
#     dynamics.fit(data)
#     dynamics.save(cfg.dyn_path)
#     print(f"Saved dynamics to {cfg.dyn_path}")

#     model_env = ModelEnv(
#         env,
#         dynamics,
#         pessimistic=cfg.pessimistic,
#         beta=cfg.beta,
#         penalty=cfg.penalty,
#     )
#     agent = Agent(model_env, cfg)

#     s = np.array([0.5, 0.35], dtype=np.float32)
#     eps = cfg.eps_start

#     for step in range(1, cfg.train_steps + 1):
#         a = agent.act(s, eps)
#         ns, r, done = agent.step(s, a)
#         s = ns

#         if done:
#             s = np.array([0.5, 0.35], dtype=np.float32)

#         eps = max(cfg.eps_end, eps * cfg.eps_decay)

#         # if step % 10_000 == 0:
#         #     print(f"step: {step}")

#         if step % cfg.eval_every == 0:
#             print(f"step: {step}")
#             print("\n--- EVAL ---")
#             eval_true_env(agent, episodes=cfg.eval_episodes, horizon=cfg.max_t)

#     agent.save(cfg.q_path)
#     print(f"Saved Q model to {cfg.q_path}")

def train_on_model_only(cfg: Config, offline_path="offline_transitions.npz"):
    data = OfflineDataset(offline_path)
    env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)

    dynamics = GaussianDynamics(reg=1e-4, mode=cfg.dynamics_mode)
    dynamics.fit(data)
    dynamics.save(cfg.dyn_path)
    print(f"Saved dynamics to {cfg.dyn_path}")

    model_env = ModelEnv(
        env,
        dynamics,
        pessimistic=cfg.pessimistic,
        beta=cfg.beta,
        penalty=cfg.penalty,
    )
    # agent = Agent(model_env, cfg)
    agent = Agent(
        model_env,
        cfg,
        bounds_json_path="single_q_bounds_from_json.json",
        summary_json_path="two_reward_dqn_outputs//two_reward_dqn_summary.json",
        q_margin_path="Q_single_region.pth",
    )

    s = np.array([0.5, 0.35], dtype=np.float32)
    eps = cfg.eps_start

    eval_returns = []

    for step in range(1, cfg.train_steps + 1):
        a = agent.act(s, eps)
        ns, r, done = agent.step(s, a)
        s = ns

        if done:
            s = np.array([0.5, 0.35], dtype=np.float32)

        eps = max(cfg.eps_end, eps * cfg.eps_decay)

        # if step % 10_000 == 0:
        #     print(f"step: {step}")

        if step % cfg.eval_every == 0:
            print(f"step: {step} \t \n--- EVAL ---")
            mean_ret = eval_true_env(agent, episodes=cfg.eval_episodes, horizon=cfg.max_t)
            eval_returns.append(mean_ret)

    agent.save(cfg.q_path)
    print(f"Saved Q model to {cfg.q_path}")

    eval_returns = np.asarray(eval_returns, dtype=np.float64)
    return agent, eval_returns


# if __name__ == "__main__":
#     import argparse

#     p = argparse.ArgumentParser()
#     p.add_argument("--pessimistic", action="store_true", help="Enable MOReL pessimism")
#     p.add_argument("--dynamics_mode", type=str, default="sa", choices=["sa", "zsa"], help="sa = P(s'|s,a), zsa = P(s'|z,s,a)")
#     p.add_argument("--offline_path", type=str, default=r"D:\Region based Lq from sources\two_reward_dqn_outputs\two_reward_dqn_raw_transitions.pkl")
#     p.add_argument("--q_path", type=str, default="morel_q.pth")
#     p.add_argument("--dyn_path", type=str, default="morel_dynamics.npz")
#     args = p.parse_args()

#     cfg = Config(
#         pessimistic=bool(args.pessimistic),
#         dynamics_mode=args.dynamics_mode,
#         q_path=args.q_path,
#         dyn_path=args.dyn_path,
#     )

#     train_on_model_only(cfg, offline_path=args.offline_path)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--pessimistic", action="store_true", help="Enable MOReL pessimism")
    p.add_argument("--dynamics_mode", type=str, default="sa", choices=["sa", "zsa"], help="sa = P(s'|s,a), zsa = P(s'|z,s,a)")
    p.add_argument("--offline_path", type=str, default=r"D:\Region based Lq from sources\two_reward_dqn_outputs\two_reward_dqn_raw_transitions.pkl")
    p.add_argument("--q_path", type=str, default="morel_q.pth")
    p.add_argument("--dyn_path", type=str, default="morel_dynamics.npz")
    p.add_argument("--returns_path", type=str, default="QM_morel_returns.npy")
    p.add_argument("--returns_plot_path", type=str, default="morel_returns.png")
    args = p.parse_args()

    cfg = Config(
        pessimistic=bool(args.pessimistic),
        dynamics_mode=args.dynamics_mode,
        q_path=args.q_path,
        dyn_path=args.dyn_path,
        returns_path=args.returns_path,
        returns_plot_path=args.returns_plot_path,
    )

    agent, eval_returns = train_on_model_only(cfg, offline_path=args.offline_path)

    np.save(cfg.returns_path, eval_returns)
    print(f"Saved eval returns to {cfg.returns_path}")

    x = np.arange(1, len(eval_returns) + 1) * cfg.eval_every
    plt.figure()
    plt.plot(x, eval_returns)
    plt.xlabel("Model env steps")
    plt.ylabel("Test return (R1+R2)")
    plt.title("QM Model-based training: true-env eval")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(cfg.returns_plot_path, dpi=150)
    plt.show()

    print(f"Saved eval plot to {cfg.returns_plot_path}")
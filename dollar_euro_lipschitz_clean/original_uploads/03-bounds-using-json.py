import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ContinuousDollarEuroEnv_radial import ContinuousDollarEuroEnv


# =============================
# Config
# =============================
@dataclass
class Config:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 256
    train_iters: int = 80000

    min_samples: int = 2

    # This should be the resolved bounds JSON built from two_reward_dqn_summary.json
    bounds_json_path: str = "single_q_bounds_from_json.json"

    save_path: str = "Q_single_region.pth"


# =============================
# Q Network
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
# Region Bounds from resolved JSON
# =============================
class RegionBounds:
    """
    Loads already-resolved transition bounds from JSON.

    Expected structure:
        {
          "by_region_action": {
            "1": {
              "0": {
                "n_total": ...,
                "mean_delta": [...],
                "student_t_conf_radius_mean_delta": [...],
                "radius_scalar": ...
              },
              ...
            },
            ...
          }
        }

    No raw dataset is used here.
    """

    def __init__(self, min_samples=2):
        self.min_samples = min_samples
        self.cell_stats = {}
        self.region_stats = {}

    def fit(self, json_path):
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as f:
            bounds = json.load(f)

        by_region_action = bounds["by_region_action"]

        # ---- region-action stats ----
        for region_key, action_map in by_region_action.items():
            r = int(region_key)

            all_means = []
            all_radii = []
            all_counts = []

            for action_key, entry in action_map.items():
                a = int(action_key)

                mean = np.asarray(entry["mean_delta"], dtype=np.float32)
                radius = np.asarray(entry["student_t_conf_radius_mean_delta"], dtype=np.float32)
                n = int(entry["n_total"])

                self.cell_stats[(r, a)] = {
                    "mean": mean,
                    "radius": radius,
                    "n": n
                }

                if n > 0:
                    all_means.append(mean)
                    all_radii.append(radius)
                    all_counts.append(n)

            # ---- fallback region stats ----
            # If region-action entry has too few samples, use region aggregate.
            # Since only resolved per-action bounds are saved, region fallback is
            # built by weighted averaging means and taking conservative radius.
            if len(all_counts) > 0:
                counts = np.asarray(all_counts, dtype=np.float64)
                means = np.stack(all_means, axis=0)
                radii = np.stack(all_radii, axis=0)

                total_n = int(np.sum(counts))
                weighted_mean = np.sum(means * counts[:, None], axis=0) / np.sum(counts)

                # Conservative fallback: max radius over actions
                region_radius = np.max(radii, axis=0)

                self.region_stats[r] = {
                    "mean": weighted_mean.astype(np.float32),
                    "radius": region_radius.astype(np.float32),
                    "n": total_n
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

        # Global-safe default
        return {
            "mean": np.zeros(2, dtype=np.float32),
            "radius": np.ones(2, dtype=np.float32) * 0.05,
            "n": 0
        }


# =============================
# Learner (single Q)
# =============================
class SingleQLearner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False)

        self.bounds = RegionBounds(cfg.min_samples)
        self.bounds.fit(cfg.bounds_json_path)

        self.q = QNet().to(self.device)
        self.q_tgt = QNet().to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

    def get_region(self, s):
        x, y = s[:, 0], s[:, 1]
        r = torch.zeros_like(x, dtype=torch.long)

        r[(x >= 0.5) & (y >= 0.5)] = 1
        r[(x < 0.5) & (y >= 0.5)] = 2
        r[(x < 0.5) & (y < 0.5)] = 3
        r[(x >= 0.5) & (y < 0.5)] = 4

        return r.unsqueeze(1)

    def compute_reward_done(self, next_states):
        rewards = []
        dones = []

        tau = self.env.tau
        rbf_sigma = self.env.rbf_sigma

        def rect_center(rect):
            x0, x1, y0, y1 = rect
            return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)

        c_d = rect_center(self.env.rect_dollar)
        c_b = rect_center(self.env.rect_both)
        c_e = rect_center(self.env.rect_euro)

        def kernel(center, x):
            d2 = np.sum((x - center) ** 2)
            return np.exp(-d2 / (2.0 * (rbf_sigma ** 2)))

        for s in next_states:
            s_np = s.detach().cpu().numpy()

            terminal_type = self.env._which_terminal(s_np)
            done = terminal_type != "none"

            kd = kernel(c_d, s_np)
            kb = kernel(c_b, s_np)
            ke = kernel(c_e, s_np)

            r1 = 1.0 * kd + 0.6 * kb
            r2 = 0.6 * kb + 1.0 * ke

            half_tau = 0.5 * tau
            r1 -= half_tau
            r2 -= half_tau

            if not done:
                if r1 > 0:
                    r1 = 0.0
                if r2 > 0:
                    r2 = 0.0

            reward = r1 + r2

            rewards.append(reward)
            dones.append(float(done))

        return (
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),
        )

    def sample_batch(self):
        s = torch.rand(self.cfg.batch_size, 2, device=self.device)
        a = torch.randint(0, 4, (self.cfg.batch_size, 1), device=self.device)

        region = self.get_region(s)

        means = []
        for i in range(len(s)):
            stats = self.bounds.get(int(region[i].item()), int(a[i].item()))
            means.append(stats["mean"])

        mean = torch.tensor(np.array(means), dtype=torch.float32, device=self.device)

        # Bounds logic uses only the resolved mean next-state change.
        next_state = torch.clamp(s + mean, 0.0, 1.0)

        r, done = self.compute_reward_done(next_state)

        return s, a, r, next_state, done

    def train(self):
        for it in range(1, self.cfg.train_iters + 1):
            s, a, r, ns, done = self.sample_batch()

            with torch.no_grad():
                q_next = self.q_tgt(ns).max(1, keepdim=True)[0]
                target = r + self.cfg.gamma * (1 - done) * q_next

            q_pred = self.q(s).gather(1, a)
            loss = F.mse_loss(q_pred, target)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            for tgt, src in zip(self.q_tgt.parameters(), self.q.parameters()):
                tgt.data.copy_(0.99 * tgt.data + 0.01 * src.data)

            if it % 5000 == 0:
                print(f"[{it}] loss={loss.item():.6f}")

        torch.save(self.q.state_dict(), self.cfg.save_path)
        print("Saved:", self.cfg.save_path)


# =============================
# Run
# =============================
if __name__ == "__main__":
    cfg = Config()
    learner = SingleQLearner(cfg)

    start = time.time()
    learner.train()
    print("Done:", time.time() - start)
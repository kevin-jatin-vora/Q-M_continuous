import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
from scipy.stats import t as student_t

import torch
import torch.nn as nn

from ContinuousDollarEuroEnv_radial import (
    ContinuousDollarEuroEnv,
    ScalarizeReward
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Region Bounds
# ============================================================

class RegionBounds:

    def __init__(self, alpha=0.05, min_samples=2):

        self.alpha = alpha
        self.min_samples = min_samples

        self.cell_stats = {}
        self.region_stats = {}

    def fit(self, path):

        data = np.load(path)

        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        regions = data["region"]

        delta = next_states - states

        # ----------------------------------------------------
        # region-action stats
        # ----------------------------------------------------

        for r in [1,2,3,4]:

            for a in [0,1,2,3]:

                idx = np.where(
                    (regions == r) &
                    (actions == a)
                )[0]

                if len(idx) == 0:
                    continue

                d = delta[idx]

                mean = d.mean(axis=0)

                if len(idx) >= 2:

                    std = d.std(axis=0, ddof=1)

                    n = len(idx)

                    tcrit = student_t.ppf(
                        1 - self.alpha/2,
                        n - 1
                    )

                    radius = tcrit * std * np.sqrt(1 + 1/n)

                else:

                    radius = np.ones(2) * 0.05

                self.cell_stats[(r,a)] = {
                    "mean": mean.astype(np.float32),
                    "radius": radius.astype(np.float32),
                    "n": len(idx)
                }

        # ----------------------------------------------------
        # region fallback
        # ----------------------------------------------------

        for r in [1,2,3,4]:

            idx = np.where(regions == r)[0]

            if len(idx) < 2:
                continue

            d = delta[idx]

            mean = d.mean(axis=0)

            std = d.std(axis=0, ddof=1)

            n = len(idx)

            tcrit = student_t.ppf(
                1 - self.alpha/2,
                n - 1
            )

            radius = tcrit * std * np.sqrt(1 + 1/n)

            self.region_stats[r] = {
                "mean": mean.astype(np.float32),
                "radius": radius.astype(np.float32),
                "n": n
            }

    def get(self, region, action):

        if (region, action) in self.cell_stats:

            cell = self.cell_stats[(region, action)]

            if cell["n"] >= self.min_samples:
                return cell

        if region in self.region_stats:
            return self.region_stats[region]

        return {
            "mean": np.zeros(2, dtype=np.float32),
            "radius": np.ones(2, dtype=np.float32) * 0.05
        }


# ============================================================
# Lipschitz estimation
# ============================================================

def estimate_lipschitz_constants_from_dataset(
    path,
    gamma=0.99,
    confidence=0.95,
    bound="upper"
):

    data = np.load(path)

    states = data["states"].astype(np.float64)
    actions = data["actions"]
    next_states = data["next_states"].astype(np.float64)
    rewards = data["rewards"].astype(np.float64)
    regions = data["region"]

    scalar_rewards = rewards.sum(axis=1)

    lr_samples = []
    lf_samples = []

    for r in [1,2,3,4]:

        for a in [0,1,2,3]:

            idx = np.where(
                (regions == r) &
                (actions == a)
            )[0]

            if len(idx) < 2:
                continue

            s = states[idx]
            ns = next_states[idx]
            rew = scalar_rewards[idx]

            for i in range(len(idx)):

                for j in range(i+1, len(idx)):

                    ds = np.linalg.norm(s[i] - s[j])

                    if ds < 1e-8:
                        continue

                    dr = abs(rew[i] - rew[j])

                    df = np.linalg.norm(ns[i] - ns[j])

                    lr_samples.append(dr / ds)
                    lf_samples.append(df / ds)

    lr_samples = np.asarray(lr_samples)
    lf_samples = np.asarray(lf_samples)

    def estimate(samples):

        mean = float(np.mean(samples))

        std = float(np.std(samples, ddof=1))

        se = std / np.sqrt(len(samples))

        alpha = 1.0 - confidence

        tcrit = float(
            student_t.ppf(
                1.0 - alpha / 2.0,
                len(samples)-1
            )
        )

        if bound == "upper":
            return mean + tcrit * se

        return mean

    Lr = estimate(lr_samples)
    Lf = estimate(lf_samples)

    Lf = min(Lf, (1.0 - 1e-6)/gamma)

    Lq = Lr / (1.0 - gamma * Lf)

    return float(Lr), float(Lf), float(Lq)


# ============================================================
# Allowed action computation
# ============================================================

class AllowedActions:

    def __init__(
        self,
        env,
        offline_path,
        q_margin_path,
        gamma=0.99
    ):

        self.env = env

        self.q_margin = QNet().to(device)

        # --------------------------------------------
        # load SAME pth into q_margin
        # --------------------------------------------

        state = torch.load(
            q_margin_path,
            map_location=device
        )

        if isinstance(state, dict) and "q" in state:
            self.q_margin.load_state_dict(state["q"])
        else:
            self.q_margin.load_state_dict(state)

        self.q_margin.eval()

        self.bounds = RegionBounds()

        self.bounds.fit(offline_path)

        self.Lr, self.Lf, self.Lq = \
            estimate_lipschitz_constants_from_dataset(
                offline_path,
                gamma=gamma
            )

        self.Lq /= 10.0

        self.gamma = gamma

        self.tol = 1e-5

        print(
            f"Lr={self.Lr:.4f} "
            f"Lf={self.Lf:.4f} "
            f"Lq={self.Lq:.4f}"
        )

    def region(self, s):
        return self.env._region_id(s)

    def allowed(self, s_np):

        s_t = torch.tensor(
            s_np,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            q = self.q_margin(s_t)[0]

        region = self.region(s_np)

        ub = torch.zeros(4, device=device)
        lb = torch.zeros(4, device=device)

        for a in range(4):

            stats = self.bounds.get(region, a)

            radius = stats["radius"]

            rad = np.linalg.norm(radius)

            margin = (
                self.Lr * rad +
                ((self.Lq * rad)/(1-self.gamma))
            )

            ub[a] = q[a] + margin
            lb[a] = q[a] - margin

        best_lb = lb.max()

        mask = ub >= (best_lb - self.tol)

        allowed = torch.where(mask)[0].cpu().numpy()

        return allowed, q.cpu().numpy()


# ============================================================
# Main
# ============================================================

Q_PATH = "Q_single_region.pth"
OFFLINE_PATH = "offline_transitions.npz"

env = ContinuousDollarEuroEnv(
    render_mode="human",
    auto_render=True,
    fps=60,
    substeps=12
)

helper = AllowedActions(
    env=env,
    offline_path=OFFLINE_PATH,
    q_margin_path=Q_PATH,
    gamma=0.99
)

obs, info = env.reset(seed=0)

done = False

t = 0
total_reward = 0.0

# ============================================================
# random initialization
# ============================================================

allowed, qvals = helper.allowed(obs)

action = random.choice(allowed)

print(f"initial allowed = {allowed}")

while not done:

    # --------------------------------------------------------
    # recompute allowed actions every step
    # --------------------------------------------------------

    allowed, qvals = helper.allowed(obs)

    # --------------------------------------------------------
    # choose ONLY from allowed actions
    # --------------------------------------------------------

    action = random.choice(allowed)

    print(
        f"t={t} "
        f"allowed={allowed} "
        f"action={action} "
        f"q={np.round(qvals,3)}"
    )

    obs, rew, terminated, truncated, info = env.step(action)

    r1, r2 = rew

    total_reward += (r1 + r2)

    done = terminated or truncated

    t += 1

print("\nEpisode finished")
print("steps =", t)
print("return =", total_reward)

env.close()
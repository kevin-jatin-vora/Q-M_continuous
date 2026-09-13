import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

# =============================
# Config
# =============================
GAMMA = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q_PATH = "Q_single_region.pth"
DATASET_PATH = "offline_transitions.npz"

# If you already know these from data.py, set them directly.
# Otherwise, replace with your estimated values.
Lr = 1.39919
Lq = 70

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
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)

# =============================
# Region Bounds
# =============================
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

        for r in [1, 2, 3, 4]:
            for a in [0, 1, 2, 3]:
                idx = np.where((regions == r) & (actions == a))[0]
                if len(idx) == 0:
                    continue

                d = delta[idx]
                mean = d.mean(axis=0)

                if len(idx) >= 2:
                    std = d.std(axis=0, ddof=1)
                    n = len(idx)
                    tcrit = student_t.ppf(1 - self.alpha / 2, n - 1)
                    radius = tcrit * std * np.sqrt(1 + 1 / n)
                else:
                    radius = np.ones(2, dtype=np.float32) * 0.05

                self.cell_stats[(r, a)] = {
                    "mean": mean.astype(np.float32),
                    "radius": radius.astype(np.float32),
                    "n": len(idx),
                }

        for r in [1, 2, 3, 4]:
            idx = np.where(regions == r)[0]
            if len(idx) < 2:
                continue

            d = delta[idx]
            mean = d.mean(axis=0)
            std = d.std(axis=0, ddof=1)
            n = len(idx)
            tcrit = student_t.ppf(1 - self.alpha / 2, n - 1)
            radius = tcrit * std * np.sqrt(1 + 1 / n)

            self.region_stats[r] = {
                "mean": mean.astype(np.float32),
                "radius": radius.astype(np.float32),
                "n": n,
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
            "radius": np.ones(2, dtype=np.float32) * 0.05,
        }

# =============================
# Helpers
# =============================
def region_from_state(s):
    x, y = float(s[0]), float(s[1])
    if x >= 0.5 and y >= 0.5:
        return 1
    if x < 0.5 and y >= 0.5:
        return 2
    if x < 0.5 and y < 0.5:
        return 3
    return 4

def load_q_model(path=Q_PATH):
    q = QNet().to(device)
    q.load_state_dict(torch.load(path, map_location=device))
    q.eval()
    return q

@torch.no_grad()
def prune_mask(q, bounds, s_tensor, Lr, Lq, gamma=GAMMA, tol=1e-5):
    """
    s_tensor: shape (1, 2)
    returns: bool tensor of shape (1, 4)
    """
    q_vals = q(s_tensor)  # frozen Q from Q_single_region.pth

    B, A = q_vals.shape
    ub = torch.zeros_like(q_vals)
    lb = torch.zeros_like(q_vals)

    for i in range(B):
        state_np = s_tensor[i].detach().cpu().numpy()
        r = region_from_state(state_np)

        margins = []
        for a in range(A):
            stats = bounds.get(r, a)
            radius = stats["radius"]
            rad = np.linalg.norm(radius)
            margin = Lr * rad + ((Lq * rad) / (1.0 - gamma))
            margins.append(margin)

        margins = torch.tensor(margins, dtype=torch.float32, device=device)
        ub[i] = q_vals[i] + margins
        lb[i] = q_vals[i] - margins

    ub = torch.clamp(ub, -100, 100)
    lb = torch.clamp(lb, -100, 100)

    best_lb = lb.max(dim=1, keepdim=True)[0]
    mask = ub >= (best_lb - tol)

    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty] = True

    return mask

# =============================
# Heatmap
# =============================
def plot_remaining_actions_heatmap(grid_size=201, Lr=1.0, Lq=1.0):
    q = load_q_model(Q_PATH)

    bounds = RegionBounds(alpha=0.05, min_samples=2)
    bounds.fit(DATASET_PATH)

    xs = np.linspace(0.0, 1.0, grid_size)
    ys = np.linspace(0.0, 1.0, grid_size)

    remaining = np.zeros((grid_size, grid_size), dtype=np.int32)

    with torch.no_grad():
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                s = torch.tensor([[x, y]], dtype=torch.float32, device=device)
                mask = prune_mask(q, bounds, s, Lr=Lr, Lq=Lq)
                remaining[grid_size - 1 - j, i] = int(mask.sum().item())

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        remaining,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="viridis",
        vmin=0,
        vmax=4,
        interpolation="nearest",
        aspect="equal",
    )
    plt.axvline(0.5, color="white", linestyle="--", linewidth=1)
    plt.axhline(0.5, color="white", linestyle="--", linewidth=1)
    plt.colorbar(im, label="Remaining actions after pruning")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Action Remaining After Pruning (Q_single_region.pth)")
    plt.tight_layout()
    plt.show()

    return remaining

# =============================
# Run
# =============================
if __name__ == "__main__":
    plot_remaining_actions_heatmap(grid_size=201, Lr=Lr, Lq=Lq)
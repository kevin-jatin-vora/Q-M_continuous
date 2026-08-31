import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from dollar_euro_lipschitz.bounds import REGION_CENTER, RegionActionBounds
from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    format_training_args,
    load_config,
    resolve_sigma,
    validate_bounds_sigma,
)
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.rewards import scalar_reward_from_next_states


def main():
    parser = argparse.ArgumentParser(description="Train Q_single on mean transition dynamics.")
    add_environment_arguments(parser)
    parser.add_argument("--bounds", default=None)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "q_single_region.pth"))
    parser.add_argument("--iters", type=int, default=10_000)
    add_training_arguments(parser, "batch_size", "gamma", "lr")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    apply_training_defaults(args, config)
    bounds_path = Path(args.bounds or ROOT / config["bounds_json"])
    validate_bounds_sigma(
        bounds_path,
        sigma,
        allow_mismatch=args.allow_radius_sigma_mismatch,
        context="center-Q mean-transition training",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bounds = RegionActionBounds(bounds_path)
    env = ContinuousDollarEuroEnv(
        render_mode=None, auto_render=False, **env_kwargs(config, sigma)
    )
    print(
        f"dynamics sigma={sigma:.10g}; center Q uses the fixed mean deltas from {bounds_path}; "
        f"{format_training_args(args)}"
    )
    mean_deltas = np.asarray(
        [
            [bounds.get(region, action)["mean"] for action in range(4)]
            for region in range(1, 5)
        ],
        dtype=np.float32,
    )
    q = QNet().to(device)
    q_tgt = QNet().to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    for it in range(1, args.iters + 1):
        states_np = np.random.rand(args.batch_size, 2).astype(np.float32)
        actions_np = np.random.randint(0, 4, size=(args.batch_size, 1), dtype=np.int64)
        right = states_np[:, 0] >= REGION_CENTER[0]
        top = states_np[:, 1] >= REGION_CENTER[1]
        region_indices = np.where(
            top,
            np.where(right, 0, 1),
            np.where(right, 3, 2),
        )
        means = mean_deltas[region_indices, actions_np[:, 0]]
        next_np = np.clip(states_np + means, 0.0, 1.0).astype(np.float32)
        rewards_np, dones_np = scalar_reward_from_next_states(next_np, env)

        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        next_states = torch.as_tensor(next_np, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            target = rewards + args.gamma * (1.0 - dones) * q_tgt(next_states).max(1, keepdim=True)[0]
        pred = q(states).gather(1, actions)
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        for tgt, src in zip(q_tgt.parameters(), q.parameters()):
            tgt.data.lerp_(src.data, 0.01)

        if it % max(1, args.iters // 10) == 0:
            print(f"iter={it} loss={loss.item():.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Train Q_single on mean dynamics s' = s + delta_mean(region(s), a)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from cartpole_ra.bounds import RegionActionBounds
from cartpole_ra.config import add_training_arguments, apply_training_defaults, load_config
from cartpole_ra.env import make_env
from cartpole_ra.models import QNet
from cartpole_ra.regions import StatePartition
from cartpole_ra.replay import ReplayBuffer


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_next_state(state, action, part: StatePartition, bounds: RegionActionBounds):
    region = part.region_of(state)
    mean = bounds.get(region, int(action))["mean"]
    return np.asarray(state, dtype=np.float32) + np.asarray(mean, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train CartPole Q_single on mean dynamics.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    add_training_arguments(parser)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bounds", default=str(ROOT / "data" / "transition_bounds.json"))
    parser.add_argument("--regions", default=str(ROOT / "data" / "regions.json"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "q_single.pth"))
    args = parser.parse_args()
    config = load_config(args.config)
    apply_training_defaults(args, config)
    if args.iters is None:
        args.iters = int(config.get("q_single_iters", config.get("center_q_iters", 30_000)))

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    part = StatePartition.load(args.regions)
    bounds = RegionActionBounds(args.bounds)
    env = make_env(reward_mode="classic", config=config, seed=args.seed)
    q = QNet().to(device)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    mem = ReplayBuffer(args.buffer_size, state_dim=4)

    s, _ = env.reset(seed=args.seed)
    eps = float(args.eps_start)
    episode_step = 0
    for step in range(1, args.iters + 1):
        episode_step += 1
        if np.random.rand() < eps:
            a = int(env.action_space.sample())
        else:
            with torch.inference_mode():
                a = int(torch.argmax(q(torch.as_tensor(s, device=device).unsqueeze(0))[0]).item())
        ns_real, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ns_mean = mean_next_state(s, a, part, bounds)
        mem.add(s, a, r, ns_mean, done)
        s = ns_real
        if done or episode_step >= args.max_t:
            s, _ = env.reset()
            episode_step = 0
            eps = max(args.eps_end, eps * args.eps_decay)

        if len(mem) >= args.batch_size and step % args.update_every == 0:
            states, actions, rewards, next_states, dones = mem.sample(args.batch_size, device)
            with torch.no_grad():
                target = rewards + (1.0 - dones) * args.gamma * q(next_states).max(1, keepdim=True).values
            current = q(states).gather(1, actions)
            loss = F.mse_loss(current, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if step % 5000 == 0:
            print(f"q_single step={step} eps={eps:.3f} buffer={len(mem)}")

    env.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

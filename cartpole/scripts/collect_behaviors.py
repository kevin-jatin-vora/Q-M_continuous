"""Train B1/B2 behavior DQNs and dump transitions for region/stats fitting."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from cartpole_ra.config import apply_training_defaults, load_config
from cartpole_ra.env import make_env
from cartpole_ra.models import QNet
from cartpole_ra.replay import ReplayBuffer
from record_policy_videos import record_greedy_video


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Agent:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        for p in self.qt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.mem = ReplayBuffer(args.buffer_size, state_dim=4)
        self.t = 0

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(2)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            return int(self.q(s).argmax(dim=1).item())

    def step(self, s, a, r, ns, done):
        self.mem.add(s, a, r, ns, done)
        self.t += 1
        if len(self.mem) < self.args.batch_size or self.t % self.args.update_every != 0:
            return
        states, actions, rewards, next_states, dones = self.mem.sample(self.args.batch_size, self.device)
        with torch.no_grad():
            next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
            target = rewards + self.args.gamma * self.qt(next_states).gather(1, next_actions) * (1.0 - dones)
        pred = self.q(states).gather(1, actions)
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for tgt, src in zip(self.qt.parameters(), self.q.parameters()):
                tgt.data.lerp_(src.data, self.args.tau)


def train_behavior(mode: str, args, out_dir: Path, seed: int):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(
        reward_mode=mode,
        seed=seed,
        max_episode_steps=args.max_t,
        angle_threshold_rad=float(getattr(args, "angle_thresh", 0.174)),
        reward_shape=str(getattr(args, "reward_shape", "step")),
        reward_kappa=float(getattr(args, "reward_kappa", 0.174)),
    )
    agent = Agent(args, device)
    eps = float(args.eps_start)
    records = []
    for ep in range(1, args.behavior_episodes + 1):
        s, _ = env.reset(seed=seed + ep)
        for _ in range(args.max_t):
            a = agent.act(s, eps)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.step(s, a, r, ns, done)
            records.append(
                {
                    "state": np.asarray(s, dtype=np.float32),
                    "action": int(a),
                    "reward": float(r),
                    "next_state": np.asarray(ns, dtype=np.float32),
                    "done": bool(done),
                    "behavior": mode,
                }
            )
            s = ns
            if done:
                break
        eps = max(args.eps_end, eps * args.eps_decay)
    env.close()

    states = np.stack([r["state"] for r in records])
    actions = np.asarray([r["action"] for r in records], dtype=np.int64)
    rewards = np.asarray([r["reward"] for r in records], dtype=np.float32)
    next_states = np.stack([r["next_state"] for r in records])
    dones = np.asarray([r["done"] for r in records], dtype=np.bool_)
    np.savez_compressed(
        out_dir / f"transitions_{mode}.npz",
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
    torch.save(agent.q.state_dict(), out_dir / f"q_{mode}.pth")
    print(f"{mode}: {len(records)} transitions -> {out_dir / f'transitions_{mode}.npz'}")
    if bool(getattr(args, "record_videos", False)):
        record_greedy_video(
            out_dir / f"q_{mode}.pth",
            out_dir / "videos",
            name_prefix=f"behavior_{mode}",
            reward_mode=mode,
            config={
                "angle_thresh": float(getattr(args, "angle_thresh", 0.174)),
                "reward_shape": str(getattr(args, "reward_shape", "step")),
                "reward_kappa": float(getattr(args, "reward_kappa", 0.174)),
            },
            episodes=int(getattr(args, "video_episodes", 2)),
            max_steps=int(args.max_t),
            seed=seed,
        )
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "behaviors"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--no-videos", action="store_true", help="skip behavior policy videos")
    args = parser.parse_args()
    config = load_config(args.config)
    apply_training_defaults(args, config)
    args.behavior_episodes = int(
        args.episodes if args.episodes is not None else config.get("behavior_episodes", 200)
    )
    args.max_t = int(config.get("behavior_max_steps", args.max_t))
    args.angle_thresh = float(config.get("angle_thresh", 0.174))
    args.reward_shape = str(config.get("reward_shape", "step"))
    args.reward_kappa = float(config.get("reward_kappa", 0.174))
    args.video_episodes = int(config.get("video_episodes", 2))
    args.record_videos = (not args.no_videos) and bool(config.get("record_behavior_videos", True))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n1 = train_behavior("b1", args, out_dir, args.seed)
    n2 = train_behavior("b2", args, out_dir, args.seed + 1000)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "b1_transitions": n1,
                "b2_transitions": n2,
                "seed_b1": args.seed,
                "seed_b2": args.seed + 1000,
                "reward_notes": {
                    "b1": "angle reward favoring negative pole angle",
                    "b2": "angle reward favoring positive pole angle",
                    "shape": args.reward_shape,
                    "kappa": args.reward_kappa,
                    "angle_thresh": args.angle_thresh,
                },
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()

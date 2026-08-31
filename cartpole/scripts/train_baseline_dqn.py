"""Train baseline Double DQN on classic CartPole."""

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

from cartpole_ra.config import (
    add_training_arguments,
    apply_training_defaults,
    format_training_args,
    load_config,
)
from cartpole_ra.env import make_env
from cartpole_ra.models import QNet
from cartpole_ra.replay import ReplayBuffer


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
        if random.random() < float(eps):
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


def evaluate(agent, episodes, max_t, config):
    env = make_env(reward_mode="classic", config=config, seed=None)
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        total = 0.0
        for _ in range(max_t):
            a = agent.act(s, eps=0.0)
            ns, r, terminated, truncated, _ = env.step(a)
            total += float(r)
            s = ns
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="Baseline CartPole DQN.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    add_training_arguments(parser)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default=str(ROOT / "outputs" / "dqn"))
    args = parser.parse_args()
    config = load_config(args.config)
    apply_training_defaults(args, config)
    if args.steps is None:
        args.steps = int(config.get("agent_steps", 100_000))
    if args.eval_every is None:
        args.eval_every = int(config.get("eval_every", 5_000))
    if args.eval_episodes is None:
        args.eval_episodes = int(config.get("eval_episodes", 20))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(reward_mode="classic", config=config, seed=args.seed)
    agent = Agent(args, device)
    eps = float(args.eps_start)
    print(f"device={device}; {format_training_args(args)}")

    cold = evaluate(agent, args.eval_episodes, args.max_t, config)
    print(f"step=0 cold_start_eval={cold:.3f}")

    s, _ = env.reset(seed=args.seed)
    returns = []
    episode_step = 0
    n_episodes = 0
    for step in range(1, args.steps + 1):
        episode_step += 1
        a = agent.act(s, eps)
        ns, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        agent.step(s, a, r, ns, done)
        s = ns
        if done or episode_step >= args.max_t:
            s, _ = env.reset()
            episode_step = 0
            n_episodes += 1
            eps = max(args.eps_end, eps * args.eps_decay)
        if step % args.eval_every == 0:
            score = evaluate(agent, args.eval_episodes, args.max_t, config)
            returns.append(score)
            print(f"step={step} eps={eps:.3f} episodes={n_episodes} eval={score:.3f}")

    env.close()
    torch.save(agent.q.state_dict(), out_prefix.with_suffix(".pth"))
    np.save(out_prefix.with_suffix(".npy"), np.asarray(returns, dtype=np.float32))
    with out_prefix.with_name(out_prefix.name + "_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "training_seed": args.seed,
                "evaluation_seeded": False,
                "evaluation_episodes": args.eval_episodes,
                "steps": args.steps,
                "cold_start_eval": cold,
            },
            handle,
            indent=2,
        )
    print(f"saved {out_prefix.with_suffix('.pth')}")


if __name__ == "__main__":
    main()

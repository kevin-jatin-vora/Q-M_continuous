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

from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv, ScalarizeReward
from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    format_training_args,
    load_config,
    resolve_sigma,
)
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.replay import ReplayBuffer


def make_env(environment, seed=None):
    env = ScalarizeReward(
        ContinuousDollarEuroEnv(
            render_mode=None, auto_render=False, seed=seed, **environment
        )
    )
    return env


class Agent:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        for parameter in self.qt.parameters():
            parameter.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.mem = ReplayBuffer(args.buffer_size)
        self.t = 0

    def act(self, state, eps):
        if random.random() < eps:
            return random.randint(0, 3)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            return int(self.q(s).argmax(dim=1).item())

    def step(self, s, a, r, ns, done):
        self.mem.add(s, a, r, ns, done)
        self.t = (self.t + 1) % self.args.update_every
        if self.t == 0 and len(self.mem) > self.args.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.mem.sample(self.args.batch_size, self.device)
        with torch.no_grad():
            best_next = self.q(next_states).argmax(dim=1, keepdim=True)
            q_next = self.qt(next_states).gather(1, best_next)
            target = rewards + self.args.gamma * q_next * (1.0 - dones)
        pred = self.q(states).gather(1, actions)
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        for tgt, src in zip(self.qt.parameters(), self.q.parameters()):
            tgt.data.lerp_(src.data, self.args.tau)


def evaluate(agent, episodes, horizon, environment, eval_seed):
    env = make_env(environment, seed=eval_seed)
    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=eval_seed)
        total = 0.0
        for _ in range(horizon):
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
    parser = argparse.ArgumentParser(description="Train baseline Double DQN on the real environment.")
    add_environment_arguments(parser)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=0)
    add_training_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default=str(ROOT / "outputs" / "dqn"))
    args = parser.parse_args()
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    apply_training_defaults(args, config)
    environment = env_kwargs(config, sigma)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"dynamics sigma={sigma:.10g}; env_horizon={environment['horizon']}; "
        f"{format_training_args(args)}"
    )
    env = make_env(environment, seed=args.seed)
    agent = Agent(args, device)

    s, _ = env.reset(seed=args.seed)
    eps = args.eps_start
    returns = []
    episode_step = 0

    for step in range(1, args.steps + 1):
        episode_step += 1
        a = agent.act(s, eps)
        ns, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        agent.step(s, a, r, ns, done)
        s = ns
        if done or episode_step % args.max_t == 0:
            s, _ = env.reset()
            episode_step = 0
            eps = max(args.eps_end, eps * args.eps_decay)
        if step % args.eval_every == 0:
            score = evaluate(agent, args.eval_episodes, args.max_t, environment, args.eval_seed)
            returns.append(score)
            print(f"step={step} eps={eps:.3f} eval_return={score:.3f}")

    env.close()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.q.state_dict(), out_prefix.with_suffix(".pth"))
    np.save(out_prefix.with_suffix(".npy"), np.asarray(returns, dtype=np.float32))
    with out_prefix.with_name(out_prefix.name + "_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dynamics_sigma": sigma,
                "training_seed": args.seed,
                "evaluation_seed_start": args.eval_seed,
                "evaluation_episodes": args.eval_episodes,
                "steps": args.steps,
                "evaluation_interval": args.eval_every,
                "gamma": args.gamma,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "buffer_size": args.buffer_size,
                "target_tau": args.tau,
                "update_every": args.update_every,
                "epsilon_start": args.eps_start,
                "epsilon_end": args.eps_end,
                "epsilon_decay": args.eps_decay,
                "max_steps_per_episode": args.max_t,
                "env_horizon": environment["horizon"],
            },
            handle,
            indent=2,
        )
    print(f"saved {out_prefix.with_suffix('.pth')} and {out_prefix.with_suffix('.npy')}")


if __name__ == "__main__":
    main()

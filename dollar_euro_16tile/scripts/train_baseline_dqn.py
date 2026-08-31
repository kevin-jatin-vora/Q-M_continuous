import argparse
import json
import os
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
    apply_resolved_determinism,
    apply_resolved_deterministic_sigma_scale,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    format_training_args,
    load_config,
    resolve_determinism,
    resolve_deterministic_sigma_scale,
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


def evaluate(agent, episodes, horizon, environment):
    """Greedy eval with unseeded resets (independent stochastic episodes)."""
    env = make_env(environment, seed=None)
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qnet_fingerprint(module) -> str:
    weight = next(module.parameters()).detach().float().cpu().reshape(-1)
    return f"norm={float(weight.norm()):.6g} sum={float(weight.sum()):.6g}"


def clear_prior_run_artifacts(out_prefix: Path) -> None:
    out_prefix = Path(out_prefix)
    for path in (
        out_prefix.with_suffix(".pth"),
        out_prefix.with_suffix(".npy"),
        out_prefix.with_name(out_prefix.name + "_manifest.json"),
    ):
        if path.is_file():
            path.unlink()
            print(f"removed leftover {path.name}")


def begin_fresh_run(args, environment, device):
    """Full reset for one independent seed: RNG, env, nets, optimizer, buffer, epsilon."""
    seed_everything(args.seed)
    env = make_env(environment, seed=args.seed)
    agent = Agent(args, device)
    eps = float(args.eps_start)
    print("=" * 60)
    print(f"FRESH RUN RESET  training_seed={args.seed}")
    print(f"  epsilon reset to {eps:.3f}  (start={args.eps_start}, end={args.eps_end}, decay={args.eps_decay})")
    print(f"  replay buffer size={len(agent.mem)} (empty)")
    print(f"  QNet: {qnet_fingerprint(agent.q)}")
    print("=" * 60)
    return env, agent, eps


def main():
    parser = argparse.ArgumentParser(description="Train baseline Double DQN on the real environment.")
    add_environment_arguments(parser)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=None)
    add_training_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default=str(ROOT / "outputs" / "dqn"))
    args = parser.parse_args()
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    determinism = resolve_determinism(config, args.determinism)
    apply_resolved_determinism(config, determinism)
    det_sigma_scale = resolve_deterministic_sigma_scale(
        config, args.deterministic_sigma_scale
    )
    apply_resolved_deterministic_sigma_scale(config, det_sigma_scale)
    apply_training_defaults(args, config)
    if args.eval_episodes is None:
        args.eval_episodes = int(config.get("eval_episodes", 30))
    environment = env_kwargs(config, sigma)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    clear_prior_run_artifacts(out_prefix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Avoid BLAS oversubscription when many seed jobs run in parallel.
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    print(
        f"dynamics sigma={sigma:.10g}; determinism={determinism:.4g}; "
        f"deterministic_sigma_scale={det_sigma_scale:.4g}; "
        f"device={device}; eval_episodes={args.eval_episodes} (unseeded); "
        f"env_horizon={environment['horizon']}; {format_training_args(args)}"
    )

    # 1) Reset everything for this seed (including epsilon).
    env, agent, eps = begin_fresh_run(args, environment, device)

    # 2) Eval before any learning (must look untrained).
    cold = evaluate(agent, args.eval_episodes, args.max_t, environment)
    print(f"step=0 eps={eps:.3f} episodes=0 cold_start_eval={cold:.3f} (before any training)")
    # Note: some random initializations greedily walk to Both by luck (argmax prefers UP),
    # so cold_start can land near -50 without loading weights. Fingerprint + empty buffer
    # are the fresh-start guarantees; do not abort on cold_start magnitude.

    # 3) Learn, then eval on the schedule.
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
        if done or episode_step % args.max_t == 0:
            s, _ = env.reset()
            episode_step = 0
            n_episodes += 1
            eps = max(args.eps_end, eps * args.eps_decay)
        if step % args.eval_every == 0:
            score = evaluate(agent, args.eval_episodes, args.max_t, environment)
            returns.append(score)
            print(
                f"step={step} eps={eps:.3f} episodes={n_episodes} "
                f"buffer={len(agent.mem)} eval_return={score:.3f}"
            )

    env.close()
    torch.save(agent.q.state_dict(), out_prefix.with_suffix(".pth"))
    np.save(out_prefix.with_suffix(".npy"), np.asarray(returns, dtype=np.float32))
    with out_prefix.with_name(out_prefix.name + "_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dynamics_sigma": sigma,
                "deterministic_sigma_scale": det_sigma_scale,
                "training_seed": args.seed,
                "evaluation_seeded": False,
                "evaluation_episodes": args.eval_episodes,
                "steps": args.steps,
                "evaluation_interval": args.eval_every,
                "cold_start_eval": cold,
                "episodes_completed": n_episodes,
                "final_epsilon": eps,
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

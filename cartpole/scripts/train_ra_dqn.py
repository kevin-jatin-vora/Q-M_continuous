"""Train RA-DQN on classic CartPole with region-based pruning."""

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

from cartpole_ra.bounds import allowed_mask, build_margin_table
from cartpole_ra.config import (
    add_training_arguments,
    apply_training_defaults,
    format_training_args,
    load_config,
)
from cartpole_ra.env import make_env
from cartpole_ra.models import QNet
from cartpole_ra.regions import StatePartition
from cartpole_ra.replay import ReplayBuffer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class PrunedAgent:
    def __init__(self, args, device, part: StatePartition):
        self.args = args
        self.device = device
        self.part = part
        self.q = QNet().to(device)
        self.qt = QNet().to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        for p in self.qt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.mem = ReplayBuffer(args.buffer_size, state_dim=4)
        self.t = 0

        self.q_single = QNet().to(device)
        self.q_single.load_state_dict(torch.load(args.q_single, map_location=device))
        self.q_single.eval()
        for p in self.q_single.parameters():
            p.requires_grad_(False)

        self.legacy_margin = bool(args.legacy_margin)
        self.clamp = not bool(args.no_q_clamp)
        table = build_margin_table(
            args.bounds,
            args.lipschitz,
            args.lq_source,
            args.gamma,
            self.legacy_margin,
            n_regions=part.n_regions,
        )
        self.margin_table = torch.as_tensor(table, dtype=torch.float32, device=device)
        self.rank_coef = float(args.rank_coef)

    @torch.inference_mode()
    def allowed_mask(self, states_np: np.ndarray):
        states = torch.as_tensor(np.asarray(states_np, dtype=np.float32), dtype=torch.float32, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        q_center = self.q_single(states)
        regions = torch.as_tensor(
            self.part.regions_of(states.detach().cpu().numpy()),
            dtype=torch.long,
            device=self.device,
        )
        margins = self.margin_table[regions]
        return allowed_mask(q_center, margins, tol=self.args.tol, clamp=self.clamp)

    def act(self, state, eps: float) -> int:
        mask = self.allowed_mask(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]
        allowed = mask.detach().cpu().numpy().astype(bool)
        if random.random() < float(eps):
            idxs = np.flatnonzero(allowed)
            return int(np.random.choice(idxs if idxs.size else np.arange(2)))
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q = self.q(s)[0]
            q = q.masked_fill(~mask, -1e9)
            return int(torch.argmax(q).item())

    def step(self, s, a, r, ns, done):
        self.mem.add(s, a, r, ns, done)
        self.t += 1
        if len(self.mem) >= self.args.batch_size and self.t % self.args.update_every == 0:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.mem.sample(
            self.args.batch_size, self.device
        )
        with torch.no_grad():
            next_mask = self.allowed_mask(next_states.detach().cpu().numpy())
            next_q_online = self.q(next_states).masked_fill(~next_mask, -1e9)
            next_actions = next_q_online.argmax(1, keepdim=True)
            next_q = self.qt(next_states).gather(1, next_actions)
            target = rewards + (1.0 - dones) * self.args.gamma * next_q
        current = self.q(states).gather(1, actions)
        loss = F.mse_loss(current, target)
        if self.rank_coef > 0:
            mask = self.allowed_mask(states.detach().cpu().numpy())
            q_all = self.q(states)
            q_allowed = q_all.masked_fill(~mask, -1e9)
            log_prob = F.log_softmax(q_allowed, dim=1)
            loss = loss - self.rank_coef * log_prob.gather(1, actions).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for tgt, src in zip(self.qt.parameters(), self.q.parameters()):
                tgt.data.lerp_(src.data, self.args.tau)


def main():
    parser = argparse.ArgumentParser(description="CartPole RA-DQN.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    add_training_arguments(parser)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bounds", default=str(ROOT / "data" / "transition_bounds.json"))
    parser.add_argument("--lipschitz", default=str(ROOT / "data" / "lipschitz_constants.json"))
    parser.add_argument("--regions", default=str(ROOT / "data" / "regions.json"))
    parser.add_argument("--q-single", default=str(ROOT / "outputs" / "q_single.pth"))
    parser.add_argument("--lq-source", choices=["empirical", "theoretical"], default="theoretical")
    parser.add_argument("--legacy-margin", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-q-clamp", action="store_true")
    parser.add_argument("--rank-coef", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    apply_training_defaults(args, config)
    if args.steps is None:
        args.steps = int(config.get("agent_steps", 100_000))
    if args.eval_every is None:
        args.eval_every = int(config.get("eval_every", 5_000))
    if args.eval_episodes is None:
        args.eval_episodes = int(config.get("eval_episodes", 20))
    if args.rank_coef is None:
        args.rank_coef = float(config.get("rank_coef", 0.001))
    if args.legacy_margin is None:
        args.legacy_margin = bool(config.get("legacy_margin", False))
    out_prefix = Path(args.out_prefix) if args.out_prefix else ROOT / "outputs" / f"ra_dqn_{args.lq_source}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    part = StatePartition.load(args.regions)
    env = make_env(reward_mode="classic", config=config, seed=args.seed)
    agent = PrunedAgent(args, device, part)
    eps = float(args.eps_start)
    print(f"device={device}; regions={part.n_regions}; lq={args.lq_source}; {format_training_args(args)}")

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
                "lq_source": args.lq_source,
                "n_regions": part.n_regions,
                "cold_start_eval": cold,
                "rank_coef": args.rank_coef,
            },
            handle,
            indent=2,
        )
    print(f"saved {out_prefix.with_suffix('.pth')}")


if __name__ == "__main__":
    main()

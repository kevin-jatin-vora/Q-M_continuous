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

from dollar_euro_lipschitz.bounds import LipschitzConstants, REGION_CENTER, RegionActionBounds, action_margin, allowed_mask
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
    validate_pruning_provenance,
)
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv, ScalarizeReward
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.replay import ReplayBuffer


def make_env(environment, seed=None):
    return ScalarizeReward(
        ContinuousDollarEuroEnv(
            render_mode=None, auto_render=False, seed=seed, **environment
        )
    )


class PrunedAgent:
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

        q_single_path = Path(args.q_single)
        if not q_single_path.exists():
            raise FileNotFoundError(
                f"Missing {q_single_path}. Run scripts/train_q_single.py first."
            )
        self.q_single = QNet().to(device)
        self.q_single.load_state_dict(torch.load(q_single_path, map_location=device))
        self.q_single.eval()
        for parameter in self.q_single.parameters():
            parameter.requires_grad_(False)

        self.bounds = RegionActionBounds(args.bounds)
        self.constants = LipschitzConstants(args.lipschitz, source=args.lq_source)
        # Empirical and theoretical both use Bellman margin; only Lq source differs.
        if args.legacy_margin is None:
            args.legacy_margin = False
        self.legacy_margin = bool(args.legacy_margin)
        self.clamp = not bool(args.no_q_clamp)
        self.margin_table = torch.as_tensor(
            [
                [
                    action_margin(*self.constants.get(region, action), self.bounds.get(region, action)["radius_scalar"], args.gamma, self.legacy_margin)
                    for action in range(4)
                ]
                for region in range(1, 5)
            ],
            dtype=torch.float32,
            device=device,
        )
        self.pruned_last_episode = 0

    @torch.inference_mode()
    def allowed_mask(self, states):
        q_center = self.q_single(states)
        right = states[:, 0] >= REGION_CENTER[0]
        top = states[:, 1] >= REGION_CENTER[1]
        region_indices = torch.where(
            top,
            torch.where(right, 0, 1),
            torch.where(right, 3, 2),
        )
        margins = self.margin_table[region_indices]
        return allowed_mask(q_center, margins, tol=self.args.tol, clamp=self.clamp)

    def act(self, state, eps):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q_vals = self.q(s)[0]
            allowed = self.allowed_mask(s)[0]
        pruned = int((~allowed).sum().item())
        if pruned:
            self.pruned_last_episode += pruned
        if random.random() < eps:
            idx = torch.nonzero(allowed, as_tuple=False).view(-1)
            return int(idx[torch.randint(idx.numel(), (1,), device=self.device)].item())
        return int(q_vals.masked_fill(~allowed, -1e9).argmax().item())

    def step(self, s, a, r, ns, done):
        self.mem.add(s, a, r, ns, done)
        self.t = (self.t + 1) % self.args.update_every
        if self.t == 0 and len(self.mem) > self.args.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.mem.sample(self.args.batch_size, self.device)
        with torch.no_grad():
            q_local_next = self.q(next_states)
            allowed_next = self.allowed_mask(next_states)
            best_next = q_local_next.masked_fill(~allowed_next, -1e9).argmax(dim=1, keepdim=True)
            q_next = self.qt(next_states).gather(1, best_next)
            target = rewards + self.args.gamma * q_next * (1.0 - dones)

        q_all = self.q(states)
        pred = q_all.gather(1, actions)
        td_loss = F.mse_loss(pred, target)

        allowed_curr = self.allowed_mask(states)
        pruned_curr = ~allowed_curr
        best_allowed = q_all.masked_fill(~allowed_curr, -1e9).argmax(dim=1, keepdim=True)
        q_best_allowed = q_all.gather(1, best_allowed)
        rank_violation = F.relu(q_all - q_best_allowed + self.args.rank_margin) * pruned_curr.float()
        rank_loss = rank_violation.sum() / pruned_curr.float().sum().clamp_min(1.0)

        loss = td_loss + self.args.rank_coef * rank_loss
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
    parser = argparse.ArgumentParser(description="Train RA-DQN with JSON Lipschitz pruning.")
    add_environment_arguments(parser)
    parser.add_argument("--bounds", default=None)
    parser.add_argument("--lipschitz", default=None)
    parser.add_argument("--q-single", default=None)
    parser.add_argument("--lq-source", choices=["empirical", "theoretical"], default="empirical")
    parser.add_argument(
        "--legacy-margin",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, use older additive margin Lr*r + Lq*r/(1-gamma). Default: Bellman (Lr+gamma*Lq)*r/(1-gamma) for both Lq sources.",
    )
    parser.add_argument("--no-q-clamp", action="store_true", help="Do not clamp Q intervals to [-100, 100]")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=0)
    add_training_arguments(parser)
    parser.add_argument("--rank-coef", type=float, default=0.01)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    apply_training_defaults(args, config)
    environment = env_kwargs(config, sigma)
    args.bounds = str(Path(args.bounds or ROOT / config["bounds_json"]))
    args.lipschitz = str(Path(args.lipschitz or ROOT / config["lipschitz_json"]))
    args.q_single = str(Path(args.q_single or ROOT / config["q_single_path"]))
    validate_bounds_sigma(
        args.bounds,
        sigma,
        allow_mismatch=args.allow_radius_sigma_mismatch,
        context="RA-DQN pruning",
    )
    validate_pruning_provenance(args.bounds, args.lipschitz, sigma)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"dynamics sigma={sigma:.10g}; env_horizon={environment['horizon']}; "
        f"{format_training_args(args)}"
    )
    env = make_env(environment, seed=args.seed)
    agent = PrunedAgent(args, device)
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
            agent.pruned_last_episode = 0
        if step % args.eval_every == 0:
            score = evaluate(agent, args.eval_episodes, args.max_t, environment, args.eval_seed)
            returns.append(score)
            print(f"step={step} eps={eps:.3f} eval_return={score:.3f}")

    env.close()
    prefix = args.out_prefix or str(ROOT / "outputs" / f"ra_dqn_{args.lq_source}")
    out_prefix = Path(prefix)
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
                "bounds": args.bounds,
                "lipschitz": args.lipschitz,
                "center_q": args.q_single,
                "lq_source": args.lq_source,
            },
            handle,
            indent=2,
        )
    print(f"saved {out_prefix.with_suffix('.pth')} and {out_prefix.with_suffix('.npy')}")


if __name__ == "__main__":
    main()

"""Record greedy DQN / RA-DQN policy rollouts as MP4 videos for CartPole."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartpole_ra.bounds import allowed_mask, build_margin_table
from cartpole_ra.config import load_config
from cartpole_ra.env import make_env
from cartpole_ra.models import QNet
from cartpole_ra.regions import StatePartition


def greedy_action(model: QNet, state, device: torch.device) -> int:
    with torch.inference_mode():
        q = model(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())


class PrunedGreedyPolicy:
    """Match RA-DQN eval: argmax Q among actions allowed by Q_single +- margins."""

    def __init__(
        self,
        *,
        q_checkpoint: Path,
        q_single_path: Path,
        bounds_path: Path,
        lipschitz_path: Path,
        regions_path: Path,
        lq_source: str,
        gamma: float,
        legacy_margin: bool,
        tol: float,
        clamp: bool,
        device: torch.device,
    ):
        self.device = device
        self.tol = float(tol)
        self.clamp = bool(clamp)
        self.part = StatePartition.load(regions_path)

        self.q = QNet().to(device)
        self.q.load_state_dict(torch.load(q_checkpoint, map_location=device))
        self.q.eval()

        self.q_single = QNet().to(device)
        self.q_single.load_state_dict(torch.load(q_single_path, map_location=device))
        self.q_single.eval()

        table = build_margin_table(
            bounds_path,
            lipschitz_path,
            lq_source,
            gamma,
            legacy_margin,
            n_regions=self.part.n_regions,
        )
        self.margin_table = torch.as_tensor(table, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def __call__(self, state) -> int:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.q(s)[0]
        q_center = self.q_single(s)
        regions = torch.as_tensor(
            self.part.regions_of(s.detach().cpu().numpy()),
            dtype=torch.long,
            device=self.device,
        )
        margins = self.margin_table[regions]
        allowed = allowed_mask(q_center, margins, tol=self.tol, clamp=self.clamp)[0]
        return int(q_vals.masked_fill(~allowed, -1e9).argmax().item())


def record_rollout(
    *,
    choose_action,
    output: Path,
    seed: int,
    fps: int = 30,
    max_steps: int = 500,
    episodes: int = 2,
    config: Optional[dict] = None,
) -> Path:
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    env = make_env(reward_mode="classic", config=config, render_mode="rgb_array")
    all_frames = []
    total_steps = 0
    returns = []

    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed) + ep)
        frame = env.render()
        if frame is not None:
            all_frames.append(frame)

        ep_ret = 0.0
        done = False
        steps = 0
        while not done and steps < int(max_steps):
            action = int(choose_action(obs))
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += float(r)
            steps += 1
            frame = env.render()
            if frame is not None:
                all_frames.append(frame)
            done = bool(terminated or truncated)

        total_steps += steps
        returns.append(ep_ret)

    env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    if all_frames:
        imageio.mimsave(output, all_frames, fps=fps)
        print(f"saved {output.resolve()}  episodes={episodes} avg_return={np.mean(returns):.1f} frames={len(all_frames)}")
    else:
        print(f"warning: no frames captured for {output}")
    return output


def _resolve_checkpoint(experiment_dir: Path, stem: str, seed: int) -> Optional[Path]:
    search_roots = [experiment_dir / "models", experiment_dir]
    for root in search_roots:
        primary = root / f"{stem}.pth"
        if primary.is_file() and primary.stat().st_size > 0:
            return primary
        seeded = root / f"{stem}_seed{seed}.pth"
        if seeded.is_file() and seeded.stat().st_size > 0:
            return seeded
    return None


def main():
    parser = argparse.ArgumentParser(description="Record CartPole policy videos for DQN and RA-DQN.")
    parser.add_argument("--experiment-dir", required=True, help="Directory containing models/ and data/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--skip-empirical", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    video_dir = exp_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    run_cfg_path = exp_dir / "run_config.json"
    if run_cfg_path.is_file():
        with run_cfg_path.open("r", encoding="utf-8") as handle:
            run_cfg = json.load(handle)
        cfg_file = run_cfg.get("config")
        config = load_config(cfg_file) if cfg_file else load_config()
    else:
        config = load_config()

    gamma = float(config.get("gamma", 0.97))
    legacy_margin = bool(config.get("legacy_margin", False))
    device = torch.device("cpu")

    # 1. Baseline DQN
    dqn_ckpt = _resolve_checkpoint(exp_dir, "dqn", args.seed)
    if dqn_ckpt:
        model = QNet().to(device)
        model.load_state_dict(torch.load(dqn_ckpt, map_location=device))
        model.eval()
        record_rollout(
            choose_action=lambda s: greedy_action(model, s, device),
            output=video_dir / "dqn_policy.mp4",
            seed=args.seed,
            fps=args.fps,
            max_steps=args.max_steps,
            episodes=args.episodes,
            config=config,
        )

    # Resolve bounds, lipschitz, regions, q_single for RA-DQN
    bounds_path = exp_dir / "data" / "transition_bounds.json"
    lipschitz_path = exp_dir / "data" / "lipschitz_constants.json"
    regions_path = exp_dir / "data" / "regions.json"
    q_single_path = exp_dir / "models" / "q_single.pth"

    ra_prereqs = [bounds_path, lipschitz_path, regions_path, q_single_path]
    if all(p.is_file() for p in ra_prereqs):
        # 2. Theoretical RA-DQN
        th_ckpt = _resolve_checkpoint(exp_dir, "ra_dqn_theoretical", args.seed)
        if th_ckpt:
            policy_th = PrunedGreedyPolicy(
                q_checkpoint=th_ckpt,
                q_single_path=q_single_path,
                bounds_path=bounds_path,
                lipschitz_path=lipschitz_path,
                regions_path=regions_path,
                lq_source="theoretical",
                gamma=gamma,
                legacy_margin=legacy_margin,
                tol=1e-5,
                clamp=True,
                device=device,
            )
            record_rollout(
                choose_action=policy_th,
                output=video_dir / "ra_dqn_theoretical_policy.mp4",
                seed=args.seed,
                fps=args.fps,
                max_steps=args.max_steps,
                episodes=args.episodes,
                config=config,
            )

        # 3. Empirical RA-DQN
        if not args.skip_empirical:
            emp_ckpt = _resolve_checkpoint(exp_dir, "ra_dqn_empirical", args.seed)
            if emp_ckpt:
                policy_emp = PrunedGreedyPolicy(
                    q_checkpoint=emp_ckpt,
                    q_single_path=q_single_path,
                    bounds_path=bounds_path,
                    lipschitz_path=lipschitz_path,
                    regions_path=regions_path,
                    lq_source="empirical",
                    gamma=gamma,
                    legacy_margin=legacy_margin,
                    tol=1e-5,
                    clamp=True,
                    device=device,
                )
                record_rollout(
                    choose_action=policy_emp,
                    output=video_dir / "ra_dqn_empirical_policy.mp4",
                    seed=args.seed,
                    fps=args.fps,
                    max_steps=args.max_steps,
                    episodes=args.episodes,
                    config=config,
                )


if __name__ == "__main__":
    main()

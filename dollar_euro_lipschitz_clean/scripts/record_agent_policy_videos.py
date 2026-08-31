"""Record greedy DQN / RA-DQN policy rollouts as MP4 videos (rgb_array + imageio)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.bounds import (
    LipschitzConstants,
    REGION_CENTER,
    RegionActionBounds,
    action_margin,
    allowed_mask,
)
from dollar_euro_lipschitz.config import env_kwargs, load_config
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
from dollar_euro_lipschitz.models import QNet


def greedy_action(model: QNet, state, device: torch.device) -> int:
    with torch.inference_mode():
        q = model(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())


class PrunedGreedyPolicy:
    """Match RA-DQN eval: argmax Q among actions allowed by Q_single ± margins."""

    def __init__(
        self,
        *,
        q_checkpoint: Path,
        q_single_path: Path,
        bounds_path: Path,
        lipschitz_path: Path,
        lq_source: str,
        gamma: float,
        tol: float,
        clamp: bool,
        device: torch.device,
    ):
        self.device = device
        self.tol = float(tol)
        self.clamp = bool(clamp)
        self.q = QNet().to(device)
        self.q.load_state_dict(torch.load(q_checkpoint, map_location=device, weights_only=True))
        self.q.eval()
        self.q_single = QNet().to(device)
        self.q_single.load_state_dict(torch.load(q_single_path, map_location=device, weights_only=True))
        self.q_single.eval()
        bounds = RegionActionBounds(bounds_path)
        constants = LipschitzConstants(lipschitz_path, source=lq_source)
        self.margin_table = torch.as_tensor(
            [
                [
                    action_margin(
                        *constants.get(region, action),
                        bounds.get(region, action)["radius_scalar"],
                        gamma,
                        False,
                    )
                    for action in range(4)
                ]
                for region in range(1, 5)
            ],
            dtype=torch.float32,
            device=device,
        )

    @torch.inference_mode()
    def __call__(self, state) -> int:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.q(s)[0]
        q_center = self.q_single(s)
        right = s[:, 0] >= REGION_CENTER[0]
        top = s[:, 1] >= REGION_CENTER[1]
        region_indices = torch.where(top, torch.where(right, 0, 1), torch.where(right, 3, 2))
        margins = self.margin_table[region_indices]
        allowed = allowed_mask(q_center, margins, tol=self.tol, clamp=self.clamp)[0]
        return int(q_vals.masked_fill(~allowed, -1e9).argmax().item())


def record_rollout(
    *,
    choose_action,
    output: Path,
    environment: dict,
    seed: int,
    fps: int,
    substeps: int,
    title: str,
) -> Path:
    env = ContinuousDollarEuroEnv(
        render_mode="rgb_array",
        auto_render=False,
        fps=fps,
        substeps=substeps,
        **environment,
    )
    obs, _ = env.reset(seed=seed)
    if env._renderer is None:
        env.render()
    env._renderer.ax.set_title(title)

    frames = []
    prev = np.asarray(obs, dtype=np.float64)
    frames.append(env.render())

    done = False
    steps = 0
    total_r1 = 0.0
    total_r2 = 0.0
    info = {"terminal": "none"}
    while not done:
        action = int(choose_action(obs))
        obs, reward_vec, terminated, truncated, info = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float64).reshape(-1)
        total_r1 += float(reward_vec[0])
        total_r2 += float(reward_vec[1])
        nxt = np.asarray(obs, dtype=np.float64)
        for i in range(1, substeps + 1):
            alpha = i / float(substeps)
            pos = (1.0 - alpha) * prev + alpha * nxt
            env._renderer.update(pos)
            frames.append(env.render())
        prev = nxt
        done = bool(terminated or truncated)
        steps += 1

    env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps)
    print(
        f"saved {output.resolve()}  steps={steps} frames={len(frames)} "
        f"return_r1={total_r1:.3f} return_r2={total_r2:.3f} terminal={info.get('terminal')}"
    )
    return output


def _resolve_checkpoint(experiment_dir: Path, stem: str, seed: int) -> Optional[Path]:
    primary = experiment_dir / f"{stem}.pth"
    if primary.is_file() and primary.stat().st_size > 0:
        return primary
    seeded = experiment_dir / f"{stem}_seed{seed}.pth"
    if seeded.is_file() and seeded.stat().st_size > 0:
        return seeded
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed and preferred agent seed checkpoint")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--no-q-clamp", action="store_true")
    parser.add_argument(
        "--skip-empirical",
        action="store_true",
        help="Do not record empirical RA-DQN even if its checkpoint exists",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    run_config_path = experiment_dir / "run_config.json"
    if not run_config_path.is_file():
        raise SystemExit(f"missing run_config.json in {experiment_dir}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    config_path = Path(run_config.get("config_path", ROOT / "configs" / "default.json"))
    if not config_path.is_file():
        config_path = ROOT / "configs" / "default.json"
    config = load_config(str(config_path))
    sigma = float(run_config.get("effective_sigma", config["environment"]["sigma"]))
    gamma = float(
        run_config.get(
            "effective_gamma",
            run_config.get("training", {}).get("gamma", config.get("gamma", 0.99)),
        )
    )
    environment = env_kwargs(config, sigma)
    device = torch.device("cpu")
    clamp = not bool(args.no_q_clamp)

    print(f"experiment: {experiment_dir}")
    print(
        f"sigma={sigma:.17g} gamma={gamma:.17g} horizon={environment['horizon']} "
        f"fps={args.fps} substeps={args.substeps}"
    )

    recorded = []
    dqn_ckpt = _resolve_checkpoint(experiment_dir, "dqn", args.seed)
    if dqn_ckpt is None:
        raise SystemExit(f"missing DQN checkpoint dqn.pth or dqn_seed{args.seed}.pth in {experiment_dir}")
    model = QNet().to(device)
    model.load_state_dict(torch.load(dqn_ckpt, map_location=device, weights_only=True))
    model.eval()
    out = record_rollout(
        choose_action=lambda s: greedy_action(model, s, device),
        output=experiment_dir / "dqn_policy.mp4",
        environment=environment,
        seed=args.seed,
        fps=args.fps,
        substeps=args.substeps,
        title=f"DQN greedy (sigma={sigma:g})",
    )
    recorded.append(out)

    bounds = experiment_dir / "transition_bounds.json"
    lipschitz = experiment_dir / "lipschitz_constants.json"
    q_single = experiment_dir / "q_single_region.pth"
    for path in (bounds, lipschitz, q_single):
        if not path.is_file():
            raise SystemExit(f"missing pruning artifact for RA-DQN video: {path}")

    ra_jobs = [("theoretical", "ra_dqn_theoretical", "ra_dqn_theoretical_policy.mp4")]
    if not args.skip_empirical:
        ra_jobs.append(("empirical", "ra_dqn_empirical", "ra_dqn_empirical_policy.mp4"))

    for lq_source, stem, filename in ra_jobs:
        ckpt = _resolve_checkpoint(experiment_dir, stem, args.seed)
        if ckpt is None:
            print(f"skip {filename}: missing {stem}.pth / {stem}_seed{args.seed}.pth")
            continue
        policy = PrunedGreedyPolicy(
            q_checkpoint=ckpt,
            q_single_path=q_single,
            bounds_path=bounds,
            lipschitz_path=lipschitz,
            lq_source=lq_source,
            gamma=gamma,
            tol=args.tol,
            clamp=clamp,
            device=device,
        )
        out = record_rollout(
            choose_action=policy,
            output=experiment_dir / filename,
            environment=environment,
            seed=args.seed,
            fps=args.fps,
            substeps=args.substeps,
            title=f"RA-DQN {lq_source} (sigma={sigma:g})",
        )
        recorded.append(out)

    print(f"recorded {len(recorded)} agent policy video(s)")


if __name__ == "__main__":
    main()

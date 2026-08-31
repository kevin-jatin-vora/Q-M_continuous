"""Record greedy R1/R2 policy rollouts as MP4 videos (rgb_array + imageio)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.config import env_kwargs, load_config
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
from dollar_euro_lipschitz.models import QNet


def greedy_action(model: QNet, state, device: torch.device) -> int:
    with torch.inference_mode():
        q = model(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())


def record_policy_video(
    *,
    checkpoint: Path,
    output: Path,
    environment: dict,
    seed: int,
    fps: int,
    substeps: int,
    title: str,
    device: torch.device,
) -> Path:
    model = QNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

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
    while not done:
        action = greedy_action(model, obs, device)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment folder containing q1_dqn_r1.pth / q2_dqn_r2.pth and run_config.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=12)
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
    environment = env_kwargs(config, sigma)

    q1 = experiment_dir / "q1_dqn_r1.pth"
    q2 = experiment_dir / "q2_dqn_r2.pth"
    for path in (q1, q2):
        if not path.is_file():
            raise SystemExit(f"missing checkpoint: {path}")

    device = torch.device("cpu")
    print(f"experiment: {experiment_dir}")
    print(f"sigma={sigma:.17g} horizon={environment['horizon']} fps={args.fps} substeps={args.substeps}")

    record_policy_video(
        checkpoint=q1,
        output=experiment_dir / "r1_policy.mp4",
        environment=environment,
        seed=args.seed,
        fps=args.fps,
        substeps=args.substeps,
        title=f"R1 greedy policy (sigma={sigma:g})",
        device=device,
    )
    record_policy_video(
        checkpoint=q2,
        output=experiment_dir / "r2_policy.mp4",
        environment=environment,
        seed=args.seed,
        fps=args.fps,
        substeps=args.substeps,
        title=f"R2 greedy policy (sigma={sigma:g})",
        device=device,
    )


if __name__ == "__main__":
    main()

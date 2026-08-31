"""Record greedy policy videos (behavior B1/B2 or a trained agent)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cartpole_ra.env import make_env
from cartpole_ra.models import QNet


def record_greedy_video(
    weights_path,
    out_dir,
    name_prefix: str,
    reward_mode: str = "classic",
    config: Optional[dict] = None,
    episodes: int = 2,
    max_steps: int = 500,
    seed: int = 0,
) -> Optional[Path]:
    """Roll out a greedy policy with rgb_array rendering and save MP4."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("video skipped (imageio unavailable)")
            return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    q = QNet().to(device)
    q.load_state_dict(torch.load(weights_path, map_location=device))
    q.eval()

    wrapper = make_env(reward_mode=reward_mode, config=config, render_mode="rgb_array")
    all_frames = []
    returns = []

    for ep in range(int(episodes)):
        obs, _ = wrapper.reset(seed=int(seed) + ep)
        frame = wrapper.render()
        if frame is not None:
            all_frames.append(frame)
        ep_ret = 0.0
        done = False
        steps = 0
        while not done and steps < int(max_steps):
            with torch.inference_mode():
                a = int(torch.argmax(q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]).item())
            obs, r, terminated, truncated, _ = wrapper.step(a)
            ep_ret += float(r)
            steps += 1
            frame = wrapper.render()
            if frame is not None:
                all_frames.append(frame)
            done = bool(terminated or truncated)
        returns.append(ep_ret)

    wrapper.close()
    out_file = out_dir / f"{name_prefix}.mp4"
    if all_frames:
        imageio.mimsave(out_file, all_frames, fps=30)
        print(f"saved {out_file.resolve()} episodes={episodes} avg_return={np.mean(returns):.1f}")
        return out_file
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Record a greedy CartPole policy video.")
    parser.add_argument("weights")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "videos"))
    parser.add_argument("--name-prefix", default="policy")
    parser.add_argument("--reward-mode", default="classic")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    args = parser.parse_args()

    from cartpole_ra.config import load_config

    record_greedy_video(
        args.weights,
        args.out_dir,
        args.name_prefix,
        reward_mode=args.reward_mode,
        config=load_config(args.config),
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

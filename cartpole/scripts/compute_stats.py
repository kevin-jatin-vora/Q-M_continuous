"""Compute per-(region, action) transition bounds and Lipschitz constants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cartpole_ra.bounds import (
    cell_delta_stats,
    estimate_cell_lipschitz,
    write_bounds_json,
    write_lipschitz_json,
)
from cartpole_ra.config import ACTION_DIM, load_config
from cartpole_ra.models import QNet
from cartpole_ra.regions import StatePartition


def load_pooled(behavior_dir: Path):
    parts = []
    for name in ("transitions_b1.npz", "transitions_b2.npz"):
        path = behavior_dir / name
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/collect_behaviors.py first")
        parts.append(np.load(path))
        print(f"loaded {path.name}: {parts[-1]['states'].shape[0]} transitions")
    return {
        "states": np.concatenate([p["states"] for p in parts], axis=0),
        "actions": np.concatenate([p["actions"] for p in parts], axis=0),
        "rewards": np.concatenate([p["rewards"] for p in parts], axis=0),
        "next_states": np.concatenate([p["next_states"] for p in parts], axis=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--behavior-dir", default=str(ROOT / "outputs" / "behaviors"))
    parser.add_argument("--regions", default=str(ROOT / "data" / "regions.json"))
    parser.add_argument("--bounds-out", default=str(ROOT / "data" / "transition_bounds.json"))
    parser.add_argument("--lipschitz-out", default=str(ROOT / "data" / "lipschitz_constants.json"))
    parser.add_argument("--q-single", default=None, help="Optional Q_single weights for empirical Lq")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    confidence = float(config.get("confidence_level", 0.95))
    gamma = float(config.get("gamma", 0.99))

    part = StatePartition.load(args.regions)
    pooled = load_pooled(Path(args.behavior_dir))
    regions = part.regions_of(pooled["states"])
    actions = pooled["actions"].astype(np.int64)

    device = torch.device("cpu")
    q_net = None
    q_path = Path(args.q_single) if args.q_single else ROOT / "outputs" / "q_single.pth"
    if q_path.is_file():
        q_net = QNet().to(device)
        q_net.load_state_dict(torch.load(q_path, map_location=device))
        q_net.eval()
        print(f"using Q_single for empirical Lq: {q_path}")
    else:
        print("Q_single not found; empirical Lq will be 0 until recomputed")

    by_region_action = {}
    lip_rows = []
    for rid in range(part.n_regions):
        by_region_action[rid] = {}
        for action in range(ACTION_DIM):
            mask = (regions == rid) & (actions == action)
            idx = np.flatnonzero(mask)
            stats = cell_delta_stats(
                pooled["states"][idx],
                pooled["next_states"][idx],
                confidence_level=confidence,
            )
            by_region_action[rid][action] = stats
            if q_net is not None and idx.size:
                with torch.inference_mode():
                    q_all = q_net(
                        torch.as_tensor(pooled["states"][idx], dtype=torch.float32)
                    )
                    q_vals = q_all.gather(
                        1, torch.as_tensor(actions[idx], dtype=torch.long).unsqueeze(1)
                    ).squeeze(1).numpy()
            else:
                q_vals = np.zeros(idx.size, dtype=np.float64)
            lip = estimate_cell_lipschitz(
                pooled["states"][idx],
                pooled["next_states"][idx],
                pooled["rewards"][idx],
                q_vals,
                gamma=gamma,
                seed=args.seed + 1000 * rid + action,
            )
            lip_rows.append(
                {
                    "region": int(rid),
                    "action": int(action),
                    **lip,
                }
            )

    write_bounds_json(
        args.bounds_out,
        by_region_action,
        metadata={
            "env": "CartPole-v1",
            "regions": str(args.regions),
            "confidence_level": confidence,
            "note": "CartPole is deterministic; fine regions => small delta_std => small r.",
        },
    )
    write_lipschitz_json(
        args.lipschitz_out,
        lip_rows,
        gamma=gamma,
        metadata={
            "estimation": "10% trimmed pairwise ratios inside each (region, action)",
            "Lf": "||s'_i - s'_j|| / ||s_i - s_j||",
            "LQ_bellman_bound": "Lr / (1 - gamma*Lf) when defined",
        },
    )
    radii = [
        by_region_action[r][a]["pruning_radius"]
        for r in range(part.n_regions)
        for a in range(ACTION_DIM)
        if by_region_action[r][a]["n_total"] > 0
    ]
    undefined = sum(1 for row in lip_rows if row["LQ_bellman_bound"] is None)
    print(f"saved {args.bounds_out}")
    print(f"saved {args.lipschitz_out}")
    if radii:
        print(
            f"pruning_radius median={np.median(radii):.4g} max={np.max(radii):.4g}; "
            f"undefined Bellman Lq cells={undefined}/{len(lip_rows)}"
        )


if __name__ == "__main__":
    main()

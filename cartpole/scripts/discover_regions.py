"""Discover CartPole regions from pooled B1/B2 transitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cartpole_ra.config import load_config
from cartpole_ra.regions import fit_adaptive_partition, fit_grid_partition


def load_pooled(behavior_dir: Path):
    parts = []
    for name in ("transitions_b1.npz", "transitions_b2.npz"):
        path = behavior_dir / name
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/collect_behaviors.py first")
        parts.append(np.load(path))
    states = np.concatenate([p["states"] for p in parts], axis=0)
    next_states = np.concatenate([p["next_states"] for p in parts], axis=0)
    return states, next_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--behavior-dir", default=str(ROOT / "outputs" / "behaviors"))
    parser.add_argument("--out", default=str(ROOT / "data" / "regions.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    states, next_states = load_pooled(Path(args.behavior_dir))

    method = str(config.get("region_method", "adaptive_tree"))
    if method == "grid":
        bins = tuple(int(x) for x in config.get("grid_bins", [2, 2, 4, 2]))
        part = fit_grid_partition(states, bins_per_dim=bins)
    else:
        part = fit_adaptive_partition(
            states,
            next_states,
            min_leaf=int(config.get("region_min_leaf", 40)),
            max_depth=int(config.get("region_max_depth", 8)),
            max_regions=int(config.get("region_max_regions", 48)),
            var_eps=float(config.get("region_var_eps", 1e-6)),
        )
    part.save(args.out)
    regions = part.regions_of(states)
    print(f"saved {args.out}")
    print(f"method={part.method} n_regions={part.n_regions}")
    uniq, counts = np.unique(regions, return_counts=True)
    print(f"occupied regions={uniq.size}  min_count={int(counts.min())} max_count={int(counts.max())}")


if __name__ == "__main__":
    main()

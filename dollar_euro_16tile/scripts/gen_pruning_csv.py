"""
Temp script: save CSV with per-(x,y,action) pruning details for each sigma* experiment.
Uses theoretical values from the JSON files only (no overrides).

Columns:
  x, y, tile, region, action,
  Q, Lr_sum, Lr1, Lr2, Lf_sum, Lq_empirical, Lq_theoretical,
  radius, gamma,
  margin_theoretical, margin_empirical,
  UB_theoretical, LB_theoretical,
  UB_empirical, LB_empirical
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\dollar_euro_lipschitz_clean\dollar_euro_16tile")
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.bounds import (
    RegionActionBounds,
    set_default_determinism,
    action_margin,
)
from dollar_euro_lipschitz.layout import (
    N_TILES,
    tile_category_map,
    tile_ids_from_states,
)

GRID_SIZE = 201


def load_lip_lookup(lip_path):
    """Build (tile, action) -> dict lookup from lipschitz_constants.json."""
    with open(lip_path, "r") as f:
        data = json.load(f)
    lookup = {}
    for row in data["constants"]:
        key = (int(row["tile"]), int(row["action"]))
        lookup[key] = {
            "Lr1": float(row["Lr1"]),
            "Lr2": float(row["Lr2"]),
            "Lr_sum": float(row["Lr_sum"]),
            "Lf_sum": float(row["Lf_sum"]),
            "Lq_empirical": float(row["Lq_empirical_sum"]),
            "Lq_theoretical": float(row["LQ_bellman_bound"]),
        }
    return lookup


def generate_csv(exp_dir):
    lip_path = exp_dir / "data" / "lipschitz_constants.json"
    bounds_path = exp_dir / "data" / "transition_bounds.json"
    q_path = exp_dir / "models" / "q_single_region.pth"
    rc_path = exp_dir / "data" / "run_config.json"

    if not all(p.exists() for p in [lip_path, bounds_path, q_path, rc_path]):
        print(f"  Missing files, skipping {exp_dir.name}")
        return

    with open(rc_path, "r") as f:
        rc = json.load(f)
    gamma = float(rc["effective_gamma"])
    determinism = float(rc["effective_determinism"])
    set_default_determinism(determinism)

    # Load Q model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNet().to(device)
    model.load_state_dict(torch.load(q_path, map_location=device))
    model.eval()

    # Build grid states
    coords = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    states_np = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    states_t = torch.as_tensor(states_np, device=device)
    with torch.inference_mode():
        q_all = model(states_t).cpu().numpy()  # (N, 4)

    # Tile and region mapping
    tile_ids = tile_ids_from_states(states_np)  # (N,)
    tile_map = tile_category_map(determinism)
    regions = tile_map[tile_ids]  # (N,)

    # Lipschitz lookup and bounds
    lip_lookup = load_lip_lookup(lip_path)
    bounds = RegionActionBounds(bounds_path)

    # Extract sigma tag for filename
    parts = exp_dir.name.split("_cfg_")[0]
    out_path = exp_dir / "plots" / f"pruning_details_{parts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "x", "y", "tile", "region", "action",
        "Q",
        "Lr_sum", "Lr1", "Lr2",
        "Lf_sum",
        "Lq_empirical", "Lq_theoretical",
        "radius", "gamma",
        "margin_theoretical", "margin_empirical",
        "UB_theoretical", "LB_theoretical",
        "UB_empirical", "LB_empirical",
    ]

    n_states = states_np.shape[0]
    print(f"  Writing {n_states * 4} rows to {out_path.name} ...")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_states):
            x_val = float(states_np[i, 0])
            y_val = float(states_np[i, 1])
            tile = int(tile_ids[i])
            region = int(regions[i])

            for action in range(4):
                q_val = float(q_all[i, action])
                lip = lip_lookup.get((tile, action))
                if lip is None:
                    continue

                radius = float(bounds.get(region, action)["pruning_radius"])

                margin_theo = action_margin(
                    lip["Lr_sum"], lip["Lq_theoretical"], radius, gamma, False
                )
                margin_emp = action_margin(
                    lip["Lr_sum"], lip["Lq_empirical"], radius, gamma, False
                )

                writer.writerow({
                    "x": f"{x_val:.5f}",
                    "y": f"{y_val:.5f}",
                    "tile": tile,
                    "region": region,
                    "action": action,
                    "Q": f"{q_val:.6f}",
                    "Lr_sum": f"{lip['Lr_sum']:.6f}",
                    "Lr1": f"{lip['Lr1']:.6f}",
                    "Lr2": f"{lip['Lr2']:.6f}",
                    "Lf_sum": f"{lip['Lf_sum']:.6f}",
                    "Lq_empirical": f"{lip['Lq_empirical']:.6f}",
                    "Lq_theoretical": f"{lip['Lq_theoretical']:.6f}",
                    "radius": f"{radius:.6f}",
                    "gamma": f"{gamma}",
                    "margin_theoretical": f"{margin_theo:.6f}",
                    "margin_empirical": f"{margin_emp:.6f}",
                    "UB_theoretical": f"{q_val + margin_theo:.6f}",
                    "LB_theoretical": f"{q_val - margin_theo:.6f}",
                    "UB_empirical": f"{q_val + margin_emp:.6f}",
                    "LB_empirical": f"{q_val - margin_emp:.6f}",
                })

    print(f"  Saved: {out_path}")


def main():
    base = ROOT / "outputs" / "experiments"
    exp_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("sigma")
    ])

    for exp_dir in exp_dirs:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_dir.name}")
        print(f"{'='*60}")
        generate_csv(exp_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()

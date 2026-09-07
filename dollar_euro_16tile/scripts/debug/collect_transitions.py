"""Collect raw transitions and transition_bounds for LC variants diagnostic."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
from dollar_euro_lipschitz.layout import layout_summary
import train_reward_components as trc


def json_dump(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def make_run_dir(provenance):
    """Per-configuration output dir, kept in sync with run_diagnostic_experiment.py."""
    tag = (
        f"sig{provenance['sigma']:g}_gam{provenance['gamma']:g}_"
        f"det{provenance['determinism']:g}_dsig{provenance['deterministic_sigma_scale']:g}_"
        f"steps{provenance['steps']}_seed{provenance['seed']}_"
        f"r2off{provenance['r2_seed_offset']}"
    )
    run_id = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return ROOT / "outputs" / "temp_cross_category_lipschitz" / f"{tag}_{run_id}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser)
    parser.add_argument("--steps", type=int, default=150_000)
    add_training_arguments(parser)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--min-cell-samples", type=int, default=2)
    parser.add_argument("--include-boundary-transitions", action="store_true")
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--run-dir", default=None, help="Explicit per-run output directory (overrides provenance-based default)")
    
    args, _ = parser.parse_known_args()
    
    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    determinism = resolve_determinism(config, args.determinism)
    apply_resolved_determinism(config, determinism)
    det_sigma_scale = resolve_deterministic_sigma_scale(config, args.deterministic_sigma_scale)
    apply_resolved_deterministic_sigma_scale(config, det_sigma_scale)
    args.deterministic_sigma_scale = det_sigma_scale
    apply_training_defaults(args, config, role="component")
    
    environment = env_kwargs(config, sigma)
    gamma = float(args.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    provenance = {
        "sigma": sigma,
        "gamma": gamma,
        "determinism": determinism,
        "deterministic_sigma_scale": det_sigma_scale,
        "steps": args.steps,
        "seed": args.seed,
        "r2_seed_offset": args.r2_seed_offset,
        "include_boundary_transitions": args.include_boundary_transitions,
        "min_cell_samples": args.min_cell_samples,
    }
    
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = make_run_dir(provenance)
    shared_dir = run_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.resolve()}")
    
    npz_out = shared_dir / "raw_transitions.npz"
    bounds_out = shared_dir / "transition_bounds.json"
    prov_out = shared_dir / "provenance.json"
    
    if args.reuse_data and npz_out.exists() and bounds_out.exists() and prov_out.exists():
        with prov_out.open("r") as f:
            old_prov = json.load(f)
        if all(abs(float(old_prov.get(k, -1)) - float(v)) < 1e-9 if isinstance(v, (int, float)) else old_prov.get(k) == v for k, v in provenance.items()):
            print(f"Reusing existing transitions and bounds in {shared_dir.resolve()}")
            return
        print("Provenance mismatch. Re-collecting data.")
    
    print(f"Collecting R1 and R2 behavior transitions (seed={args.seed}, steps={args.steps})...")
    layout = layout_summary(determinism)
    cat_counts = {int(k): int(v) for k, v in layout["counts"].items()}
    
    q1, records1, tile_records1, next_tile_records1, seconds1 = trc.train_component(
        args, environment, 0, args.seed, device
    )
    q2, records2, tile_records2, next_tile_records2, seconds2 = trc.train_component(
        args, environment, 1, args.seed + args.r2_seed_offset, device
    )
    print(f"R1: {seconds1:.1f}s, R2: {seconds2:.1f}s")
    
    bounds = trc.build_bounds(records1, records2, args, sigma, category_counts=cat_counts)
    json_dump(bounds, bounds_out)
    
    # Also save q1.q, q2.q to compute diagnostic Lq empirical if desired by production functions later
    torch.save(q1.q.state_dict(), shared_dir / "q1_dqn_r1.pth")
    torch.save(q2.q.state_dict(), shared_dir / "q2_dqn_r2.pth")

    npz_arrays = {}
    for behavior, (records, tile_records, next_tile_records) in ((1, (records1, tile_records1, next_tile_records1)), (2, (records2, tile_records2, next_tile_records2))):
        for region in range(1, 6):
            for action in range(4):
                s, ns, r = records.get((region, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                prefix = f"r{behavior}_c{region}_a{action}"
                npz_arrays[f"{prefix}_states"] = s
                npz_arrays[f"{prefix}_next"] = ns
                npz_arrays[f"{prefix}_rewards"] = r

        for tile_id in range(16):
            s, ns, r = next_tile_records.get(tile_id, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            prefix = f"r{behavior}_nxt_t{tile_id}"
            npz_arrays[f"{prefix}_states"] = s
            npz_arrays[f"{prefix}_next"] = ns
            npz_arrays[f"{prefix}_rewards"] = r
            
            for action in range(4):
                s, ns, r = tile_records.get((tile_id, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                prefix = f"r{behavior}_src_t{tile_id}_a{action}"
                npz_arrays[f"{prefix}_states"] = s
                npz_arrays[f"{prefix}_next"] = ns
                npz_arrays[f"{prefix}_rewards"] = r
                
    npz_arrays["meta"] = np.array([json.dumps(provenance)])
    np.savez(npz_out, **npz_arrays)
    json_dump(provenance, prov_out)
    print(f"Saved to {shared_dir.resolve()}")

if __name__ == "__main__":
    main()

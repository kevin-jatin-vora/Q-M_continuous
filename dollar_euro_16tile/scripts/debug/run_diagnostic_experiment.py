"""Orchestrator for Lipschitz diagnostic experiments."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_determinism,
    apply_resolved_deterministic_sigma_scale,
    apply_resolved_sigma,
    apply_training_defaults,
    load_config,
    resolve_determinism,
    resolve_deterministic_sigma_scale,
    resolve_sigma,
)


def run_cmd(cmd, cwd=ROOT):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def make_run_dir(provenance):
    """Stable per-configuration output dir under temp_cross_category_lipschitz.

    Mirrors scripts/debug/collect_transitions.py so both stages agree on the
    directory without exchanging paths.  Every effective parameter that enters
    provenance (sigma, gamma, determinism, deterministic_sigma_scale, steps,
    seed, r2_seed_offset, boundary/min-cell settings) is hashed, so any change
    produces a brand-new directory and previous runs stay intact.
    """
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


def resolve_run_dir(forward_args):
    """Resolve the effective run provenance exactly like collect_transitions.py."""
    probe = argparse.ArgumentParser()
    add_environment_arguments(probe)
    probe.add_argument("--steps", type=int, default=150_000)
    add_training_arguments(probe)
    probe.add_argument("--confidence-level", type=float, default=0.95)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--r2-seed-offset", type=int, default=1_000_000)
    probe.add_argument("--min-cell-samples", type=int, default=2)
    probe.add_argument("--include-boundary-transitions", action="store_true")
    probe.add_argument("--log-every", type=int, default=10_000)
    probe.add_argument("--reuse-data", action="store_true")
    probe.add_argument("--max-pairs", type=int, default=50_000)
    probe.add_argument("--run-dir", default=None)
    parsed, _ = probe.parse_known_args(forward_args)

    config = load_config(parsed.config)
    sigma = resolve_sigma(config, parsed.sigma, parsed.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    determinism = resolve_determinism(config, parsed.determinism)
    apply_resolved_determinism(config, determinism)
    det_sigma_scale = resolve_deterministic_sigma_scale(config, parsed.deterministic_sigma_scale)
    apply_resolved_deterministic_sigma_scale(config, det_sigma_scale)
    parsed.deterministic_sigma_scale = det_sigma_scale
    apply_training_defaults(parsed, config, role="component")

    provenance = {
        "sigma": sigma,
        "gamma": float(parsed.gamma),
        "determinism": determinism,
        "deterministic_sigma_scale": det_sigma_scale,
        "steps": parsed.steps,
        "seed": parsed.seed,
        "r2_seed_offset": parsed.r2_seed_offset,
        "include_boundary_transitions": parsed.include_boundary_transitions,
        "min_cell_samples": parsed.min_cell_samples,
    }
    return make_run_dir(provenance)


def scan_options(argv, valued=(), flags=()):
    """Extract only the options the Q-bound trainer understands.

    Value options are pulled in pairs, store_true flags are passed through,
    and every other token is ignored.  Used so environment/sigma/gamma/iters
    stay consistent between data collection, constants, and Q-bound training.
    """
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in flags:
            out.append(token)
            i += 1
        elif token in valued and i + 1 < len(argv):
            out.extend((token, argv[i + 1]))
            i += 2
        else:
            i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()

    python_exe = sys.executable
    scripts_dir = ROOT / "scripts"
    debug_scripts_dir = scripts_dir / "debug"
    
    # Handle if user passed positional config (e.g. `configs\radial_match.json` instead of `--config ...`)
    forward_args = sys.argv[1:]
    if forward_args and not forward_args[0].startswith("-"):
        forward_args = ["--config", forward_args[0]] + forward_args[1:]

    run_dir = resolve_run_dir(forward_args)
    shared_dir = run_dir / "shared"
    print(f"Run directory: {run_dir.resolve()}")

    # 1. Collect transitions (pass all args through)
    collect_cmd = [
        python_exe, "-u", str(debug_scripts_dir / "collect_transitions.py"),
        "--run-dir", str(run_dir),
    ] + forward_args
    run_cmd(collect_cmd)

    # 2. Compute variants
    # pass all args so compute_lipschitz_variants.py gets --config if provided
    run_cmd([
        python_exe, "-u", str(debug_scripts_dir / "compute_lipschitz_variants.py"),
        "--run-dir", str(run_dir),
    ] + forward_args)
    
    with (shared_dir / "compute_results.json").open("r") as f:
        compute_results = json.load(f)
        
    variants = ["local_trimmed", "global_trimmed", "local_max", "global_max"]
    summary = []
    
    for v in variants:
        print(f"\n{'='*50}\nEvaluating variant: {v}\n{'='*50}")
        is_valid = compute_results.get(v, False)
        
        variant_dir = run_dir / v
        
        if not is_valid:
            print(f"Variant {v} is INVALID (gamma*Lf >= 1). Skipping training.")
            summary.append({
                "method": v,
                "valid": "no",
                "heatmap": "skipped"
            })
            continue
            
        # 3. Train Q bounds
        models_dir = variant_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # We pass a fixed seed for Q-bounds training to guarantee identical sampling
        q_bound_seed = args.seed + 999
        q_ub_path = models_dir / "q_ub_theoretical.pth"
        q_lb_path = models_dir / "q_lb_theoretical.pth"
        training_options = scan_options(
            forward_args,
            valued=(
                "--config", "--sigma", "--stochasticity-scale", "--determinism",
                "--deterministic-sigma-scale", "--gamma", "--iters", "--lq-source",
                "--target-tau", "--max-grad-norm",
            ),
            flags=("--allow-radius-sigma-mismatch", "--no-overlap-aware"),
        )
        train_cmd = [
            python_exe, "-u", str(scripts_dir / "train_q_bounds.py"),
            "--bounds", str(shared_dir / "transition_bounds.json"),
            "--lipschitz", str(variant_dir / "lipschitz_constants.json"),
            "--cross-tile-reward-lipschitz", str(variant_dir / "cross_tile_reward_lipschitz.json"),
            "--cross-category-dynamics-lipschitz", str(variant_dir / "cross_category_dynamics_lipschitz.json"),
            "--seed", str(q_bound_seed),
            "--out-ub", str(q_ub_path),
            "--out-lb", str(q_lb_path),
        ] + training_options
        run_cmd(train_cmd)
        
        # 4. Generate heatmaps
        analysis_dir = variant_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        heatmap_cmd = [
            python_exe, "-u", str(scripts_dir / "generate_q_bound_analysis.py"),
            "--q-ub", str(q_ub_path),
            "--q-lb", str(q_lb_path),
            "--output-dir", str(analysis_dir)
        ]
        run_cmd(heatmap_cmd)
        
        summary.append({
            "method": v,
            "valid": "yes",
            "heatmap": "generated"
        })
        
    print(f"\n{'='*50}\nFINAL SUMMARY\n{'='*50}")
    print(f"{'method':<20} {'valid?':<8} {'heatmap':<15}")
    for row in summary:
        print(f"{row['method']:<20} {row['valid']:<8} {row['heatmap']:<15}")
        
    summary_out = run_dir / "experiment_summary.json"
    with summary_out.open("w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()

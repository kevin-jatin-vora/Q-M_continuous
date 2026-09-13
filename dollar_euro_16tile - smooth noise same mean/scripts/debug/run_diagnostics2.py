"""Controlled local-mean comparison: 10% trim versus minimum pair distance."""

import argparse
import json
import sys
from pathlib import Path

from run_diagnostic_experiment import (
    ROOT,
    gap_dir_name,
    resolve_run_dir,
    run_cmd,
    scan_options,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pair-distance", type=float, default=0.04)
    parser.add_argument("--coverage-bins", type=int, default=40)
    args, _ = parser.parse_known_args()
    if args.min_pair_distance <= 0.0:
        raise SystemExit("--min-pair-distance must be > 0")

    forwarded = sys.argv[1:]
    if forwarded and not forwarded[0].startswith("-"):
        forwarded = ["--config", forwarded[0], *forwarded[1:]]

    python_exe = sys.executable
    scripts_dir = ROOT / "scripts"
    debug_dir = scripts_dir / "debug"
    run_dir = resolve_run_dir(forwarded)
    shared_dir = run_dir / "shared"
    comparison_dir = run_dir / f"diag2_{gap_dir_name(args.min_pair_distance)}"
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Comparison directory: {comparison_dir.resolve()}")

    run_cmd([
        python_exe, "-u", str(debug_dir / "collect_transitions.py"),
        "--run-dir", str(run_dir), *forwarded,
    ])
    run_cmd([
        python_exe, "-u", str(debug_dir / "plot_source_state_counts.py"),
        "--run-dir", str(run_dir), "--coverage-bins", str(args.coverage_bins),
    ])

    experiments = (
        ("exp1_trimmed", "trimmed"),
        ("exp2_gap", "gap"),
    )
    training_options = scan_options(
        forwarded,
        valued=(
            "--config", "--sigma", "--stochasticity-scale", "--determinism",
            "--deterministic-sigma-scale", "--gamma", "--iters", "--lq-source",
            "--target-tau", "--max-grad-norm",
        ),
        flags=("--allow-radius-sigma-mismatch", "--no-overlap-aware"),
    )
    summary = []

    for name, pair_method in experiments:
        experiment_dir = comparison_dir / name
        print(f"\n{'=' * 60}\n{name}: local_mean ({pair_method})\n{'=' * 60}")
        compute_cmd = [
            python_exe, "-u", str(debug_dir / "compute_lipschitz_variants.py"),
            "--run-dir", str(run_dir),
            "--output-root", str(experiment_dir),
            "--variants", "local_mean",
            "--pair-method", pair_method,
            *forwarded,
        ]
        run_cmd(compute_cmd)

        with (experiment_dir / "compute_results.json").open(encoding="utf-8") as handle:
            valid = bool(json.load(handle).get("local_mean", False))
        variant_dir = experiment_dir / "local_mean"
        if not valid:
            print(f"{name} is INVALID (gamma*Lf >= 1); skipping Q-bound training.")
            summary.append({"experiment": name, "valid": False, "analysis": "skipped"})
            continue

        models_dir = variant_dir / "models"
        analysis_dir = variant_dir / "analysis"
        models_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        q_ub = models_dir / "q_ub_theoretical.pth"
        q_lb = models_dir / "q_lb_theoretical.pth"
        run_cmd([
            python_exe, "-u", str(scripts_dir / "train_q_bounds.py"),
            "--bounds", str(shared_dir / "transition_bounds.json"),
            "--lipschitz", str(variant_dir / "lipschitz_constants.json"),
            "--cross-tile-reward-lipschitz", str(variant_dir / "cross_tile_reward_lipschitz.json"),
            "--cross-category-dynamics-lipschitz", str(variant_dir / "cross_category_dynamics_lipschitz.json"),
            "--seed", str(args.seed),
            "--out-ub", str(q_ub),
            "--out-lb", str(q_lb),
            *training_options,
        ])
        run_cmd([
            python_exe, "-u", str(scripts_dir / "generate_q_bound_analysis.py"),
            "--q-ub", str(q_ub), "--q-lb", str(q_lb),
            "--output-dir", str(analysis_dir),
        ])
        summary.append({"experiment": name, "valid": True, "analysis": "generated"})

    comparison_dir.mkdir(parents=True, exist_ok=True)
    with (comparison_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nComparison complete: {comparison_dir.resolve()}")
    for row in summary:
        print(f"  {row['experiment']}: valid={row['valid']}, analysis={row['analysis']}")


if __name__ == "__main__":
    main()

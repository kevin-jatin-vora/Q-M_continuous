"""Run the debug-only raw-sample versus deterministic mean-map diagnostic."""

import argparse
import json
import sys
from pathlib import Path

from run_diagnostic_experiment import ROOT, gap_dir_name, resolve_run_dir, run_cmd, scan_options


def train_and_analyze(name, variant_dir, shared_dir, forwarded, seed, no_overlap):
    python = sys.executable
    debug = ROOT / "scripts" / "debug"
    models = variant_dir / "models"
    analysis = variant_dir / "analysis"
    models.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    q_ub = models / "q_ub_theoretical.pth"
    q_lb = models / "q_lb_theoretical.pth"
    options = scan_options(
        forwarded,
        valued=(
            "--config", "--sigma", "--stochasticity-scale", "--determinism",
            "--deterministic-sigma-scale", "--gamma", "--iters", "--lq-source",
            "--target-tau", "--max-grad-norm", "--batch-size", "--lr",
        ),
        flags=("--allow-radius-sigma-mismatch",),
    )
    command = [
        python, "-u", str(debug / "train_q_bounds_mean_dynamics.py"),
        "--debug-method", name,
        "--bounds", str(shared_dir / "transition_bounds.json"),
        "--lipschitz", str(variant_dir / "lipschitz_constants.json"),
        "--cross-tile-reward-lipschitz", str(variant_dir / "cross_tile_reward_lipschitz.json"),
        "--cross-category-dynamics-lipschitz", str(variant_dir / "cross_category_dynamics_lipschitz.json"),
        "--seed", str(seed), "--out-ub", str(q_ub), "--out-lb", str(q_lb),
    ]
    if no_overlap:
        command.append("--no-overlap-aware")
    command.extend(options)
    run_cmd(command)
    run_cmd([
        python, "-u", str(debug / "generate_q_bound_analysis_mean_dynamics.py"),
        "--q-ub", str(q_ub), "--q-lb", str(q_lb), "--output-dir", str(analysis),
    ])
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pair-distance", type=float, default=0.04)
    parser.add_argument("--coverage-bins", type=int, default=40)
    args, _ = parser.parse_known_args()
    if args.min_pair_distance <= 0:
        raise SystemExit("--min-pair-distance must be > 0")

    forwarded = sys.argv[1:]
    if forwarded and not forwarded[0].startswith("-"):
        forwarded = ["--config", forwarded[0], *forwarded[1:]]
    run_dir = resolve_run_dir(forwarded)
    shared = run_dir / "shared"
    comparison = run_dir / f"mean_dynamics_{gap_dir_name(args.min_pair_distance)}"
    raw_compute = comparison / "raw_local_mean_gap"
    debug = ROOT / "scripts" / "debug"
    python = sys.executable
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Mean-dynamics diagnostic: {comparison.resolve()}")

    run_cmd([python, "-u", str(debug / "collect_transitions.py"), "--run-dir", str(run_dir), *forwarded])
    run_cmd([
        python, "-u", str(debug / "plot_source_state_counts.py"),
        "--run-dir", str(run_dir), "--coverage-bins", str(args.coverage_bins),
        "--reuse-existing",
    ])
    run_cmd([
        python, "-u", str(debug / "compute_lipschitz_variants.py"),
        "--run-dir", str(run_dir), "--output-root", str(raw_compute),
        "--variants", "local_mean", "--pair-method", "gap", *forwarded,
    ])
    raw_variant = raw_compute / "local_mean"
    raw_results = json.loads((raw_compute / "compute_results.json").read_text(encoding="utf-8"))
    required = ("lipschitz_constants.json", "cross_tile_reward_lipschitz.json", "cross_category_dynamics_lipschitz.json")
    missing = [name for name in required if not (raw_variant / name).exists()]
    if missing:
        raise RuntimeError(
            "The raw local-mean computation did not produce the Lr carrier artifacts "
            f"needed by the mean-map experiment: {missing}"
        )

    run_cmd([
        python, "-u", str(debug / "compute_mean_dynamics_lipschitz.py"),
        "--run-dir", str(run_dir), "--baseline-dir", str(raw_variant),
        "--output-root", str(comparison), *forwarded,
    ])
    mean_results = json.loads((comparison / "mean_dynamics_compute_results.json").read_text(encoding="utf-8"))

    arms = [
        ("raw_local_mean_gap", raw_variant, bool(raw_results.get("local_mean")), False),
        ("wasserstein_kernel_local", comparison / "wasserstein_kernel_local", bool(mean_results["wasserstein_kernel_local"]), True),
        ("wasserstein_kernel_cross", comparison / "wasserstein_kernel_cross", bool(mean_results["wasserstein_kernel_cross"]), False),
    ]
    summary = []
    comparison_args = []
    for name, path, valid, no_overlap in arms:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        if not valid:
            print(f"{name} is non-contractive; skipping Q-bound training.")
            summary.append({"method": name, "valid": False, "analysis": "skipped"})
            continue
        analysis = train_and_analyze(name, path, shared, forwarded, args.seed, no_overlap)
        metrics = json.loads((analysis / "analysis_metrics.json").read_text(encoding="utf-8"))
        summary.append({"method": name, "valid": True, "analysis": "generated", **metrics})
        comparison_args.extend(("--arm", f"{name}={analysis}"))

    run_cmd([
        python, "-u", str(debug / "compare_mean_dynamics_pruning.py"),
        "--output-dir", str(comparison / "comparison"), *comparison_args,
    ])
    payload = {
        "run_dir": str(run_dir.resolve()),
        "comparison_dir": str(comparison.resolve()),
        "min_pair_distance": args.min_pair_distance,
        "mean_dynamics_validity": mean_results,
        "arms": summary,
    }
    (comparison / "experiment_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nFINAL SUMMARY")
    for row in summary:
        print(f"  {row['method']}: valid={row['valid']} analysis={row['analysis']}")
    print(f"Saved summary: {(comparison / 'experiment_summary.json').resolve()}")


if __name__ == "__main__":
    main()

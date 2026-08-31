import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.analysis import (
    generate_pruning_heatmap,
    write_table1,
    write_table2,
)
from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    apply_resolved_sigma,
    apply_training_defaults,
    artifact_sigma_tag,
    load_config,
    resolve_sigma,
    validate_bounds_sigma,
    validate_pruning_provenance,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pruning heatmap and publication CSV tables."
    )
    add_environment_arguments(parser)
    parser.add_argument("--bounds", default=None)
    parser.add_argument("--lipschitz", default=None)
    parser.add_argument("--q-single", default=None)
    parser.add_argument("--lq-source", choices=["empirical", "theoretical"], default="empirical")
    parser.add_argument(
        "--legacy-margin",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, use older additive margin. Default: Bellman (Lr+gamma*Lq)*r/(1-gamma) for both Lq sources.",
    )
    parser.add_argument("--no-q-clamp", action="store_true")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    config = load_config(args.config)
    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    apply_training_defaults(args, config)
    bounds = Path(args.bounds or ROOT / config["bounds_json"])
    lipschitz = Path(args.lipschitz or ROOT / config["lipschitz_json"])
    q_single = Path(args.q_single or ROOT / config["q_single_path"])
    gamma = float(config["gamma"] if args.gamma is None else args.gamma)
    validate_bounds_sigma(
        bounds,
        sigma,
        allow_mismatch=args.allow_radius_sigma_mismatch,
        context="pruning heatmap",
    )
    validate_pruning_provenance(bounds, lipschitz, sigma)
    if not q_single.exists():
        raise FileNotFoundError(f"Missing trained center Q model: {q_single}")

    # Empirical and theoretical both use Bellman margin; only Lq source differs.
    if args.legacy_margin is None:
        args.legacy_margin = False
    output_dir = Path(args.output_dir)
    suffix = f"{args.lq_source}_sigma_{artifact_sigma_tag(sigma)}"
    heatmaps = generate_pruning_heatmap(
        q_single,
        bounds,
        lipschitz,
        output_dir / f"pruning_heatmap_{suffix}",
        lq_source=args.lq_source,
        gamma=gamma,
        tol=args.tol,
        grid_size=args.grid_size,
        legacy_margin=bool(args.legacy_margin),
        clamp=not args.no_q_clamp,
        formats=args.formats,
        dpi=args.dpi,
    )
    table1 = write_table1(lipschitz, output_dir / f"table1_lipschitz_constants_{suffix}.csv")
    table2 = write_table2(bounds, output_dir / f"table2_transition_bounds_{suffix}.csv")
    print(f"runtime dynamics sigma={sigma:.10g}")
    for path in [*heatmaps, table1, table2]:
        print(f"saved {path.resolve()}")


if __name__ == "__main__":
    main()

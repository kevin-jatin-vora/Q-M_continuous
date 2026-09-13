"""Plot combined R1/R2 source-state sample counts, one heatmap per action."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def combined_states_for_action(npz, action: int) -> np.ndarray:
    """Combine each stored transition exactly once across behavior/category."""
    groups = []
    for behavior in (1, 2):
        for category in range(1, 6):
            states = np.asarray(
                npz[f"r{behavior}_c{category}_a{action}_states"],
                dtype=np.float64,
            )
            if states.size:
                groups.append(states.reshape(-1, 2))
    return np.concatenate(groups, axis=0) if groups else np.empty((0, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--coverage-bins", type=int, default=40)
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Keep an existing complete four-action heatmap set (useful when a PNG is open on Windows).",
    )
    args, _ = parser.parse_known_args()

    if args.coverage_bins < 1:
        raise SystemExit("--coverage-bins must be >= 1")

    shared_dir = Path(args.run_dir) / "shared"
    raw_path = shared_dir / "raw_transitions.npz"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw transition data: {raw_path}")

    shared_dir.mkdir(parents=True, exist_ok=True)
    expected = [shared_dir / f"source_state_counts_action_{action}.png" for action in range(4)]
    if args.reuse_existing and all(path.is_file() for path in expected):
        print("Reusing existing shared source-state count heatmaps:")
        for path in expected:
            print(f"  {path.resolve()}")
        return
    with np.load(raw_path, allow_pickle=False) as npz:
        states_by_action = [combined_states_for_action(npz, action) for action in range(4)]

    histograms = []
    for states in states_by_action:
        histogram, _, _ = np.histogram2d(
            states[:, 0] if states.size else np.empty(0),
            states[:, 1] if states.size else np.empty(0),
            bins=args.coverage_bins,
            range=((0.0, 1.0), (0.0, 1.0)),
        )
        histograms.append(histogram.T)

    common_max = max((float(hist.max()) for hist in histograms), default=0.0)
    common_max = max(common_max, 1.0)
    action_names = ("down", "up", "left", "right")

    for action, (name, states, histogram) in enumerate(
        zip(action_names, states_by_action, histograms)
    ):
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        image = ax.imshow(
            histogram,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="Blues",
            vmin=0.0,
            vmax=common_max,
            interpolation="nearest",
            aspect="equal",
        )
        for boundary in (0.25, 0.5, 0.75):
            ax.axvline(boundary, color="black", linewidth=0.5, alpha=0.35)
            ax.axhline(boundary, color="black", linewidth=0.5, alpha=0.35)
        ax.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="source-state x",
            ylabel="source-state y",
            title=(
                f"Combined R1 + R2 source-state samples: action {action} ({name})\n"
                f"n={states.shape[0]:,}, bins={args.coverage_bins}x{args.coverage_bins}"
            ),
        )
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("sample count per bin")
        fig.tight_layout()
        output = expected[action]
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()

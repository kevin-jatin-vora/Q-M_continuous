"""Compare pruning grids produced by valid diagnostic arms."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", action="append", default=[], help="NAME=analysis-directory")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arms = []
    for item in args.arm:
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        metrics = json.loads((path / "analysis_metrics.json").read_text(encoding="utf-8"))
        grid = np.load(path / "surviving_actions.npy")
        arms.append((name, path, metrics, grid))
    if not arms:
        json_path = output / "pruning_comparison.json"
        json_path.write_text(json.dumps({"valid_arms": []}, indent=2), encoding="utf-8")
        return

    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4.5), squeeze=False)
    for axis, (name, _, _, grid) in zip(axes[0], arms):
        image = axis.imshow(grid, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=4, aspect="equal")
        axis.set(xlabel="x", ylabel="y", title=name)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="actions surviving", shrink=0.85)
    fig.savefig(output / "pruning_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4.5), squeeze=False)
    for axis, (name, path, _, _) in zip(axes[0], arms):
        violations = np.load(path / "ordering_violations.npy")
        image = axis.imshow(violations, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=4, aspect="equal")
        axis.set(xlabel="x", ylabel="y", title=name)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="actions with Q_LB > Q_UB", shrink=0.85)
    fig.savefig(output / "ordering_violation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    reference_name, _, _, reference = arms[0]
    differences = []
    for arm_index, (name, _, _, grid) in enumerate(arms[1:], start=1):
        diff = grid.astype(np.int16) - reference.astype(np.int16)
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(diff, origin="lower", extent=[0, 1, 0, 1], vmin=-4, vmax=4, cmap="coolwarm", aspect="equal")
        axis.set(xlabel="x", ylabel="y", title=f"{name} minus {reference_name}")
        fig.colorbar(image, ax=axis, label="change in actions surviving")
        fig.tight_layout()
        difference_path = output / f"pruning_difference_{arm_index}_minus_0.png"
        fig.savefig(difference_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        differences.append({
            "file": str(difference_path.resolve()),
            "arm": name,
            "reference": reference_name,
        })

    summary = {"reference_arm": reference_name, "difference_heatmaps": differences, "valid_arms": []}
    for name, path, metrics, grid in arms:
        summary["valid_arms"].append({
            "name": name, "analysis_dir": str(path.resolve()), **metrics,
            "mean_surviving_difference_from_reference": float(np.mean(grid.astype(float) - reference.astype(float))),
        })
    (output / "pruning_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_fields = (
        "name", "mean_surviving_actions", "mean_surviving_difference_from_reference",
        "empty_action_states", "empty_action_state_percent", "ordering_violations",
        "ordering_violation_percent", "width_min", "width_mean", "width_max",
    )
    with (output / "pruning_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["valid_arms"])
    print(f"Saved pruning comparison to {output.resolve()}")


if __name__ == "__main__":
    main()

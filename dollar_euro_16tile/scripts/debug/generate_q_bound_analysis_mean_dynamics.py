"""Generate Q-bound heatmaps plus machine-readable pruning diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.q_bounds import learned_bound_allowed_mask, load_frozen_qnet


def save_img(values, path, title, label):
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
    axis.set(xlabel="x", ylabel="y", title=title)
    fig.colorbar(image, ax=axis, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-ub", required=True)
    parser.add_argument("--q-lb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ub = load_frozen_qnet(args.q_ub, QNet, device)
    lb = load_frozen_qnet(args.q_lb, QNet, device)
    coords = np.linspace(0, 1, args.grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    states = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    with torch.inference_mode():
        tensor = torch.as_tensor(states, device=device)
        qu, ql = ub(tensor), lb(tensor)
        allowed = learned_bound_allowed_mask(qu, ql, args.tol)
        widths = qu - ql
        surviving = allowed.sum(1).cpu().numpy().astype(np.int16)
        violations = (ql > qu).sum(1).cpu().numpy().astype(np.int16)
        vu = qu.max(1).values.cpu().numpy()
        vl = ql.max(1).values.cpu().numpy()
        mean_width = widths.mean(1).cpu().numpy()
        all_widths = widths.cpu().numpy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shape = (args.grid_size, args.grid_size)
    save_img(vu.reshape(shape), output / "q_ub_value_heatmap.png", "Learned upper bound: max_a Q_UB", "max_a Q_UB")
    save_img(vl.reshape(shape), output / "q_lb_value_heatmap.png", "Learned lower bound: max_a Q_LB", "max_a Q_LB")
    save_img(mean_width.reshape(shape), output / "q_bound_width_heatmap.png", "Mean action-wise Q_UB - Q_LB", "mean width")
    save_img(violations.reshape(shape), output / "q_bound_violation_heatmap.png", "Ordering violations (diagnostic only)", "count Q_LB > Q_UB")
    save_img(surviving.reshape(shape), output / "pruning_heatmap_learned_bounds.png", "Actions surviving learned Q bounds", "actions")
    np.save(output / "surviving_actions.npy", surviving.reshape(shape))
    np.save(output / "ordering_violations.npy", violations.reshape(shape))
    histogram = {str(k): int(np.sum(surviving == k)) for k in range(5)}
    metrics = {
        "grid_size": args.grid_size,
        "states": int(surviving.size),
        "action_values": int(all_widths.size),
        "ordering_violations": int(np.sum(all_widths < 0)),
        "ordering_violation_percent": float(100.0 * np.mean(all_widths < 0)),
        "width_min": float(np.min(all_widths)),
        "width_mean": float(np.mean(all_widths)),
        "width_max": float(np.max(all_widths)),
        "surviving_action_histogram": histogram,
        "mean_surviving_actions": float(np.mean(surviving)),
        "empty_action_states": int(np.sum(surviving == 0)),
        "empty_action_state_percent": float(100.0 * np.mean(surviving == 0)),
    }
    (output / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"ordering violations: {metrics['ordering_violations']}/{metrics['action_values']} ({metrics['ordering_violation_percent']:.6f}%)")
    print(f"min width={metrics['width_min']:.6g} mean width={metrics['width_mean']:.6g}")
    print(f"saved diagnostics to {output.resolve()}")


if __name__ == "__main__":
    main()

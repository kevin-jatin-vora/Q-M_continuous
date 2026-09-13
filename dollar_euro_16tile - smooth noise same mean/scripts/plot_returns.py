"""Return plots in the style of CI_DE_RT_plots5.py: mean line plus shaded 95% t CI."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sns.set_style("darkgrid")


def load_returns(path):
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise ValueError(f"{path} must be a 1D curve or a 2D array of shape (n_runs, n_evals)")
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, np.nan)
    return arr


def confidence_interval(data, confidence=0.95):
    """Half-width of a two-sided Student-t CI, matching CI_DE_RT_plots5.py."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return 0.0
    se = stats.sem(data)
    return float(se * stats.t.ppf((1.0 + confidence) / 2.0, n - 1))


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean return with a shaded 95% t CI, matching CI_DE_RT_plots5.py."
    )
    parser.add_argument("files", nargs="*", default=["outputs/dqn.npy", "outputs/ra_dqn_empirical.npy"])
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--step", type=int, default=2_000)
    parser.add_argument("--out", default="outputs/returns_plot.png")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be in (0, 1)")

    labels = args.labels or [Path(file).stem for file in args.files]
    figure, axis = plt.subplots(figsize=(8, 5))
    color_map = {}
    for index, (file, label) in enumerate(zip(args.files, labels)):
        returns = load_returns(file)
        if returns.size == 0 or returns.shape[1] == 0:
            continue
        mean_rewards = np.nanmean(returns, axis=0)
        ci_values = np.array(
            [confidence_interval(column, args.confidence) for column in returns.T],
            dtype=float,
        )
        x_axis = np.arange(mean_rewards.shape[0]) * args.step
        if label not in color_map:
            color_map[label] = plt.cm.tab10(index)
        color = color_map[label]
        axis.plot(x_axis, mean_rewards, label=label, color=color)
        axis.fill_between(
            x_axis,
            mean_rewards - ci_values,
            mean_rewards + ci_values,
            alpha=0.25,
            color=color,
        )

    axis.set_xlabel("Step")
    axis.set_ylabel("Average Return")
    axis.set_title("Dollar-Euro DQN comparison")
    axis.xaxis.set_major_locator(plt.MaxNLocator(6))
    axis.yaxis.set_major_locator(plt.MaxNLocator(6))
    axis.legend()
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(figure)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0.0
    se = stats.sem(data)
    return se * stats.t.ppf((1 + confidence) / 2.0, n - 1)


def trim_to_nonzero_runs(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    mask = np.any(arr != 0.0, axis=1)
    trimmed = arr[mask]
    if trimmed.shape[0] == 0:
        raise ValueError("No non-zero runs found in array.")
    return trimmed


def main():
    sns.set_style("darkgrid")

    files = [
        "radqn.npy",
        "dqn.npy",
    ]
    legend = ["RA-DQN", "DQN"]

    smooth_w = 8
    step = 2500

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, file in enumerate(files):
        data = np.load(file)
        data = trim_to_nonzero_runs(data)

        mean_rewards = np.mean(data, axis=0).flatten()
        smoothed_mean = np.convolve(mean_rewards, np.ones(smooth_w), "valid") / smooth_w

        ci = []
        for t in range(data.shape[1]):
            ci.append(confidence_interval(data[:, t]))
        ci = np.asarray(ci)
        smoothed_ci = np.convolve(ci, np.ones(smooth_w), "valid") / smooth_w

        x_axis = np.arange(smoothed_mean.shape[0]) * step
        ax.plot(x_axis, smoothed_mean, label=f"{legend[i]} (runs={data.shape[0]})")
        ax.fill_between(x_axis, smoothed_mean - smoothed_ci, smoothed_mean + smoothed_ci, alpha=0.25)

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Average Return", fontsize=16)
    ax.set_title("Cartpole (Non-zero runs only) - Noise=1e-5", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    # out_dir = Path("debug")
    # out_dir.mkdir(exist_ok=True)
    # out_path = out_dir / "returns_plot_margin_noise_nonzero.png"
    # plt.savefig(out_path, dpi=200)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

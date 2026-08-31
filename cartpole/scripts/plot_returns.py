"""Plot mean ± stderr evaluation curves from *_runs.npy files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="*_runs.npy arrays shaped (n_seeds, n_evals)")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        raise SystemExit("labels must match number of run files")

    plt.figure(figsize=(8, 5))
    for path, label in zip(args.runs, args.labels):
        data = np.load(path).astype(np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        mean = data.mean(axis=0)
        stderr = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0]) if data.shape[0] > 1 else np.zeros_like(mean)
        xs = (np.arange(mean.size) + 1) * int(args.step)
        plt.plot(xs, mean, label=label)
        plt.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    plt.xlabel("env steps")
    plt.ylabel("eval return (classic CartPole)")
    plt.title("CartPole DQN vs RA-DQN")
    plt.legend()
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

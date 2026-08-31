"""End-to-end CartPole Lipschitz RA-DQN experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartpole_ra.config import load_config


def artifacts_ready(paths) -> bool:
    for path in paths:
        p = Path(path)
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    return True


def run_stage(name, command, dry_run: bool, env=None):
    printable = subprocess.list2cmdline([str(c) for c in command])
    print(f"\n=== {name} ===\n{printable}", flush=True)
    if dry_run:
        return
    completed = subprocess.run([str(c) for c in command], cwd=ROOT, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed with exit code {completed.returncode}: {name}")


def _worker_env(parallel: bool):
    if not parallel:
        return None
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[key] = "1"
    return env


def train_runs(
    name,
    builder,
    prefix: Path,
    n_runs: int,
    base_seed: int,
    dry_run: bool,
    resume: bool = False,
    parallel_runs: int = 1,
):
    runs_path = prefix.with_name(prefix.name + "_runs").with_suffix(".npy")
    if resume and artifacts_ready([runs_path, prefix.with_suffix(".pth"), prefix.with_suffix(".npy")]):
        print(f"\n=== {name} skipped; artifacts already present ===")
        return runs_path

    curves_by_seed = {}
    pending = []
    for run in range(n_runs):
        seed = base_seed + run
        run_prefix = prefix.parent / f"{prefix.name}_seed{seed}"
        run_artifacts = [run_prefix.with_suffix(".pth"), run_prefix.with_suffix(".npy")]
        if resume and artifacts_ready(run_artifacts):
            print(f"\n=== {name} run {run + 1}/{n_runs} (seed={seed}) skipped; artifacts present ===")
            curves_by_seed[seed] = np.load(run_prefix.with_suffix(".npy")).astype(np.float32).reshape(-1)
            continue
        pending.append((run, seed, run_prefix))

    workers = max(1, min(int(parallel_runs) if parallel_runs > 0 else (os.cpu_count() or 1), max(1, len(pending))))
    worker_env = _worker_env(workers > 1)

    def execute(job):
        run, seed, run_prefix = job
        run_stage(
            f"{name} run {run + 1}/{n_runs} (seed={seed})",
            builder(seed, run_prefix),
            dry_run,
            env=worker_env,
        )
        if dry_run:
            return seed, None
        return seed, np.load(run_prefix.with_suffix(".npy")).astype(np.float32).reshape(-1)

    if dry_run:
        for job in pending:
            execute(job)
        return runs_path

    if workers <= 1 or len(pending) <= 1:
        for job in pending:
            seed, curve = execute(job)
            curves_by_seed[seed] = curve
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute, job) for job in pending]
            for fut in concurrent.futures.as_completed(futures):
                seed, curve = fut.result()
                curves_by_seed[seed] = curve

    curves = [curves_by_seed[base_seed + run] for run in range(n_runs)]
    length = min(c.size for c in curves)
    stacked = np.stack([c[:length] for c in curves], axis=0)
    np.save(runs_path, stacked)
    np.save(prefix.with_suffix(".npy"), stacked.mean(axis=0))
    shutil.copyfile(prefix.parent / f"{prefix.name}_seed{base_seed}.pth", prefix.with_suffix(".pth"))
    return runs_path


def main():
    parser = argparse.ArgumentParser(description="Run full CartPole RA pipeline.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--lq-source", choices=["empirical", "theoretical", "both"], default="empirical")
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=None,
        help="Concurrent seed jobs (config parallel_runs; 0=auto CPU count; 1=sequential)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing artifacts in the output directory and only run missing stages/seeds",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    python = sys.executable
    cfg = str(Path(args.config).resolve())

    if args.profile == "smoke":
        behavior_episodes, q_iters, agent_steps, eval_every, n_runs = 20, 2000, 4000, 1000, 1
    else:
        behavior_episodes = int(config.get("behavior_episodes", 200))
        q_iters = int(config.get("q_single_iters", 30_000))
        agent_steps = int(config.get("agent_steps", 100_000))
        eval_every = int(config.get("eval_every", 5_000))
        n_runs = int(args.runs if args.runs is not None else config.get("n_runs", 5))
    parallel_runs = int(
        args.parallel_runs if args.parallel_runs is not None else config.get("parallel_runs", 1)
    )

    out = Path(args.output_root) / f"cartpole_{args.profile}_seed{args.seed}"
    behaviors = out / "behaviors"
    data = out / "data"
    models = out / "models"
    plots = out / "plots"
    videos = out / "videos"
    if not args.dry_run:
        for d in (behaviors, data, models, plots, videos):
            d.mkdir(parents=True, exist_ok=True)
        with (out / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": cfg,
                    "profile": args.profile,
                    "seed": args.seed,
                    "n_runs": n_runs,
                    "parallel_runs": parallel_runs,
                    "behavior_episodes": behavior_episodes,
                    "q_single_iters": q_iters,
                    "agent_steps": agent_steps,
                    "resume": bool(args.resume),
                },
                handle,
                indent=2,
            )
    print(
        f"profile={args.profile} seed={args.seed} runs={n_runs} "
        f"parallel_runs={'auto' if parallel_runs <= 0 else parallel_runs} "
        f"resume={'yes' if args.resume else 'no'} lq_source={args.lq_source}"
    )

    active_config = cfg
    if args.profile == "smoke" and not args.dry_run:
        overlay = dict(config)
        overlay["behavior_episodes"] = behavior_episodes
        overlay["behavior_max_steps"] = 200
        overlay["q_single_iters"] = q_iters
        overlay["agent_steps"] = agent_steps
        overlay["eval_every"] = eval_every
        overlay["region_min_leaf"] = 20
        overlay["region_max_regions"] = 16
        active_path = out / "smoke_config.json"
        with active_path.open("w", encoding="utf-8") as handle:
            json.dump(overlay, handle, indent=2)
        active_config = str(active_path)

    behavior_files = [behaviors / "transitions_b1.npz", behaviors / "transitions_b2.npz"]
    if args.resume and artifacts_ready(behavior_files):
        print("\n=== 1/8 Collect B1/B2 behaviors skipped; artifacts already present ===")
    else:
        run_stage(
            "1/8 Collect B1/B2 behaviors",
            [python, "scripts/collect_behaviors.py", "--config", active_config, "--output-dir", behaviors, "--seed", args.seed],
            args.dry_run,
        )

    regions = data / "regions.json"
    if args.resume and artifacts_ready([regions]):
        print("\n=== 2/8 Discover regions skipped; artifacts already present ===")
    else:
        run_stage(
            "2/8 Discover regions",
            [python, "scripts/discover_regions.py", "--config", active_config, "--behavior-dir", behaviors, "--out", regions],
            args.dry_run,
        )

    bounds = data / "transition_bounds.json"
    lipschitz = data / "lipschitz_constants.json"
    q_single = models / "q_single.pth"
    if args.resume and artifacts_ready([bounds, lipschitz]):
        print("\n=== 3/8 Compute stats skipped; artifacts already present ===")
    else:
        run_stage(
            "3/8 Compute stats (initial, Lq_emp=0 if no Q_single)",
            [
                python, "scripts/compute_stats.py", "--config", active_config,
                "--behavior-dir", behaviors, "--regions", regions,
                "--bounds-out", bounds, "--lipschitz-out", lipschitz, "--seed", args.seed,
            ],
            args.dry_run,
        )

    if args.resume and artifacts_ready([q_single]):
        print("\n=== 4/8 Train Q_single skipped; artifacts already present ===")
        recompute_stats = False
    else:
        run_stage(
            "4/8 Train Q_single",
            [
                python, "scripts/train_q_single.py", "--config", active_config,
                "--bounds", bounds, "--regions", regions, "--out", q_single,
                "--iters", q_iters, "--seed", args.seed,
            ],
            args.dry_run,
        )
        recompute_stats = True

    if recompute_stats or not args.resume:
        run_stage(
            "5/8 Recompute stats with Q_single (empirical Lq)",
            [
                python, "scripts/compute_stats.py", "--config", active_config,
                "--behavior-dir", behaviors, "--regions", regions,
                "--bounds-out", bounds, "--lipschitz-out", lipschitz,
                "--q-single", q_single, "--seed", args.seed,
            ],
            args.dry_run,
        )
    else:
        print("\n=== 5/8 Recompute stats skipped; Q_single unchanged ===")

    baseline = models / "dqn"
    train_runs(
        "6a/8 Baseline DQN",
        lambda seed, run_prefix: [
            python, "scripts/train_baseline_dqn.py", "--config", active_config,
            "--steps", agent_steps, "--eval-every", eval_every, "--seed", seed, "--out-prefix", run_prefix,
        ],
        baseline,
        n_runs,
        args.seed,
        args.dry_run,
        resume=args.resume,
        parallel_runs=parallel_runs,
    )

    plot_files = [baseline.with_name(baseline.name + "_runs").with_suffix(".npy")]
    plot_labels = ["DQN"]
    sources = ["theoretical", "empirical"] if args.lq_source == "both" else [args.lq_source]
    for source in sources:
        prefix = models / f"ra_dqn_{source}"
        train_runs(
            f"6b/8 RA-DQN ({source})",
            lambda seed, run_prefix, src=source: [
                python, "scripts/train_ra_dqn.py", "--config", active_config,
                "--bounds", bounds, "--lipschitz", lipschitz, "--regions", regions,
                "--q-single", q_single, "--lq-source", src,
                "--steps", agent_steps, "--eval-every", eval_every, "--seed", seed,
                "--out-prefix", run_prefix,
            ],
            prefix,
            n_runs,
            args.seed,
            args.dry_run,
            resume=args.resume,
            parallel_runs=parallel_runs,
        )
        plot_files.append(prefix.with_name(prefix.name + "_runs").with_suffix(".npy"))
        plot_labels.append(f"RA-{source}")

    returns_plot = plots / "returns_plot.png"
    if args.resume and artifacts_ready([returns_plot]):
        print("\n=== 7/8 Returns plot skipped; artifacts already present ===")
    else:
        run_stage(
            "7/8 Returns plot",
            [
                python, "scripts/plot_returns.py", *[str(p) for p in plot_files],
                "--labels", *plot_labels, "--step", eval_every, "--out", returns_plot,
            ],
            args.dry_run,
        )

    # 8. Record DQN / RA-DQN policy videos
    video_ready = [videos / "dqn_policy.mp4"]
    if args.lq_source in ("theoretical", "both"):
        video_ready.append(videos / "ra_dqn_theoretical_policy.mp4")
    if args.lq_source in ("empirical", "both"):
        video_ready.append(videos / "ra_dqn_empirical_policy.mp4")

    if args.resume and artifacts_ready(video_ready):
        print("\n=== 8/8 Agent policy videos skipped; artifacts already present ===")
    else:
        video_cmd = [
            python, "scripts/record_agent_policy_videos.py",
            "--experiment-dir", out,
            "--seed", args.seed,
        ]
        if args.lq_source == "theoretical":
            video_cmd.append("--skip-empirical")
        run_stage("8/8 Record DQN / RA-DQN policy videos", video_cmd, args.dry_run)

    print(f"\nDone. Output: {out}")


if __name__ == "__main__":
    main()

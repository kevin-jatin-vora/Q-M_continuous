"""Run the complete sigma-specific experiment from CMD, Git Bash, or PowerShell.

CLI ``--sigma`` / ``--gamma`` always override ``configs/default.json`` and are
passed as literal float arguments to every stage. Empirical RA-DQN is on by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.config import (
    apply_resolved_sigma,
    artifact_sigma_tag,
    env_kwargs,
    load_config,
    resolve_sigma,
    resolved_training_values,
)
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect data, recompute radii, train DQN/RA-DQN, and write plots."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(ROOT / "configs" / "default.json"),
        help="Config JSON path (default: configs/default.json)",
    )
    parser.add_argument("--config", dest="config_option", default=None, help="Alternative to the positional config")
    parser.add_argument("--sigma", type=float, default=None, help="Override environment.sigma for every stage")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount gamma for every stage")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=None, help="Independent agent seeds to average (config n_runs if omitted)")
    parser.add_argument(
        "--from-source-json",
        action="store_true",
        help="Skip R1/R2 regeneration and use data/ bounds + Lipschitz from the original 0.00008 snapshot.",
    )
    parser.add_argument(
        "--empirical",
        dest="empirical",
        action="store_true",
        default=True,
        help="Train empirical RA-DQN and write its heatmap/plot (default: on)",
    )
    parser.add_argument(
        "--no-empirical",
        dest="empirical",
        action="store_false",
        help="Skip empirical RA-DQN",
    )
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue in an existing output directory; skip stages whose artifacts already exist",
    )
    parser.add_argument(
        "--stop-after",
        choices=["videos", "all"],
        default="all",
        help="Stop after R1/R2 + policy videos (videos) or run the full pipeline (all).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_files(paths):
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise SystemExit("error: stage did not produce a nonempty artifact:\n  " + "\n  ".join(missing))


def artifacts_ready(paths) -> bool:
    try:
        require_files(paths)
        return True
    except SystemExit:
        return False


def run_stage(name, command, dry_run):
    printable = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n=== {name} ===\n{printable}", flush=True)
    if dry_run:
        return
    completed = subprocess.run([str(part) for part in command], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def train_agent_runs(name, command_builder, prefix, n_runs, base_seed, dry_run, resume=False):
    curves = []
    prefix = Path(prefix)
    runs_path = prefix.with_name(prefix.name + "_runs").with_suffix(".npy")
    if resume and artifacts_ready([runs_path, prefix.with_suffix(".pth"), prefix.with_suffix(".npy")]):
        print(f"\n=== {name} skipped; artifacts already present ===")
        return runs_path
    for run in range(n_runs):
        seed = int(base_seed) + run
        run_prefix = prefix.parent / f"{prefix.name}_seed{seed}"
        run_artifacts = [run_prefix.with_suffix(".pth"), run_prefix.with_suffix(".npy")]
        if resume and artifacts_ready(run_artifacts):
            print(f"\n=== {name} run {run + 1}/{n_runs} (seed={seed}) skipped; artifacts already present ===")
            curves.append(np.asarray(np.load(run_prefix.with_suffix(".npy")), dtype=np.float32).reshape(-1))
            continue
        run_stage(
            f"{name} run {run + 1}/{n_runs} (seed={seed})",
            command_builder(seed, run_prefix),
            dry_run,
        )
        if dry_run:
            continue
        require_files([run_prefix.with_suffix(".pth"), run_prefix.with_suffix(".npy")])
        curves.append(np.asarray(np.load(run_prefix.with_suffix(".npy")), dtype=np.float32).reshape(-1))
    if dry_run:
        return prefix.with_name(prefix.name + "_runs").with_suffix(".npy")
    if not curves:
        raise SystemExit(f"error: {name} produced no evaluation curves")
    length = min(curve.size for curve in curves)
    stacked = np.stack([curve[:length] for curve in curves], axis=0)
    runs_path = prefix.with_name(prefix.name + "_runs").with_suffix(".npy")
    np.save(runs_path, stacked)
    np.save(prefix.with_suffix(".npy"), stacked.mean(axis=0))
    shutil.copyfile(
        prefix.parent / f"{prefix.name}_seed{base_seed}.pth",
        prefix.with_suffix(".pth"),
    )
    return runs_path


def verify_runtime_sigma(config, sigma: float) -> dict:
    environment = env_kwargs(config, sigma)
    env = ContinuousDollarEuroEnv(render_mode=None, auto_render=False, **environment)
    # Top half (regions 1–2, y>=0.5) is deterministic; check a bottom stochastic region.
    region4_var = float(env.terrain_config["regions"][4]["noise_cov"][0, 0])
    region1_var = float(env.terrain_config["regions"][1]["noise_cov"][0, 0])
    expected_bottom = 0.9 * (float(sigma) ** 2)
    if abs(float(env.sigma) - float(sigma)) > max(1e-18, abs(sigma) * 1e-12):
        raise SystemExit(
            f"error: environment used sigma={env.sigma:.17g}, but CLI/config requested {sigma:.17g}"
        )
    if abs(region1_var) > 1e-30:
        raise SystemExit(
            f"error: top region-1 must be deterministic (noise=0); got var={region1_var:.17g}"
        )
    if abs(region4_var - expected_bottom) > max(1e-30, abs(expected_bottom) * 1e-9):
        raise SystemExit(
            f"error: region-4 variance is {region4_var:.17g}, expected 0.9*sigma^2={expected_bottom:.17g}"
        )
    env_sigma = float(env.sigma)
    env.close()
    return {
        "requested_sigma": float(sigma),
        "env_sigma": env_sigma,
        "region1_variance": region1_var,
        "region1_std": region1_var ** 0.5,
        "region4_variance": region4_var,
        "region4_std": region4_var ** 0.5,
        "top_half_deterministic": True,
    }


def main():
    args = parse_args()
    config_path = Path(args.config_option or args.config)
    if not config_path.is_file():
        # CMD users often pass backslash paths; also accept relative to project root.
        candidate = ROOT / str(config_path).replace("\\", "/")
        if candidate.is_file():
            config_path = candidate
        else:
            raise SystemExit(f"error: config not found: {config_path}")
    config_path = config_path.resolve()
    config = load_config(str(config_path))
    config_sigma = float(config["environment"]["sigma"])
    config_gamma = float(config.get("gamma", 0.99))
    sigma = resolve_sigma(config, args.sigma, None)
    apply_resolved_sigma(config, sigma)
    if args.gamma is not None:
        if not (0.0 < float(args.gamma) < 1.0):
            raise SystemExit("error: --gamma must satisfy 0 < gamma < 1")
        config["gamma"] = float(args.gamma)
    gamma = float(config.get("gamma", 0.99))
    python = sys.executable
    sigma_arg = format(sigma, ".17g")
    gamma_arg = format(gamma, ".17g")
    env_args = ["--config", str(config_path), "--sigma", sigma_arg, "--gamma", gamma_arg]
    identity = (
        config_path.read_bytes()
        + b"\0effective_sigma="
        + sigma_arg.encode("ascii")
        + b"\0effective_gamma="
        + gamma_arg.encode("ascii")
    )
    config_hash = hashlib.sha256(identity).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_path.stem)
    source_tag = "_sourcejson" if args.from_source_json else ""
    output_dir = (
        Path(args.output_root)
        / f"sigma_{artifact_sigma_tag(sigma)}_cfg_{stem}_{config_hash}_{args.profile}{source_tag}_seed{args.seed}"
    )
    if args.profile == "smoke":
        component_steps, min_cell, center_iters, agent_steps, eval_every = 10_000, 2, 1_000, 2_000, 500
        grid_size, dpi = 31, 100
        default_runs = 2
    else:
        component_steps, min_cell, center_iters, agent_steps, eval_every = 150_000, 2, 80_000, 150_000, 2_000
        grid_size, dpi = 201, 300
        default_runs = int(config.get("n_runs", 5))
    n_runs = int(args.runs) if args.runs is not None else default_runs
    if n_runs < 1:
        raise SystemExit("error: --runs must be at least 1")

    training = resolved_training_values(config)
    print("=== Effective experiment configuration ===")
    print(f"config:          {config_path}")
    print(f"config sigma:    {config_sigma:.17g}")
    print(f"CLI --sigma:     {args.sigma if args.sigma is not None else '(not passed; using config)'}")
    print(f"effective sigma: {sigma:.17g}")
    print(f"config gamma:    {config_gamma:.17g}")
    print(f"CLI --gamma:     {args.gamma if args.gamma is not None else '(not passed; using config)'}")
    print(f"effective gamma: {gamma:.17g}")
    print(f"profile:         {args.profile}")
    print(f"seed:            {args.seed}")
    print(f"n_runs:          {n_runs}  (agent seeds {args.seed}..{args.seed + n_runs - 1})")
    print(f"empirical:       {'yes' if args.empirical else 'no'}")
    print(f"python:          {python}")
    print(f"output:          {output_dir}")
    print(f"resume:          {'yes' if args.resume else 'no'}")
    print(f"stop_after:      {args.stop_after}")
    print(f"dry run:         {'yes' if args.dry_run else 'no'}")
    print(f"gamma:           {training['gamma']}")
    print(f"learning_rate:   {training['lr']}")
    print(f"batch_size:      {training['batch_size']}")
    print(f"buffer_size:     {training['buffer_size']}")
    print(f"target_tau:      {training['tau']}  (DQN Polyak; environment living penalty is {config['environment']['tau']})")
    print(f"update_every:    {training['update_every']}")
    print(f"agent epsilon:   {training['eps_start']} -> {training['eps_end']} decay {training['eps_decay']}")
    print(f"agent max_t:     {training['max_t']}  (env horizon {config['environment']['horizon']})")
    print(
        f"R1/R2 max_t:     {int(config.get('component_max_steps_per_episode', 200))} "
        f"decay {config.get('component_epsilon_decay', 0.998)}"
    )
    print("pruning:          Bellman margin (Lr+gamma*Lq)*r/(1-gamma); theoretical or empirical Lq; clamp [-100,100]")
    print(f"bounds policy:   {config.get('bounds_sigma_policy', 'error')}")
    runtime = verify_runtime_sigma(config, sigma)
    print(
        f"verified env.sigma={runtime['env_sigma']:.17g}; "
        f"top-half deterministic; bottom region-4 noise std={runtime['region4_std']:.17g}"
    )

    bounds = output_dir / "transition_bounds.json"
    constants = output_dir / "lipschitz_constants.json"
    center = output_dir / "q_single_region.pth"
    baseline = output_dir / "dqn"
    theoretical = output_dir / "ra_dqn_theoretical"
    empirical = output_dir / "ra_dqn_empirical"
    analysis_tag = f"sigma_{artifact_sigma_tag(sigma)}"

    if not args.dry_run:
        if output_dir.exists():
            if not args.resume:
                raise SystemExit(f"error: output directory already exists; refusing to overwrite: {output_dir}")
            if not output_dir.is_dir():
                raise SystemExit(f"error: output path exists but is not a directory: {output_dir}")
        else:
            if args.resume:
                raise SystemExit(f"error: --resume requested but output directory does not exist: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (output_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "config_path": str(config_path),
                        "config_sigma": config_sigma,
                        "cli_sigma": args.sigma,
                        "effective_sigma": sigma,
                        "config_gamma": config_gamma,
                        "cli_gamma": args.gamma,
                        "effective_gamma": gamma,
                        "profile": args.profile,
                        "seed": args.seed,
                        "n_runs": n_runs,
                        "empirical": args.empirical,
                        "python": python,
                        "training": training,
                        "environment": config["environment"],
                        "component_max_steps_per_episode": int(config.get("component_max_steps_per_episode", 200)),
                        "component_epsilon_decay": float(config.get("component_epsilon_decay", 0.998)),
                        "from_source_json": bool(args.from_source_json),
                        "pruning_radius": "student_t_mean_ci",
                        "bounds_sigma_policy": config.get("bounds_sigma_policy", "error"),
                        "runtime_check": runtime,
                        "top_half_deterministic": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_source_json:
        source_sigma = 0.00008
        if abs(float(sigma) - source_sigma) > max(1e-15, source_sigma * 1e-9):
            raise SystemExit(
                "error: --from-source-json uses the original snapshot at sigma=0.00008; "
                f"got {sigma:.17g}"
            )
        stage1_ready = [bounds, constants]
        if args.resume and artifacts_ready(stage1_ready):
            print("\n=== 1/8 R1/R2 collection skipped; existing source JSON snapshot artifacts present ===")
        else:
            if not args.dry_run:
                shutil.copyfile(ROOT / "data" / "single_q_bounds_from_json.json", bounds)
                shutil.copyfile(ROOT / "data" / "lipschitz_constants_from_doc.json", constants)
            print("\n=== 1/8 R1/R2 collection skipped; using original source JSON snapshot ===")
            require_files(stage1_ready) if not args.dry_run else None
    else:
        stage1_ready = [
            output_dir / "q1_dqn_r1.pth",
            output_dir / "q2_dqn_r2.pth",
            bounds,
            constants,
            output_dir / "component_training_manifest.json",
        ]
        if args.resume and artifacts_ready(stage1_ready):
            print("\n=== 1/8 R1/R2 behavior training skipped; artifacts already present ===")
        else:
            run_stage(
                "1/8 R1/R2 behavior training, transition collection, and regeneration",
                [
                    python, "scripts/train_reward_components.py", *env_args,
                    "--steps", component_steps, "--min-cell-samples", min_cell,
                    "--seed", args.seed, "--output-dir", output_dir,
                ],
                args.dry_run,
            )
            require_files(stage1_ready) if not args.dry_run else None

    video_paths = [output_dir / "r1_policy.mp4", output_dir / "r2_policy.mp4"]
    if args.from_source_json:
        print("\n=== 1b/8 R1/R2 policy videos skipped (from-source-json has no R1/R2 checkpoints) ===")
    elif args.resume and artifacts_ready(video_paths):
        print("\n=== 1b/8 R1/R2 policy videos skipped; artifacts already present ===")
    else:
        run_stage(
            "1b/8 Record R1/R2 greedy policy videos",
            [
                python, "scripts/record_component_policy_videos.py",
                "--experiment-dir", output_dir,
                "--seed", args.seed,
            ],
            args.dry_run,
        )
        require_files(video_paths) if not args.dry_run else None

    if args.stop_after == "videos":
        if args.dry_run:
            print("\nDry run complete; stopped after R1/R2 videos.")
        else:
            print(f"\nStopped after R1/R2 videos: {output_dir}")
            print(f"  {video_paths[0]}")
            print(f"  {video_paths[1]}")
            print("Resume the rest with the same args plus --resume (omit --stop-after videos).")
        return

    if args.resume and artifacts_ready([center]):
        print("\n=== 2/8 Center-Q training skipped; artifacts already present ===")
    else:
        run_stage(
            "2/8 Center-Q training",
            [
                python, "scripts/train_q_single.py", *env_args,
                "--bounds", bounds, "--iters", center_iters, "--seed", args.seed,
                "--out", center,
            ],
            args.dry_run,
        )
        require_files([center]) if not args.dry_run else None

    theoretical_heatmap = output_dir / f"pruning_heatmap_theoretical_{analysis_tag}.png"
    stage3_ready = [
        theoretical_heatmap,
        output_dir / f"table1_lipschitz_constants_theoretical_{analysis_tag}.csv",
        output_dir / f"table2_transition_bounds_theoretical_{analysis_tag}.csv",
    ]
    if args.resume and artifacts_ready(stage3_ready):
        print("\n=== 3/8 Theoretical pruning heatmap and CSV reports skipped; artifacts already present ===")
    else:
        run_stage(
            "3/8 Theoretical pruning heatmap and CSV reports",
            [
                python, "scripts/generate_analysis.py", *env_args,
                "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
                "--lq-source", "theoretical", "--grid-size", grid_size, "--dpi", dpi,
                "--output-dir", output_dir,
            ],
            args.dry_run,
        )
        require_files(stage3_ready) if not args.dry_run else None
    print(f"heatmap ready (inspect before agent training): {theoretical_heatmap}", flush=True)

    if args.empirical:
        empirical_heatmap = output_dir / f"pruning_heatmap_empirical_{analysis_tag}.png"
        if args.resume and artifacts_ready([empirical_heatmap]):
            print("\n=== 3b/8 Empirical pruning heatmap skipped; artifacts already present ===")
        else:
            run_stage(
                "3b/8 Empirical pruning heatmap",
                [
                    python, "scripts/generate_analysis.py", *env_args,
                    "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
                    "--lq-source", "empirical", "--grid-size", grid_size, "--dpi", dpi,
                    "--output-dir", output_dir,
                ],
                args.dry_run,
            )
            require_files([empirical_heatmap]) if not args.dry_run else None
        print(f"heatmap ready (inspect before agent training): {empirical_heatmap}", flush=True)

    baseline_runs = train_agent_runs(
        "4/8 Baseline DQN training",
        lambda seed, run_prefix: [
            python, "scripts/train_baseline_dqn.py", *env_args,
            "--steps", agent_steps, "--eval-every", eval_every, "--seed", seed,
            "--eval-seed", 0,
            "--out-prefix", run_prefix,
        ],
        baseline,
        n_runs,
        args.seed,
        args.dry_run,
        resume=args.resume,
    )

    theoretical_runs = train_agent_runs(
        "5/8 Theoretical RA-DQN training",
        lambda seed, run_prefix: [
            python, "scripts/train_ra_dqn.py", *env_args,
            "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
            "--lq-source", "theoretical", "--steps", agent_steps,
            "--eval-every", eval_every, "--seed", seed,
            "--eval-seed", 0,
            "--out-prefix", run_prefix,
        ],
        theoretical,
        n_runs,
        args.seed,
        args.dry_run,
        resume=args.resume,
    )

    empirical_runs = None
    if args.empirical:
        empirical_runs = train_agent_runs(
            "6/8 Empirical RA-DQN training",
            lambda seed, run_prefix: [
                python, "scripts/train_ra_dqn.py", *env_args,
                "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
                "--lq-source", "empirical", "--steps", agent_steps,
                "--eval-every", eval_every, "--seed", seed,
                "--eval-seed", 0,
                "--out-prefix", run_prefix,
            ],
            empirical,
            n_runs,
            args.seed,
            args.dry_run,
            resume=args.resume,
        )
    else:
        print("\n=== 6/8 Empirical RA-DQN training (skipped; default is on, pass --no-empirical to skip) ===")

    plot_files = [baseline_runs, theoretical_runs]
    plot_labels = ["DQN", "RA-DQN-Theoretical"]
    if args.empirical:
        plot_files.append(empirical_runs)
        plot_labels.append("RA-DQN-Empirical")
    returns_plot = output_dir / "returns_plot.png"
    if args.resume and artifacts_ready([returns_plot]):
        print("\n=== 7/8 Returns plot skipped; artifacts already present ===")
    else:
        run_stage(
            "7/8 Returns plot",
            [
                python, "scripts/plot_returns.py", *plot_files,
                "--labels", *plot_labels, "--step", eval_every,
                "--out", returns_plot,
            ],
            args.dry_run,
        )
        require_files([returns_plot]) if not args.dry_run else None
    if args.empirical:
        dqn_empirical_plot = output_dir / "returns_plot_dqn_empirical.png"
        if args.resume and artifacts_ready([dqn_empirical_plot]):
            print("\n=== 7b/8 DQN vs empirical returns plot skipped; artifacts already present ===")
        else:
            run_stage(
                "7b/8 DQN vs empirical returns plot",
                [
                    python, "scripts/plot_returns.py",
                    baseline_runs, empirical_runs,
                    "--labels", "DQN", "RA-DQN-Empirical", "--step", eval_every,
                    "--out", dqn_empirical_plot,
                ],
                args.dry_run,
            )
            require_files([dqn_empirical_plot]) if not args.dry_run else None

    agent_video_paths = [output_dir / "dqn_policy.mp4", output_dir / "ra_dqn_theoretical_policy.mp4"]
    if args.empirical:
        agent_video_paths.append(output_dir / "ra_dqn_empirical_policy.mp4")
    if args.resume and artifacts_ready(agent_video_paths):
        print("\n=== 8/8 Agent policy videos skipped; artifacts already present ===")
    else:
        video_cmd = [
            python, "scripts/record_agent_policy_videos.py",
            "--experiment-dir", output_dir,
            "--seed", args.seed,
        ]
        if not args.empirical:
            video_cmd.append("--skip-empirical")
        run_stage("8/8 Record DQN / RA-DQN greedy policy videos", video_cmd, args.dry_run)
        require_files(agent_video_paths) if not args.dry_run else None

    if args.dry_run:
        print("\nDry run complete; no experiment output was created.")
    else:
        print(f"\nExperiment complete: {output_dir}")


if __name__ == "__main__":
    main()

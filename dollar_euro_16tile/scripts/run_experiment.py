"""Run the complete sigma-specific experiment from CMD, Git Bash, or PowerShell.

CLI ``--sigma`` / ``--gamma`` always override ``configs/default.json`` and are
passed as literal float arguments to every stage. Empirical RA-DQN is on by default.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.config import (
    apply_resolved_determinism,
    apply_resolved_deterministic_sigma_scale,
    apply_resolved_sigma,
    artifact_sigma_tag,
    env_kwargs,
    load_config,
    resolve_determinism,
    resolve_deterministic_sigma_scale,
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
    parser.add_argument(
        "--determinism",
        type=float,
        default=None,
        help="Fraction of 16 tiles that are deterministic category 5 (0..1); remainder split among radial categories 1–4.",
    )
    parser.add_argument(
        "--deterministic-sigma-scale",
        type=float,
        default=None,
        help=(
            "Category-5 noise std as a multiple of sigma. 0 (default) = exactly "
            "deterministic; 0.1 = 10x quieter than sigma; 0.01 = 100x quieter."
        ),
    )
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Explicit step budget for R1/R2 component collection and (if "
            "trained) agent training, overriding profile/config defaults "
            "(smoke: 10000/2000, full: 150000/300000). When set, the "
            "effective value also namespaces the experiment directory."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=None, help="Independent agent seeds to average (config n_runs if omitted)")
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=None,
        help="Concurrent seed jobs (config parallel_runs; 0=auto CPU count; 1=sequential)",
    )
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
    parser.add_argument(
        "--rank-coef",
        type=float,
        default=None,
        help="RA-DQN rank loss weight (default: config rank_coef or 0.001)",
    )
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue in an existing output directory; skip stages whose artifacts already exist",
    )
    parser.add_argument(
        "--stop-after",
        choices=["videos", "heatmap", "all"],
        default="all",
        help="Stop after R1/R2 videos, after pruning heatmap, or run the full pipeline.",
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


def run_stage(name, command, dry_run, env=None):
    printable = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n=== {name} ===\n{printable}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed with exit code {completed.returncode}: {name}")


def _resolve_parallel_workers(parallel_runs: int, n_jobs: int) -> int:
    if n_jobs <= 0:
        return 1
    if parallel_runs <= 0:
        # Auto: use CPU count on CPU-only machines. On CUDA, stay sequential unless
        # the user sets an explicit parallel_runs > 1 (shared-GPU jobs thrash/OOM).
        workers = os.cpu_count() or 1
        try:
            import torch

            if torch.cuda.is_available():
                workers = 1
        except Exception:
            pass
    else:
        workers = int(parallel_runs)
    return max(1, min(workers, n_jobs))


def _worker_env(parallel: bool) -> dict | None:
    """Pin BLAS/torch to 1 thread per process when running seeds concurrently."""
    if not parallel:
        return None
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[key] = "1"
    return env


def train_agent_runs(
    name,
    command_builder,
    prefix,
    n_runs,
    base_seed,
    dry_run,
    resume=False,
    parallel_runs=1,
):
    curves_by_seed = {}
    prefix = Path(prefix)
    runs_path = prefix.with_name(prefix.name + "_runs").with_suffix(".npy")
    if resume and artifacts_ready([runs_path, prefix.with_suffix(".pth"), prefix.with_suffix(".npy")]):
        print(f"\n=== {name} skipped; artifacts already present ===")
        return runs_path

    pending = []
    for run in range(n_runs):
        seed = int(base_seed) + run
        run_prefix = prefix.parent / f"{prefix.name}_seed{seed}"
        run_artifacts = [run_prefix.with_suffix(".pth"), run_prefix.with_suffix(".npy")]
        if resume and artifacts_ready(run_artifacts):
            print(f"\n=== {name} run {run + 1}/{n_runs} (seed={seed}) skipped; artifacts already present ===")
            curves_by_seed[seed] = np.asarray(np.load(run_prefix.with_suffix(".npy")), dtype=np.float32).reshape(-1)
            continue
        pending.append((run, seed, run_prefix))

    workers = _resolve_parallel_workers(int(parallel_runs), len(pending))
    worker_env = _worker_env(workers > 1)
    if pending and not dry_run:
        print(
            f"\n=== {name}: {len(pending)} job(s), parallel_workers={workers} "
            f"(independent seeds; same math as sequential) ===",
            flush=True,
        )

    def _execute(job):
        run, seed, run_prefix = job
        run_stage(
            f"{name} run {run + 1}/{n_runs} (seed={seed})",
            command_builder(seed, run_prefix),
            dry_run,
            env=worker_env,
        )
        if dry_run:
            return seed, None
        require_files([run_prefix.with_suffix(".pth"), run_prefix.with_suffix(".npy")])
        curve = np.asarray(np.load(run_prefix.with_suffix(".npy")), dtype=np.float32).reshape(-1)
        return seed, curve

    if dry_run:
        for job in pending:
            _execute(job)
        return prefix.with_name(prefix.name + "_runs").with_suffix(".npy")

    if workers <= 1 or len(pending) <= 1:
        for job in pending:
            seed, curve = _execute(job)
            curves_by_seed[seed] = curve
    else:
        # Threads only schedule subprocesses; each seed is its own process.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_execute, job) for job in pending]
            for fut in concurrent.futures.as_completed(futures):
                seed, curve = fut.result()
                curves_by_seed[seed] = curve

    curves = [curves_by_seed[int(base_seed) + run] for run in range(n_runs)]
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
    if abs(float(env.sigma) - float(sigma)) > max(1e-18, abs(sigma) * 1e-12):
        raise SystemExit(
            f"error: environment used sigma={env.sigma:.17g}, but CLI/config requested {sigma:.17g}"
        )
    det_cov = np.asarray(env.terrain_config["regions"][5]["noise_cov"], dtype=np.float64)
    det_scale = float(env.deterministic_sigma_scale)
    expected_det_var = (det_scale * float(sigma)) ** 2
    expected_det_cov = np.array([[expected_det_var, 0.0], [0.0, expected_det_var]], dtype=np.float64)
    if np.any(np.abs(det_cov - expected_det_cov) > max(1e-30, abs(expected_det_var) * 1e-9)):
        raise SystemExit(
            f"error: category-5 covariance is {det_cov.tolist()}, expected "
            f"(deterministic_sigma_scale*sigma)^2={expected_det_var:.17g} on the diagonal "
            f"(scale={det_scale:.17g}, sigma={float(sigma):.17g})"
        )
    cat1_var = float(env.terrain_config["regions"][1]["noise_cov"][0, 0])
    expected = 1.0 * (float(sigma) ** 2)
    if abs(cat1_var - expected) > max(1e-30, abs(expected) * 1e-9):
        raise SystemExit(
            f"error: category-1 variance is {cat1_var:.17g}, expected 1.0*sigma^2={expected:.17g}"
        )
    cat4_var = float(env.terrain_config["regions"][4]["noise_cov"][0, 0])
    expected4 = 0.9 * (float(sigma) ** 2)
    if abs(cat4_var - expected4) > max(1e-30, abs(expected4) * 1e-9):
        raise SystemExit(
            f"error: category-4 variance is {cat4_var:.17g}, expected 0.9*sigma^2={expected4:.17g}"
        )
    counts = env.layout_info["counts"]
    env_sigma = float(env.sigma)
    env_det = float(env.determinism)
    env_det_sigma = float(env.deterministic_sigma)
    env.close()
    return {
        "requested_sigma": float(sigma),
        "env_sigma": env_sigma,
        "determinism": env_det,
        "deterministic_sigma_scale": det_scale,
        "deterministic_sigma": env_det_sigma,
        "category5_variance": float(det_cov[0, 0]),
        "tile_counts": counts,
        "category1_variance": cat1_var,
        "category1_std": float(np.sqrt(max(cat1_var, 0.0))),
        "category4_variance": cat4_var,
        "category4_std": float(np.sqrt(max(cat4_var, 0.0))),
        "deterministic_category": 5,
        "n_categories": 5,
        "n_tiles": 16,
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
    config_determinism = float(config["environment"].get("determinism", 0.25))
    config_det_sigma_scale = float(
        config["environment"].get("deterministic_sigma_scale", 0.0)
    )
    sigma = resolve_sigma(config, args.sigma, None)
    apply_resolved_sigma(config, sigma)
    determinism = resolve_determinism(config, args.determinism)
    apply_resolved_determinism(config, determinism)
    det_sigma_scale = resolve_deterministic_sigma_scale(
        config, args.deterministic_sigma_scale
    )
    apply_resolved_deterministic_sigma_scale(config, det_sigma_scale)
    if args.gamma is not None:
        if not (0.0 < float(args.gamma) < 1.0):
            raise SystemExit("error: --gamma must satisfy 0 < gamma < 1")
        config["gamma"] = float(args.gamma)
    gamma = float(config.get("gamma", 0.99))
    python = sys.executable
    sigma_arg = format(sigma, ".17g")
    gamma_arg = format(gamma, ".17g")
    det_arg = format(determinism, ".17g")
    det_sigma_scale_arg = format(det_sigma_scale, ".17g")
    env_args = [
        "--config", str(config_path),
        "--sigma", sigma_arg,
        "--gamma", gamma_arg,
        "--determinism", det_arg,
        "--deterministic-sigma-scale", det_sigma_scale_arg,
    ]
    # ------------------------------------------------------------
    # Profile / steps resolution
    # ------------------------------------------------------------

    if args.profile == "smoke":
        default_component_steps, min_cell, center_iters = 10_000, 2, 1_000
        default_agent_steps, eval_every = 2_000, 500
        grid_size, dpi = 31, 100
        default_runs = 2
    else:
        default_component_steps, min_cell, center_iters = 150_000, 2, 80_000
        default_agent_steps, eval_every = 300_000, 20_000
        grid_size, dpi = 201, 300
        default_runs = int(config.get("n_runs", 30))

    if args.steps is not None:
        if args.steps <= 0:
            raise SystemExit("error: --steps must be >= 1")
        component_steps = int(args.steps)
        agent_steps = int(args.steps)
    else:
        component_steps = int(config.get("component_steps", default_component_steps))
        agent_steps = int(config.get("agent_steps", default_agent_steps))

    center_iters = int(config.get("center_q_iters", center_iters))
    eval_every = int(config.get("eval_every", eval_every))
    eval_episodes = int(config.get("eval_episodes", 30))

    # ------------------------------------------------------------
    # Experiment directory identity
    # ------------------------------------------------------------

    identity = (
        config_path.read_bytes()
        + b"\0effective_sigma="
        + sigma_arg.encode("ascii")
        + b"\0effective_gamma="
        + gamma_arg.encode("ascii")
        + b"\0effective_determinism="
        + det_arg.encode("ascii")
        + b"\0effective_deterministic_sigma_scale="
        + det_sigma_scale_arg.encode("ascii")
    )
    if args.steps is not None:
        identity += (
            b"\0effective_steps="
            + str(int(agent_steps)).encode("ascii")
        )
    config_hash = hashlib.sha256(identity).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_path.stem)
    source_tag = "_sourcejson" if args.from_source_json else ""
    det_tag = f"_det{int(round(determinism * 100)):02d}"
    if det_sigma_scale > 0.0:
        det_tag += f"_dsig{artifact_sigma_tag(det_sigma_scale)}"
    steps_tag = f"_steps{int(agent_steps)}" if args.steps is not None else ""
    output_dir = (
        Path(args.output_root)
        / f"sigma_{artifact_sigma_tag(sigma)}{det_tag}{steps_tag}_cfg_{stem}_{config_hash}_{args.profile}{source_tag}_seed{args.seed}"
    )
    data_dir = output_dir / "data"
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    legacy_margin = bool(config.get("legacy_margin", False))
    legacy_margin_args = ["--legacy-margin"] if legacy_margin else ["--no-legacy-margin"]
    include_boundary = not bool(config.get("exclude_boundary_affected_transitions", True))
    boundary_args = ["--include-boundary-transitions"] if include_boundary else []
    n_runs = int(args.runs) if args.runs is not None else default_runs
    if n_runs < 1:
        raise SystemExit("error: --runs must be at least 1")
    parallel_runs = int(
        args.parallel_runs if args.parallel_runs is not None else config.get("parallel_runs", 0)
    )
    parallel_workers_hint = (
        f"auto(~{os.cpu_count() or 1})" if parallel_runs <= 0 else str(parallel_runs)
    )

    training = resolved_training_values(config)
    print("=== Effective experiment configuration ===")
    print(f"config:          {config_path}")
    print(f"config sigma:    {config_sigma:.17g}")
    print(f"CLI --sigma:     {args.sigma if args.sigma is not None else '(not passed; using config)'}")
    print(f"effective sigma: {sigma:.17g}")
    print(f"config determinism: {config_determinism:.17g}")
    print(f"CLI --determinism:  {args.determinism if args.determinism is not None else '(not passed; using config)'}")
    print(f"effective determinism: {determinism:.17g}")
    print(f"config det sigma scale: {config_det_sigma_scale:.17g}")
    print(
        f"CLI --deterministic-sigma-scale: "
        f"{args.deterministic_sigma_scale if args.deterministic_sigma_scale is not None else '(not passed; using config)'}"
    )
    print(
        f"effective det sigma scale: {det_sigma_scale:.17g}  "
        f"(category-5 noise std = {det_sigma_scale * sigma:.17g}"
        f"{'; exactly deterministic' if det_sigma_scale == 0.0 else ''})"
    )
    print(f"config gamma:    {config_gamma:.17g}")
    print(f"CLI --gamma:     {args.gamma if args.gamma is not None else '(not passed; using config)'}")
    print(f"effective gamma: {gamma:.17g}")
    print(f"profile:         {args.profile}")
    print(f"seed:            {args.seed}")
    print(f"n_runs:          {n_runs}  (agent seeds {args.seed}..{args.seed + n_runs - 1})")
    print(f"parallel_runs:   {parallel_workers_hint}  (independent seed jobs)")
    print(f"agent_steps:     {agent_steps}")
    print(f"component_steps: {component_steps}")
    print(f"eval_every:      {eval_every}")
    print(f"eval_episodes:   {eval_episodes}  (unseeded greedy rollouts)")
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
    print(f"legacy_margin:   {legacy_margin}  (radial QM_using json uses legacy Lr*r + Lq*r/(1-gamma))")
    conf_level = float(config.get("confidence_level", 0.95))
    print(f"RA pruning r:  Student-t L2 radius (confidence_level={conf_level})")
    rank_coef = float(args.rank_coef if args.rank_coef is not None else config.get("rank_coef", 0.001))
    print(f"rank_coef:       {rank_coef}")
    print("Q_single:        s'=clip(s+delta_mean)")
    env_cfg = config["environment"]
    print(
        f"reward peaks:    alpha={float(env_cfg.get('reward_alpha', 2.0)):.4g} "
        f"beta={float(env_cfg.get('reward_beta', 1.2)):.4g} "
        f"(tau={float(env_cfg.get('tau', 2.4)):.4g})"
    )
    print(f"stats boundary:  {'include all transitions' if include_boundary else 'exclude boundary cancel/clip'}")
    print(f"bounds policy:   {config.get('bounds_sigma_policy', 'error')}")
    runtime = verify_runtime_sigma(config, sigma)
    print(
        f"verified env.sigma={runtime['env_sigma']:.17g}; determinism={runtime['determinism']:.4g}; "
        f"cat5 noise std={runtime['deterministic_sigma']:.17g}; "
        f"tile counts={runtime['tile_counts']}; cat1 noise std={runtime['category1_std']:.17g}"
    )

    bounds = data_dir / "transition_bounds.json"
    constants = data_dir / "lipschitz_constants.json"
    center = models_dir / "q_single_region.pth"
    baseline = models_dir / "dqn"
    theoretical = models_dir / "ra_dqn_theoretical"
    empirical = models_dir / "ra_dqn_empirical"
    analysis_tag = f"sigma_{artifact_sigma_tag(sigma)}_det{int(round(determinism * 100)):02d}"

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
        for sub in (data_dir, models_dir, plots_dir, videos_dir):
            sub.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (data_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "config_path": str(config_path),
                        "config_sigma": config_sigma,
                        "cli_sigma": args.sigma,
                        "effective_sigma": sigma,
                        "config_determinism": config_determinism,
                        "cli_determinism": args.determinism,
                        "effective_determinism": determinism,
                        "config_deterministic_sigma_scale": config_det_sigma_scale,
                        "cli_deterministic_sigma_scale": args.deterministic_sigma_scale,
                        "effective_deterministic_sigma_scale": det_sigma_scale,
                        "effective_deterministic_sigma": det_sigma_scale * sigma,
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
                        "legacy_margin": legacy_margin,
                        "include_boundary_transitions_in_stats": include_boundary,
                        "from_source_json": bool(args.from_source_json),
                        "confidence_level": conf_level,
                        "pruning_radius": "student_t_l2",
                        "q_single": "delta_mean",
                        "bounds_sigma_policy": config.get("bounds_sigma_policy", "error"),
                        "runtime_check": runtime,
                        "layout": "16-tile / 5-category",
                        "output_layout": {
                            "data": str(data_dir),
                            "models": str(models_dir),
                            "plots": str(plots_dir),
                            "videos": str(videos_dir),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            for sub in (data_dir, models_dir, plots_dir, videos_dir):
                sub.mkdir(parents=True, exist_ok=True)

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
            models_dir / "q1_dqn_r1.pth",
            models_dir / "q2_dqn_r2.pth",
            bounds,
            constants,
            data_dir / "component_training_manifest.json",
        ]
        if args.resume and artifacts_ready(stage1_ready):
            print("\n=== 1/8 R1/R2 behavior training skipped; artifacts already present ===")
        else:
            run_stage(
                "1/8 R1/R2 behavior training, transition collection, and regeneration",
                [
                    python, "scripts/train_reward_components.py", *env_args,
                    "--steps", component_steps, "--min-cell-samples", min_cell,
                    "--confidence-level", config.get("confidence_level", 0.95),
                    *boundary_args,
                    "--seed", args.seed, "--output-dir", output_dir,
                ],
                args.dry_run,
            )
            require_files(stage1_ready) if not args.dry_run else None

    video_paths = [videos_dir / "r1_policy.mp4", videos_dir / "r2_policy.mp4"]
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

    theoretical_heatmap = plots_dir / f"pruning_heatmap_theoretical_{analysis_tag}.png"
    stage3_ready = [
        theoretical_heatmap,
        plots_dir / f"table1_lipschitz_constants_{analysis_tag}.csv",
        plots_dir / f"table2_transition_bounds_{analysis_tag}.csv",
    ]
    if args.resume and artifacts_ready(stage3_ready):
        print("\n=== 3/8 Theoretical pruning heatmap and CSV reports skipped; artifacts already present ===")
    else:
        run_stage(
            "3/8 Theoretical pruning heatmap and CSV reports",
            [
                python, "scripts/generate_analysis.py", *env_args,
                "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
                "--lq-source", "theoretical", *legacy_margin_args,
                "--grid-size", grid_size, "--dpi", dpi,
                "--output-dir", plots_dir,
            ],
            args.dry_run,
        )
        require_files(stage3_ready) if not args.dry_run else None
    print(f"heatmap ready (inspect before agent training): {theoretical_heatmap}", flush=True)

    if args.empirical:
        empirical_heatmap = plots_dir / f"pruning_heatmap_empirical_{analysis_tag}.png"
        if args.resume and artifacts_ready([empirical_heatmap]):
            print("\n=== 3b/8 Empirical pruning heatmap skipped; artifacts already present ===")
        else:
            run_stage(
                "3b/8 Empirical pruning heatmap",
                [
                    python, "scripts/generate_analysis.py", *env_args,
                    "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
                    "--lq-source", "empirical", *legacy_margin_args,
                    "--grid-size", grid_size, "--dpi", dpi,
                    "--output-dir", plots_dir,
                ],
                args.dry_run,
            )
            require_files([empirical_heatmap]) if not args.dry_run else None
        print(f"heatmap ready (inspect before agent training): {empirical_heatmap}", flush=True)

    if args.stop_after == "heatmap":
        if args.dry_run:
            print("\nDry run complete; stopped after heatmap.")
        else:
            print(f"\nStopped after heatmap: {output_dir}")
            print(f"  {theoretical_heatmap}")
            if args.empirical:
                print(f"  {plots_dir / f'pruning_heatmap_empirical_{analysis_tag}.png'}")
            print("Resume agent training with the same args plus --resume (omit --stop-after heatmap).")
        return

    baseline_runs = train_agent_runs(
        "4/8 Baseline DQN training",
        lambda seed, run_prefix: [
            python, "scripts/train_baseline_dqn.py", *env_args,
            "--steps", agent_steps, "--eval-every", eval_every,
            "--eval-episodes", eval_episodes,
            "--seed", seed,
            "--out-prefix", run_prefix,
        ],
        baseline,
        n_runs,
        args.seed,
        args.dry_run,
        resume=args.resume,
        parallel_runs=parallel_runs,
    )

    theoretical_runs = train_agent_runs(
        "5/8 Theoretical RA-DQN training",
        lambda seed, run_prefix: [
            python, "scripts/train_ra_dqn.py", *env_args,
            "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
            "--lq-source", "theoretical", *legacy_margin_args,
            "--steps", agent_steps,
            "--eval-every", eval_every,
            "--eval-episodes", eval_episodes,
            "--seed", seed,
            "--rank-coef", rank_coef,
            "--out-prefix", run_prefix,
        ],
        theoretical,
        n_runs,
        args.seed,
        args.dry_run,
        resume=args.resume,
        parallel_runs=parallel_runs,
    )

    empirical_runs = None
    if args.empirical:
        empirical_runs = train_agent_runs(
            "6/8 Empirical RA-DQN training",
            lambda seed, run_prefix: [
                python, "scripts/train_ra_dqn.py", *env_args,
                "--bounds", bounds, "--lipschitz", constants, "--q-single", center,
                "--lq-source", "empirical", *legacy_margin_args,
                "--steps", agent_steps,
                "--eval-every", eval_every,
                "--eval-episodes", eval_episodes,
                "--seed", seed,
                "--rank-coef", rank_coef,
                "--out-prefix", run_prefix,
            ],
            empirical,
            n_runs,
            args.seed,
            args.dry_run,
            resume=args.resume,
            parallel_runs=parallel_runs,
        )
    else:
        print("\n=== 6/8 Empirical RA-DQN training (skipped; default is on, pass --no-empirical to skip) ===")

    plot_files = [baseline_runs, theoretical_runs]
    plot_labels = ["DQN", "RA-DQN-Theoretical"]
    if args.empirical:
        plot_files.append(empirical_runs)
        plot_labels.append("RA-DQN-Empirical")
    returns_plot = plots_dir / "returns_plot.png"
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
        dqn_empirical_plot = plots_dir / "returns_plot_dqn_empirical.png"
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

    agent_video_paths = [videos_dir / "dqn_policy.mp4", videos_dir / "ra_dqn_theoretical_policy.mp4"]
    if args.empirical:
        agent_video_paths.append(videos_dir / "ra_dqn_empirical_policy.mp4")
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

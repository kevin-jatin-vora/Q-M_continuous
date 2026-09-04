"""Run a fresh/full experiment using learned Q_UB/Q_LB bounds for pruning.

Pipeline
--------
1. Reuse original run_experiment.py for:
      - R1/R2 component training
      - transition_bounds.json
      - lipschitz_constants.json
      - component-policy videos

2. Train learned Q_UB / Q_LB.

3. Analyze learned Q bounds.

4. Train baseline DQN over N independent seeds.

5. Train RA-DQN over the same seeds using frozen learned Q_UB/Q_LB
   for pruning. This does NOT use Q_single +/- analytical margins.

6. Rebuild aggregate return arrays and plot DQN vs learned-bound RA-DQN.

7. Record representative seed policy videos:
      - DQN greedy policy
      - RA-DQN greedy policy subject to learned Q_UB/Q_LB pruning

Resume behavior
---------------
If --resume is supplied:

- Existing R1/R2/component artifacts are reused by original run_experiment.py.
- Existing Q_UB/Q_LB checkpoints are reused.
- Existing complete DQN seed runs are skipped individually.
- Existing complete RA-DQN seed runs are skipped individually.
- Missing requested seeds are trained.
- Aggregate *_runs.npy files are ALWAYS rebuilt from all requested seeds.
- Return plots are regenerated.
- Existing policy videos are reused if already present.

Example:
    First:
        --runs 1 --seed 0

    Later:
        --runs 30 --seed 0 --resume

    This reuses seed 0 and trains seeds 1..29.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


# ================================================================
# Project paths
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


# ================================================================
# Project imports
# ================================================================

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
)

from dollar_euro_lipschitz.models import QNet

from dollar_euro_lipschitz.q_bounds import (
    learned_bound_allowed_mask,
    load_frozen_qnet,
)

# Reuse ONLY the original rendering/video function.
#
# We intentionally do NOT use its old PrunedGreedyPolicy because
# that policy uses Q_single +/- analytical margins.
from record_agent_policy_videos import record_rollout


# ================================================================
# General helpers
# ================================================================


def run_stage(
    name,
    cmd,
    dry_run=False,
    env=None,
):
    printable = subprocess.list2cmdline(
        [str(x) for x in cmd]
    )

    print(
        f"\n=== {name} ===\n{printable}",
        flush=True,
    )

    if dry_run:
        return

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=ROOT,
        env=env,
        check=False,
    )

    if result.returncode:
        raise SystemExit(result.returncode)


def file_ready(path):
    path = Path(path)

    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def artifacts_ready(paths):
    return all(
        file_ready(path)
        for path in paths
    )


def require_files(paths):
    missing = [
        str(path)
        for path in paths
        if not file_ready(path)
    ]

    if missing:
        raise SystemExit(
            "Missing expected artifact(s):\n  "
            + "\n  ".join(missing)
        )


def validate_cross_category_artifact(
    path, expected_sigma, expected_det_sigma_scale, expected_gamma, kind="cross"
):
    """Validate a v3 cross (tile-reward or category-dynamics) Lipschitz artifact.

    ``kind`` is one of "cross_tile" | "cross_category" | "cross" (generic).

    Raises a clear error on missing/incompatible schema so that --resume does
    not silently reuse an experiment produced by an incompatible (older)
    pipeline.
    """
    try:
        import json as _json
        data = _json.load(open(path, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Cross Lipschitz artifact could not be read: {path}\n  {e}"
        )

    issues = []

    def check(ok, msg):
        if not ok:
            issues.append(msg)

    check(
        "sigma" in data,
        "missing 'sigma' provenance",
    )
    check(
        "deterministic_sigma_scale" in data,
        "missing 'deterministic_sigma_scale' provenance",
    )
    check(
        "gamma" in data,
        "missing 'gamma' provenance",
    )
    check(
        "tile_map" in data,
        "missing 'tile_map' provenance",
    )
    if kind == "cross_tile":
        check(
            "neighbor_tile_pairs" in data,
            "missing 'neighbor_tile_pairs' provenance",
        )
        check(
            isinstance(data.get("neighbor_tile_pairs"), list),
            "'neighbor_tile_pairs' must be a list",
        )
    elif kind == "cross_category":
        check(
            "neighbor_category_pairs" in data,
            "missing 'neighbor_category_pairs' provenance",
        )
        check(
            isinstance(data.get("neighbor_category_pairs"), list),
            "'neighbor_category_pairs' must be a list",
        )
    check(
        "lipschitz_method_version" in data,
        "missing 'lipschitz_method_version' provenance",
    )
    if data.get("lipschitz_method_version") != 3:
        issues.append(
            "Old Lipschitz artifacts use incompatible reward grouping/"
            "cross-boundary semantics; regenerate from scratch. "
            f"(artifact version={data.get('lipschitz_method_version')!r}, "
            "required=3)"
        )

    if data.get("sigma") is not None and expected_sigma is not None:
        if abs(float(data["sigma"]) - float(expected_sigma)) > 1e-12:
            issues.append(
                f"sigma mismatch: artifact={data['sigma']} command={expected_sigma}"
            )
    if (
        data.get("deterministic_sigma_scale") is not None
        and expected_det_sigma_scale is not None
    ):
        if abs(
            float(data["deterministic_sigma_scale"])
            - float(expected_det_sigma_scale)
        ) > 1e-12:
            issues.append(
                "deterministic_sigma_scale mismatch: "
                f"artifact={data['deterministic_sigma_scale']} "
                f"command={expected_det_sigma_scale}"
            )
    if data.get("gamma") is not None and expected_gamma is not None:
        if abs(float(data["gamma"]) - float(expected_gamma)) > 1e-12:
            issues.append(
                f"gamma mismatch: artifact={data['gamma']} command={expected_gamma}"
            )

    if issues:
        raise SystemExit(
            "Cross Lipschitz artifact is incompatible (provenance/schema "
            "mismatch). Refusing to resume on an experiment produced by an "
            f"incompatible pipeline:\n  {path}\n  "
            + "\n  ".join(f"- {msg}" for msg in issues)
        )

    return data


# ================================================================
# CLI
# ================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "config",
        nargs="?",
        default=str(
            ROOT
            / "configs"
            / "default.json"
        ),
    )

    parser.add_argument(
        "--config",
        dest="config_option",
        default=None,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--determinism",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--deterministic-sigma-scale",
        type=float,
        default=None,
        help=(
            "Multiplier of base sigma for category 5. "
            "Actual category-5 sigma = "
            "sigma * deterministic_sigma_scale."
        ),
    )

    parser.add_argument(
        "--profile",
        choices=[
            "smoke",
            "full",
        ],
        default="full",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base seed. With --runs N, agent seeds are "
            "seed, seed+1, ..., seed+N-1."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=None,
    )

    # Kept for CLI compatibility.
    # Runs are currently launched sequentially.
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--rank-coef",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        default=str(
            ROOT
            / "outputs"
            / "experiments"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--q-bound-iters",
        type=int,
        default=120000,
    )

    parser.add_argument(
        "--q-bound-tau",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lq-source",
        choices=[
            "theoretical",
            "empirical",
        ],
        default="theoretical",
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--video-substeps",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--pruning-tol",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


# ================================================================
# Experiment-directory identity
#
# This follows the original run_experiment.py naming/hash logic.
# ================================================================


def experiment_identity(
    config_path,
    sigma,
    gamma,
    determinism,
    det_sigma_scale,
    profile,
    seed,
    output_root,
):
    sigma_arg = format(
        sigma,
        ".17g",
    )

    gamma_arg = format(
        gamma,
        ".17g",
    )

    det_arg = format(
        determinism,
        ".17g",
    )

    det_sigma_scale_arg = format(
        det_sigma_scale,
        ".17g",
    )

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

    config_hash = hashlib.sha256(
        identity
    ).hexdigest()[:12]

    stem = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        config_path.stem,
    )

    det_tag = (
        f"_det"
        f"{int(round(determinism * 100)):02d}"
    )

    if det_sigma_scale > 0.0:
        det_tag += (
            f"_dsig"
            f"{artifact_sigma_tag(det_sigma_scale)}"
        )

    return (
        Path(output_root)
        / (
            f"sigma_"
            f"{artifact_sigma_tag(sigma)}"
            f"{det_tag}"
            f"_cfg_{stem}_"
            f"{config_hash}_"
            f"{profile}_seed{seed}"
        )
    )


# ================================================================
# Multi-seed agent training + resume
# ================================================================


def train_runs(
    name,
    script,
    prefix,
    common_args,
    n_runs,
    base_seed,
    dry_run,
    resume,
):
    """Train/check N independent seeds and rebuild aggregate arrays.

    Requested seeds are:

        base_seed
        base_seed + 1
        ...
        base_seed + n_runs - 1

    With --resume, a seed is skipped only if BOTH its .pth and .npy
    artifacts are present and nonempty.

    After all requested seeds are available, the aggregate arrays are
    rebuilt from the individual seed .npy files. This is important when
    continuing from --runs 1 to --runs 30.
    """

    prefix = Path(prefix)

    seed_curve_paths = []
    seed_model_paths = []

    for index in range(n_runs):
        seed = (
            base_seed
            + index
        )

        run_prefix = (
            prefix.parent
            / f"{prefix.name}_seed{seed}"
        )

        model_path = (
            run_prefix.with_suffix(
                ".pth"
            )
        )

        curve_path = (
            run_prefix.with_suffix(
                ".npy"
            )
        )

        complete = artifacts_ready(
            [
                model_path,
                curve_path,
            ]
        )

        if resume and complete:
            print(
                f"\n=== {name} "
                f"seed={seed} skipped; "
                f"complete artifacts present ===",
                flush=True,
            )

        else:
            run_stage(
                (
                    f"{name} "
                    f"run {index + 1}/{n_runs} "
                    f"seed={seed}"
                ),
                [
                    sys.executable,
                    script,
                    *common_args,
                    "--seed",
                    seed,
                    "--out-prefix",
                    run_prefix,
                ],
                dry_run,
            )

        seed_model_paths.append(
            model_path
        )

        seed_curve_paths.append(
            curve_path
        )

        if not dry_run:
            require_files(
                [
                    model_path,
                    curve_path,
                ]
            )

    runs_path = (
        prefix.with_name(
            prefix.name
            + "_runs"
        )
        .with_suffix(
            ".npy"
        )
    )

    mean_path = (
        prefix.with_suffix(
            ".npy"
        )
    )

    representative_model = (
        prefix.with_suffix(
            ".pth"
        )
    )

    if dry_run:
        return (
            runs_path,
            representative_model,
        )

    # ------------------------------------------------------------
    # ALWAYS rebuild aggregates from all requested completed seeds.
    # ------------------------------------------------------------

    curves = []

    for curve_path in seed_curve_paths:
        require_files(
            [curve_path]
        )

        curve = np.asarray(
            np.load(
                curve_path
            ),
            dtype=np.float32,
        ).reshape(-1)

        curves.append(
            curve
        )

    if not curves:
        raise SystemExit(
            f"No completed runs available for {name}"
        )

    common_length = min(
        curve.size
        for curve in curves
    )

    if common_length <= 0:
        raise SystemExit(
            f"Empty return curve found for {name}"
        )

    stacked = np.stack(
        [
            curve[:common_length]
            for curve in curves
        ],
        axis=0,
    )

    np.save(
        runs_path,
        stacked,
    )

    np.save(
        mean_path,
        stacked.mean(
            axis=0
        ),
    )

    # Representative model is always the base-seed model.
    base_seed_model = (
        prefix.parent
        / (
            f"{prefix.name}"
            f"_seed{base_seed}.pth"
        )
    )

    require_files(
        [base_seed_model]
    )

    shutil.copyfile(
        base_seed_model,
        representative_model,
    )

    print(
        f"\nRebuilt {name} aggregates:"
    )

    print(
        f"  runs:  {runs_path}"
    )

    print(
        f"  shape: {stacked.shape}"
    )

    print(
        f"  mean:  {mean_path}"
    )

    print(
        f"  representative model: "
        f"{representative_model}"
    )

    return (
        runs_path,
        representative_model,
    )


# ================================================================
# Learned-bound RA policy used ONLY for video recording
# ================================================================


class LearnedBoundVideoPolicy:
    """Greedy RA-DQN policy with frozen learned Q_UB/Q_LB pruning.

    This matches the learned-bound pruning concept used by
    train_ra_dqn_bounds.py:

        upper = Q_UB(s, :)
        lower = Q_LB(s, :)
        allowed = learned_bound_allowed_mask(upper, lower)

    The trained RA-DQN Q network then chooses the largest Q value among
    the allowed actions.

    No Q_single.
    No analytical margin.
    """

    def __init__(
        self,
        *,
        agent_checkpoint,
        q_ub_checkpoint,
        q_lb_checkpoint,
        tol,
        device,
    ):
        self.device = device
        self.tol = float(tol)

        self.q = QNet().to(
            device
        )

        self.q.load_state_dict(
            torch.load(
                agent_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )

        self.q.eval()

        self.q_ub = load_frozen_qnet(
            q_ub_checkpoint,
            QNet,
            device,
        )

        self.q_lb = load_frozen_qnet(
            q_lb_checkpoint,
            QNet,
            device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        state,
    ):
        states = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = self.q(
            states
        )[0]

        upper = self.q_ub(
            states
        )

        lower = self.q_lb(
            states
        )

        allowed = (
            learned_bound_allowed_mask(
                upper,
                lower,
                tol=self.tol,
            )[0]
        )

        return int(
            q_values.masked_fill(
                ~allowed,
                -1e9,
            )
            .argmax()
            .item()
        )


# ================================================================
# Agent policy videos
# ================================================================


def record_agent_videos(
    *,
    output_dir,
    config,
    sigma,
    seed,
    dqn_checkpoint,
    ra_checkpoint,
    q_ub_checkpoint,
    q_lb_checkpoint,
    lq_source,
    tol,
    fps,
    substeps,
    resume,
    dry_run,
):
    videos_dir = (
        output_dir
        / "videos"
    )

    dqn_video = (
        videos_dir
        / "dqn_policy.mp4"
    )

    ra_video = (
        videos_dir
        / (
            "ra_dqn_learned_bounds_"
            f"{lq_source}_policy.mp4"
        )
    )

    if (
        resume
        and artifacts_ready(
            [
                dqn_video,
                ra_video,
            ]
        )
    ):
        print(
            "\n=== 7/7 Agent policy videos "
            "skipped; artifacts already present ==="
        )

        return (
            dqn_video,
            ra_video,
        )

    print(
        "\n=== 7/7 Record DQN / learned-bound "
        "RA-DQN policy videos ===",
        flush=True,
    )

    if dry_run:
        print(
            f"would write {dqn_video}"
        )
        print(
            f"would write {ra_video}"
        )

        return (
            dqn_video,
            ra_video,
        )

    require_files(
        [
            dqn_checkpoint,
            ra_checkpoint,
            q_ub_checkpoint,
            q_lb_checkpoint,
        ]
    )

    videos_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    environment = env_kwargs(
        config,
        sigma,
    )

    # ------------------------------------------------------------
    # DQN video
    # ------------------------------------------------------------

    dqn = QNet().to(
        device
    )

    dqn.load_state_dict(
        torch.load(
            dqn_checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    dqn.eval()

    @torch.inference_mode()
    def choose_dqn_action(
        state,
    ):
        state_tensor = (
            torch.as_tensor(
                state,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
        )

        return int(
            dqn(
                state_tensor
            )
            .argmax(
                dim=1
            )
            .item()
        )

    record_rollout(
        choose_action=choose_dqn_action,
        output=dqn_video,
        environment=environment,
        seed=seed,
        fps=fps,
        substeps=substeps,
        title=(
            f"DQN greedy "
            f"(sigma={sigma:g})"
        ),
    )

    # ------------------------------------------------------------
    # Learned-bound RA-DQN video
    # ------------------------------------------------------------

    ra_policy = (
        LearnedBoundVideoPolicy(
            agent_checkpoint=(
                ra_checkpoint
            ),
            q_ub_checkpoint=(
                q_ub_checkpoint
            ),
            q_lb_checkpoint=(
                q_lb_checkpoint
            ),
            tol=tol,
            device=device,
        )
    )

    record_rollout(
        choose_action=ra_policy,
        output=ra_video,
        environment=environment,
        seed=seed,
        fps=fps,
        substeps=substeps,
        title=(
            "RA-DQN learned-bound greedy "
            f"({lq_source}, sigma={sigma:g})"
        ),
    )

    require_files(
        [
            dqn_video,
            ra_video,
        ]
    )

    return (
        dqn_video,
        ra_video,
    )


# ================================================================
# Main
# ================================================================


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Resolve config path
    # ------------------------------------------------------------

    config_path = Path(
        args.config_option
        or args.config
    )

    if not config_path.is_file():
        candidate = (
            ROOT
            / str(
                config_path
            ).replace(
                "\\",
                "/",
            )
        )

        if candidate.is_file():
            config_path = (
                candidate
            )

        else:
            raise SystemExit(
                f"config not found: "
                f"{config_path}"
            )

    config_path = (
        config_path.resolve()
    )

    config = load_config(
        str(config_path)
    )

    # ------------------------------------------------------------
    # Resolve environment settings exactly as project does
    # ------------------------------------------------------------

    sigma = resolve_sigma(
        config,
        args.sigma,
        None,
    )

    apply_resolved_sigma(
        config,
        sigma,
    )

    determinism = (
        resolve_determinism(
            config,
            args.determinism,
        )
    )

    apply_resolved_determinism(
        config,
        determinism,
    )

    det_sigma_scale = (
        resolve_deterministic_sigma_scale(
            config,
            args.deterministic_sigma_scale,
        )
    )

    apply_resolved_deterministic_sigma_scale(
        config,
        det_sigma_scale,
    )

    if args.gamma is not None:
        gamma = float(
            args.gamma
        )

        if not (
            0.0
            < gamma
            < 1.0
        ):
            raise SystemExit(
                "--gamma must satisfy "
                "0 < gamma < 1"
            )

        config["gamma"] = (
            gamma
        )

    gamma = float(
        config.get(
            "gamma",
            0.99,
        )
    )

    # Actual category-5 sigma.
    deterministic_sigma = (
        sigma
        * det_sigma_scale
    )

    # ------------------------------------------------------------
    # Exact experiment directory
    # ------------------------------------------------------------

    output_dir = (
        experiment_identity(
            config_path,
            sigma,
            gamma,
            determinism,
            det_sigma_scale,
            args.profile,
            args.seed,
            args.output_root,
        )
    )

    data_dir = (
        output_dir
        / "data"
    )

    models_dir = (
        output_dir
        / "models"
    )

    plots_dir = (
        output_dir
        / "plots"
    )

    videos_dir = (
        output_dir
        / "videos"
    )

    # ------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------

    if args.profile == "smoke":
        agent_steps = 2_000
        eval_every = 500
        default_runs = 2

    else:
        agent_steps = 300_000
        eval_every = 20_000

        default_runs = int(
            config.get(
                "n_runs",
                30,
            )
        )

    agent_steps = int(
        config.get(
            "agent_steps",
            agent_steps,
        )
    )

    eval_every = int(
        config.get(
            "eval_every",
            eval_every,
        )
    )

    eval_episodes = int(
        config.get(
            "eval_episodes",
            30,
        )
    )

    n_runs = (
        int(args.runs)
        if args.runs is not None
        else default_runs
    )

    if n_runs <= 0:
        raise SystemExit(
            "--runs must be >= 1"
        )

    rank_coef = float(
        args.rank_coef
        if args.rank_coef is not None
        else config.get(
            "rank_coef",
            0.001,
        )
    )

    # ------------------------------------------------------------
    # Environment CLI forwarded consistently to all training stages
    # ------------------------------------------------------------

    env_args = [
        "--config",
        config_path,
        "--sigma",
        format(
            sigma,
            ".17g",
        ),
        "--gamma",
        format(
            gamma,
            ".17g",
        ),
        "--determinism",
        format(
            determinism,
            ".17g",
        ),
        "--deterministic-sigma-scale",
        format(
            det_sigma_scale,
            ".17g",
        ),
    ]

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print(
        "=== Learned-bound full experiment ==="
    )

    print(
        f"output:                    "
        f"{output_dir}"
    )

    print(
        f"sigma:                     "
        f"{sigma:.10g}"
    )

    print(
        f"determinism:               "
        f"{determinism:.10g}"
    )

    print(
        f"det sigma scale:           "
        f"{det_sigma_scale:.10g}"
    )

    print(
        f"actual category-5 sigma:   "
        f"{deterministic_sigma:.10g}"
    )

    print(
        f"gamma:                     "
        f"{gamma:.10g}"
    )

    print(
        f"Q-bound iterations:        "
        f"{args.q_bound_iters}"
    )

    print(
        f"Q-bound target tau:        "
        f"{args.q_bound_tau}"
    )

    print(
        f"Q-bound source:            "
        f"{args.lq_source}"
    )

    print(
        f"agent steps/run:           "
        f"{agent_steps}"
    )

    print(
        f"agent runs requested:      "
        f"{n_runs}"
    )

    print(
        f"agent seeds:               "
        f"{args.seed}.."
        f"{args.seed + n_runs - 1}"
    )

    print(
        f"resume:                    "
        f"{args.resume}"
    )

    print(
        "RA pruning:                "
        "frozen learned Q_UB/Q_LB"
    )

    print(
        "old Q_single margin:       "
        "NOT USED"
    )

    # ============================================================
    # 1/7 Original R1/R2 generation + component videos
    # ============================================================

    original_cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        str(config_path),
        "--sigma",
        sigma,
        "--gamma",
        gamma,
        "--determinism",
        determinism,
        "--deterministic-sigma-scale",
        det_sigma_scale,
        "--profile",
        args.profile,
        "--seed",
        args.seed,
        "--runs",
        n_runs,
        "--output-root",
        args.output_root,
        "--stop-after",
        "videos",
    ]

    if args.resume:
        original_cmd.append(
            "--resume"
        )

    run_stage(
        (
            "1/7 Generate R1/R2 data, "
            "transition bounds, Lipschitz constants, "
            "component videos"
        ),
        original_cmd,
        args.dry_run,
    )

    bounds_path = (
        data_dir
        / "transition_bounds.json"
    )

    lipschitz_path = (
        data_dir
        / "lipschitz_constants.json"
    )

    cross_tile_reward_path = (
        data_dir
        / "cross_tile_reward_lipschitz.json"
    )

    cross_cat_dynamics_path = (
        data_dir
        / "cross_category_dynamics_lipschitz.json"
    )

    noise_ratio_path = (
        data_dir
        / "category_noise_action_ratio.json"
    )

    run_config_path = (
        data_dir
        / "run_config.json"
    )

    if not args.dry_run:
        require_files(
            [
                bounds_path,
                lipschitz_path,
                cross_tile_reward_path,
                cross_cat_dynamics_path,
                noise_ratio_path,
                run_config_path,
            ]
        )

        validate_cross_category_artifact(
            cross_tile_reward_path, sigma, det_sigma_scale, gamma, kind="cross_tile"
        )
        validate_cross_category_artifact(
            cross_cat_dynamics_path, sigma, det_sigma_scale, gamma, kind="cross_category"
        )

        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        videos_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # 2/7 Learned Q_UB / Q_LB
    # ============================================================

    q_ub_path = (
        models_dir
        / (
            f"q_ub_"
            f"{args.lq_source}.pth"
        )
    )

    q_lb_path = (
        models_dir
        / (
            f"q_lb_"
            f"{args.lq_source}.pth"
        )
    )

    q_manifest_path = (
        models_dir
        / (
            f"q_bounds_"
            f"{args.lq_source}_"
            f"manifest.json"
        )
    )

    if (
        args.resume
        and artifacts_ready(
            [
                q_ub_path,
                q_lb_path,
            ]
        )
    ):
        print(
            "\n=== 2/7 Q-bound training skipped; "
            "Q_UB/Q_LB checkpoints present ==="
        )

    else:
        run_stage(
            "2/7 Train learned Q_UB/Q_LB",
            [
                sys.executable,
                "scripts/train_q_bounds.py",
                *env_args,
                "--bounds",
                bounds_path,
                "--lipschitz",
                lipschitz_path,
                "--cross-tile-reward-lipschitz",
                cross_tile_reward_path,
                "--cross-category-dynamics-lipschitz",
                cross_cat_dynamics_path,
                "--lq-source",
                args.lq_source,
                "--iters",
                args.q_bound_iters,
                "--target-tau",
                args.q_bound_tau,
                "--seed",
                args.seed,
                "--out-ub",
                q_ub_path,
                "--out-lb",
                q_lb_path,
                "--manifest",
                q_manifest_path,
            ],
            args.dry_run,
        )

    if not args.dry_run:
        require_files(
            [
                q_ub_path,
                q_lb_path,
            ]
        )

    # ============================================================
    # 3/7 Learned Q-bound analysis / heatmaps
    # ============================================================

    bound_plots_dir = (
        plots_dir
        / (
            "learned_q_bounds_"
            f"{args.lq_source}"
        )
    )

    run_stage(
        "3/7 Analyze learned Q bounds",
        [
            sys.executable,
            "scripts/generate_q_bound_analysis.py",
            "--q-ub",
            q_ub_path,
            "--q-lb",
            q_lb_path,
            "--output-dir",
            bound_plots_dir,
        ],
        args.dry_run,
    )

    # ============================================================
    # 4/7 Baseline DQN
    # ============================================================

    baseline_prefix = (
        models_dir
        / "dqn"
    )

    baseline_common = [
        *env_args,
        "--steps",
        agent_steps,
        "--eval-every",
        eval_every,
        "--eval-episodes",
        eval_episodes,
    ]

    (
        baseline_runs_path,
        baseline_representative_model,
    ) = train_runs(
        "4/7 Baseline DQN",
        "scripts/train_baseline_dqn.py",
        baseline_prefix,
        baseline_common,
        n_runs,
        args.seed,
        args.dry_run,
        args.resume,
    )

    # ============================================================
    # 5/7 RA-DQN using learned Q_UB/Q_LB
    # ============================================================

    ra_prefix = (
        models_dir
        / (
            "ra_dqn_learned_bounds_"
            f"{args.lq_source}"
        )
    )

    ra_common = [
        *env_args,
        "--q-ub",
        q_ub_path,
        "--q-lb",
        q_lb_path,
        "--lq-source",
        args.lq_source,
        "--steps",
        agent_steps,
        "--eval-every",
        eval_every,
        "--eval-episodes",
        eval_episodes,
        "--rank-coef",
        rank_coef,
        "--tol",
        args.pruning_tol,
    ]

    (
        ra_runs_path,
        ra_representative_model,
    ) = train_runs(
        "5/7 RA-DQN with learned Q bounds",
        "scripts/train_ra_dqn_bounds.py",
        ra_prefix,
        ra_common,
        n_runs,
        args.seed,
        args.dry_run,
        args.resume,
    )

    # ============================================================
    # 6/7 Returns plot
    # ============================================================

    returns_plot = (
        plots_dir
        / "returns_plot_learned_bounds.png"
    )

    run_stage(
        "6/7 Plot DQN vs learned-bound RA-DQN",
        [
            sys.executable,
            "scripts/plot_returns.py",
            baseline_runs_path,
            ra_runs_path,
            "--labels",
            "DQN",
            (
                "RA-DQN-Learned-"
                f"{args.lq_source}"
            ),
            "--step",
            eval_every,
            "--out",
            returns_plot,
        ],
        args.dry_run,
    )

    if not args.dry_run:
        require_files(
            [
                returns_plot,
            ]
        )

    # ============================================================
    # 7/7 Representative policy videos
    #
    # Representative = base seed.
    #
    # RA video uses learned Q_UB/Q_LB pruning directly.
    # ============================================================

    dqn_video, ra_video = (
        record_agent_videos(
            output_dir=output_dir,
            config=config,
            sigma=sigma,
            seed=args.seed,
            dqn_checkpoint=(
                baseline_representative_model
            ),
            ra_checkpoint=(
                ra_representative_model
            ),
            q_ub_checkpoint=(
                q_ub_path
            ),
            q_lb_checkpoint=(
                q_lb_path
            ),
            lq_source=(
                args.lq_source
            ),
            tol=(
                args.pruning_tol
            ),
            fps=(
                args.video_fps
            ),
            substeps=(
                args.video_substeps
            ),
            resume=(
                args.resume
            ),
            dry_run=(
                args.dry_run
            ),
        )
    )

    # ============================================================
    # Finished
    # ============================================================

    if args.dry_run:
        print(
            "\nDry run complete."
        )

        return

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LEARNED-BOUND EXPERIMENT COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"experiment:       "
        f"{output_dir}"
    )

    print(
        f"Q_UB:             "
        f"{q_ub_path}"
    )

    print(
        f"Q_LB:             "
        f"{q_lb_path}"
    )

    print(
        f"DQN runs:         "
        f"{baseline_runs_path}"
    )

    print(
        f"RA-DQN runs:      "
        f"{ra_runs_path}"
    )

    print(
        f"DQN model:        "
        f"{baseline_representative_model}"
    )

    print(
        f"RA-DQN model:     "
        f"{ra_representative_model}"
    )

    print(
        f"returns plot:     "
        f"{returns_plot}"
    )

    print(
        f"DQN video:        "
        f"{dqn_video}"
    )

    print(
        f"RA-DQN video:     "
        f"{ra_video}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()


# """Run a fresh full experiment using learned Q_UB/Q_LB bounds for pruning.

# Pipeline:
# 1) Reuse the original run_experiment.py only for R1/R2 data generation + component videos.
# 2) Train learned Q_UB/Q_LB from transition_bounds.json + lipschitz_constants.json.
# 3) Analyze learned Q bounds.
# 4) Train baseline DQN over N independent seeds.
# 5) Train RA-DQN over the same seeds using frozen Q_UB/Q_LB pruning bounds.
# 6) Plot DQN vs learned-bound RA-DQN returns.

# This intentionally does NOT call the old train_ra_dqn.py margin-pruning path.
# """
# from __future__ import annotations

# import argparse
# import hashlib
# import json
# import os
# import re
# import shutil
# import subprocess
# import sys
# from pathlib import Path

# import numpy as np

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT / "src"))

# from dollar_euro_lipschitz.config import (
    # apply_resolved_determinism,
    # apply_resolved_deterministic_sigma_scale,
    # apply_resolved_sigma,
    # artifact_sigma_tag,
    # load_config,
    # resolve_determinism,
    # resolve_deterministic_sigma_scale,
    # resolve_sigma,
# )


# def run_stage(name, cmd, dry_run=False, env=None):
    # print(f"\n=== {name} ===\n{subprocess.list2cmdline([str(x) for x in cmd])}", flush=True)
    # if dry_run:
        # return
    # r = subprocess.run([str(x) for x in cmd], cwd=ROOT, env=env, check=False)
    # if r.returncode:
        # raise SystemExit(r.returncode)


# def require_files(paths):
    # missing = [str(p) for p in paths if not Path(p).is_file() or Path(p).stat().st_size <= 0]
    # if missing:
        # raise SystemExit("Missing expected artifact(s):\n  " + "\n  ".join(missing))


# def parse_args():
    # p = argparse.ArgumentParser(description=__doc__)
    # p.add_argument("config", nargs="?", default=str(ROOT / "configs" / "default.json"))
    # p.add_argument("--config", dest="config_option", default=None)
    # p.add_argument("--sigma", type=float, default=None)
    # p.add_argument("--gamma", type=float, default=None)
    # p.add_argument("--determinism", type=float, default=None)
    # p.add_argument("--deterministic-sigma-scale", type=float, default=None)
    # p.add_argument("--profile", choices=["smoke", "full"], default="full")
    # p.add_argument("--seed", type=int, default=0)
    # p.add_argument("--runs", type=int, default=None)
    # p.add_argument("--parallel-runs", type=int, default=None)
    # p.add_argument("--rank-coef", type=float, default=None)
    # p.add_argument("--output-root", default=str(ROOT / "outputs" / "experiments"))
    # p.add_argument("--resume", action="store_true")
    # p.add_argument("--dry-run", action="store_true")
    # p.add_argument("--q-bound-iters", type=int, default=120000)
    # p.add_argument("--q-bound-tau", type=float, default=0.01)
    # p.add_argument("--lq-source", choices=["theoretical", "empirical"], default="theoretical")
    # return p.parse_args()


# def experiment_identity(config_path, sigma, gamma, determinism, det_sigma_scale, profile, seed, output_root):
    # sigma_arg = format(sigma, ".17g")
    # gamma_arg = format(gamma, ".17g")
    # det_arg = format(determinism, ".17g")
    # ds_arg = format(det_sigma_scale, ".17g")
    # identity = (
        # config_path.read_bytes()
        # + b"\0effective_sigma=" + sigma_arg.encode("ascii")
        # + b"\0effective_gamma=" + gamma_arg.encode("ascii")
        # + b"\0effective_determinism=" + det_arg.encode("ascii")
        # + b"\0effective_deterministic_sigma_scale=" + ds_arg.encode("ascii")
    # )
    # config_hash = hashlib.sha256(identity).hexdigest()[:12]
    # stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_path.stem)
    # det_tag = f"_det{int(round(determinism * 100)):02d}"
    # if det_sigma_scale > 0.0:
        # det_tag += f"_dsig{artifact_sigma_tag(det_sigma_scale)}"
    # return Path(output_root) / (
        # f"sigma_{artifact_sigma_tag(sigma)}{det_tag}_cfg_{stem}_{config_hash}_{profile}_seed{seed}"
    # )


# def train_runs(name, script, prefix, common_args, n_runs, base_seed, dry_run, resume):
    # prefix = Path(prefix)
    # curves = []
    # for i in range(n_runs):
        # seed = base_seed + i
        # run_prefix = prefix.parent / f"{prefix.name}_seed{seed}"
        # pth = run_prefix.with_suffix(".pth")
        # npy = run_prefix.with_suffix(".npy")
        # if resume and pth.is_file() and npy.is_file():
            # print(f"\n=== {name} seed={seed} skipped; artifacts present ===")
        # else:
            # run_stage(
                # f"{name} run {i+1}/{n_runs} seed={seed}",
                # [sys.executable, script, *common_args, "--seed", seed, "--out-prefix", run_prefix],
                # dry_run,
            # )
        # if not dry_run:
            # require_files([pth, npy])
            # curves.append(np.asarray(np.load(npy), dtype=np.float32).reshape(-1))

    # runs_path = prefix.with_name(prefix.name + "_runs").with_suffix(".npy")
    # if dry_run:
        # return runs_path
    # length = min(x.size for x in curves)
    # stacked = np.stack([x[:length] for x in curves], axis=0)
    # np.save(runs_path, stacked)
    # np.save(prefix.with_suffix(".npy"), stacked.mean(axis=0))
    # shutil.copyfile(prefix.parent / f"{prefix.name}_seed{base_seed}.pth", prefix.with_suffix(".pth"))
    # return runs_path


# def main():
    # a = parse_args()
    # config_path = Path(a.config_option or a.config)
    # if not config_path.is_file():
        # candidate = ROOT / str(config_path).replace("\\", "/")
        # if candidate.is_file():
            # config_path = candidate
        # else:
            # raise SystemExit(f"config not found: {config_path}")
    # config_path = config_path.resolve()
    # cfg = load_config(str(config_path))

    # sigma = resolve_sigma(cfg, a.sigma, None)
    # apply_resolved_sigma(cfg, sigma)
    # determinism = resolve_determinism(cfg, a.determinism)
    # apply_resolved_determinism(cfg, determinism)
    # ds = resolve_deterministic_sigma_scale(cfg, a.deterministic_sigma_scale)
    # apply_resolved_deterministic_sigma_scale(cfg, ds)
    # if a.gamma is not None:
        # cfg["gamma"] = float(a.gamma)
    # gamma = float(cfg.get("gamma", 0.99))

    # output_dir = experiment_identity(
        # config_path, sigma, gamma, determinism, ds, a.profile, a.seed, a.output_root
    # )
    # data = output_dir / "data"
    # models = output_dir / "models"
    # plots = output_dir / "plots"

    # if a.profile == "smoke":
        # agent_steps, eval_every, default_runs = 2000, 500, 2
    # else:
        # agent_steps, eval_every, default_runs = 300000, 20000, int(cfg.get("n_runs", 30))
    # agent_steps = int(cfg.get("agent_steps", agent_steps))
    # eval_every = int(cfg.get("eval_every", eval_every))
    # eval_episodes = int(cfg.get("eval_episodes", 30))
    # n_runs = int(a.runs) if a.runs is not None else default_runs
    # rank_coef = float(a.rank_coef if a.rank_coef is not None else cfg.get("rank_coef", 0.001))

    # env_args = [
        # "--config", config_path,
        # "--sigma", format(sigma, ".17g"),
        # "--gamma", format(gamma, ".17g"),
        # "--determinism", format(determinism, ".17g"),
        # "--deterministic-sigma-scale", format(ds, ".17g"),
    # ]

    # print("=== Learned-bound full experiment ===")
    # print(f"output:          {output_dir}")
    # print(f"sigma:           {sigma}")
    # print(f"determinism:     {determinism}")
    # print(f"det sigma scale: {ds}")
    # print(f"gamma:           {gamma}")
    # print(f"Q-bound iters:   {a.q_bound_iters}")
    # print(f"Q-bound tau:     {a.q_bound_tau}")
    # print(f"Q-bound source:  {a.lq_source}")
    # print(f"agent runs:      {n_runs}")
    # print("RA pruning:      frozen learned Q_UB/Q_LB (NO old analytical margin)")

    # # Stage 1: use the original runner for exactly its R1/R2 generation + videos.
    # original_cmd = [
        # sys.executable, "scripts/run_experiment.py", str(config_path),
        # "--sigma", sigma,
        # "--gamma", gamma,
        # "--determinism", determinism,
        # "--deterministic-sigma-scale", ds,
        # "--profile", a.profile,
        # "--seed", a.seed,
        # "--runs", n_runs,
        # "--output-root", a.output_root,
        # "--stop-after", "videos",
    # ]
    # if a.resume:
        # original_cmd.append("--resume")
    # run_stage("1/6 Generate R1/R2 data, transition bounds, Lipschitz constants, component videos", original_cmd, a.dry_run)

    # bounds = data / "transition_bounds.json"
    # lips = data / "lipschitz_constants.json"
    # if not a.dry_run:
        # require_files([bounds, lips, data / "run_config.json"])
        # models.mkdir(parents=True, exist_ok=True)
        # plots.mkdir(parents=True, exist_ok=True)

    # q_ub = models / f"q_ub_{a.lq_source}.pth"
    # q_lb = models / f"q_lb_{a.lq_source}.pth"
    # q_manifest = models / f"q_bounds_{a.lq_source}_manifest.json"

    # if a.resume and q_ub.is_file() and q_lb.is_file():
        # print("\n=== 2/6 Q-bound training skipped; checkpoints present ===")
    # else:
        # run_stage(
            # "2/6 Train learned Q_UB/Q_LB",
            # [
                # sys.executable, "scripts/train_q_bounds.py", *env_args,
                # "--bounds", bounds,
                # "--lipschitz", lips,
                # "--lq-source", a.lq_source,
                # "--iters", a.q_bound_iters,
                # "--target-tau", a.q_bound_tau,
                # "--seed", a.seed,
                # "--out-ub", q_ub,
                # "--out-lb", q_lb,
                # "--manifest", q_manifest,
            # ],
            # a.dry_run,
        # )
    # if not a.dry_run:
        # require_files([q_ub, q_lb])

    # bound_plots = plots / f"learned_q_bounds_{a.lq_source}"
    # run_stage(
        # "3/6 Analyze learned Q bounds",
        # [
            # sys.executable, "scripts/generate_q_bound_analysis.py",
            # "--q-ub", q_ub,
            # "--q-lb", q_lb,
            # "--output-dir", bound_plots,
        # ],
        # a.dry_run,
    # )

    # baseline_prefix = models / "dqn"
    # baseline_common = [
        # *env_args,
        # "--steps", agent_steps,
        # "--eval-every", eval_every,
        # "--eval-episodes", eval_episodes,
    # ]
    # baseline_runs = train_runs(
        # "4/6 Baseline DQN", "scripts/train_baseline_dqn.py", baseline_prefix,
        # baseline_common, n_runs, a.seed, a.dry_run, a.resume,
    # )

    # ra_prefix = models / f"ra_dqn_learned_bounds_{a.lq_source}"
    # ra_common = [
        # *env_args,
        # "--q-ub", q_ub,
        # "--q-lb", q_lb,
        # "--lq-source", a.lq_source,
        # "--steps", agent_steps,
        # "--eval-every", eval_every,
        # "--eval-episodes", eval_episodes,
        # "--rank-coef", rank_coef,
    # ]
    # ra_runs = train_runs(
        # "5/6 RA-DQN with learned Q bounds", "scripts/train_ra_dqn_bounds.py", ra_prefix,
        # ra_common, n_runs, a.seed, a.dry_run, a.resume,
    # )

    # returns_plot = plots / "returns_plot_learned_bounds.png"
    # run_stage(
        # "6/6 Plot DQN vs learned-bound RA-DQN",
        # [
            # sys.executable, "scripts/plot_returns.py",
            # baseline_runs, ra_runs,
            # "--labels", "DQN", f"RA-DQN-Learned-{a.lq_source}",
            # "--step", eval_every,
            # "--out", returns_plot,
        # ],
        # a.dry_run,
    # )

    # if not a.dry_run:
        # print(f"\nExperiment complete: {output_dir}")
        # print(f"Q_UB: {q_ub}")
        # print(f"Q_LB: {q_lb}")
        # print(f"RA model: {ra_prefix.with_suffix('.pth')}")
        # print(f"Returns: {returns_plot}")


# if __name__ == "__main__":
    # main()

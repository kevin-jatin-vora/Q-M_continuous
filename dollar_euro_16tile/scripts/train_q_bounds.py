"""Train learned Q upper/lower Bellman bounds for the Dollar-Euro project.

Uses the existing project artifacts and equations:

    s_bar' = clip(s + delta_mean(category, action), 0, 1)
    delta_r = Lr(tile, action) * pruning_radius(category, action)
    delta_q = Lq(tile, action) * pruning_radius(category, action)

    y_UB = R + delta_r + gamma * (1-done) * (max_a Q_UB_tgt(s_bar',a) + delta_q)
    y_LB = R - delta_r + gamma * (1-done) * (max_a Q_LB_tgt(s_bar',a) - delta_q)

Q_UB and Q_LB start identically. There is no consistency loss and no Q clipping.
Gradient norm clipping is enabled by default at 10.0 because it stabilized the fitted
Bellman updates without changing the Bellman targets.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from dollar_euro_lipschitz.bounds import RegionActionBounds
from dollar_euro_lipschitz.config import (
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_determinism,
    apply_resolved_deterministic_sigma_scale,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    load_config,
    resolve_determinism,
    resolve_deterministic_sigma_scale,
    resolve_sigma,
    validate_bounds_sigma,
    validate_pruning_provenance,
)
from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
from dollar_euro_lipschitz.layout import categories_from_states
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.q_bounds import (
    build_one_step_uncertainty_tables,
    batch_uncertainty,
)
from dollar_euro_lipschitz.rewards import scalar_reward_from_next_states


def parameters_identical(first, second):
    return all(
        torch.equal(p1.detach().cpu(), p2.detach().cpu())
        for p1, p2 in zip(first.parameters(), second.parameters())
    )


def assert_finite(name, tensor, iteration):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"Non-finite tensor at iteration {iteration}: {name}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser)

    parser.add_argument("--bounds", required=True)
    parser.add_argument("--lipschitz", required=True)
    parser.add_argument(
        "--lq-source",
        choices=["empirical", "theoretical"],
        default="theoretical",
    )
    parser.add_argument("--iters", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-ub", required=True)
    parser.add_argument("--out-lb", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=10000)

    add_training_arguments(parser, "batch_size", "gamma", "lr")
    args = parser.parse_args()

    cfg = load_config(args.config)

    sigma = resolve_sigma(cfg, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(cfg, sigma)

    determinism = resolve_determinism(cfg, args.determinism)
    apply_resolved_determinism(cfg, determinism)

    deterministic_sigma_scale = resolve_deterministic_sigma_scale(
        cfg, args.deterministic_sigma_scale
    )
    apply_resolved_deterministic_sigma_scale(
        cfg, deterministic_sigma_scale
    )

    apply_training_defaults(args, cfg)

    validate_bounds_sigma(
        args.bounds,
        sigma,
        allow_mismatch=args.allow_radius_sigma_mismatch,
        context="learned Q-bound training",
    )
    validate_pruning_provenance(args.bounds, args.lipschitz, sigma)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ContinuousDollarEuroEnv(
        render_mode=None,
        auto_render=False,
        **env_kwargs(cfg, sigma),
    )

    bounds = RegionActionBounds(args.bounds)
    mean_deltas = np.asarray(
        [
            [bounds.get(category, action)["mean"] for action in range(4)]
            for category in range(1, 6)
        ],
        dtype=np.float32,
    )

    dr_table, dq_table, _, _ = build_one_step_uncertainty_tables(
        args.bounds,
        args.lipschitz,
        args.lq_source,
        determinism,
    )

    # Exact same initialization for UB and LB.
    q_ub = QNet().to(device)
    q_lb = QNet().to(device)
    q_lb.load_state_dict(q_ub.state_dict())

    q_ub_tgt = QNet().to(device)
    q_lb_tgt = QNet().to(device)
    q_ub_tgt.load_state_dict(q_ub.state_dict())
    q_lb_tgt.load_state_dict(q_ub.state_dict())

    q_ub_tgt.eval()
    q_lb_tgt.eval()
    for parameter in q_ub_tgt.parameters():
        parameter.requires_grad_(False)
    for parameter in q_lb_tgt.parameters():
        parameter.requires_grad_(False)

    if not parameters_identical(q_ub, q_lb):
        raise RuntimeError("Q_UB_0 and Q_LB_0 are not identical")
    if not parameters_identical(q_ub, q_ub_tgt):
        raise RuntimeError("Q_UB target does not match initial Q_UB")
    if not parameters_identical(q_lb, q_lb_tgt):
        raise RuntimeError("Q_LB target does not match initial Q_LB")

    opt_ub = torch.optim.Adam(q_ub.parameters(), lr=args.lr)
    opt_lb = torch.optim.Adam(q_lb.parameters(), lr=args.lr)

    print("=== Learned Q bounds ===")
    print(
        f"device={device} sigma={sigma:g} determinism={determinism:g} "
        f"gamma={args.gamma:g} lr={args.lr:g} batch={args.batch_size}"
    )
    print(
        f"iters={args.iters} tau={args.target_tau:g} "
        f"grad_clip={args.max_grad_norm:g} Lq={args.lq_source}"
    )
    print("Q_UB_0 == Q_LB_0; no consistency loss; no Q clipping")

    for iteration in range(1, args.iters + 1):
        states_np = np.random.rand(args.batch_size, 2).astype(np.float32)
        actions_np = np.random.randint(
            0, 4, size=args.batch_size
        ).astype(np.int64)

        categories_np = categories_from_states(
            states_np,
            determinism,
            tile_map=env.tile_map,
        )
        category_indices = categories_np - 1

        next_states_np = np.clip(
            states_np + mean_deltas[category_indices, actions_np],
            0.0,
            1.0,
        ).astype(np.float32)

        rewards_np, dones_np = scalar_reward_from_next_states(
            next_states_np, env
        )
        delta_r_np, delta_q_np = batch_uncertainty(
            states_np,
            actions_np,
            dr_table,
            dq_table,
        )

        states = torch.as_tensor(
            states_np, dtype=torch.float32, device=device
        )
        next_states = torch.as_tensor(
            next_states_np, dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            actions_np, dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            rewards_np, dtype=torch.float32, device=device
        ).unsqueeze(1)
        dones = torch.as_tensor(
            dones_np, dtype=torch.float32, device=device
        ).unsqueeze(1)
        delta_r = torch.as_tensor(
            delta_r_np, dtype=torch.float32, device=device
        ).unsqueeze(1)
        delta_q = torch.as_tensor(
            delta_q_np, dtype=torch.float32, device=device
        ).unsqueeze(1)

        continuation = 1.0 - dones

        with torch.no_grad():
            max_q_ub_next = q_ub_tgt(next_states).max(
                dim=1, keepdim=True
            ).values
            max_q_lb_next = q_lb_tgt(next_states).max(
                dim=1, keepdim=True
            ).values

            target_ub = (
                rewards
                + delta_r
                + args.gamma
                * continuation
                * (max_q_ub_next + delta_q)
            )
            target_lb = (
                rewards
                - delta_r
                + args.gamma
                * continuation
                * (max_q_lb_next - delta_q)
            )

        q_ub_sa = q_ub(states).gather(1, actions)
        q_lb_sa = q_lb(states).gather(1, actions)

        loss_ub = F.mse_loss(q_ub_sa, target_ub)
        loss_lb = F.mse_loss(q_lb_sa, target_lb)

        assert_finite("target_ub", target_ub, iteration)
        assert_finite("target_lb", target_lb, iteration)
        assert_finite("loss_ub", loss_ub, iteration)
        assert_finite("loss_lb", loss_lb, iteration)

        opt_ub.zero_grad()
        loss_ub.backward()
        if args.max_grad_norm > 0:
            ub_preclip_norm = torch.nn.utils.clip_grad_norm_(
                q_ub.parameters(), max_norm=args.max_grad_norm
            )
        else:
            ub_preclip_norm = torch.tensor(float("nan"))
        opt_ub.step()

        opt_lb.zero_grad()
        loss_lb.backward()
        if args.max_grad_norm > 0:
            lb_preclip_norm = torch.nn.utils.clip_grad_norm_(
                q_lb.parameters(), max_norm=args.max_grad_norm
            )
        else:
            lb_preclip_norm = torch.tensor(float("nan"))
        opt_lb.step()

        # Polyak target update every optimizer step.
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                q_ub_tgt.parameters(), q_ub.parameters()
            ):
                target_parameter.data.lerp_(
                    online_parameter.data, args.target_tau
                )
            for target_parameter, online_parameter in zip(
                q_lb_tgt.parameters(), q_lb.parameters()
            ):
                target_parameter.data.lerp_(
                    online_parameter.data, args.target_tau
                )

        if (
            iteration == 1
            or iteration % args.progress_every == 0
            or iteration == args.iters
        ):
            print(
                f"iter={iteration:6d}/{args.iters} "
                f"loss_ub={loss_ub.item():.6f} "
                f"loss_lb={loss_lb.item():.6f} "
                f"rmse_ub={loss_ub.sqrt().item():.4f} "
                f"rmse_lb={loss_lb.sqrt().item():.4f} "
                f"grad_preclip_ub={float(ub_preclip_norm):.2f} "
                f"grad_preclip_lb={float(lb_preclip_norm):.2f} "
                f"max_q_ub_tgt={max_q_ub_next.max().item():.3f}"
            )

    env.close()

    out_ub = Path(args.out_ub)
    out_lb = Path(args.out_lb)
    out_ub.parent.mkdir(parents=True, exist_ok=True)
    out_lb.parent.mkdir(parents=True, exist_ok=True)

    torch.save(q_ub.state_dict(), out_ub)
    torch.save(q_lb.state_dict(), out_lb)

    manifest = (
        Path(args.manifest)
        if args.manifest
        else out_ub.parent / f"q_bounds_{args.lq_source}_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "method": "learned_lipschitz_bellman_q_bounds",
        "sigma": sigma,
        "determinism": determinism,
        "deterministic_sigma_scale": deterministic_sigma_scale,
        "gamma": args.gamma,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "target_tau": args.target_tau,
        "iters": args.iters,
        "seed": args.seed,
        "lq_source": args.lq_source,
        "bounds": str(Path(args.bounds).resolve()),
        "lipschitz": str(Path(args.lipschitz).resolve()),
        "initialization": {
            "q_ub_equals_q_lb": True,
            "q_ub_target_equals_q_ub": True,
            "q_lb_target_equals_q_lb": True,
        },
        "consistency_loss": False,
        "q_clipping": False,
        "gradient_clipping": args.max_grad_norm > 0,
        "max_grad_norm": args.max_grad_norm,
        "early_stopping": False,
        "best_checkpoint_replacement": False,
        "double_q": False,
        "delta_r": "Lr(tile,a) * pruning_radius(category,a)",
        "delta_q": "Lq(tile,a) * pruning_radius(category,a)",
        "ub_target": (
            "R + delta_r + gamma*(1-d)*(max_a Q_UB_target(s_bar,a) + delta_q)"
        ),
        "lb_target": (
            "R - delta_r + gamma*(1-d)*(max_a Q_LB_target(s_bar,a) - delta_q)"
        ),
    }
    manifest.write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    print(f"saved {out_ub.resolve()}")
    print(f"saved {out_lb.resolve()}")
    print(f"saved {manifest.resolve()}")


if __name__ == "__main__":
    main()

# """Train separate learned Q upper/lower bounds.

# Project formulation
# -------------------

# Center transition:

    # s_bar' = clip(s + delta_mean(category, action), 0, 1)

# One-step uncertainty:

    # delta_r = Lr(tile, action) * pruning_radius(category, action)
    # delta_q = Lq(tile, action) * pruning_radius(category, action)

# Bellman targets:

    # target_UB =
        # R(s, a, s_bar') + delta_r
        # + gamma * (1-done)
          # * (max_a' Q_UB_target(s_bar', a') + delta_q)

    # target_LB =
        # R(s, a, s_bar') - delta_r
        # + gamma * (1-done)
          # * (max_a' Q_LB_target(s_bar', a') - delta_q)

# Important:
# - Q_UB_0 == Q_LB_0 exactly.
# - Both target networks start from the same initialization.
# - No consistency/order loss.
# - No Q clipping.
# - No gradient clipping.
# - No early stopping.
# - No best-checkpoint replacement.
# - No Double-Q modification.
# - No old analytical /(1-gamma) RA margin.
# - gamma/lr/batch size/etc. come from this project's config/CLI.
# """

# import argparse
# import json
# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT / "src"))

# import numpy as np
# import torch
# import torch.nn.functional as F

# from dollar_euro_lipschitz.bounds import (
    # RegionActionBounds,
    # LipschitzConstants,
# )
# from dollar_euro_lipschitz.config import (
    # add_environment_arguments,
    # add_training_arguments,
    # apply_resolved_determinism,
    # apply_resolved_deterministic_sigma_scale,
    # apply_resolved_sigma,
    # apply_training_defaults,
    # env_kwargs,
    # load_config,
    # resolve_determinism,
    # resolve_deterministic_sigma_scale,
    # resolve_sigma,
    # validate_bounds_sigma,
    # validate_pruning_provenance,
# )
# from dollar_euro_lipschitz.env import ContinuousDollarEuroEnv
# from dollar_euro_lipschitz.layout import (
    # categories_from_states,
    # tile_ids_from_states,
# )
# from dollar_euro_lipschitz.models import QNet
# from dollar_euro_lipschitz.q_bounds import (
    # build_one_step_uncertainty_tables,
    # batch_uncertainty,
# )
# from dollar_euro_lipschitz.rewards import (
    # scalar_reward_from_next_states,
# )


# def tensor_stats(x):
    # x = x.detach().float()

    # return {
        # "min": float(x.min().item()),
        # "mean": float(x.mean().item()),
        # "max": float(x.max().item()),
        # "abs_max": float(x.abs().max().item()),
    # }


# def print_stats(name, x):
    # st = tensor_stats(x)

    # print(
        # f"  {name:<22}"
        # f" min={st['min']:12.6f}"
        # f" mean={st['mean']:12.6f}"
        # f" max={st['max']:12.6f}"
        # f" absmax={st['abs_max']:12.6f}"
    # )


# def gradient_norm(model):
    # total_sq = 0.0

    # for parameter in model.parameters():
        # if parameter.grad is None:
            # continue

        # grad = parameter.grad.detach().float()
        # total_sq += float(torch.sum(grad * grad).item())

    # return total_sq ** 0.5


# def parameters_identical(first, second):
    # return all(
        # torch.equal(
            # first_parameter.detach().cpu(),
            # second_parameter.detach().cpu(),
        # )
        # for first_parameter, second_parameter
        # in zip(first.parameters(), second.parameters())
    # )


# def assert_finite(name, tensor, iteration):
    # if torch.isfinite(tensor).all():
        # return

    # raise FloatingPointError(
        # f"Non-finite tensor detected at iteration "
        # f"{iteration}: {name}"
    # )


# def main():
    # parser = argparse.ArgumentParser(description=__doc__)

    # add_environment_arguments(parser)

    # parser.add_argument("--bounds", required=True)
    # parser.add_argument("--lipschitz", required=True)

    # parser.add_argument(
        # "--lq-source",
        # choices=["empirical", "theoretical"],
        # default="theoretical",
    # )

    # parser.add_argument(
        # "--iters",
        # type=int,
        # default=80000,
    # )

    # parser.add_argument(
        # "--seed",
        # type=int,
        # default=0,
    # )

    # parser.add_argument(
        # "--out-ub",
        # required=True,
    # )

    # parser.add_argument(
        # "--out-lb",
        # required=True,
    # )

    # parser.add_argument(
        # "--manifest",
        # default=None,
    # )

    # parser.add_argument(
        # "--target-tau",
        # type=float,
        # default=0.01,
    # )

    # parser.add_argument(
        # "--debug-every",
        # type=int,
        # default=None,
        # help="Detailed diagnostic interval. Default: iters//10.",
    # )

    # add_training_arguments(
        # parser,
        # "batch_size",
        # "gamma",
        # "lr",
    # )

    # args = parser.parse_args()

    # # ==============================================================
    # # Project configuration
    # # ==============================================================

    # cfg = load_config(args.config)

    # sigma = resolve_sigma(
        # cfg,
        # args.sigma,
        # args.stochasticity_scale,
    # )
    # apply_resolved_sigma(cfg, sigma)

    # determinism = resolve_determinism(
        # cfg,
        # args.determinism,
    # )
    # apply_resolved_determinism(
        # cfg,
        # determinism,
    # )

    # deterministic_sigma_scale = (
        # resolve_deterministic_sigma_scale(
            # cfg,
            # args.deterministic_sigma_scale,
        # )
    # )

    # apply_resolved_deterministic_sigma_scale(
        # cfg,
        # deterministic_sigma_scale,
    # )

    # apply_training_defaults(args, cfg)

    # validate_bounds_sigma(
        # args.bounds,
        # sigma,
        # allow_mismatch=args.allow_radius_sigma_mismatch,
        # context="learned Q-bound training",
    # )

    # validate_pruning_provenance(
        # args.bounds,
        # args.lipschitz,
        # sigma,
    # )

    # # ==============================================================
    # # Reproducibility
    # # ==============================================================

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(args.seed)

    # device = torch.device(
        # "cuda" if torch.cuda.is_available() else "cpu"
    # )

    # # ==============================================================
    # # Environment
    # # ==============================================================

    # env = ContinuousDollarEuroEnv(
        # render_mode=None,
        # auto_render=False,
        # **env_kwargs(cfg, sigma),
    # )

    # # ==============================================================
    # # Existing project artifacts
    # # ==============================================================

    # bounds = RegionActionBounds(args.bounds)

    # # Direct loader used only for exact diagnostic Lr/Lq lookup.
    # lipschitz = LipschitzConstants(
        # args.lipschitz,
        # source=args.lq_source,
    # )

    # # category x action x 2
    # mean_deltas = np.asarray(
        # [
            # [
                # bounds.get(category, action)["mean"]
                # for action in range(4)
            # ]
            # for category in range(1, 6)
        # ],
        # dtype=np.float32,
    # )

    # (
        # dr_table,
        # dq_table,
        # lr_table,
        # lq_table,
    # ) = build_one_step_uncertainty_tables(
        # args.bounds,
        # args.lipschitz,
        # args.lq_source,
        # determinism,
    # )

    # # ==============================================================
    # # Networks
    # #
    # # Q_UB_0 == Q_LB_0 exactly.
    # # Both target networks also equal that same Q0.
    # # ==============================================================

    # q_ub = QNet().to(device)

    # q_lb = QNet().to(device)
    # q_lb.load_state_dict(q_ub.state_dict())

    # q_ub_tgt = QNet().to(device)
    # q_lb_tgt = QNet().to(device)

    # q_ub_tgt.load_state_dict(q_ub.state_dict())
    # q_lb_tgt.load_state_dict(q_ub.state_dict())

    # q_ub_tgt.eval()
    # q_lb_tgt.eval()

    # for parameter in q_ub_tgt.parameters():
        # parameter.requires_grad_(False)

    # for parameter in q_lb_tgt.parameters():
        # parameter.requires_grad_(False)

    # if not parameters_identical(q_ub, q_lb):
        # raise RuntimeError(
            # "Q_UB_0 and Q_LB_0 are not identical."
        # )

    # if not parameters_identical(q_ub, q_ub_tgt):
        # raise RuntimeError(
            # "Q_UB target does not match initial Q_UB."
        # )

    # if not parameters_identical(q_lb, q_lb_tgt):
        # raise RuntimeError(
            # "Q_LB target does not match initial Q_LB."
        # )

    # # ==============================================================
    # # Optimizers
    # # ==============================================================

    # opt_ub = torch.optim.Adam(
        # q_ub.parameters(),
        # lr=args.lr,
    # )

    # opt_lb = torch.optim.Adam(
        # q_lb.parameters(),
        # lr=args.lr,
    # )

    # debug_every = (
        # args.debug_every
        # if args.debug_every is not None
        # else max(1, args.iters // 10)
    # )

    # # ==============================================================
    # # Startup diagnostics
    # # ==============================================================

    # print("=" * 78)
    # print("LEARNED Q-BOUND TRAINING")
    # print(f"device:               {device}")
    # print(f"seed:                 {args.seed}")
    # print(f"sigma:                {sigma:.10g}")
    # print(f"determinism:          {determinism:.10g}")
    # print(
        # f"det sigma scale:      "
        # f"{deterministic_sigma_scale:.10g}"
    # )
    # print(f"gamma:                {args.gamma:.10g}")
    # print(f"learning_rate:        {args.lr:.10g}")
    # print(f"batch_size:           {args.batch_size}")
    # print(f"target_tau:           {args.target_tau:.10g}")
    # print(f"iterations:           {args.iters}")
    # print(f"debug_every:          {debug_every}")
    # print(f"Lq source:            {args.lq_source}")
    # print(f"bounds:               {Path(args.bounds).resolve()}")
    # print(
        # f"lipschitz:            "
        # f"{Path(args.lipschitz).resolve()}"
    # )
    # print("Q_UB_0 == Q_LB_0:     YES")
    # print("UB target same init:  YES")
    # print("LB target same init:  YES")
    # print("consistency loss:     NONE")
    # print("Q clipping:           NONE")
    # print("gradient clipping:    NONE")
    # print("early stopping:       NONE")
    # print("best-checkpoint swap: NONE")
    # print("delta_r:              Lr * pruning_radius")
    # print("delta_q:              Lq * pruning_radius")
    # print("=" * 78)

    # print("\nSTATIC UNCERTAINTY TABLES")

    # print(
        # f"Lr table:      "
        # f"min={np.min(lr_table):.6g} "
        # f"mean={np.mean(lr_table):.6g} "
        # f"max={np.max(lr_table):.6g}"
    # )

    # print(
        # f"Lq table:      "
        # f"min={np.min(lq_table):.6g} "
        # f"mean={np.mean(lq_table):.6g} "
        # f"max={np.max(lq_table):.6g}"
    # )

    # print(
        # f"delta_r table: "
        # f"min={np.min(dr_table):.6g} "
        # f"mean={np.mean(dr_table):.6g} "
        # f"max={np.max(dr_table):.6g}"
    # )

    # print(
        # f"delta_q table: "
        # f"min={np.min(dq_table):.6g} "
        # f"mean={np.mean(dq_table):.6g} "
        # f"max={np.max(dq_table):.6g}"
    # )

    # # ==============================================================
    # # Training
    # # ==============================================================

    # for iteration in range(1, args.iters + 1):

        # # ----------------------------------------------------------
        # # Same synthetic state/action sampling used by this project's
        # # nominal model training.
        # # ----------------------------------------------------------

        # states_np = np.random.rand(
            # args.batch_size,
            # 2,
        # ).astype(np.float32)

        # actions_np = np.random.randint(
            # 0,
            # 4,
            # size=args.batch_size,
        # ).astype(np.int64)

        # # ----------------------------------------------------------
        # # Category lookup
        # # ----------------------------------------------------------

        # categories_np = categories_from_states(
            # states_np,
            # determinism,
            # tile_map=env.tile_map,
        # )

        # category_indices = categories_np - 1

        # # ----------------------------------------------------------
        # # Center transition
        # #
        # # s_bar' = clip(s + delta_mean(category,a), 0, 1)
        # # ----------------------------------------------------------

        # next_states_np = np.clip(
            # states_np
            # + mean_deltas[
                # category_indices,
                # actions_np,
            # ],
            # 0.0,
            # 1.0,
        # ).astype(np.float32)

        # # Reward and done correspond to the SAME center transition.
        # rewards_np, dones_np = scalar_reward_from_next_states(
            # next_states_np,
            # env,
        # )

        # # ----------------------------------------------------------
        # # One-step uncertainty
        # # ----------------------------------------------------------

        # delta_r_np, delta_q_np = batch_uncertainty(
            # states_np,
            # actions_np,
            # dr_table,
            # dq_table,
        # )

        # tile_ids_np = tile_ids_from_states(
            # states_np,
        # )

        # # ----------------------------------------------------------
        # # Torch tensors
        # # ----------------------------------------------------------

        # states = torch.as_tensor(
            # states_np,
            # dtype=torch.float32,
            # device=device,
        # )

        # next_states = torch.as_tensor(
            # next_states_np,
            # dtype=torch.float32,
            # device=device,
        # )

        # actions = torch.as_tensor(
            # actions_np,
            # dtype=torch.long,
            # device=device,
        # ).unsqueeze(1)

        # rewards = torch.as_tensor(
            # rewards_np,
            # dtype=torch.float32,
            # device=device,
        # ).unsqueeze(1)

        # dones = torch.as_tensor(
            # dones_np,
            # dtype=torch.float32,
            # device=device,
        # ).unsqueeze(1)

        # delta_r = torch.as_tensor(
            # delta_r_np,
            # dtype=torch.float32,
            # device=device,
        # ).unsqueeze(1)

        # delta_q = torch.as_tensor(
            # delta_q_np,
            # dtype=torch.float32,
            # device=device,
        # ).unsqueeze(1)

        # continuation = 1.0 - dones

        # # ==========================================================
        # # Bellman targets
        # #
        # # No separate V network.
        # #
        # # max_a' Q_UB_target(s_bar', a')
        # # max_a' Q_LB_target(s_bar', a')
        # # ==========================================================

        # with torch.no_grad():

            # q_ub_tgt_all_next = q_ub_tgt(
                # next_states
            # )

            # q_lb_tgt_all_next = q_lb_tgt(
                # next_states
            # )

            # max_q_ub_next, max_q_ub_actions = (
                # q_ub_tgt_all_next.max(
                    # dim=1,
                    # keepdim=True,
                # )
            # )

            # max_q_lb_next, max_q_lb_actions = (
                # q_lb_tgt_all_next.max(
                    # dim=1,
                    # keepdim=True,
                # )
            # )

            # target_ub = (
                # rewards
                # + delta_r
                # + args.gamma
                # * continuation
                # * (
                    # max_q_ub_next
                    # + delta_q
                # )
            # )

            # target_lb = (
                # rewards
                # - delta_r
                # + args.gamma
                # * continuation
                # * (
                    # max_q_lb_next
                    # - delta_q
                # )
            # )

        # # ==========================================================
        # # Online predictions
        # # ==========================================================

        # q_ub_all = q_ub(states)
        # q_lb_all = q_lb(states)

        # q_ub_sa = q_ub_all.gather(
            # 1,
            # actions,
        # )

        # q_lb_sa = q_lb_all.gather(
            # 1,
            # actions,
        # )

        # # ==========================================================
        # # Independent Bellman losses
        # #
        # # No consistency loss.
        # # ==========================================================

        # loss_ub = F.mse_loss(
            # q_ub_sa,
            # target_ub,
        # )

        # loss_lb = F.mse_loss(
            # q_lb_sa,
            # target_lb,
        # )

        # # ==========================================================
        # # Numerical checks
        # # ==========================================================

        # assert_finite(
            # "delta_r",
            # delta_r,
            # iteration,
        # )
        # assert_finite(
            # "delta_q",
            # delta_q,
            # iteration,
        # )
        # assert_finite(
            # "max_q_ub_next",
            # max_q_ub_next,
            # iteration,
        # )
        # assert_finite(
            # "max_q_lb_next",
            # max_q_lb_next,
            # iteration,
        # )
        # assert_finite(
            # "target_ub",
            # target_ub,
            # iteration,
        # )
        # assert_finite(
            # "target_lb",
            # target_lb,
            # iteration,
        # )
        # assert_finite(
            # "q_ub_sa",
            # q_ub_sa,
            # iteration,
        # )
        # assert_finite(
            # "q_lb_sa",
            # q_lb_sa,
            # iteration,
        # )
        # assert_finite(
            # "loss_ub",
            # loss_ub,
            # iteration,
        # )
        # assert_finite(
            # "loss_lb",
            # loss_lb,
            # iteration,
        # )

        # # ==========================================================
        # # UB update
        # # ==========================================================

        # opt_ub.zero_grad()
        # loss_ub.backward()
        
        # torch.nn.utils.clip_grad_norm_(
            # q_ub.parameters(),
            # max_norm=10.0,
        # )


        # ub_grad_norm = gradient_norm(
            # q_ub,
        # )

        # opt_ub.step()

        # # ==========================================================
        # # LB update
        # # ==========================================================

        # opt_lb.zero_grad()
        # loss_lb.backward()
        
        # torch.nn.utils.clip_grad_norm_(
            # q_lb.parameters(),
            # max_norm=10.0,
        # )

        # lb_grad_norm = gradient_norm(
            # q_lb,
        # )

        # opt_lb.step()

        # # ==========================================================
        # # Polyak target updates
        # # ==========================================================

        # with torch.no_grad():

            # for target_parameter, online_parameter in zip(
                # q_ub_tgt.parameters(),
                # q_ub.parameters(),
            # ):
                # target_parameter.data.lerp_(
                    # online_parameter.data,
                    # args.target_tau,
                # )

            # for target_parameter, online_parameter in zip(
                # q_lb_tgt.parameters(),
                # q_lb.parameters(),
            # ):
                # target_parameter.data.lerp_(
                    # online_parameter.data,
                    # args.target_tau,
                # )

        # # ==========================================================
        # # Diagnostics
        # # ==========================================================

        # if (
            # iteration == 1
            # or iteration % debug_every == 0
            # or iteration == args.iters
        # ):

            # with torch.no_grad():

                # # Re-evaluate online networks after optimizer update.
                # q_ub_debug_all = q_ub(
                    # states
                # )

                # q_lb_debug_all = q_lb(
                    # states
                # )

                # q_ub_debug_sa = (
                    # q_ub_debug_all.gather(
                        # 1,
                        # actions,
                    # )
                # )

                # q_lb_debug_sa = (
                    # q_lb_debug_all.gather(
                        # 1,
                        # actions,
                    # )
                # )

                # width_selected = (
                    # q_ub_debug_sa
                    # - q_lb_debug_sa
                # )

                # width_all = (
                    # q_ub_debug_all
                    # - q_lb_debug_all
                # )

                # selected_violations = int(
                    # (
                        # q_lb_debug_sa
                        # > q_ub_debug_sa
                    # )
                    # .sum()
                    # .item()
                # )

                # all_violations = int(
                    # (
                        # q_lb_debug_all
                        # > q_ub_debug_all
                    # )
                    # .sum()
                    # .item()
                # )

                # # --------------------------------------------------
                # # Find the largest Q_UB target-network value among
                # # every next-state/action pair in this batch.
                # # --------------------------------------------------

                # flat_index = int(
                    # torch.argmax(
                        # q_ub_tgt_all_next
                    # ).item()
                # )

                # sample_index = (
                    # flat_index // 4
                # )

                # maximizing_ub_action = (
                    # flat_index % 4
                # )

                # source_state = (
                    # states_np[
                        # sample_index
                    # ]
                # )

                # center_next_state = (
                    # next_states_np[
                        # sample_index
                    # ]
                # )

                # source_tile = int(
                    # tile_ids_np[
                        # sample_index
                    # ]
                # )

                # source_category = int(
                    # categories_np[
                        # sample_index
                    # ]
                # )

                # sampled_action = int(
                    # actions_np[
                        # sample_index
                    # ]
                # )

                # source_reward = float(
                    # rewards_np[
                        # sample_index
                    # ]
                # )

                # source_done = float(
                    # dones_np[
                        # sample_index
                    # ]
                # )

                # source_delta_r = float(
                    # delta_r_np[
                        # sample_index
                    # ]
                # )

                # source_delta_q = float(
                    # delta_q_np[
                        # sample_index
                    # ]
                # )

                # # Exact project Lipschitz constants for the SOURCE
                # # tile/action that generated this center transition.
                # source_lr, source_lq = lipschitz.get(
                    # source_tile,
                    # sampled_action,
                # )

                # # Exact project transition radius.
                # source_bound_stats = bounds.get(
                    # source_category,
                    # sampled_action,
                # )

                # source_radius = float(
                    # source_bound_stats[
                        # "pruning_radius"
                    # ]
                # )

                # ub_next_values = (
                    # q_ub_tgt_all_next[
                        # sample_index
                    # ]
                    # .detach()
                    # .cpu()
                    # .numpy()
                # )

                # lb_next_values = (
                    # q_lb_tgt_all_next[
                        # sample_index
                    # ]
                    # .detach()
                    # .cpu()
                    # .numpy()
                # )

                # # Verify that the table used in training agrees with
                # # the raw project constants.
                # reconstructed_delta_r = (
                    # source_lr
                    # * source_radius
                # )

                # reconstructed_delta_q = (
                    # source_lq
                    # * source_radius
                # )

            # print("\n" + "=" * 78)
            # print(
                # f"ITERATION "
                # f"{iteration}/{args.iters}"
            # )
            # print("=" * 78)

            # print(
                # f"loss_ub={loss_ub.item():.9f} "
                # f"loss_lb={loss_lb.item():.9f}"
            # )

            # print(
                # f"RMSE_ub="
                # f"{loss_ub.sqrt().item():.9f} "
                # f"RMSE_lb="
                # f"{loss_lb.sqrt().item():.9f}"
            # )

            # print(
                # f"grad_norm_ub="
                # f"{ub_grad_norm:.9f} "
                # f"grad_norm_lb="
                # f"{lb_grad_norm:.9f}"
            # )

            # print("\nINPUT / UNCERTAINTY")

            # print_stats(
                # "reward",
                # rewards,
            # )
            # print_stats(
                # "delta_r",
                # delta_r,
            # )
            # print_stats(
                # "delta_q",
                # delta_q,
            # )
            # print_stats(
                # "continuation",
                # continuation,
            # )

            # print(
                # "\nTARGET NETWORK BOOTSTRAP"
            # )

            # print_stats(
                # "max Q_UB_tgt next",
                # max_q_ub_next,
            # )
            # print_stats(
                # "max Q_LB_tgt next",
                # max_q_lb_next,
            # )

            # print("\nBELLMAN TARGETS")

            # print_stats(
                # "target_UB",
                # target_ub,
            # )
            # print_stats(
                # "target_LB",
                # target_lb,
            # )

            # print(
                # "\nONLINE Q(s,a) AFTER UPDATE"
            # )

            # print_stats(
                # "Q_UB(s,a)",
                # q_ub_debug_sa,
            # )
            # print_stats(
                # "Q_LB(s,a)",
                # q_lb_debug_sa,
            # )

            # print(
                # "\nBOUND WIDTH AFTER UPDATE"
            # )

            # print_stats(
                # "UB-LB selected",
                # width_selected,
            # )
            # print_stats(
                # "UB-LB all actions",
                # width_all,
            # )

            # print(
                # "\nORDERING DIAGNOSTIC ONLY"
            # )

            # print(
                # "  selected violations: "
                # f"{selected_violations}/"
                # f"{q_ub_debug_sa.numel()} "
                # f"("
                # f"{100.0 * selected_violations / max(q_ub_debug_sa.numel(), 1):.6f}"
                # f"%)"
            # )

            # print(
                # "  all-action violations: "
                # f"{all_violations}/"
                # f"{q_ub_debug_all.numel()} "
                # f"("
                # f"{100.0 * all_violations / max(q_ub_debug_all.numel(), 1):.6f}"
                # f"%)"
            # )

            # print(
                # "  NOTE: diagnostic only; "
                # "NO consistency loss."
            # )

            # # ======================================================
            # # UB spike diagnostic
            # # ======================================================

            # print(
                # "\nMAXIMUM Q_UB TARGET-NETWORK LOCATION"
            # )

            # print(
                # "  source state s       = "
                # f"[{source_state[0]:.8f}, "
                # f"{source_state[1]:.8f}]"
            # )

            # print(
                # f"  sampled action a     = "
                # f"{sampled_action}"
            # )

            # print(
                # f"  source tile          = "
                # f"{source_tile}"
            # )

            # print(
                # f"  source category      = "
                # f"{source_category}"
            # )

            # print(
                # "  center next s_bar'   = "
                # f"[{center_next_state[0]:.8f}, "
                # f"{center_next_state[1]:.8f}]"
            # )

            # print(
                # f"  reward               = "
                # f"{source_reward:.9f}"
            # )

            # print(
                # f"  done                 = "
                # f"{source_done:.0f}"
            # )

            # print(
                # f"  Lr(tile,a)           = "
                # f"{source_lr:.9f}"
            # )

            # print(
                # f"  Lq(tile,a)           = "
                # f"{source_lq:.9f}"
            # )

            # print(
                # f"  pruning_radius       = "
                # f"{source_radius:.9f}"
            # )

            # print(
                # f"  delta_r used         = "
                # f"{source_delta_r:.9f}"
            # )

            # print(
                # f"  Lr * radius          = "
                # f"{reconstructed_delta_r:.9f}"
            # )

            # print(
                # f"  delta_q used         = "
                # f"{source_delta_q:.9f}"
            # )

            # print(
                # f"  Lq * radius          = "
                # f"{reconstructed_delta_q:.9f}"
            # )

            # print(
                # "  Q_UB_tgt(s_bar', :)  = "
                # + np.array2string(
                    # ub_next_values,
                    # precision=6,
                # )
            # )

            # print(
                # "  Q_LB_tgt(s_bar', :)  = "
                # + np.array2string(
                    # lb_next_values,
                    # precision=6,
                # )
            # )

            # print(
                # f"  maximizing UB action = "
                # f"{maximizing_ub_action}"
            # )

            # print(
                # f"  max Q_UB_tgt         = "
                # f"{ub_next_values[maximizing_ub_action]:.9f}"
            # )

            # print("=" * 78, flush=True)

    # # ==============================================================
    # # Save FINAL networks exactly as trained.
    # #
    # # No early stopping.
    # # No best-checkpoint substitution.
    # # ==============================================================

    # env.close()

    # out_ub = Path(
        # args.out_ub
    # )

    # out_lb = Path(
        # args.out_lb
    # )

    # out_ub.parent.mkdir(
        # parents=True,
        # exist_ok=True,
    # )

    # out_lb.parent.mkdir(
        # parents=True,
        # exist_ok=True,
    # )

    # torch.save(
        # q_ub.state_dict(),
        # out_ub,
    # )

    # torch.save(
        # q_lb.state_dict(),
        # out_lb,
    # )

    # manifest = (
        # Path(args.manifest)
        # if args.manifest
        # else out_ub.parent
        # / (
            # f"q_bounds_"
            # f"{args.lq_source}_"
            # f"manifest.json"
        # )
    # )

    # manifest_data = {
        # "method": (
            # "learned_lipschitz_"
            # "bellman_q_bounds"
        # ),
        # "sigma": sigma,
        # "determinism": determinism,
        # "deterministic_sigma_scale": (
            # deterministic_sigma_scale
        # ),
        # "gamma": args.gamma,
        # "learning_rate": args.lr,
        # "batch_size": args.batch_size,
        # "target_tau": args.target_tau,
        # "iters": args.iters,
        # "seed": args.seed,
        # "lq_source": args.lq_source,
        # "bounds": str(
            # Path(args.bounds).resolve()
        # ),
        # "lipschitz": str(
            # Path(args.lipschitz).resolve()
        # ),
        # "initialization": {
            # "q_ub_equals_q_lb": True,
            # "q_ub_target_equals_q_ub": True,
            # "q_lb_target_equals_q_lb": True,
        # },
        # "consistency_loss": False,
        # "q_clipping": False,
        # "gradient_clipping": False,
        # "early_stopping": False,
        # "best_checkpoint_replacement": False,
        # "double_q": False,
        # "delta_r": (
            # "Lr(tile,a) * "
            # "pruning_radius(category,a)"
        # ),
        # "delta_q": (
            # "Lq(tile,a) * "
            # "pruning_radius(category,a)"
        # ),
        # "ub_target": (
            # "R + delta_r + "
            # "gamma*(1-d)*"
            # "(max_a Q_UB_target(s_bar,a) "
            # "+ delta_q)"
        # ),
        # "lb_target": (
            # "R - delta_r + "
            # "gamma*(1-d)*"
            # "(max_a Q_LB_target(s_bar,a) "
            # "- delta_q)"
        # ),
    # }

    # manifest.parent.mkdir(
        # parents=True,
        # exist_ok=True,
    # )

    # manifest.write_text(
        # json.dumps(
            # manifest_data,
            # indent=2,
        # ),
        # encoding="utf-8",
    # )

    # print()
    # print(
        # f"saved {out_ub.resolve()}"
    # )
    # print(
        # f"saved {out_lb.resolve()}"
    # )
    # print(
        # f"saved {manifest.resolve()}"
    # )


# if __name__ == "__main__":
    # main()
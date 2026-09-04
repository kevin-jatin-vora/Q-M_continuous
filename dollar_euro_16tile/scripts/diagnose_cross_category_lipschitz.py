"""Diagnostic: compare within-category Lipschitz constants vs cross-category ones.

For each action, the existing pipeline computes per-category / per-tile Lipschitz
constants from free (s, a, s') transitions. The concern under test is whether
pairing samples from *physically neighboring dynamics categories* (categories that
share an edge in the 4x4 tile layout) yields substantially larger Lipschitz
constants than the within-category constants used by the bounds machinery.

This script reuses the exact R1/R2 behavior-training, free-transition and
boundary filtering, 10% trimmed-mean pairwise-ratio methodology from
``train_reward_components.py``, and additionally computes cross-category constants
by pairing free transitions drawn from two neighboring categories under the same
action.

Output is written ONLY under ``outputs/temp_cross_category_lipschitz/``. No
existing file (and no ``outputs/experiments`` directory) is modified.

Combination rules copied from the existing within-category logic:
  Lf_sum = max(Lf1, Lf2)
  Lr_sum = Lr1 + Lr2
  Lq_th   = Lr_sum / (1 - gamma * Lf_sum)   (infinite-horizon Bellman bound)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dollar_euro_lipschitz.config import (  # noqa: E402
    add_environment_arguments,
    add_training_arguments,
    apply_resolved_determinism,
    apply_resolved_deterministic_sigma_scale,
    apply_resolved_sigma,
    apply_training_defaults,
    env_kwargs,
    format_training_args,
    load_config,
    resolve_determinism,
    resolve_deterministic_sigma_scale,
    resolve_sigma,
)
from dollar_euro_lipschitz.layout import (  # noqa: E402
    GRID_SIZE,
    N_TILES,
    layout_summary,
)

import train_reward_components as trc  # noqa: E402


def discover_neighbor_pairs(tile_map: np.ndarray) -> list:
    """Auto-discover physically neighboring category pairs in the 4x4 tile grid.

    Two categories are neighbors if any tile of category A shares a grid edge
    (4-connectivity) with a tile of category B. Returns sorted unordered canonical
    pairs with no same-category edges.
    """
    pairs = set()
    grid = GRID_SIZE
    for tid in range(int(N_TILES)):
        iy, ix = divmod(tid, grid)
        cat = int(tile_map[tid])
        if ix + 1 < grid:
            ncat = int(tile_map[tid + 1])
            if cat != ncat:
                pairs.add(tuple(sorted((cat, ncat))))
        if iy + 1 < grid:
            ncat = int(tile_map[tid + grid])
            if cat != ncat:
                pairs.add(tuple(sorted((cat, ncat))))
    return sorted(pairs)


def cross_category_constants(set_a, set_b, model, action, device, max_pairs, seed):
    """Cross-category 10% trimmed pairwise Lipschitz ratios for one behavior.

    ``set_a`` / ``set_b`` are ``(states, next_states, rewards)`` free-transition
    arrays for the two categories under ``action``. Pairs are formed with one
    sample from each category, mirroring ``component_constants`` (1e-12 distance
    filter, 10% trimmed mean, capped unique pair count).
    """
    states_a, next_a, rewards_a = set_a
    states_b, next_b, rewards_b = set_b
    n_a = int(states_a.shape[0])
    n_b = int(states_b.shape[0])
    empty = {
        "n_a": n_a,
        "n_b": n_b,
        "Lr": 0.0,
        "Lf": 0.0,
        "Lq": 0.0,
        "pairs_used": 0,
    }
    if n_a == 0 or n_b == 0:
        return empty
    states_a64 = states_a.astype(np.float64, copy=False)
    next_a64 = next_a.astype(np.float64, copy=False)
    rewards_a64 = rewards_a.astype(np.float64, copy=False)
    states_b64 = states_b.astype(np.float64, copy=False)
    next_b64 = next_b.astype(np.float64, copy=False)
    rewards_b64 = rewards_b.astype(np.float64, copy=False)
    with torch.inference_mode():
        qa = (
            model(torch.as_tensor(states_a, dtype=torch.float32, device=device))[:, action]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        qb = (
            model(torch.as_tensor(states_b, dtype=torch.float32, device=device))[:, action]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 41 * action)
    total = n_a * n_b
    if max_pairs is None or total <= int(max_pairs):
        idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
    else:
        idx_a = np.empty(int(max_pairs), dtype=np.int64)
        idx_b = np.empty(int(max_pairs), dtype=np.int64)
        used = set()
        k = 0
        while k < int(max_pairs):
            i = int(rng.integers(0, n_a))
            j = int(rng.integers(0, n_b))
            key = i * n_b + j
            if key in used:
                continue
            used.add(key)
            idx_a[k] = i
            idx_b[k] = j
            k += 1
    distances_src = np.linalg.norm(states_a64[idx_a] - states_b64[idx_b], axis=1)  # ||s1 - s2||
    distances_nxt = np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1)  # ||s1' - s2'||
    valid = distances_src > 1e-12
    distances_src = distances_src[valid]
    distances_nxt = distances_nxt[valid]
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    if idx_a.size == 0:
        return empty
    return {
        "n_a": n_a,
        "n_b": n_b,
        "Lr": trc.trimmed_mean(np.abs(rewards_a64[idx_a] - rewards_b64[idx_b]) / distances_nxt),
        "Lf": trc.trimmed_mean(
            np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1) / distances_src
        ),
        "Lq": trc.trimmed_mean(np.abs(qa[idx_a] - qb[idx_b]) / distances_src),
        "pairs_used": int(valid.sum()),
    }


def bellman_lq(lr_sum, lf_sum, gamma):
    denominator = 1.0 - gamma * lf_sum
    if denominator > 0:
        return lr_sum * lf_sum / denominator
    return None


def build_within_category(cat1, cat2, gamma):
    out = {}
    for action in range(4):
        per_action = {}
        for region in range(1, 6):
            first = cat1.get((region, action), {"n": 0, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0})
            second = cat2.get((region, action), {"n": 0, "Lr": 0.0, "Lf": 0.0, "Lq": 0.0})
            lf_sum = max(float(first["Lf"]), float(second["Lf"]))
            lr_sum = float(first["Lr"]) + float(second["Lr"])
            per_action[str(region)] = {
                "Lr1": float(first["Lr"]),
                "Lr2": float(second["Lr"]),
                "Lr_sum": lr_sum,
                "Lf1": float(first["Lf"]),
                "Lf2": float(second["Lf"]),
                "Lf_sum": lf_sum,
                "Lq_emp1": float(first["Lq"]),
                "Lq_emp2": float(second["Lq"]),
                "Lq_th": bellman_lq(lr_sum, lf_sum, gamma),
                "n_r1": int(first["n"]),
                "n_r2": int(second["n"]),
            }
        out[str(action)] = per_action
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default=str(ROOT / "configs" / "default.json"),
        help="Config JSON path (default: configs/default.json)",
    )
    parser.add_argument(
        "--profile", choices=["smoke", "full"], default="full",
        help="smoke uses 10k steps/behavior; full uses 150k (config component_steps overrides)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Only 1 run is supported (diagnostic is single-shot)")
    parser.add_argument("--seed", type=int, default=0)
    add_environment_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--min-cell-samples", type=int, default=2)
    parser.add_argument("--r2-seed-offset", type=int, default=1_000_000)
    parser.add_argument(
        "--include-boundary-transitions",
        action="store_true",
        help="Include boundary cancel/clip steps in delta and Lipschitz stats.",
    )
    parser.add_argument("--log-every", type=int, default=10_000)
    args = parser.parse_args()

    if args.runs not in (None, 1):
        raise SystemExit("error: --runs must be 1 for this single-shot diagnostic")

    args.config_at = Path(args.config)
    config = load_config(args.config_at)

    sigma = resolve_sigma(config, args.sigma, args.stochasticity_scale)
    apply_resolved_sigma(config, sigma)
    determinism = resolve_determinism(config, args.determinism)
    apply_resolved_determinism(config, determinism)
    det_sigma_scale = resolve_deterministic_sigma_scale(config, args.deterministic_sigma_scale)
    apply_resolved_deterministic_sigma_scale(config, det_sigma_scale)
    args.deterministic_sigma_scale = det_sigma_scale
    apply_training_defaults(args, config, role="component")
    environment = env_kwargs(config, sigma)
    gamma = float(args.gamma)

    if args.profile == "smoke":
        steps = 10_000
    else:
        steps = 150_000
    steps = int(config.get("component_steps", steps))
    args.steps = steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / "outputs" / "temp_cross_category_lipschitz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cross-category Lipschitz diagnostic ===")
    print(f"config:               {args.config_at}")
    print(f"effective sigma:      {sigma:.17g}")
    print(f"effective determinism:{determinism:.17g}")
    print(f"det sigma scale:      {det_sigma_scale:.17g}")
    print(f"effective gamma:      {gamma:.17g}")
    print(f"profile:              {args.profile}")
    print(f"steps/behavior:       {args.steps}")
    print(f"max_pairs:            {args.max_pairs}")
    print(f"seed:                 {args.seed}")
    print(f"device:               {device}")
    print(f"training:             {format_training_args(args)}")
    print(f"output:               {output_dir}")
    print(f"tile layout:          {layout_summary(determinism)['counts']}")

    tile_map = np.asarray(layout_summary(determinism)["tile_map_row_major"], dtype=np.int32)
    neighbor_pairs = discover_neighbor_pairs(tile_map)
    print("\n--- Physically neighboring category pairs (edge-adjacent 4x4 tiles) ---")
    for pair in neighbor_pairs:
        a, b = pair
        tiles_a = [t for t in range(N_TILES) if int(tile_map[t]) == a]
        tiles_b = [t for t in range(N_TILES) if int(tile_map[t]) == b]
        print(f"  cat {a} (tiles {tiles_a}) <-> cat {b} (tiles {tiles_b})")

    q1, records1, tile_records1, next_tile_records1, seconds1 = trc.train_component(
        args, environment, 0, args.seed, device
    )
    q2, records2, tile_records2, next_tile_records2, seconds2 = trc.train_component(
        args, environment, 1, args.seed + args.r2_seed_offset, device
    )
    print(f"\nR1 behavior train/collect: {seconds1:.1f}s; R2: {seconds2:.1f}s")

    cat1 = trc.component_constants(records1, q1.q, device, args.max_pairs, args.seed)
    cat2 = trc.component_constants(records2, q2.q, device, args.max_pairs, args.seed)

    within = build_within_category(cat1, cat2, gamma)

    print("\n=== Within-category constants (per action) ===")
    print(
        f"  {'a':>2} {'region':>6} {'Lr_sum':>10} {'Lf_sum':>10} {'Lq_th':>10} "
        f"{'Lf1':>10} {'Lf2':>10} {'n_r1':>7} {'n_r2':>7}"
    )
    for action in range(4):
        for region in range(1, 6):
            r = within[str(action)][str(region)]
            lq = "n/a" if r["Lq_th"] is None else f"{r['Lq_th']:.6g}"
            print(
                f"  {action:>2} {region:>6} {r['Lr_sum']:>10.6g} {r['Lf_sum']:>10.6g} "
                f"{lq:>10} {r['Lf1']:>10.6g} {r['Lf2']:>10.6g} "
                f"{r['n_r1']:>7} {r['n_r2']:>7}"
            )

    cross = {}
    print("\n=== Cross-category constants (per neighbor pair, per action) ===")
    header_a = (
        f"  {'a':>2} {'pair':>9} {'n_a1':>6} {'n_b1':>6} {'n_a2':>6} {'n_b2':>6} "
        f"{'Lr1':>9} {'Lr2':>9} {'Lr_sum':>9} {'Lf1':>9} {'Lf2':>9} {'Lf_sum':>9} "
        f"{'Lq_th':>10}"
    )
    print(header_a)
    for a, b in neighbor_pairs:
        for action in range(4):
            set_a1 = records1.get((a, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_b1 = records1.get((b, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_a2 = records2.get((a, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_b2 = records2.get((b, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            x1 = cross_category_constants(
                set_a1, set_b1, q1.q, action, device, args.max_pairs, args.seed
            )
            x2 = cross_category_constants(
                set_a2, set_b2, q2.q, action, device, args.max_pairs, args.seed + 7
            )
            lf_sum = max(x1["Lf"], x2["Lf"])
            lr_sum = x1["Lr"] + x2["Lr"]
            row = {
                "pair": [a, b],
                "Lr1": x1["Lr"],
                "Lr2": x2["Lr"],
                "Lr_sum": lr_sum,
                "Lf1": x1["Lf"],
                "Lf2": x2["Lf"],
                "Lf_sum": lf_sum,
                "Lq_emp1": x1["Lq"],
                "Lq_emp2": x2["Lq"],
                "Lq_th": bellman_lq(lr_sum, lf_sum, gamma),
                "n_a_r1": x1["n_a"],
                "n_b_r1": x1["n_b"],
                "n_a_r2": x2["n_a"],
                "n_b_r2": x2["n_b"],
                "pairs_used_r1": x1["pairs_used"],
                "pairs_used_r2": x2["pairs_used"],
            }
            cross.setdefault(f"{a}-{b}", {})[str(action)] = row
            lq = "n/a" if row["Lq_th"] is None else f"{row['Lq_th']:.6g}"
            print(
                f"  {action:>2} {a}-{b:>6} {x1['n_a']:>6} {x1['n_b']:>6} {x2['n_a']:>6} {x2['n_b']:>6} "
                f"{x1['Lr']:>9.6g} {x2['Lr']:>9.6g} {lr_sum:>9.6g} "
                f"{x1['Lf']:>9.6g} {x2['Lf']:>9.6g} {lf_sum:>9.6g} {lq:>10}"
            )

    print("\n=== Category + neighbor summary (max Lf_sum / max Lq_th) ===")
    all_within_lq = [
        within[str(a)][str(r)]["Lq_th"]
        for a in range(4)
        for r in range(1, 6)
        if within[str(a)][str(r)]["Lq_th"] is not None
    ]
    # collect within max across actions per region and cross max per pair
    within_max_lf = {}
    within_max_lq = {}
    for r in range(1, 6):
        vals = [within[str(a)][str(r)]["Lf_sum"] for a in range(4)]
        lqs = [within[str(a)][str(r)]["Lq_th"] for a in range(4) if within[str(a)][str(r)]["Lq_th"] is not None]
        within_max_lf[str(r)] = max(vals)
        within_max_lq[str(r)] = max(lqs) if lqs else None
    cross_max_lf = {}
    cross_max_lq = {}
    for pk, per_action in cross.items():
        vals = [per_action[str(a)]["Lf_sum"] for a in range(4)]
        lqs = [per_action[str(a)]["Lq_th"] for a in range(4) if per_action[str(a)]["Lq_th"] is not None]
        cross_max_lf[pk] = max(vals)
        cross_max_lq[pk] = max(lqs) if lqs else None
    print(f"  {'entity':>14} {'max_Lf_sum':>12} {'max_Lq_th':>12}")
    for r in range(1, 6):
        lq = "n/a" if within_max_lq[str(r)] is None else f"{within_max_lq[str(r)]:.6g}"
        print(f"  {'within cat ' + str(r):>14} {within_max_lf[str(r)]:>12.6g} {lq:>12}")
    for pk in sorted(cross):
        lq = "n/a" if cross_max_lq[pk] is None else f"{cross_max_lq[pk]:.6g}"
        print(f"  {'cross ' + pk:>14} {cross_max_lf[pk]:>12.6g} {lq:>12}")

    print("\n=== Maximum Lf report ===")
    all_lf_entries = []
    for r in range(1, 6):
        all_lf_entries.append((within_max_lf[str(r)], f"within-cat {r}"))
    for pk in sorted(cross):
        all_lf_entries.append((cross_max_lf[pk], f"cross {pk}"))
    all_lf_entries.sort(reverse=True, key=lambda t: t[0])
    print(f"  {'rank':>4} {'Lf_sum':>10} {'source':>16} {'gamma*Lf<1?':>10}")
    for rank, (lf, src) in enumerate(all_lf_entries, start=1):
        ok = (gamma * lf) < 1.0
        print(
            f"  {rank:>4} {lf:>10.6g} {src:>16} {('yes' if ok else 'NO'):>10}"
        )
    any_invalid = not all((gamma * lf) < 1.0 for lf, _ in all_lf_entries)
    worst_invalid = next((f"{lf:.6g} ({src})" for lf, src in all_lf_entries if (gamma * lf) >= 1.0), None)
    if any_invalid:
        print(
            f"\n  WARNING: at least one entity has gamma*Lf_sum >= 1, so its Bellman "
            f"Lq bound is invalid (diverging). Worst: {worst_invalid}"
        )

    print("\n=== Saving raw transitions + cross-category Lipschitz ===")
    npz_out = output_dir / "raw_transitions.npz"
    json_out = output_dir / "cross_category_lipschitz.json"
    npz_arrays = {}
    for behavior, records in ((1, records1), (2, records2)):
        for region in range(1, 6):
            for action in range(4):
                s, ns, r = records.get(
                    (region, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)))
                )
                prefix = f"r{behavior}_c{region}_a{action}"
                npz_arrays[f"{prefix}_states"] = s
                npz_arrays[f"{prefix}_next"] = ns
                npz_arrays[f"{prefix}_rewards"] = r
    npz_arrays["meta"] = np.array(
        [json.dumps({
            "sigma": sigma, "gamma": gamma, "determinism": determinism,
            "deterministic_sigma_scale": det_sigma_scale,
            "steps": args.steps, "seed": args.seed, "profile": args.profile,
        })]
    )
    np.savez(npz_out, **npz_arrays)

    payload = {
        "source": "scripts/diagnose_cross_category_lipschitz.py",
        "metadata": {
            "config": str(args.config_at),
            "sigma": sigma,
            "gamma": gamma,
            "determinism": determinism,
            "deterministic_sigma_scale": det_sigma_scale,
            "deterministic_sigma": det_sigma_scale * sigma,
            "steps_per_behavior": args.steps,
            "seed": args.seed,
            "seed_r1": args.seed,
            "seed_r2": args.seed + args.r2_seed_offset,
            "profile": args.profile,
            "max_pairs": args.max_pairs,
            "device": str(device),
            "neighbor_pairs": [list(p) for p in neighbor_pairs],
            "tile_map": tile_map.tolist(),
            "method": {
                "Lr_Lf_Lq": (
                    "10% trimmed pairwise ratios on free (non-boundary) transitions; "
                    "Lr_sum = Lr1+Lr2, Lf_sum = max(Lf1,Lf2), "
                    "Lq_th = Lr_sum*Lf_sum/(1-gamma*Lf_sum). "
                    "NOTE: diagnostic-only legacy COMBINED (category-pooled) estimate; "
                    "the production pipeline uses v3 with Lr grouped by NEXT-state tile "
                    "and Lf grouped by source-category/action."
                ),
                "within": "single-category constants from R1/R2 behavior (pooled over the category's tiles).",
                "cross": "diagnostic constants from pairing samples across two physically neighboring categories under the same action (non-production).",
            },
            "lipschitz_method_version": 3,
        },
        "within_category": within,
        "cross_category": cross,
        "summary": {
            "within_max_Lf_by_category": {k: v for k, v in within_max_lf.items()},
            "within_max_Lq_by_category": {
                k: (None if v is None else v) for k, v in within_max_lq.items()
            },
            "cross_max_Lf_by_pair": {k: v for k, v in cross_max_lf.items()},
            "cross_max_Lq_by_pair": {
                k: (None if v is None else v) for k, v in cross_max_lq.items()
            },
            "max_Lf_rankings": [
                {"rank": rank, "Lf_sum": lf, "source": src, "gamma_Lf_lt_1": (gamma * lf) < 1.0}
                for rank, (lf, src) in enumerate(all_lf_entries, start=1)
            ],
            "any_gamma_Lf_invalid": any_invalid,
            "worst_invalid": worst_invalid,
        },
    }
    with json_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"saved: {npz_out.resolve()}")
    print(f"saved: {json_out.resolve()}")


if __name__ == "__main__":
    main()

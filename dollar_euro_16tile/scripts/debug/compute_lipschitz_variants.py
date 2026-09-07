"""Compute four LC variants from collected transitions."""

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_ROOT = ROOT / "outputs" / "temp_cross_category_lipschitz"

from dollar_euro_lipschitz.layout import discover_neighbor_pairs, discover_tile_neighbor_pairs, layout_summary, tile_ids_from_states
from dollar_euro_lipschitz.config import load_config, resolve_determinism
from dollar_euro_lipschitz.models import QNet
import train_reward_components as trc


def json_dump(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def global_trimmed(values, ratio=0.1):
    return trc.trimmed_mean(values, ratio)


def global_max(values, ratio=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(values)) if values.size else 0.0


def trimmed_max(values, ratio=0.1):
    """Trim the extreme ``ratio`` from each tail, then take the max remainder.

    Same trimming rule as ``train_reward_components.trimmed_mean`` (sort,
    drop int(n*ratio) from both ends); the "max" diagnostic variants use this
    so the single most extreme pairwise ratio cannot dominate the constant.
    """
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    trim = int(values.size * ratio)
    core = values[trim : values.size - trim] if values.size > 2 * trim else values
    return float(np.max(core)) if core.size else 0.0


def future_lq_finite(main_rows, cross_cat_rows, gamma):
    """Whether every future action can be bounded by the overlap-aware trainer.

    ``OverlapAwareConstants.Lf_eff(mask, b)`` takes the max over the
    ordinary per-category Lf plus the cross-category Lf for action ``b``;
    the worst case over any tile mask is exactly the global per-action max.
    Returns ``(finite, worst_by_action)`` where finite means
    ``gamma * worst(b) < 1`` for all actions.
    """
    worst = {}
    for r in main_rows:
        a = int(r["action"])
        worst[a] = max(worst.get(a, 0.0), float(r["Lf_sum"]))
    for r in cross_cat_rows:
        a = int(r["action"])
        worst[a] = max(worst.get(a, 0.0), float(r["Lf_sum"]))
    return all(gamma * worst.get(b, 0.0) < 1.0 - 1e-9 for b in range(4)), worst


def compute_lr_lf_custom(
    set_a, set_b, max_pairs, seed, estimator="trimmed_mean", distance_type="source",
    is_lr=False
):
    """Generic pairwise ratio computer for Lr or Lf."""
    states_a, next_a, rewards_a = set_a
    states_b, next_b, rewards_b = set_b
    n_a = int(states_a.shape[0])
    n_b = int(states_b.shape[0])
    empty = {"n_a": n_a, "n_b": n_b, "L": 0.0, "pairs_used": 0}
    if n_a == 0 or n_b == 0:
        return empty
        
    states_a64 = states_a.astype(np.float64, copy=False)
    next_a64 = next_a.astype(np.float64, copy=False)
    rewards_a64 = rewards_a.astype(np.float64, copy=False)
    states_b64 = states_b.astype(np.float64, copy=False)
    next_b64 = next_b.astype(np.float64, copy=False)
    rewards_b64 = rewards_b.astype(np.float64, copy=False)
    
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 41 * max(min(n_a, n_b), 0))
    total = n_a * n_b
    
    if max_pairs is None or total <= int(max_pairs):
        idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
    else:
        n_pairs = int(max_pairs)
        flat = rng.choice(total, size=n_pairs, replace=False)
        idx_a = flat // n_b
        idx_b = flat % n_b
        
    if distance_type == "source":
        distances = np.linalg.norm(states_a64[idx_a] - states_b64[idx_b], axis=1)
    else:
        distances = np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1)
        
    valid = distances > 1e-12
    distances = distances[valid]
    if distances.size == 0:
        return empty
        
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    
    if is_lr:
        numerators = np.abs(rewards_a64[idx_a] - rewards_b64[idx_b])
    else:
        numerators = np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1)
        
    ratios = numerators / distances
    
    if estimator == "max":
        val = global_max(ratios)
    elif estimator == "trimmed_max":
        val = trimmed_max(ratios)
    else:
        val = global_trimmed(ratios)
        
    return {
        "n_a": n_a,
        "n_b": n_b,
        "L": val,
        "pairs_used": int(valid.sum()),
    }


def compute_variant(
    variant_name,
    records1, records2,
    tile_records1, tile_records2,
    next_tile_records1, next_tile_records2,
    prov, tile_map, args, q1_q, q2_q, device
):
    print(f"Computing variant: {variant_name}")
    out_dir = OUT_ROOT / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gamma = prov["gamma"]
    sigma = prov["sigma"]
    det_scale = prov["deterministic_sigma_scale"]
    
    if variant_name == "local_trimmed":
        # Direct reuse of exact production trc logic
        tile_reward1 = trc.reward_lipschitz_per_tile(next_tile_records1, q1_q, device, args.max_pairs, args.seed)
        tile_reward2 = trc.reward_lipschitz_per_tile(next_tile_records2, q2_q, device, args.max_pairs, args.seed + 7)
        category_dynamics1 = trc.dynamics_lipschitz_per_category(records1, q1_q, device, args.max_pairs, args.seed)
        category_dynamics2 = trc.dynamics_lipschitz_per_category(records2, q2_q, device, args.max_pairs, args.seed + 7)
        empirical_q1 = trc.empirical_q_lipschitz_per_tile_action(tile_records1, q1_q, device, args.max_pairs, args.seed + 11)
        empirical_q2 = trc.empirical_q_lipschitz_per_tile_action(tile_records2, q2_q, device, args.max_pairs, args.seed + 13)
        
        constants = trc.build_lipschitz(tile_reward1, tile_reward2, category_dynamics1, category_dynamics2, empirical_q1, empirical_q2, args, sigma, tile_map)
        json_dump(constants, out_dir / "lipschitz_constants.json")
        
        neigh_tile_pairs = discover_tile_neighbor_pairs()
        cross_tile_rows = []
        for tile_i, tile_j in neigh_tile_pairs:
            set_a1 = next_tile_records1.get(tile_i, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_b1 = next_tile_records1.get(tile_j, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_a2 = next_tile_records2.get(tile_i, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_b2 = next_tile_records2.get(tile_j, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            x1 = trc.cross_tile_reward_constants(set_a1, set_b1, args.max_pairs, args.seed + 21)
            x2 = trc.cross_tile_reward_constants(set_a2, set_b2, args.max_pairs, args.seed + 23)
            cross_tile_rows.append({
                "tile_i": tile_i, "tile_j": tile_j, "region_i": int(tile_map[tile_i]), "region_j": int(tile_map[tile_j]),
                "n_i_r1": x1["n_a"], "n_j_r1": x1["n_b"], "n_i_r2": x2["n_a"], "n_j_r2": x2["n_b"],
                "pairs_r1": x1["pairs_used"], "pairs_r2": x2["pairs_used"],
                "Lr1": x1["Lr"], "Lr2": x2["Lr"], "Lr_sum": x1["Lr"] + x2["Lr"],
            })
        json_dump({
            "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
            "dynamics_sigma": sigma,
            "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
            "neighbor_tile_pairs": [list(p) for p in neigh_tile_pairs], "Lr_grouping": "next_state_tile_pair",
            "constants": cross_tile_rows
        }, out_dir / "cross_tile_reward_lipschitz.json")
        
        neighbor_cat_pairs = discover_neighbor_pairs(tile_map)
        cross_cat_rows = []
        for pair_info in neighbor_cat_pairs:
            ci, cj = pair_info["category_i"], pair_info["category_j"]
            for action in range(4):
                set_a1 = records1.get((ci, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_b1 = records1.get((cj, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_a2 = records2.get((ci, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_b2 = records2.get((cj, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                x1 = trc.cross_category_dynamics_constants(set_a1, set_b1, args.max_pairs, args.seed + 31)
                x2 = trc.cross_category_dynamics_constants(set_a2, set_b2, args.max_pairs, args.seed + 33)
                cross_cat_rows.append({
                    "category_i": ci, "category_j": cj, "action": action, "tile_edges": pair_info["tile_edges"],
                    "n_i_r1": x1["n_a"], "n_j_r1": x1["n_b"], "n_i_r2": x2["n_a"], "n_j_r2": x2["n_b"],
                    "pairs_r1": x1["pairs_used"], "pairs_r2": x2["pairs_used"],
                    "Lf1": x1["Lf"], "Lf2": x2["Lf"], "Lf_sum": max(x1["Lf"], x2["Lf"]),
                })
        json_dump({
            "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
            "dynamics_sigma": sigma,
            "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
            "neighbor_category_pairs": neighbor_cat_pairs, "Lf_grouping": "source_category_action_pair",
            "constants": cross_cat_rows
        }, out_dir / "cross_category_dynamics_lipschitz.json")
        
        finite_future, worst_by_action = future_lq_finite(constants["constants"], cross_cat_rows, gamma)
        if not finite_future:
            offenders = [b for b, v in worst_by_action.items() if gamma * v >= 1.0 - 1e-9]
            print(
                f"  Variant {variant_name} cannot train Q bounds: future action(s) "
                f"{offenders} non-contractive (worst gamma*Lf_eff: "
                f"{ {b: round(gamma * v, 6) for b, v in worst_by_action.items()} })."
            )
            json_dump({
                "error": "INVALID_NONCONTRACTIVE_FUTURE_ACTION",
                "gamma": gamma,
                "worst_lf_by_action": worst_by_action,
                "gamma_times_worst": {b: gamma * v for b, v in worst_by_action.items()},
            }, out_dir / "INVALID_NONCONTRACTIVE_FUTURE_ACTION.json")
            return False
        
        return True # local_trimmed always valid
        
    # For global/max variants
    estimator = "trimmed_max" if "max" in variant_name else "trimmed_mean"
    is_global = "global" in variant_name
    
    # 1. Main constants
    rows = []
    
    # If global, compute one global L_r for behavior 1 and 2
    if is_global:
        # pool all next_tile_records
        s1 = np.concatenate([next_tile_records1.get(t, (np.empty((0,2)),)*3)[0] for t in range(16)])
        ns1 = np.concatenate([next_tile_records1.get(t, (np.empty((0,2)),)*3)[1] for t in range(16)])
        r1 = np.concatenate([next_tile_records1.get(t, (np.empty((0,2)),)*3)[2] for t in range(16)])
        s2 = np.concatenate([next_tile_records2.get(t, (np.empty((0,2)),)*3)[0] for t in range(16)])
        ns2 = np.concatenate([next_tile_records2.get(t, (np.empty((0,2)),)*3)[1] for t in range(16)])
        r2 = np.concatenate([next_tile_records2.get(t, (np.empty((0,2)),)*3)[2] for t in range(16)])
        
        global_lr1 = compute_lr_lf_custom((s1, ns1, r1), (s1, ns1, r1), args.max_pairs, args.seed, estimator, "next", is_lr=True)["L"]
        global_lr2 = compute_lr_lf_custom((s2, ns2, r2), (s2, ns2, r2), args.max_pairs, args.seed + 7, estimator, "next", is_lr=True)["L"]
    
    for tile in range(16):
        region = int(tile_map[tile])
        if is_global:
            lr1 = global_lr1
            lr2 = global_lr2
        else:
            set1 = next_tile_records1.get(tile, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            set2 = next_tile_records2.get(tile, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            lr1 = compute_lr_lf_custom(set1, set1, args.max_pairs, args.seed, estimator, "next", is_lr=True)["L"]
            lr2 = compute_lr_lf_custom(set2, set2, args.max_pairs, args.seed + 7, estimator, "next", is_lr=True)["L"]
            
        lr_sum = lr1 + lr2
        
        for action in range(4):
            if is_global:
                s1 = np.concatenate([records1.get((c, action), (np.empty((0,2)),)*3)[0] for c in range(1,6)])
                ns1 = np.concatenate([records1.get((c, action), (np.empty((0,2)),)*3)[1] for c in range(1,6)])
                r1 = np.concatenate([records1.get((c, action), (np.empty((0,2)),)*3)[2] for c in range(1,6)])
                s2 = np.concatenate([records2.get((c, action), (np.empty((0,2)),)*3)[0] for c in range(1,6)])
                ns2 = np.concatenate([records2.get((c, action), (np.empty((0,2)),)*3)[1] for c in range(1,6)])
                r2 = np.concatenate([records2.get((c, action), (np.empty((0,2)),)*3)[2] for c in range(1,6)])
                set1 = (s1, ns1, r1)
                set2 = (s2, ns2, r2)
            else:
                set1 = records1.get((region, action), (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
                set2 = records2.get((region, action), (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
                
            lf1 = compute_lr_lf_custom(set1, set1, args.max_pairs, args.seed, estimator, "source", is_lr=False)["L"]
            lf2 = compute_lr_lf_custom(set2, set2, args.max_pairs, args.seed + 7, estimator, "source", is_lr=False)["L"]
            lf_sum = max(lf1, lf2)
            
            denom = 1.0 - gamma * lf_sum
            if denom > 0:
                bellman = lr_sum * lf_sum / denom
            else:
                bellman = None
                
            rows.append({
                "tile": tile, "region": region, "action": action,
                "lr_source": "next_tile", "Lr1": lr1, "Lr2": lr2, "Lr_sum": lr_sum, "Lr_action_dependence": False,
                "Lf1": lf1, "Lf2": lf2, "Lf_sum": lf_sum, "Lq_empirical_sum": 0.0,
                "LQ_bellman_bound": bellman,
            })
            
    is_valid = True
    bad_rows = []
    for r in rows:
        if r["LQ_bellman_bound"] is None or not math.isfinite(r["LQ_bellman_bound"]):
            is_valid = False
            bad_rows.append(r)
            
    if not is_valid:
        print(f"  Variant {variant_name} is INVALID! gamma*Lf >= 1.")
        json_dump({"error": "INVALID_THEORETICAL_BOUND", "bad_rows": bad_rows}, out_dir / "INVALID_THEORETICAL_BOUND.json")
        return False
        
    json_dump({
        "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
        "method_notes": "trimmed_max: drop extreme 10% from each tail, then max of remaining pairwise ratios",
        "Lr_grouping": "next_state_tile", "constants": rows
    }, out_dir / "lipschitz_constants.json")
    
    # Cross artifacts
    neigh_tile_pairs = discover_tile_neighbor_pairs()
    cross_tile_rows = []
    for tile_i, tile_j in neigh_tile_pairs:
        if is_global:
            # same as global
            ct_lr1 = global_lr1
            ct_lr2 = global_lr2
        else:
            set_a1 = next_tile_records1.get(tile_i, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            set_b1 = next_tile_records1.get(tile_j, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            set_a2 = next_tile_records2.get(tile_i, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            set_b2 = next_tile_records2.get(tile_j, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            ct_lr1 = compute_lr_lf_custom(set_a1, set_b1, args.max_pairs, args.seed + 21, estimator, "next", is_lr=True)["L"]
            ct_lr2 = compute_lr_lf_custom(set_a2, set_b2, args.max_pairs, args.seed + 23, estimator, "next", is_lr=True)["L"]
            
        cross_tile_rows.append({
            "tile_i": tile_i, "tile_j": tile_j, "region_i": int(tile_map[tile_i]), "region_j": int(tile_map[tile_j]),
            "Lr1": ct_lr1, "Lr2": ct_lr2, "Lr_sum": ct_lr1 + ct_lr2,
        })
    json_dump({
        "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
        "neighbor_tile_pairs": [list(p) for p in neigh_tile_pairs], "Lr_grouping": "next_state_tile_pair",
        "constants": cross_tile_rows
    }, out_dir / "cross_tile_reward_lipschitz.json")
    
    neighbor_cat_pairs = discover_neighbor_pairs(tile_map)
    cross_cat_rows = []
    for pair_info in neighbor_cat_pairs:
        ci, cj = pair_info["category_i"], pair_info["category_j"]
        for action in range(4):
            if is_global:
                # same as global lf1
                s1 = np.concatenate([records1.get((c, action), (np.empty((0,2)),)*3)[0] for c in range(1,6)])
                ns1 = np.concatenate([records1.get((c, action), (np.empty((0,2)),)*3)[1] for c in range(1,6)])
                r1 = np.concatenate([records1.get((c, action), (np.empty((0,2)),)*3)[2] for c in range(1,6)])
                s2 = np.concatenate([records2.get((c, action), (np.empty((0,2)),)*3)[0] for c in range(1,6)])
                ns2 = np.concatenate([records2.get((c, action), (np.empty((0,2)),)*3)[1] for c in range(1,6)])
                r2 = np.concatenate([records2.get((c, action), (np.empty((0,2)),)*3)[2] for c in range(1,6)])
                cc_lf1 = compute_lr_lf_custom((s1, ns1, r1), (s1, ns1, r1), args.max_pairs, args.seed + 31, estimator, "source", is_lr=False)["L"]
                cc_lf2 = compute_lr_lf_custom((s2, ns2, r2), (s2, ns2, r2), args.max_pairs, args.seed + 33, estimator, "source", is_lr=False)["L"]
            else:
                set_a1 = records1.get((ci, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_b1 = records1.get((cj, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_a2 = records2.get((ci, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_b2 = records2.get((cj, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                cc_lf1 = compute_lr_lf_custom(set_a1, set_b1, args.max_pairs, args.seed + 31, estimator, "source", is_lr=False)["L"]
                cc_lf2 = compute_lr_lf_custom(set_a2, set_b2, args.max_pairs, args.seed + 33, estimator, "source", is_lr=False)["L"]
                
            cross_cat_rows.append({
                "category_i": ci, "category_j": cj, "action": action, "tile_edges": pair_info["tile_edges"],
                "Lf1": cc_lf1, "Lf2": cc_lf2, "Lf_sum": max(cc_lf1, cc_lf2),
            })
    json_dump({
        "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
        "neighbor_category_pairs": neighbor_cat_pairs, "Lf_grouping": "source_category_action_pair",
        "constants": cross_cat_rows
    }, out_dir / "cross_category_dynamics_lipschitz.json")
    
    finite_future, worst_by_action = future_lq_finite(rows, cross_cat_rows, gamma)
    if not finite_future:
        offenders = [b for b, v in worst_by_action.items() if gamma * v >= 1.0 - 1e-9]
        print(
            f"  Variant {variant_name} cannot train Q bounds: future action(s) "
            f"{offenders} non-contractive (worst gamma*Lf_eff: "
            f"{ {b: round(gamma * v, 6) for b, v in worst_by_action.items()} })."
        )
        json_dump({
            "error": "INVALID_NONCONTRACTIVE_FUTURE_ACTION",
            "gamma": gamma,
            "worst_lf_by_action": worst_by_action,
            "gamma_times_worst": {b: gamma * v for b, v in worst_by_action.items()},
        }, out_dir / "INVALID_NONCONTRACTIVE_FUTURE_ACTION.json")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--run-dir", default=None, help="Per-run output directory (defaults to legacy temp_cross_category_lipschitz)")
    args, _ = parser.parse_known_args()
    
    global OUT_ROOT
    if args.run_dir:
        OUT_ROOT = Path(args.run_dir)
        shared_dir = OUT_ROOT / "shared"
    else:
        OUT_ROOT = ROOT / "outputs" / "temp_cross_category_lipschitz"
        shared_dir = OUT_ROOT / "shared"
    with (shared_dir / "provenance.json").open("r") as f:
        prov = json.load(f)
        
    args.gamma = prov["gamma"]
    args.min_cell_samples = prov["min_cell_samples"]
    args.deterministic_sigma_scale = prov["deterministic_sigma_scale"]
    
    config = load_config(args.config)
    determinism = resolve_determinism(config, prov["determinism"])
    layout = layout_summary(determinism)
    tile_map = np.asarray(layout["tile_map_row_major"], dtype=np.int32)
    
    npz = np.load(shared_dir / "raw_transitions.npz", allow_pickle=True)
    
    records1 = {}
    records2 = {}
    for region in range(1, 6):
        for action in range(4):
            records1[(region, action)] = (
                npz[f"r1_c{region}_a{action}_states"], npz[f"r1_c{region}_a{action}_next"], npz[f"r1_c{region}_a{action}_rewards"]
            )
            records2[(region, action)] = (
                npz[f"r2_c{region}_a{action}_states"], npz[f"r2_c{region}_a{action}_next"], npz[f"r2_c{region}_a{action}_rewards"]
            )
            
    tile_records1 = {}
    tile_records2 = {}
    next_tile_records1 = {}
    next_tile_records2 = {}
    for tile_id in range(16):
        next_tile_records1[tile_id] = (npz[f"r1_nxt_t{tile_id}_states"], npz[f"r1_nxt_t{tile_id}_next"], npz[f"r1_nxt_t{tile_id}_rewards"])
        next_tile_records2[tile_id] = (npz[f"r2_nxt_t{tile_id}_states"], npz[f"r2_nxt_t{tile_id}_next"], npz[f"r2_nxt_t{tile_id}_rewards"])
        for action in range(4):
            tile_records1[(tile_id, action)] = (npz[f"r1_src_t{tile_id}_a{action}_states"], npz[f"r1_src_t{tile_id}_a{action}_next"], npz[f"r1_src_t{tile_id}_a{action}_rewards"])
            tile_records2[(tile_id, action)] = (npz[f"r2_src_t{tile_id}_a{action}_states"], npz[f"r2_src_t{tile_id}_a{action}_next"], npz[f"r2_src_t{tile_id}_a{action}_rewards"])
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1_q = QNet().to(device)
    q2_q = QNet().to(device)
    q1_q.load_state_dict(torch.load(shared_dir / "q1_dqn_r1.pth", map_location=device))
    q2_q.load_state_dict(torch.load(shared_dir / "q2_dqn_r2.pth", map_location=device))
    
    variants = ["local_trimmed", "global_trimmed", "local_max", "global_max"]
    results = {}
    
    for v in variants:
        valid = compute_variant(
            v,
            records1, records2,
            tile_records1, tile_records2,
            next_tile_records1, next_tile_records2,
            prov, tile_map, args, q1_q, q2_q, device
        )
        results[v] = valid
        
    json_dump(results, shared_dir / "compute_results.json")

if __name__ == "__main__":
    main()

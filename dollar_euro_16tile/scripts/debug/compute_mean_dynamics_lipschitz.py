"""Build debug-only Lipschitz artifacts for the estimated deterministic mean map.

The ordinary dynamics map is f_hat(s,a) = s + mu_hat[category(s),a], so its
within-category Lipschitz constant is exactly one.  Cross-category values are
finite-resolution sensitivities computed from source-state pairs after applying
``--min-pair-distance``.  Observed noisy successors never enter either Lf.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.bounds import TRANSITION_STAT_DECIMALS, quantize_transition_stat
from dollar_euro_lipschitz.config import load_config, resolve_determinism
from dollar_euro_lipschitz.layout import discover_neighbor_pairs, layout_summary


def json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_records(npz):
    records = {}
    for category in range(1, 6):
        for action in range(4):
            by_behavior = []
            for behavior in (1, 2):
                prefix = f"r{behavior}_c{category}_a{action}"
                by_behavior.append((npz[f"{prefix}_states"], npz[f"{prefix}_next"]))
            records[(category, action)] = by_behavior
    return records


def displacement_summary(states, next_states):
    n = int(states.shape[0])
    if n == 0:
        return {"n": 0, "mean": [0.0, 0.0], "std": [0.0, 0.0], "covariance": [[0.0, 0.0], [0.0, 0.0]]}
    deltas = next_states.astype(np.float64) - states.astype(np.float64)
    mean = quantize_transition_stat(deltas.mean(axis=0))
    std = quantize_transition_stat(deltas.std(axis=0, ddof=1) if n > 1 else np.zeros(2))
    covariance = np.cov(deltas, rowvar=False, ddof=1) if n > 1 else np.zeros((2, 2))
    covariance = quantize_transition_stat(covariance)
    return {"n": n, "mean": mean.tolist(), "std": std.tolist(), "covariance": covariance.tolist()}


def fit_mean_models(records, represented):
    models = {}
    means = {}
    for category in range(1, 6):
        models[str(category)] = {}
        for action in range(4):
            (s1, n1), (s2, n2) = records[(category, action)]
            r1 = displacement_summary(s1, n1)
            r2 = displacement_summary(s2, n2)
            states = np.concatenate((s1, s2), axis=0)
            next_states = np.concatenate((n1, n2), axis=0)
            pooled = displacement_summary(states, next_states)
            means[(category, action)] = np.asarray(pooled["mean"], dtype=np.float64)
            models[str(category)][str(action)] = {
                "represented_in_layout": category in represented,
                "R1": r1,
                "R2": r2,
                "pooled": pooled,
            }
    return models, means


def sampled_cross_pairs(states_i, states_j, max_pairs, min_distance, seed, mu_i, mu_j):
    ni, nj = int(states_i.shape[0]), int(states_j.shape[0])
    empty = {
        "n_i": ni, "n_j": nj, "pairs_considered": 0,
        "pairs_rejected_below_gap": 0, "pairs_used": 0,
        "Lf": 0.0, "witness": None,
    }
    if ni == 0 or nj == 0:
        return empty
    total = ni * nj
    rng = np.random.default_rng(seed)
    if max_pairs is None or total <= int(max_pairs):
        flat = np.arange(total, dtype=np.int64)
    else:
        flat = rng.choice(total, size=int(max_pairs), replace=False)
    ii, jj = flat // nj, flat % nj
    si = states_i[ii].astype(np.float64, copy=False)
    sj = states_j[jj].astype(np.float64, copy=False)
    source_distance = np.linalg.norm(si - sj, axis=1)
    valid = source_distance >= float(min_distance)
    used = int(valid.sum())
    result = {
        **empty,
        "pairs_considered": int(flat.size),
        "pairs_rejected_below_gap": int(flat.size - used),
        "pairs_used": used,
    }
    if used == 0:
        return result
    ii, jj = ii[valid], jj[valid]
    si, sj, source_distance = si[valid], sj[valid], source_distance[valid]
    mapped_i = si + mu_i
    mapped_j = sj + mu_j
    mapped_distance = np.linalg.norm(mapped_i - mapped_j, axis=1)
    ratios = mapped_distance / source_distance
    pos = int(np.argmax(ratios))
    result["Lf"] = float(ratios[pos])
    result["witness"] = {
        "index_i": int(ii[pos]), "index_j": int(jj[pos]),
        "source_state_i": si[pos].tolist(), "source_state_j": sj[pos].tolist(),
        "mean_displacement_i": mu_i.tolist(), "mean_displacement_j": mu_j.tolist(),
        "mapped_state_i": mapped_i[pos].tolist(), "mapped_state_j": mapped_j[pos].tolist(),
        "source_distance": float(source_distance[pos]),
        "mapped_distance": float(mapped_distance[pos]),
        "pair_ratio": float(ratios[pos]),
    }
    return result


def verify_within_pairs(states, max_pairs, min_distance, seed, mean):
    """Numerically verify the analytic value using sampled source pairs."""
    n = int(states.shape[0])
    if n < 2:
        return {"n": n, "pairs_considered": 0, "pairs_used": 0, "ratio_min": None, "ratio_mean": None, "ratio_max": None, "max_abs_error_from_one": None}
    total = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if max_pairs is None or total <= int(max_pairs):
        ii, jj = np.triu_indices(n, k=1)
    else:
        # Sampling ordered unequal indices is sufficient for this identity check.
        count = int(max_pairs)
        ii = rng.integers(0, n, size=count, dtype=np.int64)
        jj = rng.integers(0, n - 1, size=count, dtype=np.int64)
        jj = jj + (jj >= ii)
    left = states[ii].astype(np.float64, copy=False)
    right = states[jj].astype(np.float64, copy=False)
    source_distance = np.linalg.norm(left - right, axis=1)
    valid = source_distance >= float(min_distance)
    if not np.any(valid):
        return {"n": n, "pairs_considered": int(source_distance.size), "pairs_used": 0, "ratio_min": None, "ratio_mean": None, "ratio_max": None, "max_abs_error_from_one": None}
    left, right, source_distance = left[valid], right[valid], source_distance[valid]
    ratios = np.linalg.norm((left + mean) - (right + mean), axis=1) / source_distance
    return {
        "n": n, "pairs_considered": int(valid.size), "pairs_used": int(valid.sum()),
        "ratio_min": float(np.min(ratios)), "ratio_mean": float(np.mean(ratios)),
        "ratio_max": float(np.max(ratios)),
        "max_abs_error_from_one": float(np.max(np.abs(ratios - 1.0))),
    }


def ordinary_artifact(baseline, models, represented, gamma):
    output = dict(baseline)
    rows = []
    for row in baseline["constants"]:
        updated = dict(row)
        lf = 1.0 if int(row["region"]) in represented else 0.0
        updated.update({
            "Lf1": lf, "Lf2": lf, "Lf_sum": lf,
            "LQ_bellman_bound": (
                float(row["Lr_sum"]) * lf / (1.0 - gamma * lf) if lf else 0.0
            ),
            "Lf_estimator": "analytic_constant_translation",
        })
        rows.append(updated)
    output.update({
        "source": "scripts/debug/compute_mean_dynamics_lipschitz.py",
        "estimator": "deterministic_mean_map",
        "Lf_definition": "||f_hat(s1,a)-f_hat(s2,a)|| / ||s1-s2||",
        "Lf_grouping": "source_category_action",
        "Lf_noise_treatment": "Observed successor noise is excluded; f_hat(s,a)=s+mu_hat[c,a].",
        "transition_stat_decimals": TRANSITION_STAT_DECIMALS,
        "mean_dynamics_models": models,
        "constants": rows,
    })
    output["method_notes"] = dict(output.get("method_notes", {}))
    output["method_notes"]["Lf"] = (
        "Within each represented category/action, mu_hat is constant in s, so "
        "f_hat has Jacobian I and Lf=1 exactly. Lf1/Lf2 are compatibility aliases "
        "for the same pooled mean-map constant, not separate noisy estimates."
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--min-pair-distance", type=float, default=0.04)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.min_pair_distance <= 0:
        raise SystemExit("--min-pair-distance must be > 0")

    run_dir = Path(args.run_dir)
    shared = run_dir / "shared"
    baseline_dir = Path(args.baseline_dir)
    output_root = Path(args.output_root)
    provenance = json.loads((shared / "provenance.json").read_text(encoding="utf-8"))
    gamma = float(provenance["gamma"])
    config = load_config(args.config)
    determinism = resolve_determinism(config, provenance["determinism"])
    tile_map = np.asarray(layout_summary(determinism)["tile_map_row_major"], dtype=np.int32)
    represented = set(int(x) for x in tile_map.ravel())

    with np.load(shared / "raw_transitions.npz", allow_pickle=True) as npz:
        records = load_records(npz)
        models, means = fit_mean_models(records, represented)

        within_checks = []
        for category in range(1, 6):
            for action in range(4):
                pooled = np.concatenate((records[(category, action)][0][0], records[(category, action)][1][0]), axis=0)
                within_checks.append({
                    "category": category, "action": action,
                    "represented": category in represented,
                    "analytic_Lf": 1.0 if category in represented else 0.0,
                    "sample_pair_verification": verify_within_pairs(
                        pooled, args.max_pairs, args.min_pair_distance,
                        args.seed + 211 * category + 17 * action,
                        means[(category, action)],
                    ),
                })

        cross_rows = []
        neighbor_pairs = discover_neighbor_pairs(tile_map)
        for pair_number, pair_info in enumerate(neighbor_pairs):
            ci, cj = int(pair_info["category_i"]), int(pair_info["category_j"])
            for action in range(4):
                pooled_i = np.concatenate((records[(ci, action)][0][0], records[(ci, action)][1][0]), axis=0)
                pooled_j = np.concatenate((records[(cj, action)][0][0], records[(cj, action)][1][0]), axis=0)
                result = sampled_cross_pairs(
                    pooled_i, pooled_j, args.max_pairs, args.min_pair_distance,
                    args.seed + 1009 * pair_number + 37 * action,
                    means[(ci, action)], means[(cj, action)],
                )
                lf = float(result["Lf"])
                jump = float(np.linalg.norm(means[(ci, action)] - means[(cj, action)]))
                cross_rows.append({
                    "category_i": ci, "category_j": cj, "action": action,
                    "tile_edges": pair_info["tile_edges"],
                    "n_i_pooled": result["n_i"], "n_j_pooled": result["n_j"],
                    "pairs_considered": result["pairs_considered"],
                    "pairs_rejected_below_gap": result["pairs_rejected_below_gap"],
                    "pairs_used": result["pairs_used"],
                    "Lf_pooled": lf, "Lf1": lf, "Lf2": lf, "Lf_sum": lf,
                    "mean_displacement_jump_norm": jump,
                    "unrestricted_global_mean_map_lipschitz": "infinite" if jump > 1e-12 else 1.0,
                    "Lf_witness": result["witness"],
                })

    baseline = json.loads((baseline_dir / "lipschitz_constants.json").read_text(encoding="utf-8"))
    ordinary = ordinary_artifact(baseline, models, represented, gamma)
    cross = {
        "lipschitz_method_version": 3,
        "source": "scripts/debug/compute_mean_dynamics_lipschitz.py",
        "gamma": gamma, "sigma": provenance["sigma"],
        "deterministic_sigma_scale": provenance["deterministic_sigma_scale"],
        "estimator": "gap_restricted_cross_category_mean_map_max",
        "min_pair_distance": args.min_pair_distance,
        "max_pairs": args.max_pairs,
        "Lf_grouping": "source_category_action_pair",
        "interpretation": (
            "Finite-resolution cross-category sensitivity of f_hat, not a global "
            "Lipschitz constant. Noise in sampled successors is excluded."
        ),
        "neighbor_category_pairs": neighbor_pairs,
        "constants": cross_rows,
    }
    models_payload = {
        "source": "scripts/debug/compute_mean_dynamics_lipschitz.py",
        "definition": "mu_hat[c,a] = mean(s'-s | source category c, action a), pooled over R1 and R2",
        "transition_stat_decimals": TRANSITION_STAT_DECIMALS,
        "models": models,
    }
    within = {
        "definition": "f_hat(s,a)=s+mu_hat[c,a]",
        "analytic_Lf": 1.0,
        "reason": "The same constant displacement cancels when two mapped states are subtracted.",
        "min_pair_distance_for_numerical_verification": args.min_pair_distance,
        "checks": within_checks,
    }

    local_dir = output_root / "mean_dynamics_local"
    cross_dir = output_root / "mean_dynamics_cross_gap"
    for destination in (local_dir, cross_dir):
        destination.mkdir(parents=True, exist_ok=True)
        json_dump(ordinary, destination / "lipschitz_constants.json")
        json_dump(cross, destination / "cross_category_dynamics_lipschitz.json")
        shutil.copy2(baseline_dir / "cross_tile_reward_lipschitz.json", destination / "cross_tile_reward_lipschitz.json")
        json_dump(models_payload, destination / "mean_dynamics_models.json")
        json_dump(within, destination / "within_category_mean_dynamics.json")

    worst = {action: 1.0 for action in range(4)}
    for row in cross_rows:
        action = int(row["action"])
        worst[action] = max(worst[action], float(row["Lf_sum"]))
    cross_valid = all(gamma * value < 1.0 - 1e-9 for value in worst.values())
    results = {
        "mean_dynamics_local": True,
        "mean_dynamics_cross_gap": cross_valid,
        "gamma": gamma,
        "worst_cross_aware_Lf_by_action": worst,
        "gamma_times_worst_cross_aware_Lf": {a: gamma * v for a, v in worst.items()},
    }
    json_dump(results, output_root / "mean_dynamics_compute_results.json")
    if not cross_valid:
        offenders = [row for row in cross_rows if gamma * float(row["Lf_sum"]) >= 1.0]
        json_dump({"error": "INVALID_NONCONTRACTIVE_FUTURE_ACTION", "offenders": offenders, **results}, cross_dir / "INVALID_NONCONTRACTIVE_FUTURE_ACTION.json")
        print("mean_dynamics_cross_gap is non-contractive; its Q-bound training will be skipped.")
        for row in offenders:
            print(f"  categories={row['category_i']}-{row['category_j']} action={row['action']} Lf={row['Lf_sum']:.9g} gamma*Lf={gamma*row['Lf_sum']:.9g}")
    print(f"Saved mean-dynamics artifacts to {output_root.resolve()}")


if __name__ == "__main__":
    main()

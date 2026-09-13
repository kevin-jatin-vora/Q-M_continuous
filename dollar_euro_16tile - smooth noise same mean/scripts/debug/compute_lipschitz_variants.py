"""Compute four LC variants from collected transitions."""

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_ROOT = ROOT / "outputs" / "temp_cross_category_lipschitz"

from dollar_euro_lipschitz.layout import discover_neighbor_pairs, discover_tile_neighbor_pairs, layout_summary, tile_ids_from_states
from dollar_euro_lipschitz.config import load_config, resolve_determinism
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.env import noise_multiplier_lipschitz
import train_reward_components as trc


def json_dump(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def gap_dir_name(min_pair_distance):
    value = format(float(min_pair_distance), ".10g")
    return "gap_" + value.replace("-", "m").replace(".", "p")


def global_max(values, ratio=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(values)) if values.size else 0.0


def global_mean(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.mean(values)) if values.size else 0.0


def _bilinear_category_basis(states, tile_map):
    """Bilinear weights mapping five fitted center-category sigmas to states."""
    states = np.asarray(states, dtype=np.float64)
    categories = np.asarray(tile_map, dtype=np.int64).reshape(4, 4) - 1
    scaled_x = np.clip((states[:, 0] - 0.125) / 0.25, 0.0, 3.0)
    scaled_y = np.clip((states[:, 1] - 0.125) / 0.25, 0.0, 3.0)
    ix = np.minimum(np.floor(scaled_x).astype(np.int64), 2)
    iy = np.minimum(np.floor(scaled_y).astype(np.int64), 2)
    tx = scaled_x - ix
    ty = scaled_y - iy
    basis = np.zeros((states.shape[0], 5), dtype=np.float64)
    rows = np.arange(states.shape[0])
    for cats, weights in (
        (categories[iy, ix], (1.0 - tx) * (1.0 - ty)),
        (categories[iy, ix + 1], tx * (1.0 - ty)),
        (categories[iy + 1, ix], (1.0 - tx) * ty),
        (categories[iy + 1, ix + 1], tx * ty),
    ):
        np.add.at(basis, (rows, cats), weights)
    return basis


def fit_sample_transition_model(records1, records2, provenance, tile_map):
    """Fit common action means and the five center-category noise scales."""
    category_grid = np.asarray(tile_map, dtype=np.int64).reshape(4, 4)
    all_states = []
    all_residuals = []
    action_means = []
    action_rows = []
    for action in range(4):
        chunks = []
        for records in (records1, records2):
            for category in range(1, 6):
                states, next_states, _ = records[(category, action)]
                if states.shape[0]:
                    chunks.append((states.astype(np.float64), next_states.astype(np.float64)))
        if not chunks:
            continue
        states = np.concatenate([chunk[0] for chunk in chunks], axis=0)
        next_states = np.concatenate([chunk[1] for chunk in chunks], axis=0)
        deltas = next_states - states
        mean_displacement = deltas.mean(axis=0)
        residuals = deltas - mean_displacement
        all_states.append(states)
        all_residuals.append(residuals)
        action_means.append(mean_displacement)
        action_rows.append({
            "action": action,
            "n": int(states.shape[0]),
            "sample_mean_displacement": mean_displacement.tolist(),
        })
    states = np.concatenate(all_states, axis=0)
    residuals = np.concatenate(all_residuals, axis=0)
    energy = np.sum(residuals * residuals, axis=1)
    basis = _bilinear_category_basis(states, category_grid)
    active = np.flatnonzero(basis.sum(axis=0) > 0.0)
    observed_rms = float(np.sqrt(np.mean(energy) / 2.0))

    if float(provenance["sigma"]) == 0.0 or observed_rms < 1e-10:
        fitted = np.zeros(5, dtype=np.float64)
        fit_success = True
        fit_message = "zero-noise numerical-floor branch"
    else:
        x = basis[:, active]
        initial = np.full(active.size, max(observed_rms, 1e-12), dtype=np.float64)

        def objective(log_scales):
            scales = np.exp(log_scales)
            local = np.maximum(x @ scales, 1e-15)
            value = np.mean(2.0 * np.log(local) + energy / (2.0 * local * local))
            derivative = 2.0 / local - energy / (local ** 3)
            gradient = scales * (x.T @ derivative) / energy.size
            return float(value), gradient

        result = minimize(
            objective, np.log(initial), jac=True, method="L-BFGS-B",
            bounds=[(-30.0, 1.0)] * active.size,
            options={"maxiter": 200, "ftol": 1e-13},
        )
        fitted = np.zeros(5, dtype=np.float64)
        fitted[active] = np.exp(result.x)
        fit_success = bool(result.success)
        fit_message = str(result.message)

    fitted_grid = fitted[category_grid - 1]
    sample_k_sigma = noise_multiplier_lipschitz(fitted_grid)
    sample_kp = float(np.sqrt(1.0 + 2.0 * sample_k_sigma ** 2))
    fitted_local_sigma = basis @ fitted
    positive = fitted_local_sigma > 1e-15
    if np.any(positive):
        normalized = residuals[positive] / fitted_local_sigma[positive, None]
        normalized_mean = normalized.mean(axis=0).tolist()
        normalized_covariance = np.cov(normalized, rowvar=False).tolist()
    else:
        normalized_mean, normalized_covariance = [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]

    model = {
        "category_sigma": fitted,
        "category_grid": category_grid,
        "action_mean_displacement": np.asarray(action_means, dtype=np.float64),
    }
    payload = {
        "method": "data_fitted_common_mean_and_bilinear_isotropic_gaussian_noise",
        "boundary_affected_transitions_excluded": not bool(
            provenance.get("include_boundary_transitions", False)
        ),
        "mean_map_Lf": 1.0,
        "theoretical_sigma_lipschitz": float(provenance["sigma_lipschitz"]),
        "theoretical_wasserstein_kernel_lipschitz": float(
            provenance["wasserstein_kernel_lipschitz"]
        ),
        "fit_success": fit_success,
        "fit_message": fit_message,
        "sample_fitted_category_sigma": {
            str(category): float(fitted[category - 1]) for category in range(1, 6)
        },
        "sample_estimated_sigma_lipschitz": sample_k_sigma,
        "sample_estimated_wasserstein_kernel_lipschitz": sample_kp,
        "normalized_residual_mean": normalized_mean,
        "normalized_residual_covariance": normalized_covariance,
        "by_action": action_rows,
        "interpretation": (
            "The fitted model supplies the per-pair Wasserstein ratios used by all "
            "four diagnostic variants. The theoretical global value is verification only."
        ),
    }
    return model, payload


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
    set_a, set_b, max_pairs, seed, estimator="mean", distance_type="source",
    is_lr=False, min_pair_distance=0.04,
):
    """Compute an Lr/Lf ratio using either trimming or a distance gap.

    Reward Lr uses next-state distance; dynamics Lf uses source-state distance.
    ``max_pairs`` caps candidate pairs before the distance filter is applied.
    ``trimmed_mean`` uses the production 10% trim and only excludes numerical
    zero-distance pairs. ``mean`` and ``max`` use ``min_pair_distance``.
    """
    states_a, next_a, rewards_a = set_a
    states_b, next_b, rewards_b = set_b
    n_a = int(states_a.shape[0])
    n_b = int(states_b.shape[0])
    empty = {
        "n_a": n_a,
        "n_b": n_b,
        "L": 0.0,
        "pairs_considered": 0,
        "pairs_rejected_below_gap": 0,
        "pairs_used": 0,
        "witness": None,
    }
    if n_a == 0 or n_b == 0:
        return empty
        
    states_a64 = states_a.astype(np.float64, copy=False)
    next_a64 = next_a.astype(np.float64, copy=False)
    rewards_a64 = rewards_a.astype(np.float64, copy=False)
    states_b64 = states_b.astype(np.float64, copy=False)
    next_b64 = next_b.astype(np.float64, copy=False)
    rewards_b64 = rewards_b.astype(np.float64, copy=False)
    
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 41 * max(min(n_a, n_b), 0))
    same_sample_set = (
        states_a is states_b and next_a is next_b and rewards_a is rewards_b
    )
    total = n_a * (n_a - 1) // 2 if same_sample_set else n_a * n_b

    if same_sample_set:
        idx_a, idx_b = trc.unique_unordered_pairs(n_a, max_pairs, rng)
    elif max_pairs is None or total <= int(max_pairs):
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
        
    pairs_considered = int(distances.size)
    if estimator == "trimmed_mean":
        valid = distances > 1e-12
    else:
        # A deterministic run can have an exact zero pruning radius. In that
        # case, match production and reject numerical zero-distance pairs so
        # identical states never create a 0/0 ratio.
        valid = trc.pair_distance_mask(distances, min_pair_distance)
    pairs_rejected = int(pairs_considered - valid.sum())
    distances = distances[valid]
    if distances.size == 0:
        return {
            **empty,
            "pairs_considered": pairs_considered,
            "pairs_rejected_below_gap": pairs_rejected,
        }
        
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    
    if is_lr:
        numerators = np.abs(rewards_a64[idx_a] - rewards_b64[idx_b])
    else:
        numerators = np.linalg.norm(next_a64[idx_a] - next_b64[idx_b], axis=1)
        
    ratios = numerators / distances

    witness_pos = int(np.argmax(ratios))
    witness_left = int(idx_a[witness_pos])
    witness_right = int(idx_b[witness_pos])
    witness = {
        "index_a": witness_left,
        "index_b": witness_right,
        "source_state_a": states_a64[witness_left].tolist(),
        "source_state_b": states_b64[witness_right].tolist(),
        "next_state_a": next_a64[witness_left].tolist(),
        "next_state_b": next_b64[witness_right].tolist(),
        "denominator_distance": float(distances[witness_pos]),
        "numerator": float(numerators[witness_pos]),
        "pair_ratio": float(ratios[witness_pos]),
    }

    if estimator == "max":
        val = global_max(ratios)
    elif estimator == "trimmed_mean":
        val = trc.trimmed_mean(ratios, ratio=0.1)
    else:
        val = global_mean(ratios)
        
    return {
        "n_a": n_a,
        "n_b": n_b,
        "L": val,
        "pairs_considered": pairs_considered,
        "pairs_rejected_below_gap": pairs_rejected,
        "pairs_used": int(valid.sum()),
        "witness": witness,
    }


def compute_wasserstein_custom(
    set_a, set_b, action, model, max_pairs, seed, estimator, min_pair_distance,
):
    """Data-fitted Gaussian W2 ratio for source-state pairs under one action."""
    states_a = np.asarray(set_a[0], dtype=np.float64)
    states_b = np.asarray(set_b[0], dtype=np.float64)
    n_a, n_b = int(states_a.shape[0]), int(states_b.shape[0])
    empty = {
        "n_a": n_a, "n_b": n_b, "L": 0.0, "pairs_considered": 0,
        "pairs_rejected_below_gap": 0, "pairs_used": 0, "witness": None,
    }
    if n_a == 0 or n_b == 0:
        return empty
    same_sample_set = set_a[0] is set_b[0]
    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b + 41 * int(action))
    total = n_a * (n_a - 1) // 2 if same_sample_set else n_a * n_b
    if same_sample_set:
        idx_a, idx_b = trc.unique_unordered_pairs(n_a, max_pairs, rng)
    elif max_pairs is None or total <= int(max_pairs):
        idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
    else:
        flat = rng.choice(total, size=int(max_pairs), replace=False)
        idx_a, idx_b = flat // n_b, flat % n_b

    distances = np.linalg.norm(states_a[idx_a] - states_b[idx_b], axis=1)
    pairs_considered = int(distances.size)
    if estimator == "trimmed_mean":
        valid = distances > 1e-12
    else:
        valid = trc.pair_distance_mask(distances, min_pair_distance)
    pairs_rejected = int(pairs_considered - valid.sum())
    idx_a, idx_b, distances = idx_a[valid], idx_b[valid], distances[valid]
    if distances.size == 0:
        return {
            **empty, "pairs_considered": pairs_considered,
            "pairs_rejected_below_gap": pairs_rejected,
        }

    sigma_a = _bilinear_category_basis(states_a[idx_a], model["category_grid"]) @ model["category_sigma"]
    sigma_b = _bilinear_category_basis(states_b[idx_b], model["category_grid"]) @ model["category_sigma"]
    # The data-fitted action displacement is common to both states and cancels.
    mean_distance = distances
    wasserstein_distance = np.sqrt(mean_distance ** 2 + 2.0 * (sigma_a - sigma_b) ** 2)
    ratios = wasserstein_distance / distances
    witness_pos = int(np.argmax(ratios))
    left, right = int(idx_a[witness_pos]), int(idx_b[witness_pos])
    displacement = model["action_mean_displacement"][int(action)]
    witness = {
        "index_a": left, "index_b": right,
        "source_state_a": states_a[left].tolist(),
        "source_state_b": states_b[right].tolist(),
        "fitted_mean_next_a": (states_a[left] + displacement).tolist(),
        "fitted_mean_next_b": (states_b[right] + displacement).tolist(),
        "fitted_sigma_a": float(sigma_a[witness_pos]),
        "fitted_sigma_b": float(sigma_b[witness_pos]),
        "source_distance": float(distances[witness_pos]),
        "wasserstein_distance": float(wasserstein_distance[witness_pos]),
        "pair_ratio": float(ratios[witness_pos]),
    }
    if estimator == "max":
        value = global_max(ratios)
    elif estimator == "trimmed_mean":
        value = trc.trimmed_mean(ratios, ratio=0.1)
    else:
        value = global_mean(ratios)
    return {
        "n_a": n_a, "n_b": n_b, "L": value,
        "pairs_considered": pairs_considered,
        "pairs_rejected_below_gap": pairs_rejected,
        "pairs_used": int(distances.size), "witness": witness,
    }


def compute_wasserstein_reports(records, model, max_pairs, seed, estimator, min_pair_distance):
    report = {}
    for (category, action), sample_set in sorted(records.items()):
        result = compute_wasserstein_custom(
            sample_set, sample_set, action, model, max_pairs,
            seed + 17 * category + 31 * action, estimator, min_pair_distance,
        )
        report[(category, action)] = {
            "n": result["n_a"], "Lf": result["L"],
            "pairs_used": result["pairs_used"], "witness": result["witness"],
        }
    return report


def compute_empirical_q_custom(
    records, model, device, max_pairs, seed, estimator, min_pair_distance,
):
    """Per-source-tile/action empirical Q Lipschitz diagnostic."""
    report = {}
    model.eval()
    for (tile_id, action), (states, next_states, rewards) in sorted(records.items()):
        n_rows = int(states.shape[0])
        if n_rows:
            with torch.inference_mode():
                q_values = (
                    model(torch.as_tensor(states, dtype=torch.float32, device=device))[:, action]
                    .detach().cpu().numpy().astype(np.float64, copy=False)
                )
            q_records = (states, next_states, q_values)
            result = compute_lr_lf_custom(
                q_records, q_records, max_pairs,
                seed + 19 * n_rows + 37 * tile_id + 41 * action,
                estimator=estimator,
                distance_type="source",
                is_lr=True,
                min_pair_distance=min_pair_distance,
            )
        else:
            result = compute_lr_lf_custom(
                (states, next_states, rewards), (states, next_states, rewards),
                max_pairs, seed, estimator=estimator,
                min_pair_distance=min_pair_distance,
            )
        report[(tile_id, action)] = {
            "n": n_rows,
            "Lq": result["L"],
            **{k: result[k] for k in (
                "pairs_considered", "pairs_rejected_below_gap", "pairs_used"
            )},
            "witness": result["witness"],
        }
    return report


def compute_local_reports(
    records, next_tile_records, tile_records, model, device,
    max_pairs, seed, estimator, min_pair_distance,
):
    tile_reward = {}
    for tile_id, sample_set in sorted(next_tile_records.items()):
        result = compute_lr_lf_custom(
            sample_set, sample_set, max_pairs, seed + 31 * tile_id,
            estimator=estimator, distance_type="next", is_lr=True,
            min_pair_distance=min_pair_distance,
        )
        tile_reward[tile_id] = {
            "n": result["n_a"], "Lr": result["L"],
            **{k: result[k] for k in (
                "pairs_considered", "pairs_rejected_below_gap", "pairs_used"
            )},
            "witness": result["witness"],
        }

    category_dynamics = {}
    for (category, action), sample_set in sorted(records.items()):
        result = compute_lr_lf_custom(
            sample_set, sample_set, max_pairs, seed + 31 * action,
            estimator=estimator, distance_type="source", is_lr=False,
            min_pair_distance=min_pair_distance,
        )
        category_dynamics[(category, action)] = {
            "n": result["n_a"], "Lf": result["L"],
            **{k: result[k] for k in (
                "pairs_considered", "pairs_rejected_below_gap", "pairs_used"
            )},
            "witness": result["witness"],
        }

    empirical_q = compute_empirical_q_custom(
        tile_records, model, device, max_pairs, seed + 11,
        estimator, min_pair_distance,
    )
    return tile_reward, category_dynamics, empirical_q


def _print_lf_witness(*, variant, category, action, behavior, lf, gamma, witness):
    print(
        f"  INVALID Lf witness [{variant}]: category={category} action={action} "
        f"behavior={behavior} Lf={lf:.17g} gamma*Lf={gamma * lf:.17g} "
        f"(required gamma*Lf < 1)"
    )
    if witness is None:
        print("    no eligible pair witness was recorded")
        return
    print(f"    pair indices:      {witness['index_a']}, {witness['index_b']}")
    print(f"    source state A:    {witness['source_state_a']}")
    print(f"    source state B:    {witness['source_state_b']}")
    if "fitted_mean_next_a" in witness:
        print(f"    fitted mean next A:{witness['fitted_mean_next_a']}")
        print(f"    fitted mean next B:{witness['fitted_mean_next_b']}")
        print(f"    fitted sigma A/B:  {witness['fitted_sigma_a']:.17g}, {witness['fitted_sigma_b']:.17g}")
        print(f"    source distance:   {witness['source_distance']:.17g}")
        print(f"    W2 distance:       {witness['wasserstein_distance']:.17g}")
    else:
        print(f"    next state A:      {witness['next_state_a']}")
        print(f"    next state B:      {witness['next_state_b']}")
        print(f"    source distance:   {witness['denominator_distance']:.17g}")
        print(f"    next distance:     {witness['numerator']:.17g}")
    print(f"    pair Lf ratio:     {witness['pair_ratio']:.17g}")


def invalid_local_dynamics(category_dynamics1, category_dynamics2, tile_map, gamma, variant):
    """Print one witness per represented category/action with gamma*Lf >= 1."""
    invalid = []
    for category in sorted(set(int(value) for value in np.asarray(tile_map).ravel())):
        for action in range(4):
            first = category_dynamics1[(category, action)]
            second = category_dynamics2[(category, action)]
            behavior, selected = ("R1", first) if first["Lf"] >= second["Lf"] else ("R2", second)
            lf = float(selected["Lf"])
            if gamma * lf >= 1.0:
                row = {
                    "category": category,
                    "action": action,
                    "behavior": behavior,
                    "Lf": lf,
                    "gamma_times_Lf": gamma * lf,
                    "witness": selected.get("witness"),
                }
                invalid.append(row)
                _print_lf_witness(
                    variant=variant, category=category, action=action,
                    behavior=behavior, lf=lf, gamma=gamma,
                    witness=selected.get("witness"),
                )
    return invalid


def compute_variant(
    variant_name,
    records1, records2,
    tile_records1, tile_records2,
    next_tile_records1, next_tile_records2,
    prov, tile_map, args, q1_q, q2_q, device, wasserstein_model
):
    print(f"Computing variant: {variant_name}")
    out_dir = OUT_ROOT / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gamma = prov["gamma"]
    sigma = prov["sigma"]
    det_scale = prov["deterministic_sigma_scale"]
    theoretical_kernel_lf = float(prov["wasserstein_kernel_lipschitz"])
    
    estimator = (
        "trimmed_mean"
        if args.pair_method == "trimmed"
        else ("max" if "max" in variant_name else "mean")
    )

    if variant_name == "local_mean":
        tile_reward1, category_dynamics1, empirical_q1 = compute_local_reports(
            records1, next_tile_records1, tile_records1, q1_q, device,
            args.max_pairs, args.seed, estimator, args.min_pair_distance,
        )
        tile_reward2, category_dynamics2, empirical_q2 = compute_local_reports(
            records2, next_tile_records2, tile_records2, q2_q, device,
            args.max_pairs, args.seed + 7, estimator, args.min_pair_distance,
        )
        category_dynamics1 = compute_wasserstein_reports(
            records1, wasserstein_model, args.max_pairs, args.seed,
            estimator, args.min_pair_distance,
        )
        category_dynamics2 = compute_wasserstein_reports(
            records2, wasserstein_model, args.max_pairs, args.seed + 7,
            estimator, args.min_pair_distance,
        )

        constants = trc.build_lipschitz(
            tile_reward1, tile_reward2, category_dynamics1, category_dynamics2,
            empirical_q1, empirical_q2, args, sigma, tile_map,
        )
        for row in constants["constants"]:
            row["sample_wasserstein_Lf1"] = row.pop("raw_sampled_Lf1")
            row["sample_wasserstein_Lf2"] = row.pop("raw_sampled_Lf2")
            row["sample_wasserstein_aggregate"] = max(
                row["sample_wasserstein_Lf1"], row["sample_wasserstein_Lf2"],
            )
            row["theoretical_global_K_P_verification"] = theoretical_kernel_lf
        constants["source"] = "scripts/debug/compute_lipschitz_variants.py"
        constants["estimator"] = estimator
        constants["pair_method"] = args.pair_method
        constants["min_pair_distance"] = (
            None if args.pair_method == "trimmed" else args.min_pair_distance
        )
        constants["min_pair_distance_source"] = (
            None if args.pair_method == "trimmed"
            else "minimum represented pruning radius"
        )
        constants["operational_dynamics_note"] = (
            "Lf1/Lf2/Lf_sum are data-fitted Wasserstein estimates at this variant's "
            "local/mean granularity. The theoretical global K_P is verification only."
        )
        constants["Lf_definition"] = (
            "W2(N(s+mu_hat_a,sigma_hat(s)^2 I), "
            "N(t+mu_hat_a,sigma_hat(t)^2 I)) / ||s-t||"
        )
        constants["Lf_grouping"] = "source_category_action_data_fitted_wasserstein"
        if args.pair_method == "trimmed":
            constants["trim_ratio"] = 0.1
            constants["method_notes"] = {
                "pair_filter": "Exclude only denominator distances <= 1e-12.",
                "aggregation": "10% trimmed mean of eligible pairwise ratios.",
            }
        else:
            constants["method_notes"] = {
                "pair_filter": (
                    "Reject pairwise comparisons whose denominator distance is less "
                    "than min_pair_distance; Lr uses next-state distance while Lf and "
                    "empirical Lq use source-state distance. No percentile trimming."
                ),
                "aggregation": "Arithmetic mean over eligible pairwise ratios.",
            }
        json_dump(constants, out_dir / "lipschitz_constants.json")
        
        neigh_tile_pairs = discover_tile_neighbor_pairs()
        cross_tile_rows = []
        for tile_i, tile_j in neigh_tile_pairs:
            set_a1 = next_tile_records1.get(tile_i, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_b1 = next_tile_records1.get(tile_j, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_a2 = next_tile_records2.get(tile_i, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            set_b2 = next_tile_records2.get(tile_j, (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
            x1 = compute_lr_lf_custom(
                set_a1, set_b1, args.max_pairs, args.seed + 21, estimator,
                "next", True, args.min_pair_distance,
            )
            x2 = compute_lr_lf_custom(
                set_a2, set_b2, args.max_pairs, args.seed + 23, estimator,
                "next", True, args.min_pair_distance,
            )
            cross_tile_rows.append({
                "tile_i": tile_i, "tile_j": tile_j, "region_i": int(tile_map[tile_i]), "region_j": int(tile_map[tile_j]),
                "n_i_r1": x1["n_a"], "n_j_r1": x1["n_b"], "n_i_r2": x2["n_a"], "n_j_r2": x2["n_b"],
                "pairs_considered_r1": x1["pairs_considered"], "pairs_considered_r2": x2["pairs_considered"],
                "pairs_rejected_below_gap_r1": x1["pairs_rejected_below_gap"],
                "pairs_rejected_below_gap_r2": x2["pairs_rejected_below_gap"],
                "pairs_r1": x1["pairs_used"], "pairs_r2": x2["pairs_used"],
                "Lr1": x1["L"], "Lr2": x2["L"], "Lr_sum": x1["L"] + x2["L"],
            })
        json_dump({
            "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
            "dynamics_sigma": sigma,
            "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
            "estimator": estimator, "min_pair_distance": args.min_pair_distance,
            "min_pair_distance_source": "minimum represented pruning radius",
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
                x1 = compute_wasserstein_custom(
                    set_a1, set_b1, action, wasserstein_model, args.max_pairs,
                    args.seed + 31, estimator, args.min_pair_distance,
                )
                x2 = compute_wasserstein_custom(
                    set_a2, set_b2, action, wasserstein_model, args.max_pairs,
                    args.seed + 33, estimator, args.min_pair_distance,
                )
                lf_behavior, lf_result = ("R1", x1) if x1["L"] >= x2["L"] else ("R2", x2)
                cross_cat_rows.append({
                    "category_i": ci, "category_j": cj, "action": action, "tile_edges": pair_info["tile_edges"],
                    "n_i_r1": x1["n_a"], "n_j_r1": x1["n_b"], "n_i_r2": x2["n_a"], "n_j_r2": x2["n_b"],
                    "pairs_considered_r1": x1["pairs_considered"], "pairs_considered_r2": x2["pairs_considered"],
                    "pairs_rejected_below_gap_r1": x1["pairs_rejected_below_gap"],
                    "pairs_rejected_below_gap_r2": x2["pairs_rejected_below_gap"],
                    "pairs_r1": x1["pairs_used"], "pairs_r2": x2["pairs_used"],
                    "Lf1": x1["L"], "Lf2": x2["L"],
                    "Lf_sum": max(x1["L"], x2["L"]),
                    "mean_map_Lf": 1.0,
                    "theoretical_global_K_P_verification": theoretical_kernel_lf,
                    "Lf_behavior": lf_behavior, "Lf_witness": lf_result["witness"],
                })
        json_dump({
            "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
            "dynamics_sigma": sigma,
            "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
            "theoretical_global_wasserstein_kernel_lipschitz": theoretical_kernel_lf,
            "estimator": estimator, "min_pair_distance": args.min_pair_distance,
            "min_pair_distance_source": "minimum represented pruning radius",
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
            for row in cross_cat_rows:
                if gamma * float(row["Lf_sum"]) >= 1.0:
                    _print_lf_witness(
                        variant=variant_name,
                        category=f"{row['category_i']}-{row['category_j']}",
                        action=int(row["action"]),
                        behavior=row["Lf_behavior"],
                        lf=float(row["Lf_sum"]),
                        gamma=gamma,
                        witness=row.get("Lf_witness"),
                    )
            json_dump({
                "error": "INVALID_NONCONTRACTIVE_FUTURE_ACTION",
                "gamma": gamma,
                "worst_lf_by_action": worst_by_action,
                "gamma_times_worst": {b: gamma * v for b, v in worst_by_action.items()},
            }, out_dir / "INVALID_NONCONTRACTIVE_FUTURE_ACTION.json")
            return False
        
        return True
        
    # For global/max variants
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
        
        global_lr1 = compute_lr_lf_custom((s1, ns1, r1), (s1, ns1, r1), args.max_pairs, args.seed, estimator, "next", is_lr=True, min_pair_distance=args.min_pair_distance)["L"]
        global_lr2 = compute_lr_lf_custom((s2, ns2, r2), (s2, ns2, r2), args.max_pairs, args.seed + 7, estimator, "next", is_lr=True, min_pair_distance=args.min_pair_distance)["L"]
    
    for tile in range(16):
        region = int(tile_map[tile])
        if is_global:
            lr1 = global_lr1
            lr2 = global_lr2
        else:
            set1 = next_tile_records1.get(tile, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            set2 = next_tile_records2.get(tile, (np.empty((0,2)), np.empty((0,2)), np.empty((0,))))
            lr1 = compute_lr_lf_custom(set1, set1, args.max_pairs, args.seed, estimator, "next", is_lr=True, min_pair_distance=args.min_pair_distance)["L"]
            lr2 = compute_lr_lf_custom(set2, set2, args.max_pairs, args.seed + 7, estimator, "next", is_lr=True, min_pair_distance=args.min_pair_distance)["L"]
            
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
                
            lf1_result = compute_wasserstein_custom(
                set1, set1, action, wasserstein_model, args.max_pairs,
                args.seed, estimator, args.min_pair_distance,
            )
            lf2_result = compute_wasserstein_custom(
                set2, set2, action, wasserstein_model, args.max_pairs,
                args.seed + 7, estimator, args.min_pair_distance,
            )
            sampled_lf1 = lf1_result["L"]
            sampled_lf2 = lf2_result["L"]
            lf1 = sampled_lf1
            lf2 = sampled_lf2
            lf_sum = max(lf1, lf2)
            lf_behavior, lf_result = (
                ("R1", lf1_result) if sampled_lf1 >= sampled_lf2 else ("R2", lf2_result)
            )
            
            denom = 1.0 - gamma * lf_sum
            if denom > 0:
                bellman = lr_sum * lf_sum / denom
            else:
                bellman = None
                
            rows.append({
                "tile": tile, "region": region, "action": action,
                "lr_source": "next_tile", "Lr1": lr1, "Lr2": lr2, "Lr_sum": lr_sum, "Lr_action_dependence": False,
                "Lf1": lf1, "Lf2": lf2, "Lf_sum": lf_sum, "Lq_empirical_sum": 0.0,
                "mean_map_Lf": 1.0,
                "sample_wasserstein_Lf1": sampled_lf1,
                "sample_wasserstein_Lf2": sampled_lf2,
                "sample_wasserstein_aggregate": max(sampled_lf1, sampled_lf2),
                "theoretical_global_K_P_verification": theoretical_kernel_lf,
                "LQ_bellman_bound": bellman,
                "Lf_behavior": lf_behavior,
                "Lf_witness": lf_result["witness"],
            })
            
    is_valid = True
    bad_rows = []
    for r in rows:
        if r["LQ_bellman_bound"] is None or not math.isfinite(r["LQ_bellman_bound"]):
            is_valid = False
            bad_rows.append(r)
            
    if not is_valid:
        print(f"  Variant {variant_name} is INVALID! gamma*Lf >= 1.")
        seen = set()
        for row in bad_rows:
            key = int(row["action"]) if is_global else (int(row["region"]), int(row["action"]))
            if key in seen:
                continue
            seen.add(key)
            _print_lf_witness(
                variant=variant_name,
                category="global" if is_global else key[0],
                action=int(row["action"]),
                behavior=row["Lf_behavior"],
                lf=float(row["Lf_sum"]),
                gamma=gamma,
                witness=row.get("Lf_witness"),
            )
        json_dump({"error": "INVALID_THEORETICAL_BOUND", "bad_rows": bad_rows}, out_dir / "INVALID_THEORETICAL_BOUND.json")
        return False

    for row in rows:
        row.pop("Lf_behavior", None)
        row.pop("Lf_witness", None)
        
    json_dump({
        "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
        "theoretical_global_wasserstein_kernel_lipschitz": theoretical_kernel_lf,
        "Lf_definition": "data-fitted Gaussian W2 ratio",
        "estimator": estimator,
        "min_pair_distance": args.min_pair_distance,
        "min_pair_distance_source": "minimum represented pruning radius",
        "method_notes": (
            "Reject pairs with denominator distance below min_pair_distance, then "
            f"take the {estimator}; no percentile trimming."
        ),
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
            ct_lr1 = compute_lr_lf_custom(set_a1, set_b1, args.max_pairs, args.seed + 21, estimator, "next", is_lr=True, min_pair_distance=args.min_pair_distance)["L"]
            ct_lr2 = compute_lr_lf_custom(set_a2, set_b2, args.max_pairs, args.seed + 23, estimator, "next", is_lr=True, min_pair_distance=args.min_pair_distance)["L"]
            
        cross_tile_rows.append({
            "tile_i": tile_i, "tile_j": tile_j, "region_i": int(tile_map[tile_i]), "region_j": int(tile_map[tile_j]),
            "Lr1": ct_lr1, "Lr2": ct_lr2, "Lr_sum": ct_lr1 + ct_lr2,
        })
    json_dump({
        "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
        "estimator": estimator, "min_pair_distance": args.min_pair_distance,
        "min_pair_distance_source": "minimum represented pruning radius",
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
                cc_lf1_result = compute_wasserstein_custom(
                    (s1, ns1, r1), (s1, ns1, r1), action, wasserstein_model,
                    args.max_pairs, args.seed + 31, estimator, args.min_pair_distance,
                )
                cc_lf2_result = compute_wasserstein_custom(
                    (s2, ns2, r2), (s2, ns2, r2), action, wasserstein_model,
                    args.max_pairs, args.seed + 33, estimator, args.min_pair_distance,
                )
            else:
                set_a1 = records1.get((ci, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_b1 = records1.get((cj, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_a2 = records2.get((ci, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                set_b2 = records2.get((cj, action), (np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))))
                cc_lf1_result = compute_wasserstein_custom(
                    set_a1, set_b1, action, wasserstein_model, args.max_pairs,
                    args.seed + 31, estimator, args.min_pair_distance,
                )
                cc_lf2_result = compute_wasserstein_custom(
                    set_a2, set_b2, action, wasserstein_model, args.max_pairs,
                    args.seed + 33, estimator, args.min_pair_distance,
                )

            sampled_cc_lf1 = cc_lf1_result["L"]
            sampled_cc_lf2 = cc_lf2_result["L"]
            cc_lf1 = sampled_cc_lf1
            cc_lf2 = sampled_cc_lf2
            lf_behavior, lf_result = (
                ("R1", cc_lf1_result)
                if sampled_cc_lf1 >= sampled_cc_lf2 else ("R2", cc_lf2_result)
            )
                
            cross_cat_rows.append({
                "category_i": ci, "category_j": cj, "action": action, "tile_edges": pair_info["tile_edges"],
                "Lf1": cc_lf1, "Lf2": cc_lf2, "Lf_sum": max(cc_lf1, cc_lf2),
                "mean_map_Lf": 1.0,
                "sample_wasserstein_Lf1": sampled_cc_lf1,
                "sample_wasserstein_Lf2": sampled_cc_lf2,
                "sample_wasserstein_aggregate": max(sampled_cc_lf1, sampled_cc_lf2),
                "theoretical_global_K_P_verification": theoretical_kernel_lf,
                "Lf_behavior": lf_behavior, "Lf_witness": lf_result["witness"],
            })
    json_dump({
        "lipschitz_method_version": 3, "gamma": gamma, "sigma": sigma,
        "dynamics_sigma": sigma,
        "deterministic_sigma_scale": det_scale, "deterministic_sigma": det_scale * sigma,
        "theoretical_global_wasserstein_kernel_lipschitz": theoretical_kernel_lf,
        "Lf_definition": "data-fitted Gaussian W2 ratio",
        "estimator": estimator, "min_pair_distance": args.min_pair_distance,
        "min_pair_distance_source": "minimum represented pruning radius",
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
        for row in cross_cat_rows:
            if gamma * float(row["Lf_sum"]) >= 1.0:
                _print_lf_witness(
                    variant=variant_name,
                    category=f"{row['category_i']}-{row['category_j']}",
                    action=int(row["action"]),
                    behavior=row["Lf_behavior"],
                    lf=float(row["Lf_sum"]),
                    gamma=gamma,
                    witness=row.get("Lf_witness"),
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
    parser.add_argument(
        "--pair-method", choices=("gap", "trimmed"), default="gap",
        help="Pairwise estimator: distance-gap arithmetic mean or 10%% trimmed mean.",
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=("local_mean", "global_mean", "local_max", "global_max"),
        default=("local_max", "global_max", "local_mean", "global_mean"),
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Explicit directory in which variant subdirectories are written.",
    )
    parser.add_argument("--run-dir", default=None, help="Per-run output directory (defaults to legacy temp_cross_category_lipschitz)")
    args, _ = parser.parse_known_args()
    
    if any(
        value == "--min-pair-distance" or value.startswith("--min-pair-distance=")
        for value in sys.argv[1:]
    ):
        raise SystemExit(
            "--min-pair-distance was removed; the diagnostic now derives it from "
            "the minimum represented pruning radius"
        )
    if args.pair_method == "trimmed" and tuple(args.variants) != ("local_mean",):
        raise SystemExit("--pair-method trimmed currently supports --variants local_mean only")

    global OUT_ROOT
    if args.output_root:
        if not args.run_dir:
            raise SystemExit("--output-root requires --run-dir")
        run_dir = Path(args.run_dir)
        shared_dir = run_dir / "shared"
        OUT_ROOT = Path(args.output_root)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        shared_dir = run_dir / "shared"
        OUT_ROOT = None
    else:
        run_dir = ROOT / "outputs" / "temp_cross_category_lipschitz"
        shared_dir = run_dir / "shared"
        OUT_ROOT = None
    with (shared_dir / "transition_bounds.json").open("r", encoding="utf-8") as handle:
        bounds = json.load(handle)
    args.min_pair_distance = trc.minimum_represented_pruning_radius(bounds)
    if not args.output_root:
        OUT_ROOT = run_dir / gap_dir_name(args.min_pair_distance)
    print(
        "Automatic LC pair-distance threshold: "
        f"min represented pruning radius={args.min_pair_distance:.9g}"
    )
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

    wasserstein_model, verification = fit_sample_transition_model(
        records1, records2, prov, tile_map
    )
    json_dump(verification, shared_dir / "wasserstein_sample_verification.json")
    print(
        "Wasserstein verification: "
        f"theory K_P={verification['theoretical_wasserstein_kernel_lipschitz']:.9g} "
        f"sample K_P={verification['sample_estimated_wasserstein_kernel_lipschitz']:.9g} "
        f"fit_success={verification['fit_success']}"
    )
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1_q = QNet().to(device)
    q2_q = QNet().to(device)
    q1_q.load_state_dict(torch.load(shared_dir / "q1_dqn_r1.pth", map_location=device))
    q2_q.load_state_dict(torch.load(shared_dir / "q2_dqn_r2.pth", map_location=device))
    
    variants = list(args.variants)
    results = {}
    
    for v in variants:
        valid = compute_variant(
            v,
            records1, records2,
            tile_records1, tile_records2,
            next_tile_records1, next_tile_records2,
            prov, tile_map, args, q1_q, q2_q, device, wasserstein_model
        )
        results[v] = valid
        
    json_dump(results, OUT_ROOT / "compute_results.json")

if __name__ == "__main__":
    main()

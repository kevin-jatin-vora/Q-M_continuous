from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Tuple

import numpy as np

try:
    from scipy.stats import t as scipy_student_t
except Exception:
    scipy_student_t = None


# =============================================================================
# JSON loading
# =============================================================================

def load_summary_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Small helpers
# =============================================================================

def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_vec(x: Any, dim: int = 2) -> np.ndarray:
    if x is None:
        return np.zeros(dim, dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(dim, dtype=float)
    return arr.astype(float)


def _get_nested(d: Dict[str, Any], *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            raise KeyError(f"Non-dict encountered before key {k!r}")
        if k in cur:
            cur = cur[k]
        elif isinstance(k, int) and str(k) in cur:
            cur = cur[str(k)]
        elif isinstance(k, str) and k.isdigit() and int(k) in cur:
            cur = cur[int(k)]
        else:
            raise KeyError(f"Missing key {k!r}")
    return cur


def _student_t_multiplier(confidence: float, dof: int) -> float:
    if dof <= 0:
        return 0.0
    p = 0.5 + confidence / 2.0
    if scipy_student_t is not None:
        return float(scipy_student_t.ppf(p, dof))
    return float(NormalDist().inv_cdf(p))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# =============================================================================
# Combine q1-behavior and q2-behavior transition stats
# =============================================================================

def combine_two_behavior_stats(
    stat_q1: Dict[str, Any],
    stat_q2: Dict[str, Any],
    confidence: float,
    norm_ord: int = 2,
) -> Dict[str, Any]:
    """
    Combine saved per-(region, action) transition stats from q1 and q2 behaviors.

    Expected keys in each stat:
        n
        mean_delta
        sample_std_delta

    Produces:
        n_total
        mean_delta
        sample_std_delta
        student_t_multiplier
        student_t_conf_radius_mean_delta
        radius_scalar
    """
    n1 = int(_safe_float(stat_q1.get("n", 0), 0.0))
    n2 = int(_safe_float(stat_q2.get("n", 0), 0.0))

    mu1 = _to_vec(stat_q1.get("mean_delta"), dim=2)
    mu2 = _to_vec(stat_q2.get("mean_delta"), dim=max(2, mu1.size))
    dim = max(mu1.size, mu2.size, 2)

    if mu1.size != dim:
        mu1 = _to_vec(mu1, dim=dim)
    if mu2.size != dim:
        mu2 = _to_vec(mu2, dim=dim)

    s1 = _to_vec(stat_q1.get("sample_std_delta"), dim=dim)
    s2 = _to_vec(stat_q2.get("sample_std_delta"), dim=dim)

    if n1 < 2:
        s1 = np.zeros(dim, dtype=float)
    if n2 < 2:
        s2 = np.zeros(dim, dtype=float)

    n_total = n1 + n2

    if n_total == 0:
        mean_delta = np.zeros(dim, dtype=float)
        sample_std = np.zeros(dim, dtype=float)
        t_mult = 0.0
        radius_vec = np.zeros(dim, dtype=float)
    else:
        mean_delta = (n1 * mu1 + n2 * mu2) / n_total

        if n_total > 1:
            pooled_var = (
                (n1 - 1) * (s1 ** 2)
                + (n2 - 1) * (s2 ** 2)
                + n1 * ((mu1 - mean_delta) ** 2)
                + n2 * ((mu2 - mean_delta) ** 2)
            ) / (n_total - 1)

            pooled_var = np.maximum(pooled_var, 0.0)
            sample_std = np.sqrt(pooled_var)

            t_mult = _student_t_multiplier(confidence, n_total - 1)
            radius_vec = t_mult * sample_std / math.sqrt(n_total)
        else:
            sample_std = np.zeros(dim, dtype=float)
            t_mult = 0.0
            radius_vec = np.zeros(dim, dtype=float)

    radius_scalar = float(np.linalg.norm(radius_vec, ord=norm_ord))

    return {
        "n_total": int(n_total),
        "n_q1_behavior": int(n1),
        "n_q2_behavior": int(n2),
        "mean_delta": mean_delta,
        "sample_std_delta": sample_std,
        "student_t_multiplier": float(t_mult),
        "student_t_conf_radius_mean_delta": radius_vec,
        "radius_scalar": radius_scalar,
    }


# =============================================================================
# Build bounds table from saved JSON
# =============================================================================

def build_bounds_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrites the bounds-building logic so it uses saved JSON transition stats
    instead of a dataset.

    Output structure:
        bounds["by_region_action"][region][action] = {
            n_total,
            n_q1_behavior,
            n_q2_behavior,
            mean_delta,
            sample_std_delta,
            student_t_multiplier,
            student_t_conf_radius_mean_delta,
            radius_scalar,
        }

    Notes:
    - No Lipschitz constants are used here.
    - This is a saved transition-bounds object.
    - center_next_state depends on current state, so it is computed later
      via apply_bounds_to_state(...).
    """
    confidence = float(summary["config"]["confidence"])
    norm_ord = int(summary["config"].get("norm_ord", 2))

    q1_stats = summary["q1"]["state_change_stats"]
    q2_stats = summary["q2"]["state_change_stats"]

    regions = sorted(set(q1_stats.keys()) | set(q2_stats.keys()), key=lambda x: int(x))

    out: Dict[str, Any] = {
        "source_json": summary.get("paths", {}).get("json", None),
        "config": {
            "confidence": confidence,
            "norm_ord": norm_ord,
        },
        "by_region_action": {},
    }

    for region in regions:
        out["by_region_action"][str(region)] = {}
        actions = sorted(
            set(q1_stats.get(str(region), {}).keys()) | set(q2_stats.get(str(region), {}).keys()),
            key=lambda x: int(x),
        )

        for action in actions:
            stat_q1 = _get_nested(summary, "q1", "state_change_stats", region, action)
            stat_q2 = _get_nested(summary, "q2", "state_change_stats", region, action)

            combined = combine_two_behavior_stats(
                stat_q1=stat_q1,
                stat_q2=stat_q2,
                confidence=confidence,
                norm_ord=norm_ord,
            )

            out["by_region_action"][str(region)][str(action)] = {
                "n_total": int(combined["n_total"]),
                "n_q1_behavior": int(combined["n_q1_behavior"]),
                "n_q2_behavior": int(combined["n_q2_behavior"]),
                "mean_delta": combined["mean_delta"],
                "sample_std_delta": combined["sample_std_delta"],
                "student_t_multiplier": float(combined["student_t_multiplier"]),
                "student_t_conf_radius_mean_delta": combined["student_t_conf_radius_mean_delta"],
                "radius_scalar": float(combined["radius_scalar"]),
            }

    return out


# =============================================================================
# Save bounds
# =============================================================================

def save_bounds(bounds: Dict[str, Any], output_json: str | Path, output_pkl: str | Path) -> None:
    output_json = Path(output_json)
    output_pkl = Path(output_pkl)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(bounds), f, indent=2)

    with output_pkl.open("wb") as f:
        pickle.dump(bounds, f, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# Use saved bounds for a concrete state
# =============================================================================

def apply_bounds_to_state(
    bounds: Dict[str, Any],
    state: np.ndarray | list[float],
    region: int | str,
    action: int | str,
) -> Dict[str, Any]:
    """
    Use the saved bounds table for a concrete state.

    Returns:
        center_next_state = state + mean_delta
        radius_scalar
        radius_vec
        counts
    """
    entry = _get_nested(bounds, "by_region_action", region, action)

    state_arr = np.asarray(state, dtype=float).reshape(-1)
    mean_delta = _to_vec(entry["mean_delta"], dim=state_arr.size)
    radius_vec = _to_vec(entry["student_t_conf_radius_mean_delta"], dim=state_arr.size)
    radius_scalar = float(entry["radius_scalar"])

    center_next_state = state_arr + mean_delta

    return {
        "region": int(region),
        "action": int(action),
        "state": state_arr,
        "center_next_state": center_next_state,
        "mean_delta": mean_delta,
        "radius_vec": radius_vec,
        "radius_scalar": radius_scalar,
        "n_total": int(entry["n_total"]),
        "n_q1_behavior": int(entry["n_q1_behavior"]),
        "n_q2_behavior": int(entry["n_q2_behavior"]),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    summary_path = Path("two_reward_dqn_outputs//two_reward_dqn_summary.json")
    bounds_json_path = Path("single_q_bounds_from_json.json")
    bounds_pkl_path = Path("single_q_bounds_from_json.pkl")

    summary = load_summary_json(summary_path)
    bounds = build_bounds_from_summary(summary)
    save_bounds(bounds, bounds_json_path, bounds_pkl_path)

    print("Saved bounds:")
    print("  JSON :", bounds_json_path.resolve())
    print("  PKL  :", bounds_pkl_path.resolve())

    print("\nExample lookup:")
    state = np.array([0.5, 0.35], dtype=float)
    region = 4
    action = 0

    info = apply_bounds_to_state(bounds, state, region, action)
    print("region:", info["region"])
    print("action:", info["action"])
    print("n_total:", info["n_total"])
    print("mean_delta:", info["mean_delta"])
    print("radius_vec:", info["radius_vec"])
    print("radius_scalar:", info["radius_scalar"])
    print("center_next_state:", info["center_next_state"])


if __name__ == "__main__":
    main()
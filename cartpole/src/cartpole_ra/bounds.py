"""Transition stats, Lipschitz constants, and RA pruning margins for CartPole."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import t as student_t

from .config import ACTION_DIM, STATE_DIM

PRUNE_Q_MIN = -100.0
PRUNE_Q_MAX = 100.0
DEFAULT_CONFIDENCE_LEVEL = 0.95


def student_t_half_width(
    delta_std,
    n_samples: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> np.ndarray:
    std = np.asarray(delta_std, dtype=np.float64).reshape(-1)
    n = int(n_samples)
    if n < 2 or std.size == 0:
        return np.zeros_like(std, dtype=np.float64)
    t_mult = float(student_t.ppf(0.5 + float(confidence_level) / 2.0, n - 1))
    return t_mult * std * math.sqrt(1.0 + 1.0 / n)


def student_t_radius_l2(half_width) -> float:
    hw = np.asarray(half_width, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(hw))


def cell_delta_stats(states, next_states, confidence_level=DEFAULT_CONFIDENCE_LEVEL) -> dict:
    states = np.asarray(states, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    n = int(states.shape[0])
    if n == 0:
        zeros = np.zeros(STATE_DIM, dtype=np.float64)
        return {
            "n_total": 0,
            "delta_mean": zeros,
            "delta_std": zeros,
            "student_t_half_width": zeros,
            "pruning_radius": 0.0,
        }
    deltas = next_states - states
    mean = deltas.mean(axis=0)
    std = deltas.std(axis=0, ddof=1) if n >= 2 else np.zeros(STATE_DIM, dtype=np.float64)
    hw = student_t_half_width(std, n, confidence_level)
    return {
        "n_total": n,
        "delta_mean": mean,
        "delta_std": std,
        "student_t_half_width": hw,
        "pruning_radius": student_t_radius_l2(hw),
    }


def _trimmed_ratio(numerators: np.ndarray, denominators: np.ndarray, trim_frac: float = 0.1) -> float:
    dens = np.asarray(denominators, dtype=np.float64)
    nums = np.asarray(numerators, dtype=np.float64)
    ok = dens > 1e-8
    if not np.any(ok):
        return 0.0
    ratios = nums[ok] / dens[ok]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 0.0
    arr = np.sort(ratios)
    k = int(arr.size * trim_frac)
    core = arr[k : arr.size - k] if arr.size > 2 * k + 1 else arr
    return float(np.mean(core)) if core.size else float(np.mean(arr))


def _sample_pairs(n: int, max_pairs: int = 5000, seed: int = 0):
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs * 2)
    j = rng.integers(0, n, size=n_pairs * 2)
    mask = i != j
    i, j = i[mask][:n_pairs], j[mask][:n_pairs]
    return i, j


def estimate_cell_lipschitz(
    states,
    next_states,
    rewards,
    q_values,
    gamma: float,
    seed: int = 0,
) -> dict:
    """Lr, Lf, Lq_emp for one (region, action) cell; Lq_th = Lr/(1-γ Lf)."""
    states = np.asarray(states, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    q_values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    n = int(states.shape[0])
    if n < 2:
        return {
            "n": n,
            "Lr_sum": 0.0,
            "Lf_sum": 0.0,
            "Lq_empirical_sum": 0.0,
            "LQ_bellman_bound": 0.0,
        }
    i, j = _sample_pairs(n, seed=seed)
    ds = np.linalg.norm(states[i] - states[j], axis=1)
    lr = _trimmed_ratio(np.abs(rewards[i] - rewards[j]), ds)
    lf = _trimmed_ratio(np.linalg.norm(next_states[i] - next_states[j], axis=1), ds)
    lq_emp = _trimmed_ratio(np.abs(q_values[i] - q_values[j]), ds)
    denom = 1.0 - float(gamma) * lf
    lq_th = (lr / denom) if denom > 1e-12 else None
    return {
        "n": n,
        "Lr_sum": float(lr),
        "Lf_sum": float(lf),
        "Lq_empirical_sum": float(lq_emp),
        "LQ_bellman_bound": None if lq_th is None else float(lq_th),
    }


def action_margin(lr_value, lq_value, radius, gamma, legacy_margin: bool = False) -> float:
    if lq_value is None or not math.isfinite(float(lq_value)):
        raise ValueError("Lq undefined (need gamma*Lf < 1 for theoretical source)")
    radius = float(radius)
    lr_value = float(lr_value)
    lq_value = float(lq_value)
    gamma = float(gamma)
    if legacy_margin:
        return lr_value * radius + (lq_value * radius) / (1.0 - gamma)
    return (lr_value + gamma * lq_value) * radius / (1.0 - gamma)


def allowed_mask(q_values, margins, tol: float = 1e-5, clamp: bool = True):
    import torch

    upper = q_values + margins
    lower = q_values - margins
    if clamp:
        upper = upper.clamp(PRUNE_Q_MIN, PRUNE_Q_MAX)
        lower = lower.clamp(PRUNE_Q_MIN, PRUNE_Q_MAX)
    mask = upper >= (lower.max(dim=1, keepdim=True).values - float(tol))
    empty = ~mask.any(dim=1)
    if empty.any():
        mask = mask.clone()
        mask[empty] = True
    return mask


def _json_vec(x):
    return [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


class RegionActionBounds:
    def __init__(self, json_path, min_samples: int = 2, state_dim: int = STATE_DIM):
        self.min_samples = int(min_samples)
        self.state_dim = int(state_dim)
        self.cell_stats = {}
        self.region_stats = {}
        self.fit(json_path)

    def fit(self, json_path):
        with Path(json_path).open("r", encoding="utf-8") as handle:
            bounds = json.load(handle)
        for region_key, action_map in bounds["by_region_action"].items():
            region = int(region_key)
            entries = []
            for action_key, entry in action_map.items():
                action = int(action_key)
                mean = np.asarray(entry["delta_mean"], dtype=np.float32)
                std = np.asarray(entry["delta_std"], dtype=np.float32)
                hw = np.asarray(entry["student_t_half_width"], dtype=np.float32)
                row = {
                    "mean": mean,
                    "delta_std": std,
                    "student_t_half_width": hw,
                    "pruning_radius": float(entry["pruning_radius"]),
                    "n": int(entry.get("n_total", 0)),
                }
                self.cell_stats[(region, action)] = row
                if row["n"] > 0:
                    entries.append(row)
            if entries:
                counts = np.asarray([e["n"] for e in entries], dtype=np.float64)
                means = np.stack([e["mean"] for e in entries], axis=0)
                stds = np.stack([e["delta_std"] for e in entries], axis=0)
                hws = np.stack([e["student_t_half_width"] for e in entries], axis=0)
                self.region_stats[region] = {
                    "mean": (np.sum(means * counts[:, None], axis=0) / np.sum(counts)).astype(np.float32),
                    "delta_std": np.max(stds, axis=0).astype(np.float32),
                    "student_t_half_width": np.max(hws, axis=0).astype(np.float32),
                    "pruning_radius": float(max(e["pruning_radius"] for e in entries)),
                    "n": int(np.sum(counts)),
                }

    def get(self, region: int, action: int):
        key = (int(region), int(action))
        if key in self.cell_stats:
            row = self.cell_stats[key]
            if row["n"] == 0 or row["n"] >= self.min_samples:
                return row
        if int(region) in self.region_stats:
            return self.region_stats[int(region)]
        hw = np.ones(self.state_dim, dtype=np.float32) * 0.05
        return {
            "mean": np.zeros(self.state_dim, dtype=np.float32),
            "delta_std": np.ones(self.state_dim, dtype=np.float32) * 0.05,
            "student_t_half_width": hw,
            "pruning_radius": float(student_t_radius_l2(hw)),
            "n": 0,
        }


class LipschitzConstants:
    def __init__(self, json_path, source: str = "empirical"):
        if source not in {"empirical", "theoretical"}:
            raise ValueError("source must be empirical or theoretical")
        self.source = source
        with Path(json_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.gamma = float(data.get("gamma", 0.99))
        self.cell_stats = {}
        rows = data["constants"]
        for row in rows:
            self.cell_stats[(int(row["region"]), int(row["action"]))] = row
        if source == "theoretical":
            bad = [
                r for r in rows
                if r.get("LQ_bellman_bound") is None
                or not math.isfinite(float(r["LQ_bellman_bound"]))
            ]
            good = len(rows) - len(bad)
            if good == 0:
                raise SystemExit(
                    "theoretical Lq undefined for all (region,action); "
                    "gamma*Lf >= 1 or insufficient data. Use --lq-source empirical."
                )
            if bad:
                print(
                    f"warning: theoretical Lq undefined for {len(bad)}/{len(rows)} cells; "
                    "those cells fall back to empirical Lq in margins."
                )

    def get(self, region: int, action: int) -> Tuple[float, float]:
        row = self.cell_stats.get((int(region), int(action)))
        if row is None:
            return 1.0, 1.0
        lr = float(row["Lr_sum"])
        if self.source == "empirical":
            return lr, float(row.get("Lq_empirical_sum", 0.0) or 0.0)
        raw = row.get("LQ_bellman_bound")
        if raw is None or not math.isfinite(float(raw)):
            return lr, float(row.get("Lq_empirical_sum", 0.0) or 0.0)
        return lr, float(raw)


def build_margin_table(
    bounds_path,
    lipschitz_path,
    lq_source,
    gamma,
    legacy_margin,
    n_regions: int,
) -> np.ndarray:
    bounds = RegionActionBounds(bounds_path)
    constants = LipschitzConstants(lipschitz_path, source=lq_source)
    table = np.zeros((int(n_regions), ACTION_DIM), dtype=np.float32)
    for region in range(int(n_regions)):
        for action in range(ACTION_DIM):
            radius = float(bounds.get(region, action)["pruning_radius"])
            lr_value, lq_value = constants.get(region, action)
            try:
                table[region, action] = action_margin(
                    lr_value, lq_value, radius, gamma, legacy_margin
                )
            except ValueError:
                table[region, action] = action_margin(
                    lr_value, float(constants.cell_stats.get((region, action), {}).get("Lq_empirical_sum", 1.0)),
                    radius, gamma, legacy_margin,
                ) if lq_source == "empirical" else 0.0
    return table


def write_bounds_json(path, by_region_action: dict, metadata: Optional[dict] = None) -> None:
    payload = {
        "metadata": metadata or {},
        "by_region_action": {},
    }
    for region, action_map in by_region_action.items():
        payload["by_region_action"][str(int(region))] = {}
        for action, entry in action_map.items():
            payload["by_region_action"][str(int(region))][str(int(action))] = {
                "n_total": int(entry["n_total"]),
                "delta_mean": _json_vec(entry["delta_mean"]),
                "delta_std": _json_vec(entry["delta_std"]),
                "student_t_half_width": _json_vec(entry["student_t_half_width"]),
                "pruning_radius": float(entry["pruning_radius"]),
            }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_lipschitz_json(path, rows, gamma: float, metadata: Optional[dict] = None) -> None:
    payload = {
        "gamma": float(gamma),
        "metadata": metadata or {},
        "constants": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

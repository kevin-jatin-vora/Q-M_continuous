import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


REGION_CENTER = (0.5, 0.5)
PRUNE_Q_MIN = -100.0
PRUNE_Q_MAX = 100.0
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Module default for heatmap / pruning callers that omit determinism.
_DEFAULT_DETERMINISM = 0.25


def set_default_determinism(determinism: float) -> None:
    global _DEFAULT_DETERMINISM
    from .layout import clamp_determinism

    _DEFAULT_DETERMINISM = clamp_determinism(determinism)


def region_from_state(state, center=REGION_CENTER, determinism=None) -> int:
    """Return dynamics category id (1..5) for the tile containing ``state``."""
    from .layout import category_from_state

    det = _DEFAULT_DETERMINISM if determinism is None else float(determinism)
    return category_from_state(state, det)


def student_t_half_width(
    delta_std,
    n_samples: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> np.ndarray:
    """Per-axis Student-t interval half-width for a new δ around delta_mean.

    hw = t_{1-α/2, n-1} · σ̂ · √(1 + 1/n)

    Interval for the next free displacement (not a confidence interval for the mean).
    """
    std = np.asarray(delta_std, dtype=np.float64).reshape(-1)
    n = int(n_samples)
    if n < 2 or std.size == 0:
        return np.zeros_like(std, dtype=np.float64)
    t_mult = float(student_t.ppf(0.5 + float(confidence_level) / 2.0, n - 1))
    return t_mult * std * math.sqrt(1.0 + 1.0 / n)


def student_t_radius_l2(student_t_half_width) -> float:
    """Single L2 radius r = ‖hw‖₂ used by RA-DQN margins."""
    hw = np.asarray(student_t_half_width, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(hw))


def format_undefined_bellman_lq_error(bad_cells, gamma, *, path=None) -> str:
    """Clear failure when LQ_bellman_bound is undefined (gamma * Lf >= 1)."""
    lines = [
        "error: theoretical Bellman Lq is undefined because gamma * Lf >= 1 "
        f"(gamma={float(gamma):.17g}).",
        "  LQ_bellman = Lr / (1 - gamma*Lf) requires 1 - gamma*Lf > 0.",
        "  This usually means sigma is too high or the empirical Lf estimate is too large.",
        "  Offending (region, action) cells:",
    ]
    for cell in bad_cells:
        region = cell["region"]
        action = cell["action"]
        tile = cell.get("tile")
        loc = f"tile={tile} region={region} action={action}" if tile is not None else f"region={region} action={action}"
        lf = float(cell.get("Lf_sum", float("nan")))
        lr = cell.get("Lr_sum")
        lr_s = f"{float(lr):.6g}" if lr is not None else "n/a"
        lines.append(
            f"    {loc}: Lf_sum={lf:.17g}, "
            f"gamma*Lf={float(gamma) * lf:.17g}, Lr_sum={lr_s}, "
            f"denom={1.0 - float(gamma) * lf:.17g}"
        )
    if path is not None:
        lines.append(f"  source: {path}")
    lines.append("  Fix: lower sigma, collect more/cleaner transitions, or skip theoretical RA-DQN.")
    return "\n".join(lines)


def require_finite_bellman_lq(rows, gamma, *, path=None) -> None:
    """Raise SystemExit if any cell has null/non-finite LQ_bellman_bound."""
    bad = []
    for row in rows:
        raw = row.get("LQ_bellman_bound")
        if raw is None:
            bad.append(row)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            bad.append(row)
            continue
        if not math.isfinite(value):
            bad.append(row)
    if bad:
        raise SystemExit(format_undefined_bellman_lq_error(bad, gamma, path=path))


def _require_cell_fields(entry: dict, region: int, action: int) -> None:
    required = ("delta_mean", "delta_std", "student_t_half_width", "pruning_radius")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(
            f"transition_bounds cell ({region}, {action}) missing required fields {missing}. "
            "Regenerate bounds with scripts/train_reward_components.py."
        )


class RegionActionBounds:
    """Load per-(category, action) transition stats from transition_bounds.json."""

    def __init__(self, json_path, min_samples: int = 2):
        self.min_samples = int(min_samples)
        self.cell_stats = {}
        self.region_stats = {}
        self.fit(json_path)

    def fit(self, json_path):
        with Path(json_path).open("r", encoding="utf-8") as f:
            bounds = json.load(f)

        for region_key, action_map in bounds["by_region_action"].items():
            region = int(region_key)
            entries = []
            for action_key, entry in action_map.items():
                action = int(action_key)
                _require_cell_fields(entry, region, action)
                mean = np.asarray(entry["delta_mean"], dtype=np.float32)
                delta_std = np.asarray(entry["delta_std"], dtype=np.float32)
                student_t_hw = np.asarray(entry["student_t_half_width"], dtype=np.float32)
                pruning_radius = float(entry["pruning_radius"])
                n = int(entry.get("n_total", 0))
                row = {
                    "mean": mean,
                    "delta_std": delta_std,
                    "student_t_half_width": student_t_hw,
                    "pruning_radius": pruning_radius,
                    "n": n,
                }
                self.cell_stats[(region, action)] = row
                if n > 0:
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
        fallback_hw = np.ones(2, dtype=np.float32) * 0.05
        return {
            "mean": np.zeros(2, dtype=np.float32),
            "delta_std": np.ones(2, dtype=np.float32) * 0.05,
            "student_t_half_width": fallback_hw,
            "pruning_radius": float(student_t_radius_l2(fallback_hw)),
            "n": 0,
        }


def action_margin(lr_value, lq_value, radius, gamma, legacy_margin: bool = False) -> float:
    """Pruning margin using Student-t radius r = pruning_radius."""
    if lq_value is None:
        raise ValueError(
            "action_margin received Lq=None (undefined Bellman Lq: gamma*Lf >= 1). "
            "Load LipschitzConstants with source='theoretical' to get a clear SystemExit, "
            "or lower sigma."
        )
    radius = float(radius)
    lr_value = float(lr_value)
    lq_value = float(lq_value)
    gamma = float(gamma)
    if not math.isfinite(lq_value):
        raise ValueError(
            f"action_margin received non-finite Lq={lq_value}. "
            "Theoretical Lq must be finite (requires gamma*Lf < 1)."
        )
    if legacy_margin:
        return lr_value * radius + (lq_value * radius) / (1.0 - gamma)
    return (lr_value + gamma * lq_value) * radius / (1.0 - gamma)


def allowed_mask(q_values, margins, tol: float = 1e-5, clamp: bool = True):
    """Same interval test as QM_using json_LQ_theoritical.py / QM_using json.py."""
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


class LipschitzConstants:
    def __init__(self, json_path, source: str = "empirical"):
        if source not in {"empirical", "theoretical"}:
            raise ValueError("source must be 'empirical' or 'theoretical'")
        self.source = source
        self.json_path = Path(json_path)
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.gamma = float(data.get("gamma", 0.99))
        self.cell_stats = {}
        self.category_action_stats = {}
        if "constants" in data:
            rows = data["constants"]
        elif "combined" in data:
            rows = []
            for region, action_map in data["combined"].items():
                for action, entry in action_map.items():
                    row = {"region": int(region), "action": int(action)}
                    row.update(entry)
                    rows.append(row)
        else:
            raise ValueError(f"{json_path} has neither 'constants' nor 'combined'")
        for row in rows:
            if "tile" not in row:
                raise ValueError(
                    f"{json_path} uses legacy category-only Lipschitz rows; "
                    "regenerate with scripts/train_reward_components.py."
                )
            tile = int(row["tile"])
            action = int(row["action"])
            self.cell_stats[(tile, action)] = row

        if source == "theoretical":
            require_finite_bellman_lq(rows, self.gamma, path=str(self.json_path))

        from .layout import CATEGORY_IDS

        for region in CATEGORY_IDS:
            for action in range(4):
                region_rows = [
                    r for r in rows
                    if int(r.get("region", 0)) == region and int(r.get("action", -1)) == action
                ]
                if not region_rows:
                    continue
                bellman_vals = [r.get("LQ_bellman_bound") for r in region_rows]
                finite_bellman = [
                    float(v) for v in bellman_vals if v is not None and math.isfinite(float(v))
                ]
                if source == "theoretical":
                    bellman_agg = max(float(v) for v in bellman_vals if v is not None)
                elif finite_bellman:
                    bellman_agg = max(finite_bellman)
                else:
                    bellman_agg = None
                self.category_action_stats[(region, action)] = {
                    "Lr_sum": max(float(r["Lr_sum"]) for r in region_rows),
                    "Lq_empirical_sum": max(float(r["Lq_empirical_sum"]) for r in region_rows),
                    "LQ_bellman_bound": bellman_agg,
                    "Lf_sum": float(region_rows[0]["Lf_sum"]),
                }

    def get(self, tile: int, action: int):
        row = self.cell_stats.get((int(tile), int(action)))
        if row is None:
            raise KeyError(
                f"No Lipschitz row for tile={tile} action={action} in {self.json_path}"
            )
        Lr = float(row["Lr_sum"])
        if self.source == "empirical":
            Lq = float(row["Lq_empirical_sum"])
        else:
            raw = row.get("LQ_bellman_bound")
            if raw is None or not math.isfinite(float(raw)):
                require_finite_bellman_lq([row], self.gamma, path=str(self.json_path))
            Lq = float(raw)
        return Lr, Lq


def build_margin_table(
    bounds_path,
    lipschitz_path,
    lq_source,
    gamma,
    legacy_margin,
    *,
    determinism=None,
):
    """Per-tile margin table shaped (N_TILES, 4)."""
    from .layout import N_TILES, tile_category_map

    bounds = RegionActionBounds(bounds_path)
    constants = LipschitzConstants(lipschitz_path, source=lq_source)
    det = _DEFAULT_DETERMINISM if determinism is None else float(determinism)
    tile_map = tile_category_map(det)
    table = np.empty((N_TILES, 4), dtype=np.float32)
    for tile in range(N_TILES):
        region = int(tile_map[tile])
        for action in range(4):
            radius = float(bounds.get(region, action)["pruning_radius"])
            lr_value, lq_value = constants.get(tile, action)
            table[tile, action] = action_margin(
                lr_value, lq_value, radius, gamma, legacy_margin
            )
    return table, tile_map

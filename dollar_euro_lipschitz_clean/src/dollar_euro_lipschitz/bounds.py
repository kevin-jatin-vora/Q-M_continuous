import json
import math
from pathlib import Path

import numpy as np


REGION_CENTER = (0.5, 0.5)
PRUNE_Q_MIN = -100.0
PRUNE_Q_MAX = 100.0


def region_from_state(state, center=REGION_CENTER) -> int:
    x, y = float(state[0]), float(state[1])
    cx, cy = float(center[0]), float(center[1])
    if y >= cy:
        return 1 if x >= cx else 2
    return 4 if x >= cx else 3


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
        lf = float(cell.get("Lf_sum", float("nan")))
        lr = cell.get("Lr_sum")
        lr_s = f"{float(lr):.6g}" if lr is not None else "n/a"
        lines.append(
            f"    region={region} action={action}: Lf_sum={lf:.17g}, "
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


class RegionActionBounds:
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
                mean = np.asarray(entry.get("mean_delta", [0.0, 0.0]), dtype=np.float32)
                radius_vec = np.asarray(
                    entry.get("student_t_conf_radius_mean_delta", [0.05, 0.05]),
                    dtype=np.float32,
                )
                radius_scalar = float(entry.get("radius_scalar", np.linalg.norm(radius_vec)))
                n = int(entry.get("n_total", 0))
                row = {
                    "mean": mean,
                    "radius_vec": radius_vec,
                    "radius_scalar": radius_scalar,
                    "n": n,
                }
                self.cell_stats[(region, action)] = row
                if n > 0:
                    entries.append(row)

            if entries:
                counts = np.asarray([e["n"] for e in entries], dtype=np.float64)
                means = np.stack([e["mean"] for e in entries], axis=0)
                radius_vecs = np.stack([e["radius_vec"] for e in entries], axis=0)
                self.region_stats[region] = {
                    "mean": (np.sum(means * counts[:, None], axis=0) / np.sum(counts)).astype(np.float32),
                    "radius_vec": np.max(radius_vecs, axis=0).astype(np.float32),
                    "radius_scalar": float(max(e["radius_scalar"] for e in entries)),
                    "n": int(np.sum(counts)),
                }

    def get(self, region: int, action: int):
        key = (int(region), int(action))
        if key in self.cell_stats and self.cell_stats[key]["n"] >= self.min_samples:
            return self.cell_stats[key]
        if int(region) in self.region_stats:
            return self.region_stats[int(region)]
        return {
            "mean": np.zeros(2, dtype=np.float32),
            "radius_vec": np.ones(2, dtype=np.float32) * 0.05,
            "radius_scalar": float(np.linalg.norm(np.ones(2, dtype=np.float32) * 0.05)),
            "n": 0,
        }


def action_margin(lr_value, lq_value, radius, gamma, legacy_margin: bool = False) -> float:
    """Pruning margin. Default (and correct) form is Bellman:

        m = (Lr + gamma * Lq) * r / (1 - gamma)

    Use the same formula for theoretical Lq and empirical Lq; only Lq changes.
    ``legacy_margin=True`` keeps the older additive form Lr*r + Lq*r/(1-gamma).
    """
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
        self.region_stats = {}
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
            region = int(row["region"])
            action = int(row["action"])
            self.cell_stats[(region, action)] = row

        # Fail early for theoretical pruning / RA-DQN: null Bellman Lq is not usable.
        if source == "theoretical":
            require_finite_bellman_lq(rows, self.gamma, path=str(self.json_path))

        for region in [1, 2, 3, 4]:
            region_rows = [r for (z, _), r in self.cell_stats.items() if z == region]
            if not region_rows:
                continue
            bellman_vals = [r.get("LQ_bellman_bound") for r in region_rows]
            finite_bellman = [float(v) for v in bellman_vals if v is not None and math.isfinite(float(v))]
            # Region fallback only used if a cell is missing; for theoretical source
            # all cells were already required finite above.
            if source == "theoretical":
                bellman_agg = max(float(v) for v in bellman_vals)
            elif finite_bellman:
                bellman_agg = max(finite_bellman)
            else:
                bellman_agg = None
            self.region_stats[region] = {
                "Lr_sum": max(float(r["Lr_sum"]) for r in region_rows),
                "Lq_empirical_sum": max(float(r["Lq_empirical_sum"]) for r in region_rows),
                "LQ_bellman_bound": bellman_agg,
            }

    def get(self, region: int, action: int):
        row = self.cell_stats.get((int(region), int(action)))
        if row is None:
            row = self.region_stats.get(int(region), {})
        Lr = float(row.get("Lr_sum", 0.0))
        if self.source == "empirical":
            raw = row.get("Lq_empirical_sum", 0.0)
            if raw is None:
                raise SystemExit(
                    f"error: missing Lq_empirical_sum for region={region} action={action} "
                    f"in {self.json_path}"
                )
            Lq = float(raw)
        else:
            raw = row.get("LQ_bellman_bound")
            if raw is None or not math.isfinite(float(raw)):
                require_finite_bellman_lq(
                    [row if row else {"region": region, "action": action, "Lf_sum": float("nan")}],
                    self.gamma,
                    path=str(self.json_path),
                )
            Lq = float(raw)
        return Lr, Lq

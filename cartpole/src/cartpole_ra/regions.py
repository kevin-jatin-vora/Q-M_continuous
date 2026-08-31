"""Adaptive state-space partition for CartPole regions.

Regions are leaves of a binary tree on normalized state. Splits minimize the
within-leaf variance of displacement δ = s' − s. Lookup ρ(s) is exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import STATE_DIM, STATE_NAMES

DEFAULT_SCALES = np.asarray([2.4, 3.0, 0.21, 3.0], dtype=np.float64)


@dataclass
class TreeNode:
    region_id: Optional[int] = None
    dim: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    def is_leaf(self) -> bool:
        return self.region_id is not None


@dataclass
class StatePartition:
    mean: np.ndarray
    scale: np.ndarray
    root: TreeNode
    n_regions: int
    method: str = "adaptive_tree"
    meta: Dict[str, Any] = field(default_factory=dict)

    def normalize(self, states: np.ndarray) -> np.ndarray:
        s = np.asarray(states, dtype=np.float64)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        return (s - self.mean) / np.maximum(self.scale, 1e-8)

    def region_of(self, state) -> int:
        z = self.normalize(np.asarray(state, dtype=np.float64).reshape(1, -1))[0]
        node = self.root
        while not node.is_leaf():
            if z[int(node.dim)] <= float(node.threshold):
                node = node.left
            else:
                node = node.right
        return int(node.region_id)

    def regions_of(self, states: np.ndarray) -> np.ndarray:
        z = self.normalize(states)
        out = np.empty(z.shape[0], dtype=np.int64)
        for i in range(z.shape[0]):
            node = self.root
            while not node.is_leaf():
                if z[i, int(node.dim)] <= float(node.threshold):
                    node = node.left
                else:
                    node = node.right
            out[i] = int(node.region_id)
        return out

    def to_dict(self) -> Dict[str, Any]:
        def pack(node: TreeNode) -> Dict[str, Any]:
            if node.is_leaf():
                return {"region_id": int(node.region_id)}
            return {
                "dim": int(node.dim),
                "dim_name": STATE_NAMES[int(node.dim)],
                "threshold": float(node.threshold),
                "left": pack(node.left),
                "right": pack(node.right),
            }

        return {
            "method": self.method,
            "state_dim": STATE_DIM,
            "state_names": list(STATE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "n_regions": int(self.n_regions),
            "tree": pack(self.root),
            "meta": self.meta,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatePartition":
        def unpack(obj: Dict[str, Any]) -> TreeNode:
            if "region_id" in obj:
                return TreeNode(region_id=int(obj["region_id"]))
            return TreeNode(
                dim=int(obj["dim"]),
                threshold=float(obj["threshold"]),
                left=unpack(obj["left"]),
                right=unpack(obj["right"]),
            )

        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            root=unpack(data["tree"]),
            n_regions=int(data["n_regions"]),
            method=str(data.get("method", "adaptive_tree")),
            meta=dict(data.get("meta", {})),
        )

    @classmethod
    def load(cls, path) -> "StatePartition":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _delta_var_score(deltas: np.ndarray) -> float:
    n = int(deltas.shape[0])
    if n < 2:
        return 0.0
    return float(np.mean(np.var(deltas, axis=0, ddof=1)))


def _best_split(
    z: np.ndarray,
    deltas: np.ndarray,
    min_leaf: int,
    n_thresholds: int = 16,
) -> Optional[Tuple[int, float, float]]:
    n = int(z.shape[0])
    if n < 2 * min_leaf:
        return None
    parent_score = _delta_var_score(deltas)
    best = None
    best_gain = 0.0
    for dim in range(z.shape[1]):
        col = z[:, dim]
        qs = np.linspace(0.15, 0.85, n_thresholds)
        thresholds = np.unique(np.quantile(col, qs))
        for thr in thresholds:
            left = col <= thr
            n_l = int(left.sum())
            n_r = n - n_l
            if n_l < min_leaf or n_r < min_leaf:
                continue
            score = (
                n_l * _delta_var_score(deltas[left]) + n_r * _delta_var_score(deltas[~left])
            ) / n
            gain = parent_score - score
            if best is None or score < best[2] - 1e-15:
                best = (dim, float(thr), float(score))
                best_gain = gain
    if best is None or best_gain < 1e-12:
        return None
    return best


def fit_adaptive_partition(
    states: np.ndarray,
    next_states: np.ndarray,
    *,
    min_leaf: int = 40,
    max_depth: int = 8,
    max_regions: int = 64,
    var_eps: float = 1e-6,
    scales: Optional[np.ndarray] = None,
) -> StatePartition:
    states = np.asarray(states, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"states must be (N,{STATE_DIM})")
    deltas = next_states - states
    mean = states.mean(axis=0)
    scale = np.asarray(scales if scales is not None else DEFAULT_SCALES, dtype=np.float64)
    z = (states - mean) / np.maximum(scale, 1e-8)

    next_id = 0

    def grow(idx: np.ndarray, depth: int) -> TreeNode:
        nonlocal next_id
        node = TreeNode()
        local_d = deltas[idx]
        stop = (
            depth >= max_depth
            or idx.size < 2 * min_leaf
            or next_id >= max_regions
            or _delta_var_score(local_d) <= var_eps
        )
        if stop:
            node.region_id = next_id
            next_id += 1
            return node
        split = _best_split(z[idx], local_d, min_leaf=min_leaf)
        if split is None:
            node.region_id = next_id
            next_id += 1
            return node
        dim, thr, _ = split
        left_mask = z[idx, dim] <= thr
        if int(left_mask.sum()) < min_leaf or int((~left_mask).sum()) < min_leaf:
            node.region_id = next_id
            next_id += 1
            return node
        if next_id + 2 > max_regions:
            node.region_id = next_id
            next_id += 1
            return node
        node.dim = dim
        node.threshold = thr
        node.left = grow(idx[left_mask], depth + 1)
        node.right = grow(idx[~left_mask], depth + 1)
        return node

    root = grow(np.arange(states.shape[0]), 0)
    return StatePartition(
        mean=mean,
        scale=scale,
        root=root,
        n_regions=next_id,
        method="adaptive_tree",
        meta={
            "min_leaf": int(min_leaf),
            "max_depth": int(max_depth),
            "max_regions": int(max_regions),
            "var_eps": float(var_eps),
            "n_fit_transitions": int(states.shape[0]),
            "fit_delta_var": _delta_var_score(deltas),
        },
    )


def fit_grid_partition(
    states: np.ndarray,
    *,
    bins_per_dim: Tuple[int, ...] = (2, 2, 4, 2),
    scales: Optional[np.ndarray] = None,
) -> StatePartition:
    """Axis-aligned grid on normalized state (exact lookup)."""
    states = np.asarray(states, dtype=np.float64)
    mean = states.mean(axis=0)
    scale = np.asarray(scales if scales is not None else DEFAULT_SCALES, dtype=np.float64)
    bins = tuple(int(b) for b in bins_per_dim)
    if len(bins) != STATE_DIM:
        raise ValueError("bins_per_dim must have length 4")

    def grid_id(prefix: Tuple[int, ...]) -> int:
        idx = 0
        for p, b in zip(prefix, bins):
            idx = idx * b + int(p)
        return int(idx)

    def build(dim: int, prefix: Tuple[int, ...]) -> TreeNode:
        if dim >= STATE_DIM:
            return TreeNode(region_id=grid_id(prefix))
        b = bins[dim]
        edges = np.linspace(-1.5, 1.5, b + 1)

        def make(lo: int, hi: int) -> TreeNode:
            if hi - lo == 1:
                return build(dim + 1, prefix + (lo,))
            mid = (lo + hi) // 2
            node = TreeNode(dim=dim, threshold=float(edges[mid]))
            node.left = make(lo, mid)
            node.right = make(mid, hi)
            return node

        return make(0, b)

    root = build(0, ())
    return StatePartition(
        mean=mean,
        scale=scale,
        root=root,
        n_regions=int(np.prod(bins)),
        method="grid",
        meta={"bins_per_dim": list(bins), "n_fit_states": int(states.shape[0])},
    )

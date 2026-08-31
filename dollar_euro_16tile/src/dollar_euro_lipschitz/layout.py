"""4x4 tile layout with five dynamics categories.

Categories (stats / Lipschitz are pooled by these, not by tile id):
  1–4 = ContinuousDollarEuroEnv_radial.py region templates (all stochastic)
  5   = deterministic (Sigma = 0)

``determinism`` in [0, 1] sets the fraction of the 16 tiles that are category 5.
The remaining tiles are split as evenly as possible among categories 1–4.

At determinism=0 the map matches radial's four quadrants (each quadrant is 2×2
tiles) and category lookup uses the same continuous (x, y) rules as radial.
For determinism>0 tiles are shuffled but reproducible from the determinism value.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

GRID_SIZE = 4
N_TILES = GRID_SIZE * GRID_SIZE
N_CATEGORIES = 5
STOCHASTIC_CATEGORIES = (1, 2, 3, 4)
DETERMINISTIC_CATEGORY = 5
CATEGORY_IDS = tuple(range(1, N_CATEGORIES + 1))
REGION_CENTER = 0.5
CATEGORY_NAMES = {
    1: "radial_r1_top_right",
    2: "radial_r2_top_left",
    3: "radial_r3_bottom_left",
    4: "radial_r4_bottom_right",
    5: "deterministic",
}


def clamp_determinism(determinism: float) -> float:
    value = float(determinism)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"determinism must be in [0, 1]; got {value}")
    return value


def category_from_radial_quadrant(x: float, y: float, center: float = REGION_CENTER) -> int:
    """Same quadrant rule as ContinuousDollarEuroEnv_radial._region_id."""
    cx = cy = float(center)
    xf = float(x)
    yf = float(y)
    if xf >= cx and yf >= cy:
        return 1
    if xf < cx and yf >= cy:
        return 2
    if xf < cx and yf < cy:
        return 3
    return 4


def categories_from_radial_quadrants(xs, ys, center: float = REGION_CENTER) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    cx = cy = float(center)
    cats = np.full(xs.shape, 4, dtype=np.int64)
    top = ys >= cy
    left = xs < cx
    right = xs >= cx
    cats[(right) & (top)] = 1
    cats[(left) & (top)] = 2
    cats[(left) & (~top)] = 3
    return cats


def category_counts(determinism: float, n_tiles: int = N_TILES) -> Dict[int, int]:
    det = clamp_determinism(determinism)
    n_det = int(round(det * n_tiles))
    n_det = max(0, min(n_tiles, n_det))
    n_rest = n_tiles - n_det
    base, rem = divmod(n_rest, len(STOCHASTIC_CATEGORIES))
    counts = {DETERMINISTIC_CATEGORY: n_det}
    for i, cat in enumerate(STOCHASTIC_CATEGORIES):
        counts[cat] = base + (1 if i < rem else 0)
    return counts


def radial_quadrant_tile_map(grid: int = GRID_SIZE) -> np.ndarray:
    """Fixed 4×4 map: each radial quadrant occupies one 2×2 block of tiles."""
    mapping = np.zeros(grid * grid, dtype=np.int32)
    split = grid // 2
    for tid in range(grid * grid):
        iy, ix = divmod(tid, grid)
        right = ix >= split
        top = iy >= split
        if right and top:
            mapping[tid] = 1
        elif (not right) and top:
            mapping[tid] = 2
        elif (not right) and (not top):
            mapping[tid] = 3
        else:
            mapping[tid] = 4
    return mapping


def tile_category_map(determinism: float, n_tiles: int = N_TILES) -> np.ndarray:
    """Return length-16 array: index = row-major tile id (y-major then x)."""
    det = clamp_determinism(determinism)
    if det == 0.0:
        return radial_quadrant_tile_map()
    counts = category_counts(det, n_tiles=n_tiles)
    # Seed depends only on determinism so layouts are reproducible across env creates.
    seed = (int(round(det * 10000.0)) ^ 0x16A17E) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    order = np.arange(n_tiles, dtype=np.int32)
    rng.shuffle(order)
    assignment = np.zeros(n_tiles, dtype=np.int32)
    cursor = 0
    for cat in (DETERMINISTIC_CATEGORY, *STOCHASTIC_CATEGORIES):
        for _ in range(int(counts[cat])):
            assignment[int(order[cursor])] = int(cat)
            cursor += 1
    return assignment


def tile_ids_from_states(states, grid: int = GRID_SIZE) -> np.ndarray:
    """Row-major tile id 0..grid*grid-1 for each state."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    xs = states[:, 0]
    ys = states[:, 1]
    ix = np.minimum(grid - 1, np.maximum(0, (xs * grid).astype(np.int32)))
    iy = np.minimum(grid - 1, np.maximum(0, (ys * grid).astype(np.int32)))
    ix = np.where(xs >= 1.0, grid - 1, ix)
    iy = np.where(ys >= 1.0, grid - 1, iy)
    return (iy * grid + ix).astype(np.int64)


def tile_id_from_xy(x: float, y: float, grid: int = GRID_SIZE) -> int:
    """Scalar tile id for one (x, y) point."""
    return int(tile_ids_from_states(np.array([float(x), float(y)], dtype=np.float64), grid=grid)[0])


def category_from_state(state, determinism: float, tile_map: Optional[np.ndarray] = None) -> int:
    det = clamp_determinism(determinism)
    x = float(state[0])
    y = float(state[1])
    if det == 0.0:
        return category_from_radial_quadrant(x, y)
    mapping = tile_map if tile_map is not None else tile_category_map(det)
    tid = tile_id_from_xy(x, y)
    return int(mapping[tid])


def categories_from_states(states, determinism: float, tile_map: Optional[np.ndarray] = None) -> np.ndarray:
    det = clamp_determinism(determinism)
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    xs = states[:, 0]
    ys = states[:, 1]
    if det == 0.0:
        return categories_from_radial_quadrants(xs, ys)
    mapping = tile_map if tile_map is not None else tile_category_map(det)
    ix = np.minimum(GRID_SIZE - 1, np.maximum(0, (xs * GRID_SIZE).astype(np.int32)))
    iy = np.minimum(GRID_SIZE - 1, np.maximum(0, (ys * GRID_SIZE).astype(np.int32)))
    ix = np.where(xs >= 1.0, GRID_SIZE - 1, ix)
    iy = np.where(ys >= 1.0, GRID_SIZE - 1, iy)
    tile_ids = iy * GRID_SIZE + ix
    return mapping[tile_ids].astype(np.int64)


def layout_summary(determinism: float) -> Dict[str, object]:
    det = clamp_determinism(determinism)
    mapping = tile_category_map(det)
    counts = category_counts(det)
    grid = mapping.reshape(GRID_SIZE, GRID_SIZE)
    return {
        "determinism": det,
        "grid_size": GRID_SIZE,
        "n_tiles": N_TILES,
        "n_categories": N_CATEGORIES,
        "layout_mode": "radial_quadrants" if det == 0.0 else "shuffled",
        "counts": {str(k): int(v) for k, v in counts.items()},
        "category_names": CATEGORY_NAMES,
        "tile_map_row_major": mapping.tolist(),
        "tile_map_grid_yx": grid.tolist(),
    }

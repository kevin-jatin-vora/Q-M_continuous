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
    }


# ---------------------------------------------------------------------------
# Cross-category neighbor discovery
# ---------------------------------------------------------------------------


def discover_neighbor_pairs(tile_map) -> list:
    """Auto-discover physically neighboring category pairs from the 4x4 tile layout.

    Inspects every horizontal/vertical tile adjacency. If adjacent tiles have
    different categories, canonicalizes as (min, max) and adds to the result set.

    Returns a list of dicts with keys:
      - category_i, category_j: canonical unordered pair
      - tile_edges: list of [tile_a, tile_b] pairs responsible for this adjacency
    """
    mapping = np.asarray(tile_map, dtype=np.int32).ravel()
    grid = GRID_SIZE
    raw: Dict[tuple, list] = {}
    for tid in range(int(N_TILES)):
        iy, ix = divmod(tid, grid)
        cat = int(mapping[tid])
        # right neighbor
        if ix + 1 < grid:
            ntid = tid + 1
            ncat = int(mapping[ntid])
            if cat != ncat:
                key = (min(cat, ncat), max(cat, ncat))
                raw.setdefault(key, []).append([tid, ntid])
        # down neighbor
        if iy + 1 < grid:
            ntid = tid + grid
            ncat = int(mapping[ntid])
            if cat != ncat:
                key = (min(cat, ncat), max(cat, ncat))
                raw.setdefault(key, []).append([tid, ntid])
    result = []
    for (ci, cj), edges in sorted(raw.items()):
        result.append({
            "category_i": ci,
            "category_j": cj,
            "tile_edges": edges,
        })
    return result


# ---------------------------------------------------------------------------
# Geometric ball-tile intersection
# ---------------------------------------------------------------------------


def tile_rectangles(grid: int = GRID_SIZE):
    """Return per-tile rectangles as list of (x0, x1, y0, y1) tuples.

    Tile id = iy * grid + ix where iy = row (y), ix = column (x).
    Each tile spans [ix/grid, (ix+1)/grid] in x and [iy/grid, (iy+1)/grid] in y.
    """
    rects = []
    cell = 1.0 / grid
    for tid in range(grid * grid):
        iy, ix = divmod(tid, grid)
        x0 = ix * cell
        x1 = (ix + 1) * cell
        y0 = iy * cell
        y1 = (iy + 1) * cell
        rects.append((x0, x1, y0, y1))
    return rects


_TILE_RECTS_CACHE = {}


def tile_rectangles_array(grid: int = GRID_SIZE) -> np.ndarray:
    """Precomputed per-tile rectangles as an (N_TILES, 4) float64 array.

    Columns are (x0, x1, y0, y1).  Tile id = iy * grid + ix.  Computed once and
    cached because the 4x4 grid is fixed for the whole process.
    """
    cached = _TILE_RECTS_CACHE.get(grid)
    if cached is not None:
        return cached
    cell = 1.0 / grid
    rects = np.empty((grid * grid, 4), dtype=np.float64)
    for tid in range(grid * grid):
        iy, ix = divmod(tid, grid)
        rects[tid, 0] = ix * cell
        rects[tid, 1] = (ix + 1) * cell
        rects[tid, 2] = iy * cell
        rects[tid, 3] = (iy + 1) * cell
    _TILE_RECTS_CACHE[grid] = rects
    return rects


def _min_distance_to_rect(px, py, rect):
    """Minimum Euclidean distance from point (px, py) to axis-aligned rectangle.

    rect = (x0, x1, y0, y1).  Returns 0.0 if the point is inside the rectangle.
    """
    x0, x1, y0, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return (dx * dx + dy * dy) ** 0.5


def _min_sq_distance_to_rects(cx, cy, rects):
    """Vectorized minimum squared distance from a single point to each rectangle.

    ``rects`` is an (T, 4) array with columns (x0, x1, y0, y1).  Returns an
    (T,) float64 array of squared distances (0 for a point inside a rectangle).
    """
    x0 = rects[:, 0]
    x1 = rects[:, 1]
    y0 = rects[:, 2]
    y1 = rects[:, 3]
    dx = np.maximum(x0 - cx, np.maximum(0.0, cx - x1))
    dy = np.maximum(y0 - cy, np.maximum(0.0, cy - y1))
    return dx * dx + dy * dy


def find_intersected_tiles(center, radius, grid: int = GRID_SIZE):
    """Return set of tile ids intersected by ball B(center, radius).

    Uses exact L2 distance from the ball center to each tile rectangle.
    A tile is intersected iff the minimum distance from the center to the
    tile rectangle is <= radius.
    """
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    rects = tile_rectangles_array(grid)
    sq = _min_sq_distance_to_rects(cx, cy, rects)
    tol = r + 1e-12
    return {int(tid) for tid in range(rects.shape[0]) if sq[tid] <= tol * tol}


def batched_intersected_tile_masks(centers, radii, grid: int = GRID_SIZE):
    """Vectorized tile-intersection masks for many balls.

    ``centers`` shape (B, 2); ``radii`` shape (B,) or scalar.  Returns a
    boolean (B, N_TILES) mask; mask[b, t] is True iff tile t is intersected by
    ball B(centers[b], radii[b]).  Uses squared distances, no sqrt.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    b = centers.shape[0]
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64).reshape(-1), (b,))
    rects = tile_rectangles_array(grid)
    x0 = rects[:, 0][None, :]  # (1, T)
    x1 = rects[:, 1][None, :]
    y0 = rects[:, 2][None, :]
    y1 = rects[:, 3][None, :]
    cx = centers[:, 0, None]  # (B, 1)
    cy = centers[:, 1, None]
    dx = np.maximum(x0 - cx, np.maximum(0.0, cx - x1))
    dy = np.maximum(y0 - cy, np.maximum(0.0, cy - y1))
    sq = dx * dx + dy * dy
    tol = (radii[:, None] + 1e-12) ** 2
    return sq <= tol


def find_intersected_categories(center, radius, tile_map, grid: int = GRID_SIZE):
    """Return set of category ids intersected by ball B(center, radius)."""
    mapping = np.asarray(tile_map, dtype=np.int32).ravel()
    tiles = find_intersected_tiles(center, radius, grid)
    return {int(mapping[t]) for t in tiles}


def tile_id_mask_from_tile_ids(tile_ids, n_tiles: int = N_TILES):
    """Convert an array/iterable of tile ids into a 16-bit integer mask.

    Bit ``t`` of the returned integer is 1 iff tile ``t`` is present.
    """
    mask = 0
    for t in np.asarray(tile_ids, dtype=np.int64).ravel():
        mask |= (1 << int(t))
    return mask


def tile_ids_from_mask(mask, n_tiles: int = N_TILES):
    """Expand a 16-bit integer tile mask back into a sorted tuple of tile ids."""
    return tuple(t for t in range(n_tiles) if mask & (1 << t))


def discover_tile_neighbor_pairs(grid: int = GRID_SIZE):
    """Every unique physical 4-neighbor tile pair in the grid.

    Returns a sorted list of canonicalized ``[tile_a, tile_b]`` pairs
    (tile_a < tile_b) that share an edge.  Adjacency rules follow the project's
    row-major convention (right neighbor + down neighbor).
    """
    pairs = set()
    ntiles = grid * grid
    for tid in range(ntiles):
        iy, ix = divmod(tid, grid)
        if ix + 1 < grid:
            a, b = tid, tid + 1
            pairs.add((min(a, b), max(a, b)))
        if iy + 1 < grid:
            a, b = tid, tid + grid
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def _tile_adjacencies(n_tiles: int = N_TILES, grid: int = GRID_SIZE):
    """Return {tile_id: [neighbor tile ids]} for the 4x4 grid (right + down)."""
    adj: Dict[int, list] = {t: [] for t in range(n_tiles)}
    for tid in range(n_tiles):
        iy, ix = divmod(tid, grid)
        if ix + 1 < grid:
            adj[tid].append(tid + 1)
        if iy + 1 < grid:
            adj[tid].append(tid + grid)
    return adj


# Set of unordered neighbor tile pairs: {(tile_a, tile_b), ...} canonicalized.
TILE_NEIGHBOR_PAIR_SET = frozenset(
    (min(a, b), max(a, b)) for a, b in discover_tile_neighbor_pairs()
)

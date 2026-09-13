"""Learned Lipschitz Q-bound utilities.

The learned bound targets use the nominal mean transition s_bar' and one-step
Lipschitz perturbations:
  delta_r = Lr(tile,a) * radius(category,a)
  delta_q = Lq(tile,a) * radius(category,a)      (legacy non-overlap path)
No /(1-gamma) factor is applied here; Bellman iteration propagates uncertainty.

Overlap-aware mode (when cross constants are available) keeps Lq as a
tile/action-specific state Lipschitz constant:

  Lq(T,b) = Lr(T) * Lf(C(T),b) / (1 - gamma*Lf(C(T),b))

and makes the CURRENT action and the FUTURE action play distinct roles:

  current action a:
    - selects source category
    - selects delta_mean(category, a) -> mean next state s_bar'
    - selects the Student-t uncertainty radius r = radius(source_category, a)

  next-state ball B(s_bar', r):
    - determines the intersected tile mask -> Lr_eff(mask)
    - determines the represented categories, over which Lf_eff(mask, b) is
      formed for EVERY possible future action b in {0,1,2,3}
    - future action b appears inside the Bellman continuation max_b Q(s',b),
      so the future Q Lipschitz constant must cover all next actions:

        |max_b Q(s1',b) - max_b Q(s2',b)| <= max_b |Q(s1',b)-Q(s2',b)|
        <= (max_b Lq(b)) * ||s1'-s2'||

    Lq_eff(mask,b) = Lr_eff(mask) * Lf_eff(mask,b) / (1 - gamma*Lf_eff(mask,b))
    Lq_future_eff(mask) = max_b Lq_eff(mask,b)

  Uncertainty terms (radius belongs to the CURRENT action a):
    delta_r = Lr_eff(mask) * r
    delta_q = Lq_future_eff(mask) * r

The cached geometry result depends only on the 16-bit intersected tile mask
(never on the current action or float center/radius).  The same mask returns
the same Lq_future_eff regardless of which current action produced it.
"""
from pathlib import Path
import json
import numpy as np
import torch

from .bounds import RegionActionBounds, LipschitzConstants
from .layout import (
    GRID_SIZE,
    N_TILES,
    tile_category_map,
    tile_ids_from_states,
    tile_id_mask_from_tile_ids,
    tile_ids_from_mask,
    find_intersected_tiles,
    batched_intersected_tile_masks,
)


def build_one_step_uncertainty_tables(bounds_path, lipschitz_path, lq_source, determinism):
    bounds = RegionActionBounds(bounds_path)
    lips = LipschitzConstants(lipschitz_path, source=lq_source)
    tile_map = tile_category_map(float(determinism))
    delta_r = np.empty((N_TILES, 4), dtype=np.float32)
    delta_q = np.empty((N_TILES, 4), dtype=np.float32)
    radius = np.empty((N_TILES, 4), dtype=np.float32)
    for tile in range(N_TILES):
        region = int(tile_map[tile])
        for action in range(4):
            r = float(bounds.get(region, action)["pruning_radius"])
            lr, lq = lips.get(tile, action)
            radius[tile, action] = r
            delta_r[tile, action] = float(lr) * r
            delta_q[tile, action] = float(lq) * r
    return delta_r, delta_q, radius, tile_map


def batch_uncertainty(states_np, actions_np, delta_r_table, delta_q_table):
    tiles = tile_ids_from_states(states_np)
    actions = np.asarray(actions_np, dtype=np.int64).reshape(-1)
    return delta_r_table[tiles, actions], delta_q_table[tiles, actions]


def learned_bound_allowed_mask(q_ub, q_lb, tol=1e-5):
    """Keep a iff Q_UB(s,a) >= max_b Q_LB(s,b) - tol."""
    threshold = q_lb.max(dim=1, keepdim=True).values - float(tol)
    mask = q_ub >= threshold
    empty = ~mask.any(dim=1)
    if empty.any():
        mask = mask.clone()
        mask[empty] = True
    return mask


def load_frozen_qnet(path, qnet_cls, device):
    model = qnet_cls().to(device)
    model.load_state_dict(torch.load(Path(path), map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Overlap-aware uncertainty constants
# ---------------------------------------------------------------------------


class InvalidFutureLqError(RuntimeError):
    """Raised when some future action cannot be bounded (gamma*Lf_eff(b) >= 1).

    The future Bellman max ``max_b Q(s',b)`` requires every relevant
    action-specific theoretical Q Lipschitz bound to be finite.  ``payload``
    carries provenance/diagnostic fields (mask info, offending action, etc.);
    the batch wrapper enriches it with the current sample (source state,
    current action, mean_s', radius) before it propagates.
    """

    def __init__(self, message, payload):
        super().__init__(message)
        self.payload = payload


class OverlapAwareConstants:
    """Load version-3 ordinary + cross Lipschitz constants for overlap-aware lookup.

    Logical constant sets (all ``lipschitz_method_version == 3``):
      - tile_lr[tile]: ordinary NEXT-state-tile reward Lr (action-independent)
      - cat_lf[cat, action]: ordinary source-category/action dynamics Lf
      - cross_tile_lr[(ti, tj)]: cross-tile reward Lr across a NEXT-state tile
        boundary (action-independent)
      - cross_cat_lf[(ci, cj, action)]: cross-category dynamics Lf across a
        SOURCE category boundary

    Runtime selection for ball B(s_bar', radius):
      Lr_eff(mask)   = max(lr over intersected tiles, cross-tile lr over pairs)
      Lf_eff(mask,b) = max(lf over represented categories + cross-category lf,
                           for each possible future action b in {0,1,2,3})
      Lq_eff(mask,b) = Lr_eff * Lf_eff(mask,b) / (1 - gamma*Lf_eff(mask,b))
      Lq_future_eff(mask) = max_b Lq_eff(mask,b)
    If ANY future action b has gamma*Lf_eff(mask,b) >= 1, raise
    InvalidFutureLqError (never 0, never skip/clamp/fallback).

    16-bit integer tile-mask lazy caching: the selection result depends only on
    the *set* of intersected tiles (the mask), never on the current action or
    on float center/radius.  Lr_eff depends only on the mask; Lf_eff depends on
    mask + future action b (all four computed together); Lq_future_eff depends
    only on the mask once gamma and the artifacts are fixed.
    """

    REQUIRED_VERSION = 3
    N_ACTIONS = 4

    def __init__(self, lipschitz_path, cross_tile_path, cross_cat_path, gamma):
        self.gamma = float(gamma)
        self.grid = GRID_SIZE
        # 16-bit tile-mask -> constant block lazy cache.
        self._cache = {}

        with Path(lipschitz_path).open("r", encoding="utf-8") as f:
            lips_data = json.load(f)
        _require_version(lips_data.get("lipschitz_method_version"), lipschitz_path)
        rows = lips_data.get("constants", [])

        # Ordinary NEXT-state-tile reward Lr (action-independent).
        self.tile_lr = np.full(N_TILES, np.nan, dtype=np.float64)
        # Ordinary source-category/action dynamics Lf.
        self.cat_lf = np.zeros((6, 4), dtype=np.float64)

        for row in rows:
            tile = int(row["tile"])
            region = int(row["region"])
            action = int(row["action"])
            lr = float(row["Lr_sum"])
            if not self.tile_lr[tile] == self.tile_lr[tile]:  # first assignment
                self.tile_lr[tile] = lr
            else:
                # All action rows share the same (action-independent) Lr;
                # guard against inconsistency in the artifact.
                self.tile_lr[tile] = max(self.tile_lr[tile], lr)
            lf = float(row["Lf_sum"])
            self.cat_lf[region, action] = max(self.cat_lf[region, action], lf)

        # Cross-tile reward Lr: keyed by canonical (tile_i, tile_j).
        self.cross_tile_lr = {}
        self.tile_neighbor_pairs = set()
        with Path(cross_tile_path).open("r", encoding="utf-8") as f:
            cross_tile = json.load(f)
        _require_version(cross_tile.get("lipschitz_method_version"), cross_tile_path)
        for row in cross_tile.get("constants", []):
            ti, tj = int(row["tile_i"]), int(row["tile_j"])
            key = (min(ti, tj), max(ti, tj))
            self.tile_neighbor_pairs.add(key)
            self.cross_tile_lr[key] = max(
                self.cross_tile_lr.get(key, 0.0),
                float(row["Lr_sum"]),
            )

        # Cross-category dynamics Lf: keyed by (ci, cj, action).
        self.cross_cat_lf = {}
        self.cat_neighbor_pairs = set()
        with Path(cross_cat_path).open("r", encoding="utf-8") as f:
            cross_cat = json.load(f)
        _require_version(cross_cat.get("lipschitz_method_version"), cross_cat_path)
        for row in cross_cat.get("constants", []):
            ci, cj = int(row["category_i"]), int(row["category_j"])
            action = int(row["action"])
            key = (min(ci, cj), max(ci, cj))
            self.cat_neighbor_pairs.add(key)
            self.cross_cat_lf[(ci, cj, action)] = max(
                self.cross_cat_lf.get((ci, cj, action), 0.0),
                float(row["Lf_sum"]),
            )

    def _intersected_tiles(self, center, radius, tile_map):
        return find_intersected_tiles(center, radius, self.grid)

    def _resolve_mask(self, mask_int, tile_map):
        """Return (cached) full constant block for a 16-bit tile mask."""
        mask_int = int(mask_int)
        cached = self._cache.get(mask_int)
        if cached is not None:
            return cached
        block = self._compute_for_mask(mask_int, tile_map)
        self._cache[mask_int] = block
        return block

    def _compute_for_mask(self, mask_int, tile_map):
        """Compute Lr_eff(mask) and, for every future action b in {0,1,2,3},
        Lf_eff(mask,b), Lq_eff(mask,b); then Lf_future_eff and Lq_future_eff.

        Raises InvalidFutureLqError if any future action b has
        gamma*Lf_eff(mask,b) >= 1 (the future Bellman max is then unbounded).
        """
        tiles = tile_ids_from_mask(mask_int)
        if not tiles:
            return {
                "lr_eff": 0.0, "lr_from": "none",
                "lf_by_action": [0.0] * self.N_ACTIONS,
                "lf_provenance_by_action": ["none"] * self.N_ACTIONS,
                "lq_by_action": [0.0] * self.N_ACTIONS,
                "lf_future_eff": 0.0, "lf_future_argmax": 0,
                "lq_future_eff": 0.0, "future_action_argmax": 0,
                "n_tiles": 0, "n_categories": 0,
                "tile_ids": [], "categories": [],
            }
        mapping = np.asarray(tile_map, dtype=np.int32).ravel()
        tile_set = set(tiles)
        represented = {int(mapping[t]) for t in tiles}
        rep = sorted(c for c in represented if 1 <= c <= 5)

        # Lr_eff(mask): ordinary next-tile + cross-tile reward pairs.
        lr_eff = -1.0
        lr_from = None
        for t in tiles:
            v = self.tile_lr[t]
            if v == v and v > lr_eff:
                lr_eff, lr_from = v, f"ordinary_tile_{t}"
        for (ti, tj), v in self.cross_tile_lr.items():
            if ti in tile_set and tj in tile_set and v > lr_eff:
                lr_eff, lr_from = v, f"cross_tile_{ti}-{tj}"
        if lr_eff < 0:
            lr_eff, lr_from = 0.0, "none"

        # Lf_eff(mask, b) for every possible future action b.
        lf_by = []
        lf_prov_by = []
        lq_by = []
        for b in range(self.N_ACTIONS):
            lf = -1.0
            src = None
            for c in rep:
                v = float(self.cat_lf[c, b]) if 0 <= c < self.cat_lf.shape[0] else 0.0
                if v > lf:
                    lf, src = v, f"ordinary_category_{c}"
            for (ci, cj) in self.cat_neighbor_pairs:
                if ci in represented and cj in represented:
                    v = self.cross_cat_lf.get((ci, cj, b))
                    if v is not None and v > lf:
                        lf, src = v, f"cross_category_{ci}-{cj}"
            if lf < 0:
                lf, src = 0.0, "none"
            lq = _derive_lq_future(
                lr_eff, lf, self.gamma, action_b=b,
                mask_int=mask_int, tiles=tiles, categories=rep,
                lr_from=lr_from, lf_from=src,
            )
            lf_by.append(lf)
            lf_prov_by.append(src)
            lq_by.append(lq)

        lf_by_a = np.asarray(lf_by, dtype=np.float64)
        lq_by_a = np.asarray(lq_by, dtype=np.float64)
        return {
            "lr_eff": lr_eff, "lr_from": lr_from,
            "lf_by_action": lf_by, "lf_provenance_by_action": lf_prov_by,
            "lq_by_action": lq_by,
            "lf_future_eff": float(lf_by_a.max()),
            "lf_future_argmax": int(lf_by_a.argmax()),
            "lq_future_eff": float(lq_by_a.max()),
            "future_action_argmax": int(lq_by_a.argmax()),
            "n_tiles": len(tiles), "n_categories": len(rep),
            "tile_ids": list(tiles), "categories": rep,
        }

    # -- scalar entry points (mainly for diagnostics/tests) ----------------
    def get_overlap_constants_scalar(self, center, radius, action, tile_map):
        return self.get_overlap_constants_with_provenance(center, radius, action, tile_map)

    def get_overlap_constants(self, center, radius, action, tile_map):
        """Compute effective Lr, Lq_future for ball B(center, radius)."""
        sel = self.get_overlap_constants_with_provenance(center, radius, action, tile_map)
        return sel["lr_eff"], sel["lq_future_eff"]

    def get_overlap_constants_with_provenance(self, center, radius, action, tile_map):
        """Compute effective constants with provenance for B(center, radius).

        ``action`` is the CURRENT action; it does not select Lf/Lq (the future
        action lives inside the Bellman max).  The mask determines every
        quantity, so the returned selection is independent of ``action``.
        """
        tiles = self._intersected_tiles(center, radius, tile_map)
        mask_int = tile_id_mask_from_tile_ids(sorted(tiles))
        block = self._resolve_mask(mask_int, tile_map)
        out = dict(block)
        # Backward-compatible aliases used by legacy callers/tests.
        out["lq_eff"] = block["lq_future_eff"]
        out["lf_eff"] = block["lf_future_eff"]
        out["lq_future_eff"] = block["lq_future_eff"]
        out["action"] = int(action)
        return out

    # -- batched vectorized lookups (unique-mask optimized) ----------------
    def lookups_for_masks(self, masks_int, tile_map):
        """Resolve future-aware constants for a batch of 16-bit tile masks.

        ``masks_int`` (B,) int array.  Returns
        (lr_eff (B,), lf_future_eff (B,), lq_future_eff (B,), argmax_actions (B,)).
        Every row with the same mask shares a single cached resolution, so four
        full geometry passes are avoided and per-action work is done once.
        """
        masks = np.asarray(masks_int).reshape(-1)
        unique_masks, inverse = np.unique(masks, return_inverse=True)
        n = len(masks)
        lr = np.zeros(n, dtype=np.float64)
        lf_fut = np.zeros(n, dtype=np.float64)
        lq_fut = np.zeros(n, dtype=np.float64)
        argmax = np.zeros(n, dtype=np.int64)
        for j, m in enumerate(unique_masks):
            block = self._resolve_mask(int(m), tile_map)
            lr[inverse == j] = block["lr_eff"]
            lf_fut[inverse == j] = block["lf_future_eff"]
            lq_fut[inverse == j] = block["lq_future_eff"]
            argmax[inverse == j] = block["future_action_argmax"]
        return lr, lf_fut, lq_fut, argmax


def _require_version(version, path):
    if int(version) != OverlapAwareConstants.REQUIRED_VERSION:
        raise ValueError(
            "Old Lipschitz artifacts use incompatible reward grouping/"
            "cross-boundary semantics; regenerate from scratch. "
            f"({path}: lipschitz_method_version={version!r}, "
            f"expected {OverlapAwareConstants.REQUIRED_VERSION})"
        )


def _is_finite(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _derive_lq_future(lr_eff, lf_b, gamma, action_b, mask_int, tiles, categories,
                      lr_from, lf_from):
    """Lq_eff(b) = Lr_eff * Lf_eff(b) / (1 - gamma*Lf_eff(b)).

    RAISES InvalidFutureLqError if gamma*Lf_eff(b) >= 1 for this future action
    b, because the Bellman continuation ``max_b Q(s',b)`` would then be
    unbounded.  Never returns 0, never skips/clamps/fallbacks.
    """
    gamma = float(gamma)
    denom = 1.0 - gamma * lf_b
    if denom <= 0:
        raise InvalidFutureLqError(
            "Invalid future Q Lipschitz: the future Bellman max over next "
            "actions (max_b Q(s',b)) requires EVERY relevant action-specific "
            "theoretical Q Lipschitz bound to be finite, but gamma*Lf_eff(b)>=1 "
            "for a future action in the max. Refusing to return 0 / skip / "
            "clamp / substitute another action / use a finite-horizon fallback.",
            {
                "tile_mask": int(mask_int),
                "intersected_tile_ids": list(tiles),
                "represented_categories": list(categories),
                "lr_eff": float(lr_eff),
                "lr_source": lr_from,
                "offending_future_action": int(action_b),
                "lf_eff_b": float(lf_b),
                "lf_source_b": lf_from,
                "gamma": gamma,
                "gamma_times_lf": gamma * lf_b,
                "denominator": float(denom),
            }
        )
    return lr_eff * lf_b / denom


def _format_invalid_future_lq(exc, source_state, current_action, mean_s, radius):
    """Build a human-readable message for an InvalidFutureLqError, enriching the
    mask-level payload with the current sample (source state, current action,
    mean_s', and current radius)."""
    p = exc.payload
    lines = [
        str(exc),
        "  current source state : {0}".format(
            np.asarray(source_state).tolist() if source_state is not None else "n/a"
        ),
        "  current action       : {0}".format(current_action),
        "  mean next state s_bar': {0}".format(
            np.asarray(mean_s).tolist() if mean_s is not None else "n/a"
        ),
        "  current radius       : {0:.17g}".format(radius) if radius is not None else "  current radius       : n/a",
        "  tile mask            : {0}".format(p.get("tile_mask")),
        "  intersected tile IDs : {0}".format(p.get("intersected_tile_ids")),
        "  represented cats     : {0}".format(p.get("represented_categories")),
        "  Lr_eff               : {0:.17g}".format(p.get("lr_eff", 0.0)),
        "  Lr source            : {0}".format(p.get("lr_source")),
        "  offending future action b : {0}".format(p.get("offending_future_action")),
        "  Lf_eff(b)            : {0:.17g}".format(p.get("lf_eff_b", 0.0)),
        "  Lf source(b)         : {0}".format(p.get("lf_source_b")),
        "  gamma                : {0:.17g}".format(p.get("gamma")),
        "  gamma*Lf_eff(b)      : {0:.17g}".format(p.get("gamma_times_lf")),
        "  denominator 1-gamma*Lf: {0:.17g}".format(p.get("denominator")),
    ]
    return "\n".join(lines)


def batch_uncertainty_overlap_aware(
    states_np, actions_np, mean_deltas, radius_table, tile_map,
    overlap_constants, next_states_np=None,
):
    """Compute delta_r, delta_q with the corrected future-action semantics.

    For each sample:
      1. CURRENT action a selects the source category.
      2. center s_bar' = clip(s + delta_mean[source_category, a], 0, 1)
         (or use provided next_states_np as the center).
      3. radius r = radius_table[source_category, a]  (current-action radius).
      4. Find tiles intersected by B(s_bar', r); collapse to a 16-bit mask.
      5. The mask yields Lr_eff(mask) and Lq_future_eff(mask) = max_b Lq_eff
         over ALL next actions b (current action does NOT feed Lf/Lq).
      6. delta_r = Lr_eff(mask) * r, delta_q = Lq_future_eff(mask) * r.

    Constants are resolved once per unique mask (vectorized geometry + mask
    collapse), then scattered back to batch rows and multiplied by the
    current-action radius.

    Returns (delta_r, delta_q, diag).
    """
    n = states_np.shape[0]
    delta_r_out = np.zeros(n, dtype=np.float32)
    delta_q_out = np.zeros(n, dtype=np.float32)

    if next_states_np is None:
        categories_np = np.asarray(
            [int(tile_map[int(tile_ids_from_states(states_np[i:i+1])[0])]) for i in range(n)],
            dtype=np.int32,
        )
        category_indices = categories_np - 1
        next_states_np = np.clip(
            states_np + mean_deltas[category_indices, np.asarray(actions_np, dtype=np.int64)],
            0.0, 1.0,
        ).astype(np.float32)

    tile_ids = tile_ids_from_states(states_np)
    mapping = np.asarray(tile_map, dtype=np.int64).ravel()
    actions = np.asarray(actions_np, dtype=np.int64).reshape(-1)

    # Vectorized tile-intersection masks once (geometry is action-independent).
    radii = np.asarray(
        [float(radius_table[int(mapping[int(tile_ids[i])]), int(actions[i])]) for i in range(n)],
        dtype=np.float64,
    )
    ball_masks_bool = batched_intersected_tile_masks(next_states_np, radii, overlap_constants.grid)

    # Collapse each boolean (N_TILES,) row into a 16-bit integer tile mask.
    tile_bits = np.arange(N_TILES, dtype=np.int64)
    mask_rows = (ball_masks_bool.astype(np.int64) & 1) @ (1 << tile_bits)

    # Resolve constants once per unique mask; scatter back to rows.
    unique_masks, first_idx, inverse = np.unique(
        mask_rows, return_index=True, return_inverse=True
    )
    lr_eff = np.zeros(n, dtype=np.float64)
    lf_future = np.zeros(n, dtype=np.float64)
    lq_future = np.zeros(n, dtype=np.float64)
    blocks = [None] * len(unique_masks)
    for j, m in enumerate(unique_masks):
        try:
            block = overlap_constants._resolve_mask(int(m), tile_map)
        except InvalidFutureLqError as exc:
            i = int(first_idx[j])
            raise RuntimeError(
                _format_invalid_future_lq(
                    exc, states_np[i], int(actions[i]), next_states_np[i],
                    float(radii[i]),
                )
            ) from exc
        blocks[j] = block
        sel_rows = np.nonzero(inverse == j)[0]
        lr_eff[sel_rows] = block["lr_eff"]
        lf_future[sel_rows] = block["lf_future_eff"]
        lq_future[sel_rows] = block["lq_future_eff"]

    # Diagnostics per the corrected semantics.
    diag = {
        "total": n,
        "single_tile": 0,
        "crossed_tile_boundary": 0,
        "single_category": 0,
        "crossed_category_boundary": 0,
        "max_tiles_intersected": 0,
        "max_categories_intersected": 0,
        "lr_from_ordinary_tile": 0,
        "lr_from_cross_tile": 0,
        "lf_from_ordinary_category": [0] * 4,
        "lf_from_cross_category": [0] * 4,
        "max_lr_effective": 0.0,
        "max_lf_effective_over_all_next_actions": 0.0,
        "max_lq_future_effective": 0.0,
        "future_action_argmax": [0, 0, 0, 0],
    }

    for i in range(n):
        radius = float(radii[i])
        block = blocks[int(inverse[i])]
        delta_r_out[i] = float(lr_eff[i]) * radius
        delta_q_out[i] = float(lq_future[i]) * radius

        n_tiles = block["n_tiles"]
        n_cat = block["n_categories"]
        diag["max_tiles_intersected"] = max(diag["max_tiles_intersected"], n_tiles)
        diag["max_categories_intersected"] = max(diag["max_categories_intersected"], n_cat)
        if n_tiles <= 1:
            diag["single_tile"] += 1
        else:
            diag["crossed_tile_boundary"] += 1
        if n_cat <= 1:
            diag["single_category"] += 1
        else:
            diag["crossed_category_boundary"] += 1

        lr_from = block["lr_from"]
        if lr_from and lr_from.startswith("cross_tile"):
            diag["lr_from_cross_tile"] += 1
        else:
            diag["lr_from_ordinary_tile"] += 1

        for b in range(4):
            lf_from_b = block["lf_provenance_by_action"][b]
            if lf_from_b and lf_from_b.startswith("cross_category"):
                diag["lf_from_cross_category"][b] += 1
            else:
                diag["lf_from_ordinary_category"][b] += 1

        diag["future_action_argmax"][block["future_action_argmax"]] += 1
        diag["max_lr_effective"] = max(diag["max_lr_effective"], float(lr_eff[i]))
        diag["max_lf_effective_over_all_next_actions"] = max(
            diag["max_lf_effective_over_all_next_actions"], float(lf_future[i])
        )
        diag["max_lq_future_effective"] = max(
            diag["max_lq_future_effective"], float(lq_future[i])
        )

    return delta_r_out, delta_q_out, diag

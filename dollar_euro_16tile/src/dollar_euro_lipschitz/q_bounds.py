"""Learned Lipschitz Q-bound utilities.

The learned bound targets use the nominal mean transition s_bar' and one-step
Lipschitz perturbations:
  delta_r = Lr(tile,a) * radius(category,a)
  delta_q = Lq(tile,a) * radius(category,a)
No /(1-gamma) factor is applied here; Bellman iteration propagates uncertainty.
"""
from pathlib import Path
import numpy as np
import torch

from .bounds import RegionActionBounds, LipschitzConstants
from .layout import N_TILES, tile_category_map, tile_ids_from_states


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
    # The theory should leave at least one action. Do not alter bounds; only avoid
    # a downstream argmax crash if approximation error produces an empty mask.
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

"""Focused validation for the smooth scalar-noise terrain model."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dollar_euro_lipschitz.env import (
    ContinuousDollarEuroEnv,
    interpolate_noise_multiplier,
    noise_multiplier_lipschitz,
    smooth_noise_metadata,
    tile_noise_multiplier_grid,
)
from dollar_euro_lipschitz.layout import GRID_SIZE, tile_category_map


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_tile_centers_and_continuity():
    determinism = 0.0
    det_scale = 0.001
    mapping = tile_category_map(determinism)
    grid = tile_noise_multiplier_grid(mapping, det_scale)
    expected = {1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: det_scale}
    for tile, category in enumerate(mapping):
        iy, ix = divmod(tile, GRID_SIZE)
        state = ((ix + 0.5) / GRID_SIZE, (iy + 0.5) / GRID_SIZE)
        actual = interpolate_noise_multiplier(state, grid)
        check(abs(actual - expected[int(category)]) < 1e-12, f"tile-center mismatch at {tile}")
    for boundary in (0.25, 0.5, 0.75):
        for fixed in np.linspace(0.0, 1.0, 31):
            left = interpolate_noise_multiplier((boundary - 1e-9, fixed), grid)
            right = interpolate_noise_multiplier((boundary + 1e-9, fixed), grid)
            below = interpolate_noise_multiplier((fixed, boundary - 1e-9), grid)
            above = interpolate_noise_multiplier((fixed, boundary + 1e-9), grid)
            check(abs(left - right) < 1e-7, "x-boundary noise field is discontinuous")
            check(abs(below - above) < 1e-7, "y-boundary noise field is discontinuous")


def test_contractivity_and_gradient_bound():
    for sigma, determinism, det_scale, gamma in (
        (0.0005, 0.0, 0.001, 0.96),
        (0.0, 1.0, 0.001, 0.96),
        (0.0005, 0.5, 0.001, 0.96),
    ):
        metadata = smooth_noise_metadata(sigma, determinism, det_scale)
        check(gamma * metadata["wasserstein_kernel_lipschitz"] < 1.0, "kernel is not contractive")
        if sigma == 0.0:
            check(metadata["sigma_lipschitz"] == 0.0, "sigma-zero K_sigma must be zero")
            check(metadata["wasserstein_kernel_lipschitz"] == 1.0, "sigma-zero K_P must be one")
    grid = tile_noise_multiplier_grid(tile_category_map(0.0), 0.001)
    bound = noise_multiplier_lipschitz(grid)
    epsilon = 1e-6
    for x in np.linspace(0.13, 0.87, 29):
        for y in np.linspace(0.13, 0.87, 29):
            dx = (
                interpolate_noise_multiplier((x + epsilon, y), grid)
                - interpolate_noise_multiplier((x - epsilon, y), grid)
            ) / (2 * epsilon)
            dy = (
                interpolate_noise_multiplier((x, y + epsilon), grid)
                - interpolate_noise_multiplier((x, y - epsilon), grid)
            ) / (2 * epsilon)
            check(np.hypot(dx, dy) <= bound + 1e-7, "analytic K_sigma multiplier bound failed")


def test_common_mean_direct_clip_and_isotropic_noise():
    deterministic = ContinuousDollarEuroEnv(
        sigma=0.0, determinism=0.0, deterministic_sigma_scale=0.001,
        render_mode=None, auto_render=False, seed=0,
    )
    for state in (np.array([0.2, 0.2], dtype=np.float32), np.array([0.8, 0.8], dtype=np.float32)):
        deterministic.state = state.copy()
        deterministic._terminated = False
        next_state, _, _, _, info = deterministic.step(3)
        np.testing.assert_allclose(next_state - state, [0.04, 0.0], atol=2e-7)
        check(not info["boundary_cancel"], "boundary cancellation must be disabled")
    state = np.array([0.99, 0.5], dtype=np.float32)
    deterministic.state = state.copy()
    deterministic._terminated = False
    next_state, _, _, _, info = deterministic.step(3)
    np.testing.assert_allclose(next_state, [1.0, 0.5], atol=2e-7)
    check(info["boundary_clip"] and not info["boundary_cancel"], "direct clipping was not used")
    deterministic.close()

    stochastic = ContinuousDollarEuroEnv(
        sigma=0.0005, determinism=0.0, deterministic_sigma_scale=0.001,
        render_mode=None, auto_render=False, seed=7,
    )
    source = np.array([0.5, 0.35], dtype=np.float32)
    residuals = []
    for _ in range(4000):
        stochastic.state = source.copy()
        stochastic._terminated = False
        next_state, _, _, _, _ = stochastic.step(3)
        residuals.append(next_state.astype(np.float64) - source - np.array([0.04, 0.0]))
    covariance = np.cov(np.asarray(residuals), rowvar=False)
    check(abs(covariance[0, 1]) < 0.1 * np.mean(np.diag(covariance)), "noise is not approximately uncorrelated")
    check(abs(covariance[0, 0] / covariance[1, 1] - 1.0) < 0.1, "noise is not approximately isotropic")
    stochastic.close()


def main():
    test_tile_centers_and_continuity()
    test_contractivity_and_gradient_bound()
    test_common_mean_direct_clip_and_isotropic_noise()
    print("smooth-noise validation passed")


if __name__ == "__main__":
    main()

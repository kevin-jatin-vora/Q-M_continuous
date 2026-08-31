import time
from typing import Optional, Tuple, Dict, Any

from .layout import (
    CATEGORY_NAMES,
    DETERMINISTIC_CATEGORY,
    GRID_SIZE,
    N_CATEGORIES,
    N_TILES,
    category_from_state,
    layout_summary,
    tile_category_map,
    tile_id_from_xy,
)

import gymnasium as gym
from gymnasium import spaces
import numpy as np

"""
Continuous Dollar–Euro Environment with a 4x4 tile map.

Each of the 16 tiles is painted with one of five dynamics categories:
  1–4: stochastic templates from ContinuousDollarEuroEnv_radial.py (regions 1–4)
  5:   near-deterministic; noise std = deterministic_sigma_scale * sigma
       (scale 0 = exactly deterministic, 0.1 = 10x quieter than sigma, etc.)

``determinism`` (fraction in [0,1]) sets how many tiles are category 5; the rest
are split evenly among categories 1–4. At determinism=0 the 4×4 map matches
radial's four quadrants and category lookup uses the same continuous (x,y) rule.
The same determinism always yields the same tile map. Statistics / Lipschitz
pooling use the 5 categories, not tile ids.

Dynamics: s' = clip(s + Delta(category, a) + N(0, Sigma_category), 0, 1)
with boundary cancel-then-clip. Drift is stored but not applied (matches source).
"""


def _default_config_sigma() -> float:
    try:
        from .config import load_config

        return float(load_config()["environment"]["sigma"])
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return 0.00004


def _default_determinism() -> float:
    try:
        from .config import load_config

        return float(load_config()["environment"].get("determinism", 0.25))
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return 0.25


def _default_deterministic_sigma_scale() -> float:
    try:
        from .config import load_config

        return float(load_config()["environment"].get("deterministic_sigma_scale", 0.0))
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return 0.0


def _category_templates(base_var: float, deterministic_var: float = 0.0) -> Dict[int, Dict[str, Any]]:
    """Radial regions 1–4 plus category 5, whose variance is ``deterministic_var``."""
    return {
        1: {
            "name": CATEGORY_NAMES[1],
            "action_scale": np.array([0.96, 0.96], dtype=np.float32),
            "drift": np.array([0.0, 0.0], dtype=np.float32),
            "noise_cov": np.array([[1.0 * base_var, 0.0], [0.0, 1.0 * base_var]], dtype=np.float64),
        },
        2: {
            "name": CATEGORY_NAMES[2],
            "action_scale": np.array([0.90, 1.05], dtype=np.float32),
            "drift": np.array([-0.002, 0.001], dtype=np.float32),
            "noise_cov": np.array([[1.4 * base_var, 0.0], [0.0, 1.1 * base_var]], dtype=np.float64),
        },
        3: {
            "name": CATEGORY_NAMES[3],
            "action_scale": np.array([1.08, 0.88], dtype=np.float32),
            "drift": np.array([0.001, -0.002], dtype=np.float32),
            "noise_cov": np.array(
                [[1.2 * base_var, 0.2 * base_var], [0.2 * base_var, 1.5 * base_var]],
                dtype=np.float64,
            ),
        },
        4: {
            "name": CATEGORY_NAMES[4],
            "action_scale": np.array([1.00, 1.00], dtype=np.float32),
            "drift": np.array([0.0, 0.0], dtype=np.float32),
            "noise_cov": np.array([[0.9 * base_var, 0.0], [0.0, 1.6 * base_var]], dtype=np.float64),
        },
        5: {
            "name": CATEGORY_NAMES[5],
            "action_scale": np.array([1.00, 1.00], dtype=np.float32),
            "drift": np.array([0.0, 0.0], dtype=np.float32),
            "noise_cov": np.array(
                [[float(deterministic_var), 0.0], [0.0, float(deterministic_var)]],
                dtype=np.float64,
            ),
        },
    }


class ContinuousDollarEuroEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}

    def __init__(
        self,
        *,
        step_size: float = 0.04,
        sigma: Optional[float] = None,
        determinism: Optional[float] = None,
        deterministic_sigma_scale: Optional[float] = None,
        tau: float = 4.4,
        reward_alpha: float = 2.0,
        reward_beta: float = 1.2,
        horizon: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "human",
        auto_render: bool = True,
        fps: int = 60,
        substeps: int = 12,
    ):
        super().__init__()

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        w, h = 0.1, 0.1
        self.rect_dollar = np.array([0.1 - w / 2, 0.1 + w / 2, 0.1 - h / 2, 0.1 + h / 2], dtype=np.float32)
        self.rect_euro = np.array([0.9 - w / 2, 0.9 + w / 2, 0.1 - h / 2, 0.1 + h / 2], dtype=np.float32)
        self.rect_both = np.array([0.5 - w / 2, 0.5 + w / 2, 0.9 - h / 2, 0.9 + h / 2], dtype=np.float32)
        self.start_state = np.array([0.5, 0.35], dtype=np.float32)
        self.rbf_sigma = 0.1
        alpha = float(reward_alpha)
        beta = float(reward_beta)
        tau_val = float(tau)
        half_tau = 0.5 * tau_val
        if not (alpha > beta > 0.0):
            raise ValueError(
                f"reward peaks must satisfy alpha > beta > 0 (got alpha={alpha}, beta={beta})"
            )
        if not (2.0 * beta > alpha):
            raise ValueError(
                f"joint objective needs 2*beta > alpha so Both beats Dollar/Euro "
                f"(got alpha={alpha}, beta={beta})"
            )
        # Non-terminal reward is min(raw, 0). Keep raw peaks strictly negative so
        # no state sits on a zero-cost plateau around Dollar/Euro/Both.
        if not (alpha < half_tau and beta < half_tau):
            raise ValueError(
                f"need alpha < tau/2 and beta < tau/2 so non-terminal reward stays < 0 "
                f"(got alpha={alpha}, beta={beta}, tau/2={half_tau})"
            )
        # RBF weights at (Dollar, Both, Euro): R1 prefers Dollar, R2 prefers Euro,
        # R1+R2 prefers Both when 2*beta > alpha.
        self.reward_alpha = alpha
        self.reward_beta = beta
        self._a_coeffs = np.array([alpha, beta, 0.0], dtype=np.float64)
        self._b_coeffs = np.array([0.0, beta, alpha], dtype=np.float64)

        if sigma is None:
            sigma = _default_config_sigma()
        if determinism is None:
            determinism = _default_determinism()
        if deterministic_sigma_scale is None:
            deterministic_sigma_scale = _default_deterministic_sigma_scale()
        base_step = float(step_size)
        base_sigma = float(sigma)
        base_var = base_sigma ** 2
        det_scale = float(deterministic_sigma_scale)
        if det_scale < 0.0:
            raise ValueError(
                f"deterministic_sigma_scale must be >= 0 (got {det_scale}); "
                "0 means category 5 is exactly deterministic"
            )
        self.sigma = base_sigma
        self.deterministic_sigma_scale = det_scale
        self.deterministic_sigma = det_scale * base_sigma
        self.determinism = float(determinism)
        self.tile_map = tile_category_map(self.determinism)
        self.layout_info = layout_summary(self.determinism)

        categories = _category_templates(base_var, self.deterministic_sigma ** 2)
        self.terrain_config: Dict[str, Any] = {
            "grid_size": GRID_SIZE,
            "n_tiles": N_TILES,
            "determinism": self.determinism,
            "deterministic_sigma_scale": det_scale,
            "deterministic_sigma": self.deterministic_sigma,
            "tile_map": self.tile_map.tolist(),
            "layout": self.layout_info,
            "dynamics": {
                "base_step_size": base_step,
                "boundary_mode": "cancel_then_clip",
                "action_vectors": np.array(
                    [
                        [0.0, -1.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            },
            # Keyed by category id 1..5 (not by the 16 tile ids).
            "regions": categories,
        }

        self.tau = float(tau)
        self.horizon = int(horizon)
        self.render_mode = render_mode
        self.auto_render = bool(auto_render)
        self.fps = int(fps)
        self.substeps = int(substeps)

        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.state: np.ndarray = self.start_state.copy()
        self._t = 0
        self._terminated = False
        self._renderer: Optional[_SimpleRenderer] = None
        self._cache_vectorized_tables()

    def seed(self, seed: Optional[int] = None):
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self.state = self.start_state.copy()
        self._t = 0
        self._terminated = False
        info = {"terminal": "none", "determinism": self.determinism}

        if self.render_mode == "human" and self.auto_render:
            if self._renderer is None:
                self._renderer = _SimpleRenderer(self)
                self._renderer.draw_static()
            self._renderer.update(self.state)

        return self._obs(), info

    def _cache_vectorized_tables(self):
        regions = self.terrain_config["regions"]
        self._base_step = float(self.terrain_config["dynamics"]["base_step_size"])
        self._action_vectors = np.asarray(
            self.terrain_config["dynamics"]["action_vectors"], dtype=np.float32
        )
        # Index by category id 1..N_CATEGORIES (slot 0 unused).
        table_len = N_CATEGORIES + 1
        self._action_scale = np.zeros((table_len, 2), dtype=np.float32)
        self._chol = np.zeros((table_len, 2, 2), dtype=np.float64)
        for region_id, spec in regions.items():
            self._action_scale[int(region_id)] = np.asarray(spec["action_scale"], dtype=np.float32)
            cov = np.asarray(spec["noise_cov"], dtype=np.float64)
            if np.all(cov == 0.0):
                self._chol[int(region_id)] = 0.0
            else:
                self._chol[int(region_id)] = np.linalg.cholesky(cov)
        self._rbf_centers = np.array(
            [
                [(self.rect_dollar[0] + self.rect_dollar[1]) * 0.5, (self.rect_dollar[2] + self.rect_dollar[3]) * 0.5],
                [(self.rect_both[0] + self.rect_both[1]) * 0.5, (self.rect_both[2] + self.rect_both[3]) * 0.5],
                [(self.rect_euro[0] + self.rect_euro[1]) * 0.5, (self.rect_euro[2] + self.rect_euro[3]) * 0.5],
            ],
            dtype=np.float64,
        )
        self._rbf_den = 2.0 * (self.rbf_sigma ** 2)
        self._half_tau = 0.5 * float(self.tau)
        self._term_rects = np.stack([self.rect_dollar, self.rect_euro, self.rect_both], axis=0)
        self._term_names = ("dollar", "euro", "both")
        self._zero_reward = np.zeros(2, dtype=np.float32)

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        if self._terminated:
            return self._obs(), self._zero_reward.copy(), True, False, {"terminal": "absorbing"}

        prev_state = self.state
        region_id = self._region_id(prev_state)
        tile_id = tile_id_from_xy(float(prev_state[0]), float(prev_state[1]))
        delta = self._action_vectors[action] * self._base_step * self._action_scale[region_id]
        proposed = prev_state + delta
        boundary_cancel = bool(
            proposed[0] < 0.0
            or proposed[0] > 1.0
            or proposed[1] < 0.0
            or proposed[1] > 1.0
        )
        if boundary_cancel:
            proposed = prev_state

        noise = self._chol[region_id] @ self._np_random.standard_normal(2)
        noisy = proposed + noise.astype(np.float32)
        next_state = np.clip(noisy, 0.0, 1.0)
        boundary_clip = bool(np.any(np.abs(next_state.astype(np.float64) - noisy.astype(np.float64)) > 1e-8))
        self.state = next_state

        terminal_type = self._which_terminal(next_state)
        terminated = terminal_type != "none"

        diff = next_state.astype(np.float64) - self._rbf_centers
        kernels = np.exp(-np.sum(diff * diff, axis=1) / self._rbf_den)
        r1 = float(kernels @ self._a_coeffs) - self._half_tau
        r2 = float(kernels @ self._b_coeffs) - self._half_tau
        if not terminated:
            r1 = min(r1, 0.0)
            r2 = min(r2, 0.0)
        reward_vec = np.array([r1, r2], dtype=np.float32)

        self._t += 1
        truncated = (self._t >= self.horizon) and not terminated
        self._terminated = terminated or truncated
        info = {
            "terminal": terminal_type,
            "region": region_id,
            "category": region_id,
            "tile": tile_id,
            "determinism": self.determinism,
            "boundary_cancel": boundary_cancel,
            "boundary_clip": boundary_clip,
            "boundary_affected": bool(boundary_cancel or boundary_clip),
        }

        if self.render_mode == "human" and self.auto_render:
            if self._renderer is None:
                self._renderer = _SimpleRenderer(self)
                self._renderer.draw_static()
            for i in range(1, self.substeps + 1):
                alpha = i / self.substeps
                pos = (1 - alpha) * prev_state + alpha * self.state
                self._renderer.update(pos)
                time.sleep(1.0 / max(1, self.fps))

        return self._obs(), reward_vec, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            if self.render_mode in ("human", "rgb_array"):
                self._renderer = _SimpleRenderer(self)
                self._renderer.draw_static()
        if self.render_mode == "human":
            self._renderer.update(self.state)
        elif self.render_mode == "rgb_array":
            return self._renderer.as_rgb()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _obs(self):
        return self.state.astype(np.float32, copy=False)

    def _delta(self, action: int, terrain: Dict[str, Any]) -> Tuple[float, float]:
        action_vec = self._action_vectors[int(action)]
        scale = np.asarray(terrain["action_scale"], dtype=np.float32)
        scaled = action_vec * self._base_step * scale
        return float(scaled[0]), float(scaled[1])

    def _region_id(self, p: np.ndarray) -> int:
        return category_from_state(p, self.determinism, tile_map=self.tile_map)

    def _terrain_params(self, region_id: int) -> Dict[str, Any]:
        return self.terrain_config["regions"][region_id]

    @staticmethod
    def _inside_unit_square(p: np.ndarray) -> bool:
        return (0.0 <= p[0] <= 1.0) and (0.0 <= p[1] <= 1.0)

    def _which_terminal(self, p: np.ndarray) -> str:
        x, y = float(p[0]), float(p[1])
        eps = 1e-7
        rects = self._term_rects
        inside = (
            (rects[:, 0] - eps <= x)
            & (x <= rects[:, 1] + eps)
            & (rects[:, 2] - eps <= y)
            & (y <= rects[:, 3] + eps)
        )
        hit = np.flatnonzero(inside)
        if hit.size:
            return self._term_names[int(hit[0])]
        return "none"


class ScalarizeReward(gym.Wrapper):
    def __init__(self, env: gym.Env, weights: Tuple[float, float] = (1.0, 1.0)):
        super().__init__(env)
        self.w = np.asarray(weights, dtype=np.float32)
        assert self.w.shape == (2,), "weights must be length-2"

    def step(self, action):
        obs, r_vec, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["reward_vec"] = np.asarray(r_vec, dtype=np.float32)
        scalar_r = float(np.dot(self.w, info["reward_vec"]))
        return obs, scalar_r, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class _SimpleRenderer:
    _CAT_COLORS = {
        1: (0.75, 0.85, 1.0, 0.35),
        2: (0.85, 0.75, 1.0, 0.35),
        3: (1.0, 0.85, 0.70, 0.35),
        4: (1.0, 0.75, 0.75, 0.35),
        5: (0.80, 0.90, 0.80, 0.35),
    }

    def __init__(self, env: ContinuousDollarEuroEnv):
        import matplotlib

        if env.render_mode == "rgb_array":
            matplotlib.use("Agg", force=True)
        else:
            try:
                matplotlib.use("Qt5Agg", force=True)
            except Exception:
                matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        self.env = env
        self.plt = plt
        self.Rectangle = Rectangle
        self.Circle = Circle
        self.headless = env.render_mode == "rgb_array"

        if not self.headless:
            self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"16-tile Dollar–Euro (determinism={env.determinism:.0%})"
        )
        self.ax.grid(True, alpha=0.2)
        self.agent = None

    def draw_static(self):
        cell = 1.0 / GRID_SIZE
        for tid, cat in enumerate(self.env.tile_map.tolist()):
            iy, ix = divmod(int(tid), GRID_SIZE)
            # row-major with iy increasing in +y
            x0 = ix * cell
            y0 = iy * cell
            color = self._CAT_COLORS.get(int(cat), (0.9, 0.9, 0.9, 0.3))
            self.ax.add_patch(
                self.Rectangle((x0, y0), cell, cell, facecolor=color, edgecolor="0.5", linewidth=0.6)
            )
            self.ax.text(
                x0 + 0.5 * cell,
                y0 + 0.5 * cell,
                f"C{int(cat)}",
                ha="center",
                va="center",
                fontsize=8,
                color="0.25",
            )

        def add_rect(rect, label):
            x0, x1, y0, y1 = rect
            self.ax.add_patch(self.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.5))
            self.ax.text((x0 + x1) * 0.5, (y0 + y1) * 0.5, label, ha="center", va="center", fontsize=9)

        add_rect(self.env.rect_dollar, "Dollar")
        add_rect(self.env.rect_euro, "Euro")
        add_rect(self.env.rect_both, "Both")

        if self.agent is None:
            self.agent = self.Circle((self.env.start_state[0], self.env.start_state[1]), 0.015, color="crimson")
            self.ax.add_patch(self.agent)

        self.fig.canvas.draw()
        if not self.headless:
            self.fig.canvas.flush_events()

    def update(self, pos):
        self.agent.center = (float(pos[0]), float(pos[1]))
        if self.headless:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
            self.plt.pause(1e-6)

    def as_rgb(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def close(self):
        self.plt.close(self.fig)

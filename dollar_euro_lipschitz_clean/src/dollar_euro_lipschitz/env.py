import time
from typing import Optional, Tuple, Dict, Any

from .bounds import REGION_CENTER

import gymnasium as gym
from gymnasium import spaces
import numpy as np

"""
Continuous Dollar–Euro Environment with piecewise Gaussian terrain dynamics.

- State: (x, y) ∈ [0,1]^2  (float32)
- Actions (Discrete(4), in this order): 0=Down, 1=Up, 2=Left, 3=Right
- Terrain: the unit square is split into 4 regions using the center (0.5, 0.5).
  Region is chosen from the CURRENT state only.
- Dynamics: region-specific Gaussian transition model.
  s' = clip(s + mean_action_offset(region, a) + N(0, Σ_region), 0, 1)
  Regions with y >= 0.5 (top_left / top_right) are deterministic: Σ = 0.
  Bottom regions (y < 0.5) keep scaled Gaussian noise from ``sigma``.
  Out-of-bounds proposals are cancelled before noise; ``info`` reports
  ``boundary_cancel`` / ``boundary_clip`` / ``boundary_affected``.
  Terrain drift vectors are stored in terrain_config but are not added in
  step(), matching the original uploaded environment.
- Reward design:
    * R1 and R2 are linear combinations of Gaussian RBF kernels (Dollar, Both, Euro).
    * Living penalty (half_tau = tau/2) is subtracted everywhere (including terminals)
      to remove any jump at the terminal boundary.
    * On non-terminal steps we additionally clamp any tiny positive values to 0
      to enforce the "no positive non-terminal reward" requirement.
- Returns a 2D reward vector; use ScalarizeReward to train scalar-reward agents.

All terrain parameters live in self.terrain_config so future changes are localized.
"""


def _default_config_sigma() -> float:
    try:
        from .config import load_config

        return float(load_config()["environment"]["sigma"])
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return 0.00004


class ContinuousDollarEuroEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}

    def __init__(
        self,
        *,
        step_size: float = 0.04,
        sigma: Optional[float] = None,
        tau: float = 2.4,
        horizon: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "human",
        auto_render: bool = True,
        fps: int = 60,
        substeps: int = 12,
    ):
        super().__init__()

        # Spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # Layout: Dollar (L-B), Euro (R-B), Both (T-C)
        w, h = 0.1, 0.1
        self.rect_dollar = np.array([0.1 - w / 2, 0.1 + w / 2, 0.1 - h / 2, 0.1 + h / 2], dtype=np.float32)
        self.rect_euro = np.array([0.9 - w / 2, 0.9 + w / 2, 0.1 - h / 2, 0.1 + h / 2], dtype=np.float32)
        self.rect_both = np.array([0.5 - w / 2, 0.5 + w / 2, 0.9 - h / 2, 0.9 + h / 2], dtype=np.float32)

        # Start state
        self.start_state = np.array([0.5, 0.35], dtype=np.float32)

        # RBF radius (separate from dynamics noise)
        self.rbf_sigma = 0.1

        # RBF coefficients (ordering: [Dollar, Both, Euro])
        self._a_coeffs = np.array([1.0, 0.6, 0.0], dtype=np.float64)
        self._b_coeffs = np.array([0.0, 0.6, 1.0], dtype=np.float64)

        # Single terrain config block for later edits.
        if sigma is None:
            sigma = _default_config_sigma()
        base_step = float(step_size)
        base_sigma = float(sigma)
        base_var = base_sigma ** 2
        self.sigma = base_sigma
        self.terrain_config: Dict[str, Any] = {
            "center": np.array(REGION_CENTER, dtype=np.float32),
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
            "regions": {
                # y >= 0.5: deterministic (zero process noise)
                1: {
                    "name": "top_right",
                    "action_scale": np.array([0.96, 0.96], dtype=np.float32),#np.array([1.00, 1.00], dtype=np.float32),
                    "drift": np.array([0.0, 0.0], dtype=np.float32),
                    "noise_cov": np.zeros((2, 2), dtype=np.float64),
                },
                2: {
                    "name": "top_left",
                    "action_scale": np.array([0.90, 1.05], dtype=np.float32),
                    "drift": np.array([-0.002, 0.001], dtype=np.float32),
                    "noise_cov": np.zeros((2, 2), dtype=np.float64),
                },
                # y < 0.5: stochastic; covariance scales with sigma^2
                3: {
                    "name": "bottom_left",
                    "action_scale": np.array([1.08, 0.88], dtype=np.float32),
                    "drift": np.array([0.001, -0.002], dtype=np.float32),
                    "noise_cov": np.array([[1.2 * base_var, 0.2 * base_var], [0.2 * base_var, 1.5 * base_var]], dtype=np.float64),
                },
                4: {
                    "name": "bottom_right",
                    "action_scale": np.array([1.00, 1.00], dtype=np.float32), #np.array([0.96, 0.96], dtype=np.float32),
                    "drift": np.array([0.0, 0.0], dtype=np.float32),
                    "noise_cov": np.array([[0.9 * base_var, 0.0], [0.0, 1.6 * base_var]], dtype=np.float64),
                },
            },
        }

        # Living penalty / episode config
        self.tau = float(tau)
        self.horizon = int(horizon)
        self.render_mode = render_mode
        self.auto_render = bool(auto_render)
        self.fps = int(fps)
        self.substeps = int(substeps)

        # RNG
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Episode vars
        self.state: np.ndarray = self.start_state.copy()
        self._t = 0
        self._terminated = False

        # Renderer
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
        info = {"terminal": "none"}

        if self.render_mode == "human" and self.auto_render:
            if self._renderer is None:
                self._renderer = _SimpleRenderer(self)
                self._renderer.draw_static()
            self._renderer.update(self.state)

        return self._obs(), info

    def _cache_vectorized_tables(self):
        regions = self.terrain_config["regions"]
        self._center_xy = np.asarray(self.terrain_config["center"], dtype=np.float32)
        self._base_step = float(self.terrain_config["dynamics"]["base_step_size"])
        self._action_vectors = np.asarray(
            self.terrain_config["dynamics"]["action_vectors"], dtype=np.float32
        )
        self._action_scale = np.zeros((5, 2), dtype=np.float32)
        self._chol = np.zeros((5, 2, 2), dtype=np.float64)
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
        x, y = float(p[0]), float(p[1])
        cx, cy = float(self._center_xy[0]), float(self._center_xy[1])
        if y >= cy:
            return 1 if x >= cx else 2
        return 4 if x >= cx else 3

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
    """Map vector reward (R1, R2) → scalar w · R. Keeps vector in info['reward_vec']."""

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
    def __init__(self, env: ContinuousDollarEuroEnv):
        import matplotlib

        # Headless rgb_array recording must use Agg; human mode may use an interactive backend.
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
        self.ax.set_title("Continuous Dollar–Euro (Piecewise Gaussian Terrain)")
        self.ax.grid(True, alpha=0.2)

        self.agent = None

    def draw_static(self):
        def add_rect(rect, label):
            x0, x1, y0, y1 = rect
            self.ax.add_patch(self.Rectangle((x0, y0), x1 - x0, y1 - y0, alpha=0.2))
            self.ax.text((x0 + x1) * 0.5, (y0 + y1) * 0.5, label, ha="center", va="center")

        add_rect(self.env.rect_dollar, "Dollar")
        add_rect(self.env.rect_euro, "Euro")
        add_rect(self.env.rect_both, "Both")

        if self.agent is None:
            self.agent = self.Circle((self.env.start_state[0], self.env.start_state[1]), 0.015)
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
        self.update(self.env.state)
        canvas = self.fig.canvas
        width, height = canvas.get_width_height()
        if hasattr(canvas, "tostring_rgb"):
            img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            return img.reshape((height, width, 3)).copy()
        # Matplotlib >= 3.9
        rgba = np.asarray(canvas.buffer_rgba())
        return np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass

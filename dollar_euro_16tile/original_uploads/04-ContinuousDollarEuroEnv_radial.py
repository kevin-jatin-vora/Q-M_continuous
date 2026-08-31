import time
from typing import Optional, Tuple, Dict, Any

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
  s' = clip(s + mean_action_offset(region, a) + drift(region) + N(0, Σ_region), 0, 1)
- Reward design:
    * R1 and R2 are linear combinations of Gaussian RBF kernels (Dollar, Both, Euro).
    * Living penalty (half_tau = tau/2) is subtracted everywhere (including terminals)
      to remove any jump at the terminal boundary.
    * On non-terminal steps we additionally clamp any tiny positive values to 0
      to enforce the "no positive non-terminal reward" requirement.
- Returns a 2D reward vector; use ScalarizeReward to train scalar-reward agents.

All terrain parameters live in self.terrain_config so future changes are localized.
"""


class ContinuousDollarEuroEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}

    def __init__(
        self,
        *,
        step_size: float = 0.04,
        sigma: float = 0.00008,
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
        base_step = float(step_size)
        base_sigma = float(sigma)
        base_var = base_sigma ** 2
        self.terrain_config: Dict[str, Any] = {
            "center": np.array([0.5, 0.5], dtype=np.float32),
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
                1: {
                    "name": "top_right",
                    "action_scale": np.array([0.96, 0.96], dtype=np.float32),#np.array([1.00, 1.00], dtype=np.float32),
                    "drift": np.array([0.0, 0.0], dtype=np.float32),
                    "noise_cov": np.array([[1.0 * base_var, 0.0], [0.0, 1.0 * base_var]], dtype=np.float64),
                },
                2: {
                    "name": "top_left",
                    "action_scale": np.array([0.90, 1.05], dtype=np.float32),
                    "drift": np.array([-0.002, 0.001], dtype=np.float32),
                    "noise_cov": np.array([[1.4 * base_var, 0.0], [0.0, 1.1 * base_var]], dtype=np.float64),
                },
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

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        if self._terminated:
            return self._obs(), np.array([0.0, 0.0], dtype=np.float32), True, False, {"terminal": "absorbing"}

        prev_state = self.state.copy()
        region_id = self._region_id(prev_state)
        terrain = self._terrain_params(region_id)

        # Region-specific deterministic proposal.
        dx, dy = self._delta(action, terrain)
        proposed = prev_state + np.array([dx, dy], dtype=np.float32)
        if not self._inside_unit_square(proposed):
            proposed = prev_state

        # Region-specific Gaussian transition noise.
        noise = self._np_random.multivariate_normal(
            mean=np.zeros(2, dtype=np.float64),
            cov=np.asarray(terrain["noise_cov"], dtype=np.float64),
        ).astype(np.float32)
        next_state = np.clip(proposed + noise, 0.0, 1.0)
        self.state = next_state

        # Terminal detection
        terminal_type = self._which_terminal(self.state)
        terminated = terminal_type != "none"

        # ---------- Reward computation (RBF fields + living penalty) ----------
        def rect_center(rect):
            x0, x1, y0, y1 = rect
            return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)

        c_d = rect_center(self.rect_dollar)
        c_b = rect_center(self.rect_both)
        c_e = rect_center(self.rect_euro)

        x = self.state.astype(np.float64)

        def k(center, x):
            d2 = float(np.sum((x - center) ** 2))
            return float(np.exp(-d2 / (2.0 * (self.rbf_sigma**2))))

        kd = k(c_d, x)
        kb = k(c_b, x)
        ke = k(c_e, x)

        r1_field = float(self._a_coeffs[0] * kd + self._a_coeffs[1] * kb + self._a_coeffs[2] * ke)
        r2_field = float(self._b_coeffs[0] * kd + self._b_coeffs[1] * kb + self._b_coeffs[2] * ke)

        # Living penalty everywhere.
        half_tau = 0.5 * float(self.tau)
        r1 = r1_field - half_tau
        r2 = r2_field - half_tau

        # Strict enforcement: no non-terminal component should be positive.
        if not terminated:
            if r1 > 0.0:
                r1 = 0.0
            if r2 > 0.0:
                r2 = 0.0

        reward_vec = np.array([r1, r2], dtype=np.float32)

        self._t += 1
        truncated = (self._t >= self.horizon) and not terminated
        self._terminated = terminated or truncated

        info = {"terminal": terminal_type, "region": region_id}

        # Auto-render
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
        return self.state.astype(np.float32)

    def _delta(self, action: int, terrain: Dict[str, Any]) -> Tuple[float, float]:
        action_vec = np.asarray(self.terrain_config["dynamics"]["action_vectors"], dtype=np.float32)[action]
        base_step = float(self.terrain_config["dynamics"]["base_step_size"])
        scale = np.asarray(terrain["action_scale"], dtype=np.float32)
        scaled = action_vec * base_step * scale
        return float(scaled[0]), float(scaled[1])

    def _region_id(self, p: np.ndarray) -> int:
        center = np.asarray(self.terrain_config["center"], dtype=np.float32)
        x, y = float(p[0]), float(p[1])
        cx, cy = float(center[0]), float(center[1])

        # 1 = top-right, 2 = top-left, 3 = bottom-left, 4 = bottom-right.
        if x >= cx and y >= cy:
            return 1
        if x < cx and y >= cy:
            return 2
        if x < cx and y < cy:
            return 3
        return 4

    def _terrain_params(self, region_id: int) -> Dict[str, Any]:
        return self.terrain_config["regions"][region_id]

    @staticmethod
    def _inside_unit_square(p: np.ndarray) -> bool:
        return (0.0 <= p[0] <= 1.0) and (0.0 <= p[1] <= 1.0)

    def _which_terminal(self, p: np.ndarray) -> str:
        x, y = float(p[0]), float(p[1])
        eps = 1e-7

        def in_rect(rect):
            x0, x1, y0, y1 = rect
            return (x0 - eps <= x <= x1 + eps) and (y0 - eps <= y <= y1 + eps)

        if in_rect(self.rect_dollar):
            return "dollar"
        if in_rect(self.rect_euro):
            return "euro"
        if in_rect(self.rect_both):
            return "both"
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

        try:
            matplotlib.use("Qt5Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        self.env = env
        self.plt = plt
        self.Rectangle = Rectangle
        self.Circle = Circle

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
        self.fig.canvas.flush_events()

    def update(self, pos):
        self.agent.center = (float(pos[0]), float(pos[1]))
        self.fig.canvas.draw_idle()
        self.plt.pause(1e-6)

    def as_rgb(self):
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        return img.reshape((h, w, 3))

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass

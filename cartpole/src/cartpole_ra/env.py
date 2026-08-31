"""Gymnasium CartPole wrapper with selectable reward modes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .rewards import DEFAULT_ANGLE_THRESHOLD, DEFAULT_KAPPA, shaped_reward

STATE_DIM = 4
ACTION_DIM = 2
STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")


class CartPoleRAEnv:
    """Thin wrapper: classic dynamics, shaped or classic reward."""

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        reward_mode: str = "classic",
        angle_threshold_rad: float = DEFAULT_ANGLE_THRESHOLD,
        reward_shape: str = "step",
        reward_kappa: float = DEFAULT_KAPPA,
        max_episode_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.env_id = str(env_id)
        self.reward_mode = str(reward_mode)
        self.angle_threshold_rad = float(angle_threshold_rad)
        self.reward_shape = str(reward_shape)
        self.reward_kappa = float(reward_kappa)
        self._seed = seed
        self.env = gym.make(self.env_id, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.max_episode_steps = max_episode_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        reset_seed = self._seed if seed is None else seed
        if reset_seed is not None:
            obs, info = self.env.reset(seed=int(reset_seed))
        else:
            obs, info = self.env.reset(options=options)
        self._seed = None
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, env_r, terminated, truncated, info = self.env.step(int(action))
        state = np.asarray(obs, dtype=np.float32)
        reward = shaped_reward(
            state,
            self.reward_mode,
            angle_threshold_rad=self.angle_threshold_rad,
            env_reward=float(env_r),
            shape=self.reward_shape,
            kappa=self.reward_kappa,
        )
        info = dict(info)
        info["env_reward"] = float(env_r)
        info["reward_mode"] = self.reward_mode
        return state, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(
    reward_mode: str = "classic",
    config: Optional[dict] = None,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    angle_threshold_rad: Optional[float] = None,
    reward_shape: Optional[str] = None,
    reward_kappa: Optional[float] = None,
    render_mode: Optional[str] = None,
) -> CartPoleRAEnv:
    env_cfg = {}
    if config:
        env_cfg = config.get("environment", config)
    thr = angle_threshold_rad
    if thr is None:
        thr = float(env_cfg.get("angle_threshold_rad", env_cfg.get("angle_thresh", DEFAULT_ANGLE_THRESHOLD)))
    shape = reward_shape if reward_shape is not None else str(env_cfg.get("reward_shape", "step"))
    kappa = reward_kappa if reward_kappa is not None else float(env_cfg.get("reward_kappa", DEFAULT_KAPPA))
    return CartPoleRAEnv(
        env_id=str(env_cfg.get("env_id", "CartPole-v1")),
        reward_mode=reward_mode,
        angle_threshold_rad=float(thr),
        reward_shape=shape,
        reward_kappa=float(kappa),
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        seed=seed,
    )

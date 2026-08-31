"""Behavior and task rewards for CartPole (B1 / B2 / classic).

B1/B2 encode the same *preference* as the original offline angle-threshold
shaping (threshold ≈ 0.174 rad ≈ 10°), in two shapes:

  step  — hard threshold (original): ±20 outside the band, 0 inside, then
          shifted by +0.5 and scaled by 1/20. Discontinuous at ±thr, so no
          finite reward Lipschitz constant Lr exists.
  tanh  — smooth surrogate with the same sign and saturation:
          R = (0.5 - 20*tanh(theta/kappa)) / 20   for B1 (sign flipped for B2)
          Continuous with bounded derivative, so Lr is finite:
          |dR/dtheta| <= 1 / kappa.

The behavior *objective* is unchanged (B1 still prefers negative pole angle,
B2 positive); only the grading between "inside" and "outside" the band becomes
gradual. The final DQN / RA-DQN agent optimizes classic CartPole survival.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ANGLE_THRESHOLD = 0.174
DEFAULT_KAPPA = 0.174
STEP_MAGNITUDE = 20.0
BASELINE_BONUS = 0.5


def _theta(state) -> float:
    return float(np.asarray(state, dtype=np.float64).reshape(-1)[2])


def _step_signal(theta: float, thr: float) -> float:
    """+1 when theta < -thr, -1 when theta > +thr, 0 inside the band."""
    if theta > thr:
        return -1.0
    if theta < -thr:
        return 1.0
    return 0.0


def _tanh_signal(theta: float, kappa: float) -> float:
    """Smooth version of _step_signal: -tanh(theta/kappa) in (-1, 1)."""
    return float(-np.tanh(theta / max(float(kappa), 1e-8)))


def behavior_reward(
    state,
    sign: float,
    angle_threshold_rad: float = DEFAULT_ANGLE_THRESHOLD,
    shape: str = "step",
    kappa: float = DEFAULT_KAPPA,
) -> float:
    """Angle-shaped behavior reward. ``sign=+1`` is B1, ``sign=-1`` is B2."""
    theta = _theta(state)
    shape = str(shape).lower()
    if shape in {"step", "threshold", "hard"}:
        signal = _step_signal(theta, float(angle_threshold_rad))
    elif shape in {"tanh", "smooth"}:
        signal = _tanh_signal(theta, kappa)
    else:
        raise ValueError(f"unknown reward shape: {shape}")
    raw = STEP_MAGNITUDE * float(sign) * signal
    return float((raw + BASELINE_BONUS) / STEP_MAGNITUDE)


def b1_reward(state, angle_threshold_rad=DEFAULT_ANGLE_THRESHOLD, shape="step", kappa=DEFAULT_KAPPA) -> float:
    """Favors negative pole angle."""
    return behavior_reward(state, +1.0, angle_threshold_rad, shape, kappa)


def b2_reward(state, angle_threshold_rad=DEFAULT_ANGLE_THRESHOLD, shape="step", kappa=DEFAULT_KAPPA) -> float:
    """Favors positive pole angle."""
    return behavior_reward(state, -1.0, angle_threshold_rad, shape, kappa)


def classic_reward(_state=None, env_reward: float = 1.0) -> float:
    """Standard CartPole step reward (survival)."""
    return float(env_reward)


def reward_lipschitz_bound(shape: str, kappa: float = DEFAULT_KAPPA) -> float:
    """Analytic bound on |dR/dtheta| (inf for the discontinuous step shape)."""
    shape = str(shape).lower()
    if shape in {"tanh", "smooth"}:
        return float(1.0 / max(float(kappa), 1e-8))
    return float("inf")


def shaped_reward(
    state,
    mode: str,
    angle_threshold_rad: float = DEFAULT_ANGLE_THRESHOLD,
    env_reward: float = 1.0,
    shape: str = "step",
    kappa: float = DEFAULT_KAPPA,
) -> float:
    mode = str(mode).lower()
    if mode in {"b1", "r1", "behavior1"}:
        return b1_reward(state, angle_threshold_rad, shape, kappa)
    if mode in {"b2", "r2", "behavior2"}:
        return b2_reward(state, angle_threshold_rad, shape, kappa)
    if mode in {"classic", "task", "combined"}:
        return classic_reward(state, env_reward)
    raise ValueError(f"unknown reward mode: {mode}")

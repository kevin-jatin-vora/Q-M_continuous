import numpy as np

from .env import ContinuousDollarEuroEnv


def scalar_reward_from_next_states(next_states, env: ContinuousDollarEuroEnv):
    if env is None:
        raise TypeError("scalar_reward_from_next_states requires the experiment environment")
    states = np.asarray(next_states, dtype=np.float64).reshape(-1, 2)
    centers = getattr(env, "_rbf_centers", None)
    if centers is None:
        def center(rect):
            x0, x1, y0, y1 = rect
            return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)

        centers = np.stack(
            [center(env.rect_dollar), center(env.rect_both), center(env.rect_euro)],
            axis=0,
        )
    diff = states[:, None, :] - centers[None, :, :]
    kernels = np.exp(-np.sum(diff * diff, axis=2) / (2.0 * env.rbf_sigma ** 2))
    r1 = kernels @ env._a_coeffs - 0.5 * env.tau
    r2 = kernels @ env._b_coeffs - 0.5 * env.tau

    eps = 1e-7
    rects = getattr(env, "_term_rects", np.stack([env.rect_dollar, env.rect_euro, env.rect_both], axis=0))
    terminal = (
        (states[:, 0:1] >= rects[:, 0] - eps)
        & (states[:, 0:1] <= rects[:, 1] + eps)
        & (states[:, 1:2] >= rects[:, 2] - eps)
        & (states[:, 1:2] <= rects[:, 3] + eps)
    ).any(axis=1)
    nonterminal = ~terminal
    r1[nonterminal] = np.minimum(r1[nonterminal], 0.0)
    r2[nonterminal] = np.minimum(r2[nonterminal], 0.0)
    return (r1 + r2).astype(np.float32), terminal.astype(np.float32)

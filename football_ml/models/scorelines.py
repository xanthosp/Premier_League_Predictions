"""Expected goals -> scoreline matrix -> 1X2 probabilities, with the optional Dixon-Coles low-score correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10


def score_matrix(lam_home, lam_away, rho: float = 0.0) -> np.ndarray:
    """(n, MAX_GOALS + 1, MAX_GOALS + 1) probabilities of each scoreline, renormalised after truncation."""
    goals = np.arange(MAX_GOALS + 1)
    lam_home, lam_away = np.asarray(lam_home, float)[:, None], np.asarray(lam_away, float)[:, None]
    matrix = poisson.pmf(goals, lam_home)[:, :, None] * poisson.pmf(goals, lam_away)[:, None, :]
    if rho:
        lh, la = lam_home[:, 0], lam_away[:, 0]
        matrix[:, 0, 0] *= 1 - lh * la * rho
        matrix[:, 0, 1] *= 1 + lh * rho
        matrix[:, 1, 0] *= 1 + la * rho
        matrix[:, 1, 1] *= 1 - rho
    matrix = np.clip(matrix, 0, None)
    return matrix / matrix.sum(axis=(1, 2), keepdims=True)


def outcome_probabilities(lam_home, lam_away, rho: float = 0.0) -> np.ndarray:
    """(n, 3) H/D/A probabilities."""
    m = score_matrix(lam_home, lam_away, rho)
    home = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)  # row = home goals > column = away goals
    p_home = (m * home).sum(axis=(1, 2))
    p_draw = np.trace(m, axis1=1, axis2=2)
    return np.column_stack([p_home, p_draw, 1 - p_home - p_draw])

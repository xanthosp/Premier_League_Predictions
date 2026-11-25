"""Scoring rules for 1X2 probabilities (columns ordered H, D, A) and the market blend test."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
EPS = 1e-12


def one_hot(outcomes) -> np.ndarray:
    index = np.array([OUTCOME_INDEX[o] for o in outcomes])
    out = np.zeros((len(index), 3))
    out[np.arange(len(index)), index] = 1.0
    return out


def log_losses(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match negative log likelihood of the observed outcome."""
    return -np.log(np.clip((p * y).sum(axis=1), EPS, 1.0))


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match ranked probability score for the ordered outcomes H < D < A."""
    return (((np.cumsum(p, axis=1) - np.cumsum(y, axis=1))[:, :2]) ** 2).sum(axis=1) / 2


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - y) ** 2).sum(axis=1)


def calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over all three outcomes: weighted mean |predicted - observed| per probability bin."""
    flat_p, flat_y = p.ravel(), y.ravel()
    which = np.minimum((flat_p * bins).astype(int), bins - 1)
    error = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            error += mask.sum() * abs(flat_p[mask].mean() - flat_y[mask].mean())
    return error / len(flat_p)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    flat_p, flat_y = p.ravel(), y.ravel()
    which = np.minimum((flat_p * bins).astype(int), bins - 1)
    return [{"bin": b / bins, "n": int((which == b).sum()), "predicted": float(flat_p[which == b].mean()),
             "observed": float(flat_y[which == b].mean())} for b in range(bins) if (which == b).any()]


def _blend(params, log_market, log_model):
    score = params[0] * log_market + params[1] * log_model
    score -= score.max(axis=1, keepdims=True)
    return score - np.log(np.exp(score).sum(axis=1, keepdims=True))


def blend_test(market: np.ndarray, model: np.ndarray, y: np.ndarray) -> dict:
    """Fit p ∝ market^a * model^b by maximum likelihood. b > 0 (beyond its standard error) means the model adds
    information the market does not have. Standard errors come from the numerical Hessian."""
    log_market, log_model = np.log(np.clip(market, EPS, 1)), np.log(np.clip(model, EPS, 1))

    def nll(params):
        return -(_blend(params, log_market, log_model) * y).sum()

    fit = minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
    h, hessian = 1e-4, np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            di, dj = np.eye(2)[i] * h, np.eye(2)[j] * h
            hessian[i, j] = (nll(fit.x + di + dj) - nll(fit.x + di - dj) - nll(fit.x - di + dj) + nll(fit.x - di - dj)) / (4 * h * h)
    # identical inputs (model == market) leave a and b unidentified separately: no standard errors then
    se = np.sqrt(np.diag(np.linalg.inv(hessian))) if np.linalg.cond(hessian) < 1e8 else np.full(2, np.nan)
    return {"blend_a": float(fit.x[0]), "blend_b": float(fit.x[1]), "blend_a_se": float(se[0]), "blend_b_se": float(se[1]),
            "blend_logloss": float(fit.fun / len(y))}

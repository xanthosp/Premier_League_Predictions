"""Reference models that check the harness: base rates, Elo-only ordered logit, and the market itself."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


class BaseRates:
    """Home/draw/away frequencies of the training matches. Any useful model must beat this."""
    name = "base_rates"
    params: dict = {}

    def fit(self, train: pd.DataFrame) -> None:
        self.p = train.ftr.value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0).to_numpy()

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p, (len(test), 1))


class EloOrdered:
    """Ordered logit on the pre-match Elo difference: P(A) = s(c1 - b x), P(not H) = s(c2 - b x), x = elo diff / 100.
    Home advantage lives in the cut points."""
    name = "elo_ordered"
    params: dict = {}

    @staticmethod
    def _probs(theta, x):
        c1, gap, b = theta
        away = expit(c1 - b * x)
        not_home = expit(c1 + np.exp(gap) - b * x)
        return np.column_stack([1 - not_home, not_home - away, away])

    def fit(self, train: pd.DataFrame) -> None:
        x = ((train.home_elo - train.away_elo) / 100).to_numpy()
        y = train.ftr.map({"H": 0, "D": 1, "A": 2}).to_numpy()

        def nll(theta):
            return -np.log(np.clip(self._probs(theta, x)[np.arange(len(y)), y], 1e-12, 1)).sum()

        start = getattr(self, "theta", np.array([-1.0, 0.0, 0.5]))
        self.theta = minimize(nll, start, method="BFGS").x

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self._probs(self.theta, ((test.home_elo - test.away_elo) / 100).to_numpy())


class Market:
    """The de-vigged market itself (`close` or `open`). Against the closing benchmark, `open` shows how much
    information arrives between the early prices and kickoff."""
    params: dict = {}

    def __init__(self, kind: str):
        self.kind, self.name = kind, f"market_{kind}"

    def fit(self, train: pd.DataFrame) -> None:
        self.fallback = train.ftr.value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0).to_numpy()

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        p = test[[f"{self.kind}_{o}" for o in ("H", "D", "A")]].to_numpy(dtype=float).copy()
        missing = ~np.isfinite(p).all(axis=1)
        p[missing] = self.fallback
        return p


MODELS = {"base_rates": BaseRates, "elo_ordered": EloOrdered,
          "market_open": lambda: Market("open"), "market_close": lambda: Market("close")}

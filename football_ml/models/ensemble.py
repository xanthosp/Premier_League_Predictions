"""Equal-weight average of several models' H/D/A probabilities (a linear opinion pool).

Members are refit every week like any model. Equal weights on purpose: learning weights on the validation
split would be one more thing to overfit. Members that model a single league (Dixon-Coles) are told the
evaluated league and filter the training data themselves, so the ensemble can train on all leagues.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Ensemble:
    def __init__(self, name: str, members: list):
        self.name, self.members, self.league = name, [make() for make in members], None
        self.params = {"members": [m.name for m in self.members]}

    def fit(self, train: pd.DataFrame) -> None:
        for member in self.members:
            if hasattr(member, "league"):
                member.league = self.league
            member.fit(train)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.mean([member.predict(test) for member in self.members], axis=0)

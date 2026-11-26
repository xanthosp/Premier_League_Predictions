"""Feature-based models on the as-of columns of `features_match`, all with one fixed feature list.

- PoissonGLM: two Poisson regressions (home goals, away goals) -> scoreline matrix -> 1X2;
- XGBoost: gradient-boosted trees straight on H/D/A (can find non-linear effects and interactions);
- Logistic: multinomial logistic regression on the same features, the linear reference for XGBoost.
Hyper-parameters are fixed a priori; tuning happens only on the validation split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .scorelines import outcome_probabilities

SIDE_FEATURES = ["gf", "ga", "npxgf", "npxga", "sf", "sa", "form_weight", "rest_days", "matches_14d", "prev_europe",
                 "next_europe", "missing_regulars", "missing_xg_share", "regulars_load_8d"]
FEATURES = ["elo_diff", "elo_home_exp"] + [f"{side}_{c}" for side in ("home", "away") for c in SIDE_FEATURES]
LABELS = {"H": 0, "D": 1, "A": 2}


def design(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.assign(elo_diff=frame.home_elo - frame.away_elo)
    return x.reindex(columns=FEATURES).astype(float)  # a missing feature group becomes NaN


def _linear(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


# Feature groups for ablation: "poisson_glm-no_<group>" drops one group to measure what it adds.
GROUPS = {"elo": ["elo"], "goals": ["gf", "ga"], "xg": ["npxgf", "npxga"], "shots": ["sf", "sa"],
          "schedule": ["rest_days", "matches_14d", "prev_europe", "next_europe"],
          "players": ["missing_regulars", "missing_xg_share", "regulars_load_8d"]}


def group_columns(group: str) -> list[str]:
    return [f for f in FEATURES if any(f == c or f.endswith(f"_{c}") or f.startswith(f"{c}_") for c in GROUPS[group])]


class PoissonGLM:
    def __init__(self, alpha: float = 1.0, drop: str | None = None):
        self.name = ("poisson_glm" if drop is None else f"poisson_glm-no_{drop}") + ("" if alpha == 1.0 else f"-a{alpha}")
        self.params = {"alpha": alpha} if drop is None else {"alpha": alpha, "drop": drop}
        self.columns = [f for f in FEATURES if drop is None or f not in group_columns(drop)]

    def fit(self, train: pd.DataFrame) -> None:
        x = design(train)[self.columns]
        self.home = _linear(PoissonRegressor(alpha=self.params["alpha"], max_iter=300)).fit(x, train.fthg)
        self.away = _linear(PoissonRegressor(alpha=self.params["alpha"], max_iter=300)).fit(x, train.ftag)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x = design(test)[self.columns]
        return outcome_probabilities(self.home.predict(x), self.away.predict(x))


class Logistic:
    name = "logistic"
    params = {"C": 0.1}

    def fit(self, train: pd.DataFrame) -> None:
        self.model = _linear(LogisticRegression(C=self.params["C"], max_iter=1000)).fit(design(train), train.ftr.map(LABELS))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(design(test))


class XGBoost:
    name = "xgboost"
    params = {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "subsample": 0.8,
              "colsample_bytree": 0.8, "reg_lambda": 5.0}

    def fit(self, train: pd.DataFrame) -> None:
        from xgboost import XGBClassifier
        self.model = XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=4, random_state=0, **self.params)
        self.model.fit(design(train), train.ftr.map(LABELS))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(design(test))

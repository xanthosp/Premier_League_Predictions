import numpy as np
import pytest

from football_ml.features.match_features import match_features
from football_ml.models.dixon_coles import DixonColes
from football_ml.models.scorelines import outcome_probabilities
from football_ml.models.simple import MODELS
from tests.test_features import synthetic_league


def test_scoreline_probabilities_are_valid_and_symmetric():
    p = outcome_probabilities([1.3, 2.0], [1.3, 0.8], rho=-0.1)
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()
    assert p[0, 0] == pytest.approx(p[0, 2]) and p[1, 0] > p[1, 2]


def dataset():
    games = synthetic_league(seed=3)
    data = match_features(games).merge(games[["match_id", "fthg", "ftag", "home_npxg", "away_npxg"]], on="match_id")
    data["ftr"] = np.select([data.fthg > data.ftag, data.fthg == data.ftag], ["H", "D"], "A")
    return data.rename(columns={"home_npxg": "result_home_npxg", "away_npxg": "result_away_npxg"})


def test_dixon_coles_rates_a_dominant_team_higher_and_handles_unknown_teams():
    data = dataset()
    data.loc[data.home_team == "B", "fthg"] += 3
    data.loc[data.away_team == "B", "ftag"] += 3
    model = DixonColes()
    model.fit(data)
    strength = {t: model.attack[t] - model.defence[t] for t in model.attack}
    assert max(strength, key=strength.get) == "B"
    test = data.head(2).assign(home_team="Newcomer")
    assert np.allclose(model.predict(test).sum(axis=1), 1)


@pytest.mark.parametrize("name", ["dixon_coles", "dixon_coles_npxg", "poisson_glm", "logistic", "xgboost", "elo_ordered", "ens_all"])
def test_every_model_returns_valid_probabilities(name):
    data = dataset()
    train, test = data.iloc[:45], data.iloc[45:]
    model = MODELS[name]()
    model.fit(train)
    p = model.predict(test)
    assert p.shape == (len(test), 3) and np.allclose(p.sum(axis=1), 1) and (p > 0).all()


def test_ablation_groups_cover_every_feature_exactly_once():
    from football_ml.models.ml import FEATURES, GROUPS, group_columns
    columns = [c for g in GROUPS for c in group_columns(g)]
    assert sorted(columns) == sorted(set(columns)) and set(FEATURES) - set(columns) == {"home_form_weight", "away_form_weight"}

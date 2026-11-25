import numpy as np
import pandas as pd
import pytest

from football_ml.evaluate import metrics
from football_ml.evaluate.market import market_probabilities
from football_ml.evaluate.walkforward import evaluate, weekly_folds


def test_market_probabilities_devig_and_prefer_pinnacle():
    rows = [("m1", "PS", s, 1, p) for s, p in zip("HDA", (2.0, 3.5, 4.0))]
    rows += [("m1", "BFE", s, 1, p) for s, p in zip("HDA", (2.1, 3.6, 4.2))]
    rows += [("m2", "BFE", s, 1, p) for s, p in zip("HDA", (1.5, 4.0, 7.0))]
    rows += [("m3", "PS", s, 1, p) for s, p in zip("HD", (1.5, 4.0))]  # incomplete: no away price
    odds = pd.DataFrame(rows, columns=["match_id", "bookmaker", "selection", "is_closing", "price"]).assign(market="1x2")
    out = market_probabilities(odds, "close").set_index("match_id")
    assert list(out.index) == ["m1", "m2"] and out.close_book.tolist() == ["PS", "BFE"]
    assert np.allclose(out[["close_H", "close_D", "close_A"]].sum(axis=1), 1.0)
    inverse = np.array([1 / 2.0, 1 / 3.5, 1 / 4.0])
    assert out.loc["m1", "close_H"] == pytest.approx(inverse[0] / inverse.sum())


def test_scores_on_known_values():
    y = metrics.one_hot(["H", "A"])
    uniform = np.full((2, 3), 1 / 3)
    assert metrics.log_losses(uniform, y) == pytest.approx([np.log(3)] * 2)
    sure_home = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    assert metrics.rps(sure_home, y).tolist() == [0.0, 1.0]  # a confident home call on an away win is the worst case
    assert metrics.brier(sure_home, y).tolist() == [0.0, 2.0]


def test_blend_test_detects_information_only_when_present():
    rng = np.random.default_rng(1)
    n = 20000
    truth = rng.dirichlet([4, 2.5, 3], n)
    noise = lambda: rng.dirichlet([30, 30, 30], n)  # noqa: E731
    market = truth * noise()
    market /= market.sum(axis=1, keepdims=True)
    informed = truth * noise()
    informed /= informed.sum(axis=1, keepdims=True)
    y = metrics.one_hot(["HDA"[rng.choice(3, p=p)] for p in truth])
    useful = metrics.blend_test(market, informed, y)
    useless = metrics.blend_test(market, rng.dirichlet([3, 3, 3], n), y)
    assert useful["blend_b"] > 4 * useful["blend_b_se"]
    assert abs(useless["blend_b"]) < 3 * useless["blend_b_se"]


def test_weekly_folds_never_train_on_the_test_week_or_later():
    days = pd.date_range("2019-08-01", "2020-05-31", freq="D")
    data = pd.DataFrame({"match_id": [f"m{i}" for i in range(len(days))], "date": days.strftime("%Y-%m-%d"),
                         "season": "2019-20", "league": ["E0", "E1"] * (len(days) // 2) + ["E0"] * (len(days) % 2),
                         "played": True})
    folds = list(weekly_folds(data, "E0", "validation"))
    assert len(folds) > 40
    for train, test in folds:
        monday = pd.to_datetime(test.date).min() - pd.Timedelta(days=pd.to_datetime(test.date).min().weekday())
        assert pd.to_datetime(train.date).max() < monday
        assert set(test.league) == {"E0"} and pd.to_datetime(test.date).max() < monday + pd.Timedelta(days=7)


def test_test_split_needs_explicit_confirmation(tmp_path):
    with pytest.raises(ValueError, match="confirm-test"):
        evaluate(tmp_path, model=None, split="test")

import numpy as np
import pandas as pd
import pytest

from football_ml.betting import policy
from football_ml.betting.markets import implied_rates, market_prices, over_probability, payoff
from football_ml.models.scorelines import outcome_probabilities, score_matrix


@pytest.mark.parametrize("line,hg,ag,expected", [
    (-0.5, 1, 0, 1.0), (-0.5, 1, 1, -1.0), (0.0, 1, 1, 0.0),
    (-0.25, 1, 1, -0.5),    # half lost (0.0 push, -0.5 lost)
    (-0.75, 1, 0, 0.5),     # half won (-0.5 won, -1.0 push)
    (-1.0, 2, 1, 0.0), (0.25, 1, 1, 0.5),
])
def test_asian_handicap_payoff_at_even_money(line, hg, ag, expected):
    assert payoff("ah", "H", line, 2.0, hg, ag) == pytest.approx(expected)
    assert payoff("ah", "A", line, 2.0, hg, ag) == pytest.approx(-expected)  # at even money the sides mirror


def test_1x2_and_over_under_payoff():
    assert payoff("1x2", "D", None, 3.4, 2, 2) == pytest.approx(2.4)
    assert payoff("ou25", "U", None, 1.9, 2, 1) == pytest.approx(-1.0)


def test_implied_rates_recover_the_generating_rates():
    lh, la = np.array([1.7, 0.9]), np.array([1.1, 1.6])
    m = score_matrix(lh, la)
    fit_h, fit_a = implied_rates(outcome_probabilities(lh, la), over_probability(m))
    assert np.allclose(fit_h, lh, atol=0.03) and np.allclose(fit_a, la, atol=0.03)


def test_market_prices_pick_first_book_and_devig():
    odds = pd.DataFrame({"match_id": "m1", "bookmaker": ["PS"] * 3 + ["Avg"] * 3, "market": "1x2", "is_closing": 0,
                         "selection": ["H", "D", "A"] * 2, "line": np.nan, "price": [2.0, 3.5, 4.0, 1.9, 3.4, 3.9]})
    wide = market_prices(odds, closing=False, books={"1x2": ["Avg", "PS"], "ou25": [], "ah": []})
    assert wide.loc[0, "1x2_book"] == "Avg" and wide.loc[0, "1x2_H"] == 1.9
    assert wide[["fair_1x2_H", "fair_1x2_D", "fair_1x2_A"]].sum(axis=1).iloc[0] == pytest.approx(1)


def test_market_only_policy_never_bets_into_a_margin_and_settles():
    frame = pd.DataFrame({"match_id": ["a", "b"], "model_lh": [2.5, 1.2], "model_la": [0.6, 1.2],
                          "market_lh": [1.4, 1.2], "market_la": [1.2, 1.2],
                          "1x2_H": [2.3, 2.6], "1x2_D": [3.2, 3.1], "1x2_A": [3.0, 2.6], "1x2_book": "Avg"})
    assert policy.decide(frame, w=0.0).empty  # fair market minus a margin: no value anywhere
    bets = policy.decide(frame, w=1.0)       # the model strongly prefers the home side of match a
    assert set(bets.match_id) == {"a"} and bets.selection.iloc[0] == "H" and bets.ev.iloc[0] >= policy.THRESHOLD
    settled = policy.settle(bets, [2], [0], score_matrix(np.array([1.4]), np.array([1.2])))
    assert settled.profit.iloc[0] == pytest.approx(1.3) and settled.clv.iloc[0] < 0


def test_fit_weight_prefers_the_source_that_generated_the_goals():
    rng = np.random.default_rng(0)
    truth_h, truth_a = rng.uniform(0.8, 2.2, 3000), rng.uniform(0.6, 1.8, 3000)
    hg, ag = rng.poisson(truth_h), rng.poisson(truth_a)
    noisy = (truth_h * np.exp(rng.normal(0, 0.4, 3000)), truth_a * np.exp(rng.normal(0, 0.4, 3000)))
    assert policy.fit_weight(noisy, (truth_h, truth_a), hg, ag) <= 0.2
    assert policy.fit_weight((truth_h, truth_a), noisy, hg, ag) >= 0.8


def test_stacking_recovers_known_coefficients_and_move_model_without_history_predicts_no_move():
    from football_ml.betting import strategy
    rng = np.random.default_rng(1)
    n = 4000
    frame = pd.DataFrame({"market_lh": rng.uniform(0.8, 2.4, n), "market_la": rng.uniform(0.6, 1.8, n),
                          "model_lh": rng.uniform(0.8, 2.4, n), "model_la": rng.uniform(0.6, 1.8, n)})
    frame["fthg"] = rng.poisson(np.exp(0.05 + 0.8 * np.log(frame.market_lh) + 0.3 * np.log(frame.model_lh)))
    frame["ftag"] = rng.poisson(frame.market_la)
    coef = strategy.fit_stack(frame)
    assert np.allclose(coef["h"], [0.05, 0.8, 0.3], atol=0.08) and abs(coef["a"][2]) < 0.08
    for c in strategy.MOVE_FEATURES[4:]:
        frame[c] = 0.0
    frame["close_lh"] = np.nan
    frame["close_la"] = np.nan
    assert np.allclose(strategy.MoveModel().fit(frame).predict(frame.head(3)), 0)


def test_price_card_and_move_model_round_trip():
    from football_ml.betting import strategy
    m = score_matrix(np.array([1.6]), np.array([1.1]))
    frame = pd.DataFrame({"match_id": ["a"], "ah_line": [-0.25], "1x2_H": [2.0], "ah_H": [1.9]})
    card = strategy.price_card(frame, m, 0.01).set_index(["market", "selection"])
    p_home = outcome_probabilities(np.array([1.6]), np.array([1.1]))[0, 0]
    assert card.loc[("1x2", "H"), "fair_price"] == pytest.approx(1 / p_home)
    assert card.loc[("1x2", "H"), "min_price"] == pytest.approx(1.01 / p_home)
    ah = card.loc[("ah", "H")]  # at the fair price the expected profit is 0, at the minimum it is 1%
    for price, ev in ((ah.fair_price, 0.0), (ah.min_price, 0.01)):
        assert payoff_ev(m, price) == pytest.approx(ev, abs=1e-9)
    rng = np.random.default_rng(2)
    hist = pd.DataFrame({c: rng.normal(size=300) for c in strategy.MOVE_FEATURES[4:]})
    hist = hist.assign(market_lh=rng.uniform(1, 2, 300), market_la=rng.uniform(1, 2, 300), model_lh=1.5, model_la=1.2)
    hist = hist.assign(close_lh=hist.market_lh + 0.1 * hist.shots_for, close_la=hist.market_la)
    move = strategy.MoveModel().fit(hist)
    assert np.allclose(strategy.MoveModel.from_dict(move.to_dict()).predict(hist.head(5)), move.predict(hist.head(5)))


def payoff_ev(matrix, price):
    from football_ml.betting.markets import expected_profit
    return expected_profit(matrix, "ah", "H", np.array([-0.25]), np.array([price]))[0]


def test_ledger_append_keeps_header_order_and_refuses_new_columns(tmp_path):
    from football_ml.betting import paper
    path = tmp_path / "cards.csv"
    paper._append(path, pd.DataFrame({"a": [1], "betano": [2.0], "pinnacle": [3.0]}))
    paper._append(path, pd.DataFrame({"pinnacle": [5.0], "a": [4], "betano": [6.0]}))
    assert pd.read_csv(path).to_dict("list") == {"a": [1, 4], "betano": [2.0, 6.0], "pinnacle": [3.0, 5.0]}
    with pytest.raises(ValueError):
        paper._append(path, pd.DataFrame({"a": [7], "new": [1]}))


def test_odds_api_parse_maps_names_and_selections(tmp_path):
    import json
    from football_ml.ingest import odds_api
    event = {"home_team": "Manchester United", "away_team": "Brentford", "commence_time": "2026-11-28T15:00:00Z",
             "bookmakers": [{"key": "pinnacle", "markets": [
                 {"key": "h2h", "outcomes": [{"name": "Manchester United", "price": 2.1}, {"name": "Draw", "price": 3.6},
                                             {"name": "Brentford", "price": 3.5}]},
                 {"key": "totals", "outcomes": [{"name": "Over", "price": 1.6, "point": 2.5},
                                                {"name": "Under", "price": 2.4, "point": 2.5}]}]}]}
    path = tmp_path / "20261128T0900Z.json"
    path.write_text(json.dumps([event]), encoding="utf-8")
    odds = odds_api.parse(path)
    assert set(odds.match_id) == {"E0_20261128_ManUnited_Brentford"}
    wide = odds_api.wide_for_card(odds).set_index(["market", "selection"]).pinnacle
    assert wide[("1x2", "H")] == 2.1 and wide[("1x2", "A")] == 3.5 and wide[("ou25", "U")] == 2.4

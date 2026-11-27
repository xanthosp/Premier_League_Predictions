"""The decision rule, fixed before looking at any profit (never tuned on ROI):

1. Mix model and market on the expected-goals scale: log rate = w * log(model) + (1 - w) * log(market open).
   w (weight of the model) is fitted on past matches only, by log loss of 1X2 and over/under 2.5 outcomes.
   With w = 0 we simply trust the market and never bet; the model only moves us where it has proven to help.
2. From the mixed scoreline matrix, the expected profit of every selection at the available open price.
3. Bet the best selection of each market (1x2, ou25, ah) of a match when its expected profit >= THRESHOLD.
4. Stakes: flat STAKE units, and in parallel KELLY_FRACTION Kelly on a fixed BANKROLL (capped), for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.scorelines import outcome_probabilities, score_matrix
from .markets import SELECTIONS, expected_profit, market_matrix, over_probability, payoff

THRESHOLD = 0.03
STAKE = 1.0
BANKROLL = 100.0
KELLY_FRACTION = 0.25
KELLY_CAP = 5.0
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
# Prices we bet at (the market average is what a normal bettor gets) and the sharp closing benchmark for CLV.
OPEN_BOOKS = {m: ["Avg", "B365", "PS"] for m in SELECTIONS}
CLOSE_BOOKS = {m: ["PS", "BFE", "Avg"] for m in SELECTIONS}


def blend(model_rates, market_rates, w: float):
    return tuple(np.exp(w * np.log(m) + (1 - w) * np.log(k)) for m, k in zip(model_rates, market_rates))


def outcome_log_loss(lh, la, home_goals, away_goals) -> float:
    """Mean log loss of the 1X2 outcome plus the over/under 2.5 outcome."""
    hg, ag = np.asarray(home_goals), np.asarray(away_goals)
    p = outcome_probabilities(lh, la)
    result = np.select([hg > ag, hg == ag], [0, 1], 2)
    over = over_probability(score_matrix(lh, la))
    p_ou = np.where(hg + ag > 2.5, over, 1 - over)
    return float(-(np.log(np.clip(p[np.arange(len(hg)), result], 1e-12, 1)) + np.log(np.clip(p_ou, 1e-12, 1))).mean())


def fit_weight(model_rates, market_rates, home_goals, away_goals) -> float:
    losses = [outcome_log_loss(*blend(model_rates, market_rates, w), home_goals, away_goals) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmin(losses))])


def evaluate_bets(matrix: np.ndarray, bets: pd.DataFrame) -> np.ndarray:
    """Expected profit per unit of each bet row (market, selection, line, price) under its own matrix row."""
    out = np.full(len(bets), np.nan)
    for (market, selection), rows in bets.groupby(["market", "selection"]).groups.items():
        idx = bets.index.get_indexer(rows)
        out[idx] = expected_profit(matrix[idx], market, selection, bets.line.to_numpy(float)[idx],
                                   bets.price.to_numpy(float)[idx])
    return out


def decision_matrix(frame: pd.DataFrame, w: float) -> np.ndarray:
    """The market's exact scoreline matrix, tilted cell by cell by how much the blended Poisson differs from the
    market's Poisson: with w = 0 it reproduces the fair market prices, so a margin is never mistaken for value."""
    market = (frame.market_lh.to_numpy(float), frame.market_la.to_numpy(float))
    model = (frame.model_lh.to_numpy(float), frame.model_la.to_numpy(float))
    base = market_matrix(frame) if "fair_1x2_H" in frame else score_matrix(*market)
    m = base * score_matrix(*blend(model, market, w)) / score_matrix(*market)
    return m / m.sum(axis=(1, 2), keepdims=True)


def decide(frame: pd.DataFrame, w: float, threshold: float = THRESHOLD) -> pd.DataFrame:
    """frame: one row per match with model_lh/la, market_lh/la and open prices ({market}_{sel}, ah_line, {market}_book).
    Returns one row per bet."""
    matrix = decision_matrix(frame, w)
    bets = []
    for market, selections in SELECTIONS.items():
        if f"{market}_{selections[0]}" not in frame:
            continue
        line = frame.ah_line.to_numpy(float) if market == "ah" else np.full(len(frame), np.nan)
        evs = np.column_stack([expected_profit(matrix, market, s, line, frame[f"{market}_{s}"].to_numpy(float))
                               for s in selections])
        evs = np.where(np.isfinite(evs), evs, -np.inf)
        best = evs.argmax(axis=1)
        ev = evs[np.arange(len(frame)), best]
        for i in np.flatnonzero(ev >= threshold):
            s = selections[best[i]]
            price = float(frame[f"{market}_{s}"].iloc[i])
            kelly = min(KELLY_FRACTION * ev[i] / (price - 1) * BANKROLL, KELLY_CAP)
            bets.append({"match_id": frame.match_id.iloc[i], "market": market, "selection": s,
                         "line": line[i], "price": price, "book": frame[f"{market}_book"].iloc[i],
                         "ev": round(float(ev[i]), 4), "stake": STAKE, "kelly_stake": round(kelly, 2)})
    columns = ["match_id", "market", "selection", "line", "price", "book", "ev", "stake", "kelly_stake"]
    return pd.DataFrame(bets, columns=columns)


def settle(bets: pd.DataFrame, home_goals, away_goals, close_matrix: np.ndarray | None = None) -> pd.DataFrame:
    """Profit (flat and Kelly) from the final score, and CLV = expected profit of the bet under the closing market."""
    unit = np.array([float(payoff(r.market, r.selection, r.line, r.price, hg, ag))
                     for r, hg, ag in zip(bets.itertuples(), home_goals, away_goals)])
    out = pd.DataFrame({"profit": unit * bets.stake.to_numpy(float),
                        "kelly_profit": unit * bets.kelly_stake.to_numpy(float)}, index=bets.index)
    out["clv"] = evaluate_bets(close_matrix, bets) if close_matrix is not None else np.nan
    return out

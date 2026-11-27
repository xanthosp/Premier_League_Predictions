"""Every bet is priced from one scoreline matrix, so 1X2, over/under, Asian handicap (and later any combination,
e.g. "home & under 3.5") are consistent with each other.

- `payoff`: profit per unit stake of a bet for every possible score (also used to settle real bets);
- `expected_profit`: the same, averaged over a scoreline matrix;
- `implied_rates`: the expected goals (home, away) whose Poisson matrix best reproduces a market's de-vigged
  1X2 and over/under 2.5 probabilities, so the market and our model can be mixed on the same scale;
- `market_prices`: per match, the first available bookmaker's prices for each market (1x2, ou25, ah).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.scorelines import MAX_GOALS, outcome_probabilities, score_matrix

GOALS = np.arange(MAX_GOALS + 1)
SELECTIONS = {"1x2": ["H", "D", "A"], "ou25": ["O", "U"], "ah": ["H", "A"]}


def _half_line(diff4: np.ndarray, price) -> np.ndarray:
    """Profit of one whole/half handicap line; diff4 = 4 * (goal difference + line) for the backed side."""
    return np.where(diff4 > 0, price - 1.0, np.where(diff4 == 0, 0.0, -1.0))


def payoff(market: str, selection: str, line, price, home_goals, away_goals) -> np.ndarray:
    """Profit per unit stake. line: home handicap for "ah" (football-data AHh, e.g. -0.75), ignored otherwise.
    Arrays broadcast, so one call can score a whole matrix of scores."""
    hg, ag, price = np.asarray(home_goals), np.asarray(away_goals), np.asarray(price, float)
    if market == "1x2":
        won = {"H": hg > ag, "D": hg == ag, "A": hg < ag}[selection]
    elif market == "ou25":
        won = (hg + ag > 2.5) if selection == "O" else (hg + ag < 2.5)
    elif market == "ah":
        line4 = np.rint(np.asarray(line, float) * 4).astype(int)
        margin4 = 4 * (hg - ag) if selection == "H" else 4 * (ag - hg)
        line4 = line4 if selection == "H" else -line4
        quarter = line4 % 2 != 0  # -0.25, -0.75, ...: half the stake on each neighbouring line
        low = _half_line(margin4 + line4 - quarter, price)
        high = _half_line(margin4 + line4 + quarter, price)
        return (low + high) / 2
    else:
        raise ValueError(f"unknown market {market}")
    return np.where(won, price - 1.0, -1.0)


def expected_profit(matrix: np.ndarray, market: str, selection: str, line, price) -> np.ndarray:
    """matrix: (n, G, G) scoreline probabilities; line/price: (n,). Returns (n,) expected profit per unit."""
    n = matrix.shape[0]
    line = np.broadcast_to(np.asarray(line, float), (n,))[:, None, None]
    price = np.broadcast_to(np.asarray(price, float), (n,))[:, None, None]
    profit = payoff(market, selection, line, price, GOALS[None, :, None], GOALS[None, None, :])
    return (matrix * profit).sum(axis=(1, 2))


def over_probability(matrix: np.ndarray) -> np.ndarray:
    total = GOALS[:, None] + GOALS[None, :]
    return (matrix * (total > 2.5)).sum(axis=(1, 2))


_GRID = np.round(np.arange(0.1, 4.0001, 0.02), 4)


def _grid_table():
    lh, la = np.meshgrid(_GRID, _GRID, indexing="ij")
    lh, la = lh.ravel(), la.ravel()
    p = outcome_probabilities(lh, la)
    over = over_probability(score_matrix(lh, la))
    return lh, la, np.column_stack([p, over])


def implied_rates(p_1x2: np.ndarray, p_over: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares grid fit of (home, away) expected goals to de-vigged H/D/A (+ over 2.5 where known)."""
    if not hasattr(implied_rates, "_table"):
        implied_rates._table = _grid_table()
    lh, la, table = implied_rates._table
    target = np.asarray(p_1x2, float)
    over = np.full(len(target), np.nan) if p_over is None else np.asarray(p_over, float)
    best = np.empty(len(target), int)
    for start in range(0, len(target), 500):
        t, o = target[start:start + 500], over[start:start + 500]
        err = ((table[None, :, :3] - t[:, None, :]) ** 2).sum(axis=2)
        err += np.where(np.isfinite(o)[:, None], (table[None, :, 3] - np.nan_to_num(o)[:, None]) ** 2, 0.0)
        best[start:start + 500] = err.argmin(axis=1)
    return lh[best], la[best]


def fair_matrix(lam_home, lam_away, fair_1x2: np.ndarray, fair_over: np.ndarray | None = None,
                iterations: int = 30) -> np.ndarray:
    """Poisson matrix of the implied rates, then rescaled (iterative proportional fitting) so that its H/D/A and
    over 2.5 totals equal the market's de-vigged probabilities exactly. A 2-parameter Poisson cannot match the
    market's draw price (it gives ~1.4 points fewer draws), which would otherwise look like value."""
    m = score_matrix(lam_home, lam_away)
    diff = GOALS[:, None] - GOALS[None, :]
    regions = [diff > 0, diff == 0, diff < 0]
    over = (GOALS[:, None] + GOALS[None, :]) > 2.5
    target = np.asarray(fair_1x2, float)
    o = None if fair_over is None else np.asarray(fair_over, float)
    for _ in range(iterations):
        for k, region in enumerate(regions):
            current = (m * region).sum(axis=(1, 2))
            m = np.where(region, m * (target[:, k] / current)[:, None, None], m)
        if o is not None:
            known = np.isfinite(o)
            current = (m * over).sum(axis=(1, 2))
            scale_o = np.where(known, o / current, 1.0)[:, None, None]
            scale_u = np.where(known, (1 - o) / (1 - current), 1.0)[:, None, None]
            m = np.where(over, m * scale_o, m * scale_u)
    return m


def market_prices(odds: pd.DataFrame, closing: bool, books: dict[str, list[str]]) -> pd.DataFrame:
    """Wide per-match prices from the long odds table: for each market the first bookmaker in `books[market]`
    with a complete set of prices. Columns: {market}_{selection} prices, {market}_book, ah_line, plus de-vigged
    probabilities fair_1x2_H/D/A and fair_ou25_O."""
    rows = odds[odds.is_closing.astype(bool) == closing]
    out = []
    for market, selections in SELECTIONS.items():
        sub = rows[(rows.market == market) & rows.bookmaker.isin(books[market])]
        if sub.empty:
            continue
        wide = sub.pivot_table(index=["match_id", "bookmaker"], columns="selection", values="price", aggfunc="first")
        wide = wide.reindex(columns=selections).dropna()
        if market == "ah":
            lines = sub.groupby(["match_id", "bookmaker"]).line.first()
            wide["line"] = lines.reindex(wide.index)
            wide = wide.dropna()
        wide = wide.reset_index()
        wide["rank"] = wide.bookmaker.map({b: i for i, b in enumerate(books[market])})
        best = wide.sort_values("rank").drop_duplicates("match_id").set_index("match_id")
        part = best[selections].add_prefix(f"{market}_")
        part[f"{market}_book"] = best.bookmaker
        if market == "ah":
            part["ah_line"] = best.line
        inverse = 1.0 / best[selections]
        fair = inverse.div(inverse.sum(axis=1), axis=0)
        if market == "1x2":
            part[["fair_1x2_H", "fair_1x2_D", "fair_1x2_A"]] = fair.to_numpy()
        elif market == "ou25":
            part["fair_ou25_O"] = fair["O"]
        out.append(part)
    if not out:
        return pd.DataFrame(columns=["match_id"])
    return pd.concat(out, axis=1).reset_index().rename(columns={"index": "match_id"})


def _fair(prices: pd.DataFrame):
    over = prices["fair_ou25_O"].to_numpy(float) if "fair_ou25_O" in prices else None
    return prices[["fair_1x2_H", "fair_1x2_D", "fair_1x2_A"]].to_numpy(float), over


def market_rates(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Implied expected goals from the fair 1X2 (+ O/U 2.5) columns of `market_prices`."""
    return implied_rates(*_fair(prices))


def market_matrix(prices: pd.DataFrame) -> np.ndarray:
    """The market's own scoreline matrix: implied rates, rescaled to its exact fair 1X2 and O/U 2.5."""
    return fair_matrix(prices.market_lh.to_numpy(float), prices.market_la.to_numpy(float), *_fair(prices))

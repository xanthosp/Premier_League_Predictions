"""Market probabilities without the bookmaker margin (de-vigged), per match.

Two benchmarks for the 1X2 market:
- close: sharp closing prices, Pinnacle (PS) first, Betfair Exchange (BFE) where Pinnacle is missing;
- open: the earlier prices football-data records a day or two before kickoff (market average first, then Bet365,
  then Pinnacle). This is the price available when we would place a bet.
De-vig is multiplicative (proportional): p_i = (1/o_i) / sum_j (1/o_j).
"""
from __future__ import annotations

import pandas as pd

OUTCOMES = ["H", "D", "A"]
SOURCES = {"close": (True, ["PS", "BFE"]), "open": (False, ["Avg", "B365", "PS"])}


def devig(prices: pd.DataFrame) -> pd.DataFrame:
    inverse = 1.0 / prices
    return inverse.div(inverse.sum(axis=1), axis=0)


def market_probabilities(odds: pd.DataFrame, kind: str) -> pd.DataFrame:
    """odds: the long `odds` table (any subset). Returns match_id, {kind}_H/D/A, {kind}_book, {kind}_overround."""
    closing, books = SOURCES[kind]
    rows = odds[(odds.market == "1x2") & (odds.is_closing.astype(bool) == closing) & odds.bookmaker.isin(books)]
    wide = rows.pivot_table(index=["match_id", "bookmaker"], columns="selection", values="price", aggfunc="first")
    wide = wide.reindex(columns=OUTCOMES).dropna()
    wide = wide[(wide > 1.0).all(axis=1)].reset_index()
    wide["rank"] = wide.bookmaker.map({book: i for i, book in enumerate(books)})
    best = wide.sort_values("rank").drop_duplicates("match_id").set_index("match_id")
    out = devig(best[OUTCOMES]).add_prefix(f"{kind}_")
    out[f"{kind}_book"] = best.bookmaker
    out[f"{kind}_overround"] = (1.0 / best[OUTCOMES]).sum(axis=1)
    return out.reset_index()

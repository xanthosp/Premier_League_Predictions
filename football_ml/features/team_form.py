"""Exponentially decayed team form, as of the moment before each match.

Every match becomes two team rows (one per side). For each row, a stat's form is the weighted mean of that
team's *earlier* matches in the same league, with weight 0.5 ** (days_ago / half_life). Rows without a result
(upcoming fixtures) receive form but add nothing to it. Only the same league is used: Championship numbers
say little about a promoted team's Premier League level; the Elo rating handles promoted teams instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 120.0  # fixed a priori (about one third of a season); tuned only on validation in Phase 2
FORM_STATS = ["gf", "ga", "xgf", "xga", "npxgf", "npxga", "sf", "sa"]
SOURCES = {  # stat -> (home column, away column) in the match frame, from the home team's point of view
    "goals": ("fthg", "ftag"), "xg": ("home_xg", "away_xg"), "npxg": ("home_npxg", "away_npxg"), "shots": ("hs", "as_")}


def team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: the team's stats for (f) and against (a)."""
    sides = []
    for side, own, opp in (("home", 0, 1), ("away", 1, 0)):
        frame = matches[["match_id", "league", "date"]].copy()
        frame["side"] = side
        frame["team"] = matches[f"{side}_team"]
        for (name, columns), suffix in ((item, s) for item in SOURCES.items() for s in ("f", "a")):
            column = columns[own] if suffix == "f" else columns[opp]
            frame[f"{_short(name)}{suffix}"] = pd.to_numeric(matches[column], errors="coerce")
        sides.append(frame)
    rows = pd.concat(sides, ignore_index=True)
    rows["date"] = pd.to_datetime(rows.date)
    return rows


def _short(name: str) -> str:
    return {"goals": "g", "xg": "xg", "npxg": "npxg", "shots": "s"}[name]


def decayed_form(matches: pd.DataFrame, half_life: float = HALF_LIFE_DAYS) -> pd.DataFrame:
    """One row per (match_id, side) with the decayed mean of each stat over earlier matches, the total
    weight behind the goal means (`form_weight`, about "number of recent matches")."""
    rows = team_rows(matches).sort_values(["team", "league", "date"], kind="stable").reset_index(drop=True)
    groups = [rows.team, rows.league]
    days = (rows.date - rows.groupby(groups).date.transform("min")).dt.days.to_numpy(dtype=float)
    grow = 2.0 ** (days / half_life)  # weight of match j seen from match i = grow[j] / grow[i]

    def earlier_sum(values: np.ndarray) -> np.ndarray:
        contribution = pd.Series(values * grow)
        return (contribution.groupby(groups).cumsum() - contribution).to_numpy() / grow

    out = rows[["match_id", "side"]].copy()
    for pair in (("gf", "ga"), ("xgf", "xga"), ("npxgf", "npxga"), ("sf", "sa")):
        known = rows[list(pair)].notna().all(axis=1).to_numpy(dtype=float)
        weight = earlier_sum(known)
        for stat in pair:
            total = earlier_sum(rows[stat].fillna(0).to_numpy() * known)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[stat] = np.where(weight > 0, total / weight, np.nan)
        if pair == ("gf", "ga"):
            out["form_weight"] = weight

    return out

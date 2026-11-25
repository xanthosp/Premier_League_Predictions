"""Goal-margin Elo per league, with pre-match ratings for every match (played or upcoming).

Rules, all fixed a priori:
- update: delta = K * G(margin) * (score - expected), expected uses a home advantage in rating points;
- each new season: teams that joined the league (promoted) start at the mean rating of the teams that left it
  (relegated), so a promoted side is rated like a typical relegated side; then every rating is pulled
  REGRESS of the way back to the league mean (squads change over the summer).
Leagues are independent: there are no cross-league matches to compare them.
"""
from __future__ import annotations

import pandas as pd

START = 1500.0
K = 20.0
HOME_ADV = 60.0
REGRESS = 0.2


def expected_home(home: float, away: float, home_adv: float = HOME_ADV) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home + home_adv - away) / 400.0))


def margin_multiplier(goal_difference: int) -> float:
    margin = abs(goal_difference)
    return 1.0 if margin <= 1 else 1.5 if margin == 2 else (11.0 + margin) / 8.0


def _new_season(ratings: dict, previous: set, current: set, regress: float) -> dict:
    if not previous:
        return {team: START for team in current}
    departed = previous - current
    known = [ratings[t] for t in previous & current]
    entry = sum(ratings[t] for t in departed) / len(departed) if departed else sum(known) / len(known)
    season = {team: ratings[team] if team in previous else entry for team in current}
    mean = sum(season.values()) / len(season)
    return {team: mean + (1 - regress) * (rating - mean) for team, rating in season.items()}


def elo_ratings(matches: pd.DataFrame, k: float = K, home_adv: float = HOME_ADV, regress: float = REGRESS) -> pd.DataFrame:
    """Columns: match_id, home_elo, away_elo, elo_home_exp. Rows without a result get ratings but no update."""
    records = []
    for _, league in matches.sort_values(["date", "match_id"]).groupby("league", sort=False):
        teams = {season: set(part.home_team) | set(part.away_team) for season, part in league.groupby("season")}
        ratings: dict = {}
        previous: set = set()
        season = None
        for row in league.itertuples(index=False):
            if row.season != season:
                season = row.season
                ratings = _new_season(ratings, previous, teams[season], regress)
                previous = teams[season]
            home, away = ratings[row.home_team], ratings[row.away_team]
            expected = expected_home(home, away, home_adv)
            records.append((row.match_id, home, away, expected))
            if pd.notna(row.fthg) and pd.notna(row.ftag):
                score = 1.0 if row.fthg > row.ftag else 0.5 if row.fthg == row.ftag else 0.0
                delta = k * margin_multiplier(int(row.fthg - row.ftag)) * (score - expected)
                ratings[row.home_team] = home + delta
                ratings[row.away_team] = away - delta
    return pd.DataFrame(records, columns=["match_id", "home_elo", "away_elo", "elo_home_exp"])

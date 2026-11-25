"""Player availability and load from matchday squads, as of each match.

For every team, league matches give each player two decayed numbers (half-life PLAYER_HALF_LIFE_DAYS):
- importance: mean share of the 90 minutes he played (a regular starter is close to 1);
- xg_share: his share of the team's open-play xG.
Before a match on date d we look at the team's last league squad (starters + bench) before d. A player who
mattered but was not in that squad is most likely injured, suspended or sold. Features:
- missing_regulars: sum of importance of those players (about "number of regulars missing");
- missing_xg_share: share of the team's recent xG produced by those players;
- regulars_load_8d: importance-weighted minutes/90 that regulars played in any competition in [d - 8, d).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAYER_HALF_LIFE_DAYS = 60.0
LOAD_WINDOW_DAYS = 8
PLAYER_FEATURES = ["missing_regulars", "missing_xg_share", "regulars_load_8d"]


def _team_state(league: pd.DataFrame, half_life: float):
    """Per league match (sorted by date): date, player ids, importance and xg_share after it, squad mask."""
    minutes = league.pivot_table(index=["date", "opta_match_id"], columns="player_id", values="minutes", aggfunc="sum")
    npxg = league.pivot_table(index=["date", "opta_match_id"], columns="player_id", values="npxg", aggfunc="sum")
    in_squad = minutes.notna().to_numpy()
    played = minutes.fillna(0).to_numpy() / 90.0
    xg = npxg.reindex_like(minutes).fillna(0).to_numpy()
    dates = pd.to_datetime(minutes.index.get_level_values("date"))
    grow = 2.0 ** ((dates - dates.min()).days.to_numpy(dtype=float) / half_life)[:, None]
    importance = np.cumsum(played * grow, axis=0) / np.cumsum(grow, axis=0)
    xg_total = np.cumsum(xg.sum(axis=1, keepdims=True) * grow, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xg_share = np.where(xg_total > 0, np.cumsum(xg * grow, axis=0) / xg_total, 0.0)
    return dates.to_numpy(), minutes.columns, importance, xg_share, in_squad


def player_features(targets: pd.DataFrame, squads: pd.DataFrame, half_life: float = PLAYER_HALF_LIFE_DAYS) -> pd.DataFrame:
    """targets: key columns + team, date. squads: the player_minutes table. Returns targets + PLAYER_FEATURES."""
    out = targets.reset_index(drop=True).copy()
    values = np.full((len(out), len(PLAYER_FEATURES)), np.nan)
    target_days = pd.to_datetime(out.date).to_numpy()
    window = np.timedelta64(LOAD_WINDOW_DAYS, "D")
    for team, rows in out.groupby("team").groups.items():
        team_squads = squads[squads.team == team]
        league = team_squads[team_squads.kind == "league"]
        if league.empty:
            continue
        dates, players, importance, xg_share, in_squad = _team_state(league, half_life)
        all_matches = team_squads.assign(_d=pd.to_datetime(team_squads.date)).sort_values("_d")
        load = all_matches.pivot_table(index="_d", columns="player_id", values="minutes", aggfunc="sum").reindex(
            columns=players).fillna(0)
        load_days, load_values = load.index.to_numpy(), load.to_numpy() / 90.0
        for i in rows:
            day = target_days[i]
            last = np.searchsorted(dates, day, "left") - 1
            if last < 0:
                continue
            missing = ~in_squad[last]
            weight = importance[last]
            lo, hi = np.searchsorted(load_days, day - window, "left"), np.searchsorted(load_days, day, "left")
            values[i] = [(weight * missing).sum(), (xg_share[last] * missing).sum(),
                         weight @ load_values[lo:hi].sum(axis=0) / weight.sum()]
    out[PLAYER_FEATURES] = values
    return out

"""The table `features_match`: one row per match (played or upcoming) with as-of team features.

Columns: match keys, `played`, Elo (home_elo, away_elo, elo_home_exp) and, for each side, the decayed form
(gf, ga, xgf, xga, npxgf, npxga, sf, sa), form_weight and rest_days, prefixed home_/away_.
Upcoming Premier League fixtures come from the latest FPL snapshot (`fpl_fixtures`).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..ingest.football_data import current_season_start, season_code, season_label
from .elo import elo_ratings
from .team_form import decayed_form

KEYS = ["match_id", "league", "season", "date", "home_team", "away_team"]


def match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """`matches` needs KEYS, fthg, ftag, hs, as_, home_xg, away_xg, home_npxg, away_npxg (NaN when unknown)."""
    form = decayed_form(matches)
    wide = None
    for side in ("home", "away"):
        part = form[form.side == side].drop(columns="side")
        part = part.rename(columns={c: f"{side}_{c}" for c in part.columns if c != "match_id"})
        wide = part if wide is None else wide.merge(part, on="match_id")
    out = matches[KEYS].assign(played=matches.fthg.notna() & matches.ftag.notna())
    out = out.merge(elo_ratings(matches), on="match_id").merge(wide, on="match_id")
    return out.sort_values(["date", "league", "match_id"]).reset_index(drop=True)


def upcoming_fixtures(fixtures: pd.DataFrame, known_ids: set) -> pd.DataFrame:
    """Unfinished FPL fixtures with mapped teams and a kickoff time, in the `matches` key format."""
    todo = fixtures[~fixtures.finished.astype(bool)].dropna(subset=["kickoff_time", "home_team", "away_team"])
    dates = pd.to_datetime(todo.kickoff_time.str.rstrip("Z"))
    out = pd.DataFrame({"league": "E0", "date": dates.dt.strftime("%Y-%m-%d"),
                        "home_team": todo.home_team, "away_team": todo.away_team})
    out["season"] = [season_label(season_code(current_season_start(d.date()))) for d in dates]
    out.insert(0, "match_id", "E0_" + dates.dt.strftime("%Y%m%d") + "_" + out.home_team.str.replace(" ", "")
               + "_" + out.away_team.str.replace(" ", ""))
    return out[~out.match_id.isin(known_ids)].reset_index(drop=True)


def build(root: Path) -> dict:
    with sqlite3.connect(Path(root) / "data/football.sqlite") as connection:
        played = pd.read_sql("SELECT m.match_id, m.league, m.season, m.date, m.home_team, m.away_team, m.fthg, m.ftag, "
                             "m.hs, m.as_, x.home_xg, x.away_xg, x.home_npxg, x.away_npxg "
                             "FROM matches m LEFT JOIN match_xg x USING (match_id)", connection)
        tables = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type = 'table'", connection).name)
        fixtures = pd.read_sql("SELECT * FROM fpl_fixtures", connection) if "fpl_fixtures" in tables else None
        upcoming = upcoming_fixtures(fixtures, set(played.match_id)) if fixtures is not None else pd.DataFrame(columns=KEYS)
        features = match_features(pd.concat([played, upcoming], ignore_index=True))
        features.to_sql("features_match", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_features_match ON features_match(match_id)")
    e0 = features[features.league == "E0"]
    return {"rows": len(features), "upcoming_e0": int((~e0.played).sum()),
            "e0_missing_share": e0.drop(columns=KEYS + ["played"]).isna().mean().round(3)[lambda s: s > 0].to_dict()}

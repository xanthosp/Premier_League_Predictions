"""Official Fantasy Premier League API: timestamped snapshots of player availability and fixtures.

The API only shows the *current* state (injury flags, news, chance of playing), so history exists only
if we save snapshots. Each `update` stores one snapshot in data/raw/fpl_api/<UTC timestamp>/ and `build()`
turns all snapshots into the SQLite table `fpl_availability` (one row per player per snapshot), plus
`fpl_fixtures` from the latest snapshot. FPL team ids are linked to football-data names by date and score.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .pannadata import pair_team_names

ENDPOINTS = {"bootstrap": "https://fantasy.premierleague.com/api/bootstrap-static/",
             "fixtures": "https://fantasy.premierleague.com/api/fixtures/"}
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PLAYER_FIELDS = ["id", "web_name", "first_name", "second_name", "team", "element_type", "status",
                 "chance_of_playing_this_round", "chance_of_playing_next_round", "news", "news_added",
                 "minutes", "starts", "now_cost"]


def snapshot(root: Path, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    folder = Path(root) / "data/raw/fpl_api" / now.strftime("%Y%m%dT%H%MZ")
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in ENDPOINTS.items():
        with urlopen(Request(url, headers={"User-Agent": "football-ml-research/0.2"}), timeout=60) as response:
            (folder / f"{name}.json.gz").write_bytes(gzip.compress(response.read()))
    return folder.name


def _load(folder: Path, name: str):
    return json.loads(gzip.decompress((folder / f"{name}.json.gz").read_bytes()))


def parse_players(bootstrap: dict, snapshot_utc: str) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])[PLAYER_FIELDS].rename(columns={
        "id": "fpl_player_id", "team": "fpl_team_id", "element_type": "position",
        "chance_of_playing_this_round": "chance_this_round", "chance_of_playing_next_round": "chance_next_round"})
    players["position"] = players.position.map(POSITIONS)
    players.insert(0, "snapshot_utc", snapshot_utc)
    return players


def fixtures_frame(bootstrap: dict, fixtures: list[dict]) -> pd.DataFrame:
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    frame = pd.DataFrame(fixtures)[["id", "event", "kickoff_time", "team_h", "team_a", "team_h_score", "team_a_score", "finished"]]
    frame = frame.rename(columns={"id": "fpl_fixture_id", "event": "gameweek", "team_h": "home_fpl_id", "team_a": "away_fpl_id",
                                  "team_h_score": "home_goals", "team_a_score": "away_goals"})
    frame["home_fpl_name"] = frame.home_fpl_id.map(teams)
    frame["away_fpl_name"] = frame.away_fpl_id.map(teams)
    return frame


def team_map(fixtures: pd.DataFrame, fd_matches: pd.DataFrame) -> dict[int, str]:
    """FPL team id -> football-data team name, from finished fixtures with the same date and score."""
    done = fixtures[fixtures.finished.astype(bool)].dropna(subset=["home_goals", "away_goals"])
    left = pd.DataFrame({"league": "E0", "date": pd.to_datetime(done.kickoff_time).dt.strftime("%Y-%m-%d"),
                         "home_team": done.home_fpl_id, "away_team": done.away_fpl_id,
                         "home_goals": done.home_goals.astype(int), "away_goals": done.away_goals.astype(int)})
    pairs = pair_team_names(left, fd_matches)
    return dict(zip(pairs.left, pairs.right))


def build(root: Path) -> dict:
    root = Path(root)
    folders = sorted(p for p in (root / "data/raw/fpl_api").glob("*Z") if (p / "bootstrap.json.gz").exists())
    if not folders:
        return {"snapshots": 0}
    players = pd.concat([parse_players(_load(f, "bootstrap"), f.name) for f in folders], ignore_index=True)
    latest = folders[-1]
    bootstrap = _load(latest, "bootstrap")
    fixtures = fixtures_frame(bootstrap, _load(latest, "fixtures"))
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        fd = pd.read_sql("SELECT league, date, home_team, away_team, fthg AS home_goals, ftag AS away_goals "
                         "FROM matches WHERE league = 'E0'", connection)
        mapping = team_map(fixtures, fd)
        unmapped = sorted(set(fixtures.home_fpl_name[~fixtures.home_fpl_id.isin(mapping)]))
        players["team"] = players.fpl_team_id.map(mapping)
        fixtures["home_team"] = fixtures.home_fpl_id.map(mapping)
        fixtures["away_team"] = fixtures.away_fpl_id.map(mapping)
        players.to_sql("fpl_availability", connection, if_exists="replace", index=False)
        fixtures.to_sql("fpl_fixtures", connection, if_exists="replace", index=False)
    return {"snapshots": len(folders), "latest": latest.name, "players": int((players.snapshot_utc == latest.name).sum()),
            "teams_mapped": len(mapping), "teams_unmapped": unmapped,
            "upcoming_fixtures": int((~fixtures.finished.astype(bool)).sum())}

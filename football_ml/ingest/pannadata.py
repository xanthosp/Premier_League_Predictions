"""pannadata (github.com/peteowen1/pannadata): Opta fixtures and shot events with per-shot xG.

Raw files in data/raw/pannadata/:
- fixtures.parquet: every competition the release covers (results, kickoff dates, upcoming fixtures)
- shot_events.parquet: shot events for our leagues only (the full release is ~80MB)
- lineups.parquet: matchday squads with minutes (starters and bench) for our leagues, European club
  competitions and domestic cups, used for fixture congestion and player availability

`build()` aggregates shots per team-match, links each Opta match to a football-data match and writes
the SQLite table `match_xg`. Team names are linked by evidence, not by fuzzy text: two names are paired
when the same league, date and final score identify a unique match in both sources.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

RELEASE = "https://github.com/peteowen1/pannadata/releases/download/opta-latest/{name}.parquet"
COMPETITIONS = {"EPL": "E0", "Championship": "E1", "La_Liga": "SP1", "Bundesliga": "D1", "Serie_A": "I1", "Ligue_1": "F1"}
SHOT_COLUMNS = ["match_id", "event_id", "player_id", "player_name", "team_id", "minute", "second", "x", "y",
                "is_goal", "body_part", "situation", "big_chance", "competition", "season", "xg", "xgot",
                "is_own_goal", "is_blocked"]
# Other competitions our clubs play: they cost rest days and minutes even though we do not model them.
OTHER_COMPETITIONS = {"UCL": "europe", "UEL": "europe", "Conference_League": "europe", "UEFA_Super_Cup": "europe",
                      "Club_World_Cup": "europe", "FIFA_Intercontinental_Cup": "europe",
                      "FA_Cup": "cup", "League_Cup": "cup", "Copa_del_Rey": "cup", "Supercopa": "cup", "DFB_Pokal": "cup",
                      "Coppa_Italia": "cup", "Coupe_de_France": "cup", "Trophee_Champions": "cup"}
LINEUP_COLUMNS = ["match_id", "match_date", "player_id", "player_name", "team_id", "position", "is_starter",
                  "minutes_played", "competition", "season"]
MIN_PAIR_VOTES = 3
MIN_PAIR_SHARE = 0.9


def _fetch(name: str, target: Path) -> dict:
    url = RELEASE.format(name=name)
    request = Request(url, headers={"User-Agent": "football-ml-research/0.2"})
    with urlopen(request, timeout=600) as response:
        payload = response.read()
    target.write_bytes(payload)
    return {"url": url, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def download(root: Path) -> dict:
    """Refresh all files. Large files are filtered to the competitions we use before they are stored."""
    out = Path(root) / "data/raw/pannadata"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"fixtures.parquet": _fetch("opta_fixtures", out / "fixtures.parquet")}
    with tempfile.TemporaryDirectory() as tmp:
        for release_name, stored, columns, keep in (
                ("opta_shot_events", "shot_events.parquet", SHOT_COLUMNS, set(COMPETITIONS)),
                ("opta_shots", "player_match_shots.parquet", None, set(COMPETITIONS)),
                ("opta_lineups", "lineups.parquet", LINEUP_COLUMNS, set(COMPETITIONS) | set(OTHER_COMPETITIONS))):
            full = Path(tmp) / f"{release_name}.parquet"
            source = _fetch(release_name, full)
            frame = pd.read_parquet(full, columns=columns)
            frame = frame[frame.competition.isin(keep)]
            frame.to_parquet(out / stored, index=False)
            manifest[stored] = {**source, "stored": f"filtered to {sorted(keep)}", "rows": len(frame)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {name: entry["downloaded_at_utc"] for name, entry in manifest.items()}


def team_schedule(fixtures: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Every played or scheduled match of our clubs in any league, cup or European competition, one row per team.

    kind is "league" for our six leagues, else the OTHER_COMPETITIONS label ("europe"/"cup").
    """
    keep = fixtures[fixtures.match_status.isin(["Played", "Awarded", "Fixture"])
                    & fixtures.competition.isin(set(COMPETITIONS) | set(OTHER_COMPETITIONS))]
    date = pd.to_datetime(keep.match_date.str.rstrip("Z")).dt.strftime("%Y-%m-%d")
    rows = [pd.DataFrame({"team": keep[f"{side}_team_id"].map(mapping), "date": date, "competition": keep.competition,
                          "opta_match_id": keep.match_id, "played": keep.match_status.ne("Fixture")}) for side in ("home", "away")]
    out = pd.concat(rows).dropna(subset=["team"])
    out["kind"] = out.competition.map(OTHER_COMPETITIONS).fillna("league")
    return out.drop_duplicates(["team", "opta_match_id"]).sort_values(["team", "date"]).reset_index(drop=True)


def player_minutes(lineups: pd.DataFrame, mapping: dict, shots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matchday squads of our clubs: one row per player per match (bench players have 0 minutes), with the
    player's non-penalty xG in that match (league matches only; 0 elsewhere)."""
    out = lineups.assign(team=lineups.team_id.map(mapping)).dropna(subset=["team"])
    out = out.assign(date=pd.to_datetime(out.match_date.str.rstrip("Z")).dt.strftime("%Y-%m-%d"),
                     minutes=out.minutes_played.fillna(0), kind=out.competition.map(OTHER_COMPETITIONS).fillna("league"))
    out = out.rename(columns={"match_id": "opta_match_id"})
    if shots is None:
        out["npxg"] = 0.0
    else:
        open_play = shots[~shots.is_own_goal.fillna(False).astype(bool) & shots.situation.ne("Penalty")]
        npxg = open_play.groupby(["match_id", "player_id"]).xg.sum().rename("npxg").reset_index()
        out = out.merge(npxg.rename(columns={"match_id": "opta_match_id"}), how="left", on=["opta_match_id", "player_id"])
        out["npxg"] = out.npxg.fillna(0.0)
    return out[["opta_match_id", "date", "competition", "kind", "team", "player_id", "player_name", "position",
                "is_starter", "minutes", "npxg"]]


def load_fixtures(root: Path) -> pd.DataFrame:
    fixtures = pd.read_parquet(Path(root) / "data/raw/pannadata/fixtures.parquet")
    fixtures = fixtures[fixtures.competition.isin(COMPETITIONS)].copy()
    fixtures["league"] = fixtures.competition.map(COMPETITIONS)
    fixtures["date"] = pd.to_datetime(fixtures.match_date.str.rstrip("Z")).dt.strftime("%Y-%m-%d")
    return fixtures


def team_match_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """xG, non-penalty xG, shots and big chances per (match, team). Own goals are not shots."""
    shots = shots[~shots.is_own_goal.fillna(False).astype(bool)]
    penalty = shots.situation.eq("Penalty")
    return shots.assign(npxg=shots.xg.where(~penalty, 0.0), big=shots.big_chance.fillna(False).astype(int)).groupby(
        ["match_id", "team_id"]).agg(xg=("xg", "sum"), npxg=("npxg", "sum"), shots=("xg", "size"),
                                     big_chances=("big", "sum"), xg_missing=("xg", lambda s: int(s.isna().sum()))).reset_index()


def pair_team_names(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Pair team names between two sources from matches that are unique by (league, date, score) in both.

    Both frames need: league, date, home_team, away_team, home_goals, away_goals.
    Returns one row per left name with its right name, votes and agreement share.
    """
    key = ["league", "date", "home_goals", "away_goals"]
    unique_left = left[~left.duplicated(key, keep=False)]
    unique_right = right[~right.duplicated(key, keep=False)]
    joined = unique_left.merge(unique_right, on=key, suffixes=("_l", "_r"))
    pairs = pd.concat([
        joined[["home_team_l", "home_team_r"]].set_axis(["left", "right"], axis=1),
        joined[["away_team_l", "away_team_r"]].set_axis(["left", "right"], axis=1),
    ])
    votes = pairs.value_counts().rename("votes").reset_index()
    votes["share"] = votes.votes / votes.groupby("left").votes.transform("sum")
    best = votes.sort_values("votes", ascending=False).drop_duplicates("left")
    return best[(best.votes >= MIN_PAIR_VOTES) & (best.share >= MIN_PAIR_SHARE)].sort_values("left").reset_index(drop=True)


def build(root: Path) -> dict:
    root = Path(root)
    fixtures = load_fixtures(root)
    played = fixtures[fixtures.match_status.eq("Played")].rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    for column in ("home_goals", "away_goals"):
        played[column] = pd.to_numeric(played[column], errors="coerce")
    played = played.dropna(subset=["home_goals", "away_goals"]).astype({"home_goals": int, "away_goals": int})
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        fd = pd.read_sql("SELECT match_id, league, date, home_team, away_team, fthg AS home_goals, ftag AS away_goals "
                         "FROM matches", connection)
    # Pair by Opta team id (stable across Opta renames such as "Lyon" -> "Olympique Lyonnais").
    by_id = played.drop(columns=["home_team", "away_team"]).rename(columns={"home_team_id": "home_team", "away_team_id": "away_team"})
    names = pair_team_names(by_id, fd).rename(columns={"left": "opta_team_id", "right": "fd_team"})
    opta_names = pd.concat([played[["home_team_id", "home_team"]].set_axis(["opta_team_id", "opta_name"], axis=1),
                            played[["away_team_id", "away_team"]].set_axis(["opta_team_id", "opta_name"], axis=1)])
    latest = opta_names.drop_duplicates("opta_team_id", keep="last")
    names = names.merge(latest, on="opta_team_id", how="left")
    mapping = dict(zip(names.opta_team_id, names.fd_team))

    played = played.assign(home_fd=played.home_team_id.map(mapping), away_fd=played.away_team_id.map(mapping))
    linked = played.merge(fd, left_on=["league", "date", "home_fd", "away_fd"], right_on=["league", "date", "home_team", "away_team"],
                          suffixes=("", "_fd"))
    # Kickoffs are UTC in Opta and local in football-data; retry unmatched rows one day later/earlier.
    rest = played[~played.match_id.isin(linked.match_id)]
    for shift in (1, -1):
        shifted = rest.assign(date=(pd.to_datetime(rest.date) + pd.Timedelta(days=shift)).dt.strftime("%Y-%m-%d"))
        extra = shifted.merge(fd, left_on=["league", "date", "home_fd", "away_fd"], right_on=["league", "date", "home_team", "away_team"],
                              suffixes=("", "_fd"))
        linked = pd.concat([linked, extra])
        rest = rest[~rest.match_id.isin(extra.match_id)]
    score_mismatch = int(((linked.home_goals != linked.home_goals_fd) | (linked.away_goals != linked.away_goals_fd)).sum())

    raw_shots = pd.read_parquet(root / "data/raw/pannadata/shot_events.parquet")
    shots = team_match_shots(raw_shots)
    table = linked[["match_id_fd", "match_id", "home_team_id", "away_team_id"]].rename(
        columns={"match_id_fd": "match_id", "match_id": "opta_match_id"})
    for side in ("home", "away"):
        side_shots = shots.rename(columns={c: f"{side}_{c}" for c in ("xg", "npxg", "shots", "big_chances", "xg_missing")})
        table = table.merge(side_shots, how="left", left_on=["opta_match_id", f"{side}_team_id"], right_on=["match_id", "team_id"],
                            suffixes=("", "_drop")).drop(columns=["match_id_drop", "team_id"])
    table = table.drop(columns=["home_team_id", "away_team_id"])
    # A team with no shot events in a match that has shot data took zero shots: its xG is 0, not unknown.
    has_shot_data = table.opta_match_id.isin(shots.match_id)
    stats = [f"{side}_{c}" for side in ("home", "away") for c in ("xg", "npxg", "shots", "big_chances", "xg_missing")]
    table.loc[has_shot_data, stats] = table.loc[has_shot_data, stats].fillna(0)
    all_fixtures = pd.read_parquet(root / "data/raw/pannadata/fixtures.parquet")
    schedule = team_schedule(all_fixtures, mapping)
    lineup_file = root / "data/raw/pannadata/lineups.parquet"
    squads = player_minutes(pd.read_parquet(lineup_file), mapping, raw_shots) if lineup_file.exists() else None
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        table.to_sql("match_xg", connection, if_exists="replace", index=False)
        names.to_sql("team_names_opta", connection, if_exists="replace", index=False)
        schedule.to_sql("team_schedule", connection, if_exists="replace", index=False)
        if squads is not None:
            squads.to_sql("player_minutes", connection, if_exists="replace", index=False)
            connection.execute("CREATE INDEX IF NOT EXISTS ix_player_minutes ON player_minutes(team, date)")
    with_xg = fd.assign(has_xg=fd.match_id.isin(table.dropna(subset=["home_xg"]).match_id))
    coverage = with_xg.groupby("league").has_xg.mean().round(3).to_dict()
    return {"opta_played": len(played), "linked": len(table), "unlinked_opta": len(rest), "score_mismatch": score_mismatch,
            "team_names_paired": len(names), "xg_coverage_by_league": coverage,
            "schedule_rows_by_kind": schedule.kind.value_counts().to_dict(),
            "player_minutes_rows": None if squads is None else len(squads)}

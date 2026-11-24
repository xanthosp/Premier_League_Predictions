"""Normalize only the OpenFootball results approved for this first experiment."""
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

FIELDS = ["match_id", "season", "match_date", "home_team", "away_team", "home_goals", "away_goals", "btts", "over25", "kickoff_local_unverified", "source"]


def normalize_match(row, season):
    score = row.get("score")
    score = score.get("ft") if isinstance(score, dict) else score
    if score is None:
        return None
    if not isinstance(score, list) or len(score) != 2:
        raise ValueError("Final score must be a two-integer list")
    if any(type(goal) is not int or goal < 0 for goal in score):
        raise ValueError("Goals must be nonnegative integers")
    day = date.fromisoformat(row["date"]).isoformat()
    home, away = row["team1"].strip(), row["team2"].strip()
    if not home or not away or home == away:
        raise ValueError("Two different, nonempty teams are required")
    identity = json.dumps(["openfootball", season, day, home, away])
    return dict(zip(FIELDS, [
        hashlib.sha256(identity.encode()).hexdigest()[:24], season, day, home, away,
        score[0], score[1], int(score[0] > 0 and score[1] > 0), int(sum(score) > 2),
        row.get("time", ""), "openfootball",
    ]))


def load_sources(raw_dir):
    rows, manifest, skipped = [], [], 0
    for path in sorted(Path(raw_dir).glob("openfootball_*.json")):
        match = re.fullmatch(r"openfootball_(\d{4}-\d{2})", path.stem)
        if not match:
            continue
        season = match[1]
        payload = path.read_bytes()
        document = json.loads(payload)
        for raw in document["matches"]:
            normalized = normalize_match(raw, season)
            if normalized is None:
                skipped += 1
            else:
                rows.append(normalized)
        manifest.append({"file": path.name, "season": season,
                         "sha256": hashlib.sha256(payload).hexdigest(),
                         "source_url": f"https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json",
                         "licence_as_declared": "CC0-1.0"})
    if not rows:
        raise ValueError("No completed OpenFootball matches found")
    rows.sort(key=lambda row: (row["match_date"], row["match_id"]))
    if len({row["match_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate match keys; refusing to silently drop rows")
    return rows, manifest, skipped


def prepare(root):
    root = Path(root)
    raw = root / "data/raw/openfootball"
    rows, manifest, skipped = load_sources(raw)
    season_counts = dict(sorted(Counter(row["season"] for row in rows).items()))
    teams = {s: len({r[t] for r in rows if r["season"] == s for t in ("home_team", "away_team")}) for s in season_counts}
    if any(n != 380 for n in season_counts.values()) or any(n != 20 for n in teams.values()):
        raise ValueError("Expected 380 completed matches and 20 teams per historical EPL season")
    output = root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    reports = root / "outputs"
    reports.mkdir(exist_ok=True)
    with (output / "matches.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS openfootball_matches (
            match_id TEXT PRIMARY KEY, season TEXT NOT NULL, match_date TEXT NOT NULL,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL CHECK(home_goals>=0),
            away_goals INTEGER NOT NULL CHECK(away_goals>=0),
            btts INTEGER NOT NULL CHECK(btts IN (0,1)),
            over25 INTEGER NOT NULL CHECK(over25 IN (0,1)),
            kickoff_local_unverified TEXT, source TEXT NOT NULL)""")
        connection.execute("DELETE FROM openfootball_matches")
        connection.executemany("INSERT INTO openfootball_matches VALUES (?,?,?,?,?,?,?,?,?,?,?)", [tuple(row[k] for k in FIELDS) for row in rows])
    profile = {"rows": len(rows), "columns": len(FIELDS), "seasons": season_counts,
               "teams_by_season": teams, "date_min": rows[0]["match_date"], "date_max": rows[-1]["match_date"],
               "null_counts": {key: sum(row[key] is None or row[key] == "" for row in rows) for key in FIELDS},
               "duplicate_match_keys": 0, "unplayed_rows_excluded": skipped,
               "goal_range": [min(row[t] for row in rows for t in ("home_goals", "away_goals")), max(row[t] for row in rows for t in ("home_goals", "away_goals"))],
               "mean_home_goals": sum(r["home_goals"] for r in rows) / len(rows),
               "mean_away_goals": sum(r["away_goals"] for r in rows) / len(rows),
               "sample": rows[:3],
               "limitations": ["Kickoff timezone is unverified; use calendar-date splits only.", "Only OpenFootball results feed the baseline.", "No player, injury, odds or tactical features are joined.", "Source score accuracy has not been exhaustively independently reconciled."]}
    (reports / "data_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports / "source_manifest.json").write_text(json.dumps({"prepared_at": datetime.now(timezone.utc).isoformat(), "sources": manifest}, indent=2), encoding="utf-8")
    return profile


def load_prepared(root):
    with (Path(root) / "data/processed/matches.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("home_goals", "away_goals", "btts", "over25"):
            row[key] = int(row[key])
    return sorted(rows, key=lambda row: (row["match_date"], row["match_id"]))

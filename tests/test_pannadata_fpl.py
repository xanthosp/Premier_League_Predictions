import gzip
import json
from datetime import datetime, timezone

import pandas as pd

from football_ml.ingest import fpl, pannadata


def frame(rows):
    return pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals"])


def test_pair_team_names_uses_only_unambiguous_matches():
    left = frame([("E0", f"2026-08-{d:02d}", "Spurs FC", "Hull City AFC", d % 3, 1) for d in range(1, 6)]
                 + [("E0", "2026-09-01", "X1", "Y1", 1, 1), ("E0", "2026-09-01", "X2", "Y2", 1, 1)])
    right = frame([("E0", f"2026-08-{d:02d}", "Tottenham", "Hull", d % 3, 1) for d in range(1, 6)]
                  + [("E0", "2026-09-01", "A", "B", 1, 1), ("E0", "2026-09-01", "C", "D", 1, 1)])
    pairs = pannadata.pair_team_names(left, right)
    assert dict(zip(pairs.left, pairs.right)) == {"Hull City AFC": "Hull", "Spurs FC": "Tottenham"}
    assert set(pairs.share) == {1.0}  # the two 1-1 games on 2026-09-01 are ambiguous and never vote


def test_pair_team_names_rejects_too_few_votes():
    left = frame([("E0", "2026-08-01", "Spurs FC", "Hull City AFC", 2, 1)])
    right = frame([("E0", "2026-08-01", "Tottenham", "Hull", 2, 1)])
    assert pannadata.pair_team_names(left, right).empty


def test_team_match_shots_excludes_own_goals_and_splits_penalties():
    shots = pd.DataFrame({
        "match_id": ["m"] * 4, "team_id": ["t"] * 4,
        "situation": ["OpenPlay", "Penalty", "Corner", "OpenPlay"],
        "is_own_goal": [False, False, False, True],
        "big_chance": [True, True, None, False],
        "xg": [0.3, 0.76, 0.1, None],
    })
    row = pannadata.team_match_shots(shots).iloc[0]
    assert row.shots == 3 and row.big_chances == 2 and row.xg_missing == 0
    assert abs(row.xg - 1.16) < 1e-9 and abs(row.npxg - 0.4) < 1e-9


def write_snapshot(root, stamp, status):
    folder = root / "data/raw/fpl_api" / stamp
    folder.mkdir(parents=True)
    player = {"id": 7, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka", "team": 1, "element_type": 3,
              "status": status, "chance_of_playing_this_round": None, "chance_of_playing_next_round": 0 if status == "i" else None,
              "news": "Thigh injury" if status == "i" else "", "news_added": None, "minutes": 900, "starts": 10, "now_cost": 100}
    bootstrap = {"elements": [player], "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Hull City"}]}
    fixtures = [{"id": i, "event": i, "kickoff_time": f"2026-08-{i:02d}T14:00:00Z", "team_h": 1 + i % 2, "team_a": 2 - i % 2,
                 "team_h_score": i, "team_a_score": 0, "finished": True} for i in range(1, 5)]
    (folder / "bootstrap.json.gz").write_bytes(gzip.compress(json.dumps(bootstrap).encode()))
    (folder / "fixtures.json.gz").write_bytes(gzip.compress(json.dumps(fixtures).encode()))


def test_fpl_snapshots_accumulate_into_history(tmp_path):
    import sqlite3
    write_snapshot(tmp_path, "20261101T1000Z", "a")
    write_snapshot(tmp_path, "20261108T1000Z", "i")
    fd = pd.DataFrame({"match_id": [f"m{i}" for i in range(1, 5)], "league": "E0",
                       "date": [f"2026-08-{i:02d}" for i in range(1, 5)],
                       "home_team": ["Hull" if i % 2 else "Arsenal" for i in range(1, 5)],
                       "away_team": ["Arsenal" if i % 2 else "Hull" for i in range(1, 5)], "fthg": range(1, 5), "ftag": 0})
    (tmp_path / "data").mkdir(exist_ok=True)
    with sqlite3.connect(tmp_path / "data/football.sqlite") as connection:
        fd.to_sql("matches", connection, index=False)
    summary = fpl.build(tmp_path)
    assert summary["snapshots"] == 2 and summary["teams_mapped"] == 2 and summary["teams_unmapped"] == []
    with sqlite3.connect(tmp_path / "data/football.sqlite") as connection:
        history = pd.read_sql("SELECT snapshot_utc, team, status FROM fpl_availability ORDER BY snapshot_utc", connection)
    assert history.status.tolist() == ["a", "i"] and set(history.team) == {"Arsenal"}


def test_snapshot_folder_name_is_utc_timestamp(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(fpl, "urlopen", lambda *args, **kwargs: Response())
    name = fpl.snapshot(tmp_path, now=datetime(2026, 11, 24, 18, 48, tzinfo=timezone.utc))
    assert name == "20261124T1848Z"
    assert (tmp_path / "data/raw/fpl_api/20261124T1848Z/fixtures.json.gz").exists()

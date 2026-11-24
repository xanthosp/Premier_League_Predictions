import sqlite3
from datetime import date

import pytest

from football_ml.ingest import football_data as fd


@pytest.mark.parametrize("column, expected", [
    ("B365H", ("B365", "1x2", False, "H")),
    ("B365CA", ("B365", "1x2", True, "A")),
    ("PSCH", ("PS", "1x2", True, "H")),
    ("P>2.5", ("PS", "ou25", False, "O")),        # Pinnacle O/U uses "P", merged with "PS"
    ("PC<2.5", ("PS", "ou25", True, "U")),
    ("VCH", ("VC", "1x2", False, "H")),           # "VC" ends in C but is not a closing marker
    ("VCCD", ("VC", "1x2", True, "D")),
    ("BFD", ("BF", "1x2", False, "D")),           # Betfair draw, not a "BFD" bookmaker
    ("BFDH", ("BFD", "1x2", False, "H")),
    ("BFECAHH", ("BFE", "ah", True, "H")),
    ("BbMx>2.5", ("Max", "ou25", False, "O")),
    ("AvgCAHA", ("Avg", "ah", True, "A")),
])
def test_odds_columns_are_parsed(column, expected):
    assert fd.parse_odds_column(column) == expected


@pytest.mark.parametrize("column", ["AHh", "AHCh", "BbAH", "Bb1X2", "BbOU", "HS", "HxG", "FTHG", "Referee"])
def test_non_price_columns_are_ignored(column):
    assert fd.parse_odds_column(column) is None


def test_season_helpers():
    assert fd.season_code(2016) == "1617"
    assert fd.season_code(2099) == "9900"
    assert fd.season_label("2627") == "2026-27"
    assert fd.current_season_start(date(2026, 11, 24)) == 2026
    assert fd.current_season_start(date(2027, 3, 1)) == 2026


SAMPLE = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HxG,AxG,B365H,B365D,B365A,B365CH,B365CD,B365CA,"
    "P>2.5,P<2.5,AHh,AHCh,AvgAHH,AvgAHA,AvgCAHH,AvgCAHA,\n"
    "E0,13/08/16,15:00,Arsenal,Hull,2,0,H,15,8,1.9,0.6,1.5,4.2,7.0,1.45,4.5,7.5,1.8,2.1,-1.25,-1.5,1.95,1.95,2.0,1.9,\n"
    "E0,14/08/2016,16:30,Leeds,Man City,1,1,D,9,14,,,4.0,3.6,1.9,4.2,3.7,1.85,1.7,2.2,0.75,0.75,1.9,2.0,1.92,1.98,\n"
    "E0,20/08/2016,15:00,Hull,Leeds,,,,,,,,2.0,3.4,3.8,,,,,,,,,,,,\n"
)


def write_sample(root):
    path = root / "data/raw/football_data/E0/1617_E0.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(SAMPLE.encode("utf-8"))
    return path


def test_parse_file_matches_and_odds(tmp_path):
    matches, odds = fd.parse_file(write_sample(tmp_path))
    assert len(matches) == 2                                   # unplayed fixture dropped
    assert matches.date.tolist() == ["2016-08-13", "2016-08-14"]  # both date formats
    assert matches.match_id.iloc[0] == "E0_20160813_Arsenal_Hull"
    assert matches.season.unique().tolist() == ["2016-17"]
    assert matches.hxg.isna().tolist() == [False, True]
    assert not odds.match_id.str.contains("Hull_Leeds").any()
    closing_home = odds.query("market == '1x2' and is_closing == 1 and selection == 'H'")
    assert closing_home.price.tolist() == [1.45, 4.2]
    ah = odds.query("market == 'ah' and match_id == 'E0_20160813_Arsenal_Hull'").set_index(["is_closing", "selection"])
    assert ah.loc[(0, "H"), "line"] == -1.25 and ah.loc[(1, "H"), "line"] == -1.5
    assert set(odds.query("market == 'ou25'").line) == {2.5}


def test_build_writes_tables_and_coverage(tmp_path):
    write_sample(tmp_path)
    summary = fd.build(tmp_path)
    assert summary["matches"] == 2
    with sqlite3.connect(tmp_path / "data/football.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(DISTINCT bookmaker) FROM odds").fetchone()[0] == 3
    coverage = (tmp_path / "outputs/coverage_football_data.csv").read_text()
    assert "E0,2016-17,2" in coverage

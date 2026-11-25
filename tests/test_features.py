import numpy as np
import pandas as pd
import pytest

from football_ml.features import elo
from football_ml.features.match_features import match_features
from football_ml.features.team_form import decayed_form

RESULTS = ["fthg", "ftag", "hs", "as_", "home_xg", "away_xg", "home_npxg", "away_npxg"]


def synthetic_league(seed=0):
    """Two seasons of a 6-team double round robin; team F replaces team A in the second season."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, start, teams in (("2024-25", "2024-08-10", list("ABCDE") + ["G"]), ("2025-26", "2025-08-09", list("BCDEFG"))):
        day = pd.Timestamp(start)
        for leg in range(2):
            for i, home in enumerate(teams):
                for away in teams[i + 1:]:
                    h, a = (home, away) if leg == 0 else (away, home)
                    xg_h, xg_a = rng.uniform(0.3, 2.5, 2)
                    rows.append({"match_id": f"X_{day:%Y%m%d}_{h}_{a}", "league": "X", "season": season,
                                 "date": f"{day:%Y-%m-%d}", "home_team": h, "away_team": a,
                                 "fthg": rng.poisson(xg_h), "ftag": rng.poisson(xg_a), "hs": rng.integers(5, 20),
                                 "as_": rng.integers(5, 20), "home_xg": xg_h, "away_xg": xg_a,
                                 "home_npxg": xg_h * 0.9, "away_npxg": xg_a * 0.9})
                    day += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


def test_features_ignore_everything_from_the_match_day_onwards():
    full = synthetic_league()
    cutoff = full.date.iloc[len(full) * 3 // 4]
    truncated = full[full.date <= cutoff].copy()
    truncated.loc[truncated.date == cutoff, RESULTS] = np.nan  # the cutoff matches become "upcoming"
    expected = match_features(full)
    expected = expected[expected.date <= cutoff].drop(columns="played").reset_index(drop=True)
    got = match_features(truncated).drop(columns="played").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)


def test_changing_a_result_changes_only_later_matches():
    base = synthetic_league()
    changed = base.copy()
    target = len(base) // 2
    changed.loc[target, ["fthg", "ftag", "home_xg"]] = [7, 0, 4.0]
    a, b = match_features(base), match_features(changed)
    numeric = a.select_dtypes("number").columns
    differs = ~np.isclose(a[numeric], b[numeric], equal_nan=True).all(axis=1)
    assert not differs[a.date <= base.date[target]].any()
    assert differs[a.date > base.date[target]].any()


def test_decayed_form_weights_by_half_life():
    games = pd.DataFrame({"match_id": ["m1", "m2", "m3"], "league": "X", "date": ["2025-01-01", "2025-01-11", "2025-01-21"],
                          "home_team": "A", "away_team": ["B", "C", "D"], "fthg": [3, 1, np.nan], "ftag": [0, 0, np.nan],
                          "hs": [10, 10, np.nan], "as_": [5, 5, np.nan], "home_xg": [2.0, np.nan, np.nan],
                          "away_xg": [0.5, np.nan, np.nan], "home_npxg": [2.0, np.nan, np.nan], "away_npxg": [0.5, np.nan, np.nan]})
    form = decayed_form(games, half_life=10).set_index(["match_id", "side"])
    upcoming = form.loc[("m3", "home")]
    assert upcoming.gf == pytest.approx((3 * 0.25 + 1 * 0.5) / 0.75)
    assert upcoming.form_weight == pytest.approx(0.75)
    assert upcoming.xgf == pytest.approx(2.0)  # m2 has no xG, so only m1 counts
    assert np.isnan(form.loc[("m1", "home")].gf)


def test_elo_promoted_team_inherits_relegated_rating_and_starts_at_1500():
    ratings = {"A": 1400.0, "B": 1600.0, "C": 1500.0}
    season = elo._new_season(ratings, {"A", "B", "C"}, {"B", "C", "N"}, regress=0.0)
    assert season["N"] == 1400.0 and season["B"] == 1600.0
    games = synthetic_league()
    first_season = games[games.season == "2024-25"]
    out = elo.elo_ratings(first_season)
    last = out.iloc[-1]
    assert out.home_elo.iloc[0] == elo.START and 0 < last.elo_home_exp < 1
    assert elo.margin_multiplier(1) == 1.0 and elo.margin_multiplier(-3) == 1.75


def test_schedule_features_use_all_competitions_but_not_the_match_day():
    from football_ml.features.schedule import schedule_features
    schedule = pd.DataFrame({"team": "A", "date": ["2025-01-01", "2025-01-04", "2025-01-07", "2025-01-10"],
                             "kind": ["league", "europe", "league", "europe"], "played": [True, True, True, False]})
    target = pd.DataFrame({"match_id": ["m"], "team": ["A"], "date": ["2025-01-07"]})
    row = schedule_features(target, schedule).iloc[0]
    assert row.rest_days == 3 and row.prev_europe and row.matches_14d == 2
    assert row.days_to_next == 3 and row.next_europe


def test_player_features_flag_regulars_missing_from_last_league_squad():
    from football_ml.features.players import player_features
    squads = pd.DataFrame({
        "opta_match_id": ["m1", "m1", "m2", "m2", "e1"], "team": "A",
        "date": ["2025-01-01", "2025-01-01", "2025-01-08", "2025-01-08", "2025-01-11"],
        "kind": ["league", "league", "league", "league", "europe"], "player_id": ["p1", "p2", "p1", "p3", "p1"],
        "minutes": [90, 90, 90, 90, 90], "npxg": [0.0, 1.0, 1.0, 0.0, 0.0]})
    targets = pd.DataFrame({"match_id": ["t1", "t2"], "team": "A", "date": ["2025-01-15", "2025-01-08"]})
    out = player_features(targets, squads, half_life=1e9).set_index("match_id")
    assert out.loc["t1", "missing_regulars"] == pytest.approx(0.5)   # p2 played half the minutes, absent in m2
    assert out.loc["t1", "missing_xg_share"] == pytest.approx(0.5)   # p2 produced half of the xG
    # window [01-07, 01-15): p1 (weight 1) played m2 and e1, p3 (weight 0.5) played m2 -> (2 + 0.5) / 2
    assert out.loc["t1", "regulars_load_8d"] == pytest.approx(1.25)
    assert out.loc["t2", "missing_regulars"] == 0                    # on m2's day only m1 is known

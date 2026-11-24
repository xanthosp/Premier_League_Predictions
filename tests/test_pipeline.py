import math
import tempfile
from pathlib import Path

import pytest

from football_ml.ingest.api_clients import endpoint, load_keys, scrub
from football_ml.ingest.openfootball import load_sources, normalize_match
from football_ml.models.baseline_poisson import GoalRates, binary_scores, goal_probabilities, walk_forward


def match(day, home="A", away="B", hg=1, ag=0, season="2023-24"):
    return normalize_match({"date": day, "team1": home, "team2": away, "score": [hg, ag]}, season)


def test_two_observed_score_schemas_and_pending_fixture():
    row = {"date": "2025-08-15", "team1": "A", "team2": "B", "score": [2, 1]}
    flat = normalize_match(row, "2025-26")
    assert normalize_match({**row, "score": {"ft": [2, 1]}}, "2025-26") == flat
    assert flat["btts"] == flat["over25"] == 1
    assert normalize_match({**row, "score": None}, "2025-26") is None


@pytest.mark.parametrize("score", [[-1, 0], [True, 0], [1.5, 0], [1], "2-1"])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError):
        normalize_match({"date": "2024-01-01", "team1": "A", "team2": "B", "score": score}, "2023-24")


def test_duplicate_match_does_not_silently_disappear():
    import json
    row = {"date": "2024-01-01", "team1": "A", "team2": "B", "score": [1, 0]}
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "openfootball_2023-24.json").write_text(json.dumps({"matches": [row, row]}))
        with pytest.raises(ValueError, match="Duplicate"):
            load_sources(directory)


def test_goal_probabilities_against_explicit_poisson_sum():
    home, away = 1.7, 0.9
    ph = [math.exp(-home) * home**i / math.factorial(i) for i in range(25)]
    pa = [math.exp(-away) * away**i / math.factorial(i) for i in range(25)]
    probabilities = goal_probabilities(home, away)
    btts = sum(ph[i] * pa[j] for i in range(1, 25) for j in range(1, 25))
    over = sum(ph[i] * pa[j] for i in range(25) for j in range(25) if i + j >= 3)
    assert probabilities["btts"] == pytest.approx(btts)
    assert probabilities["over25"] == pytest.approx(over)
    assert goal_probabilities(0, 0) == {"btts": 0, "over25": 0}


def test_same_day_and_future_results_cannot_change_predictions():
    seed = [match("2023-01-01"), match("2023-01-02", "C", "D")]
    first = match("2024-08-01", "A", "C", 1, 1, "2024-25")
    second = match("2024-08-01", "B", "D", 0, 0, "2024-25")
    baseline = walk_forward(seed + [first, second])
    changed = walk_forward(seed + [{**first, "home_goals": 20}, second, match("2025-01-01")])
    expected = {r["match_id"]: r for r in baseline}
    for row in changed[:2]:
        old = expected[row["match_id"]]
        assert row["p_btts"] == old["p_btts"]
        assert row["p_over25"] == old["p_over25"]
        assert row["training_matches"] == 2
        assert row["training_max_date"] < row["match_date"]


def test_unknown_teams_have_finite_shrunk_rates():
    model = GoalRates()
    model.update(match("2023-01-01", hg=0, ag=0))
    prediction = model.predict("NEW1", "NEW2", "2024-08-01")
    assert prediction["lambda_home"] > 0
    assert 0 <= prediction["p_btts"] <= 1
    with pytest.raises(ValueError):
        model.predict("A", "B", "2023-01-01")


def test_brier_and_log_loss_known_values():
    scores = binary_scores([{"btts": 1, "p_btts": 0.5}, {"btts": 0, "p_btts": 0.5}], "btts")
    assert scores["brier"] == 0.25
    assert scores["log_loss"] == pytest.approx(math.log(2))


def test_missing_key_and_query_secret_redaction():
    with pytest.raises(ValueError, match="Missing"):
        endpoint("odds", "sports", {}, {})
    secret = "test-private-key"
    public, request, _ = endpoint("odds", "odds", {}, {"ODDS_API_KEY": secret})
    assert secret not in public
    assert secret in request.full_url
    assert "regions=uk" in public and "markets=h2h%2Ctotals" in public
    assert scrub({"api_key": secret, "message": "bad " + secret}, secret) == {"api_key": "[REDACTED]", "message": "bad [REDACTED]"}


def test_env_is_parsed_as_text_without_execution(monkeypatch, tmp_path):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ODDS_API_KEY='literal$(no_execution)'\nOTHER_KEY=ignored\n", encoding="utf-8")
    assert load_keys(tmp_path)["ODDS_API_KEY"] == "literal$(no_execution)"

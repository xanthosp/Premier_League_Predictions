"""A fixed, smoothed rate-ratio baseline; not fitted Dixon-Coles or a value model."""
import csv
import json
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from ..ingest.openfootball import load_prepared


def goal_probabilities(home_rate, away_rate):
    if not all(math.isfinite(v) and v >= 0 for v in (home_rate, away_rate)):
        raise ValueError("Poisson rates must be finite and nonnegative")
    total = home_rate + away_rate
    return {"btts": (-math.expm1(-home_rate)) * (-math.expm1(-away_rate)),
            "over25": 1 - math.exp(-total) * (1 + total + total * total / 2)}


class GoalRates:
    def __init__(self, prior_games=10):
        if not math.isfinite(prior_games) or prior_games <= 0:
            raise ValueError("prior_games must be finite and positive")
        self.prior = prior_games
        self.teams = defaultdict(lambda: {k: 0 for k in ("hn", "an", "hgf", "hga", "agf", "aga")})
        self.n = self.home_goals = self.away_goals = 0
        self.latest_date = None

    def update(self, row):
        home, away = self.teams[row["home_team"]], self.teams[row["away_team"]]
        hg, ag = row["home_goals"], row["away_goals"]
        home["hn"] += 1
        home["hgf"] += hg
        home["hga"] += ag
        away["an"] += 1
        away["agf"] += ag
        away["aga"] += hg
        self.n += 1
        self.home_goals += hg
        self.away_goals += ag
        self.latest_date = max(self.latest_date or row["match_date"], row["match_date"])

    def predict(self, home_team, away_team, prediction_date):
        if not self.n or self.latest_date >= prediction_date:
            raise ValueError("Training data must exist and strictly precede the prediction date")
        # One weak global pseudo-match with one goal per side prevents zero rates.
        mu_h = (self.home_goals + 1) / (self.n + 1)
        mu_a = (self.away_goals + 1) / (self.n + 1)
        home, away = self.teams[home_team], self.teams[away_team]
        home_scoring = (home["hgf"] + self.prior * mu_h) / (home["hn"] + self.prior)
        away_conceding = (away["aga"] + self.prior * mu_h) / (away["an"] + self.prior)
        away_scoring = (away["agf"] + self.prior * mu_a) / (away["an"] + self.prior)
        home_conceding = (home["hga"] + self.prior * mu_a) / (home["hn"] + self.prior)
        lh, la = home_scoring * away_conceding / mu_h, away_scoring * home_conceding / mu_a
        return {"lambda_home": lh, "lambda_away": la,
                **{f"p_{k}": v for k, v in goal_probabilities(lh, la).items()},
                **{f"league_p_{k}": v for k, v in goal_probabilities(mu_h, mu_a).items()},
                "training_matches": self.n, "training_max_date": self.latest_date}


def walk_forward(rows, evaluation_start="2024-07-01", prior_games=10):
    model, predictions = GoalRates(prior_games), []
    rows = sorted(rows, key=lambda row: (row["match_date"], row["match_id"]))
    for day, matches in groupby(rows, key=lambda row: row["match_date"]):
        matches = list(matches)
        if day >= evaluation_start:
            # Predict the entire date before updating, so same-day results cannot leak.
            for row in matches:
                predictions.append({**row, **model.predict(row["home_team"], row["away_team"], day)})
        for row in matches:
            model.update(row)
    return predictions


def binary_scores(rows, target, prefix="p_"):
    if not rows:
        raise ValueError("Cannot score an empty evaluation set")
    probabilities = [min(1 - 1e-15, max(1e-15, row[prefix + target])) for row in rows]
    actual = [row[target] for row in rows]
    return {"n": len(rows), "observed_rate": sum(actual) / len(rows),
            "mean_prediction": sum(probabilities) / len(rows),
            "brier": sum((p - y) ** 2 for p, y in zip(probabilities, actual)) / len(rows),
            "log_loss": -sum(y * math.log(p) + (1 - y) * math.log1p(-p) for p, y in zip(probabilities, actual)) / len(rows)}


def run_baseline(root):
    root = Path(root)
    predictions = walk_forward(load_prepared(root))
    if not predictions:
        raise ValueError("No evaluation matches; prepare the historical dataset first")
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "baseline_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    scores = {}
    for season in sorted({row["season"] for row in predictions}):
        subset = [row for row in predictions if row["season"] == season]
        scores[season] = {target: {"team_rate_poisson": binary_scores(subset, target),
                                  "league_only_poisson": binary_scores(subset, target, "league_p_")}
                         for target in ("btts", "over25")}
    report = {"method": "Independent Poisson with venue-specific, smoothed team rate ratios",
              "warmup": "2022-23 and 2023-24", "validation_season": "2024-25", "test_season": "2025-26",
              "evaluation": "Walk-forward by calendar date; earlier completed dates only. Updates within evaluation seasons are allowed after the date ends.",
              "fixed_prior_games": 10, "hyperparameter_search": False,
              "predictions": len(predictions), "scores": scores,
              "limitations": ["This is an initial baseline, not evidence of betting value.", "No odds, ROI or staking simulation.", "Goal independence is an assumption; no Dixon-Coles correction.", "No time decay or player features yet.", "Future choices must not be tuned using the 2025-26 results already observed here; reserve new future data for a fresh final test."]}
    (output / "baseline_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

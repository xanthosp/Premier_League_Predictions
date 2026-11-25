"""Shared walk-forward harness and the leaderboard.

Every model implements `fit(train)` and `predict(test) -> (n, 3) array of H/D/A probabilities`. For each
calendar week (Monday to Sunday) of the evaluation split, the model is refit on every played match dated before
that Monday and predicts the week's matches. Features in `features_match` are already as of kickoff.

Splits are fixed (docs/PLAN.md): all choices are made on `validation`; `test` is for final candidates only and
needs an explicit flag; 2026-27 onwards is the untouched forward period.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .market import OUTCOMES, market_probabilities
from .metrics import blend_test, brier, calibration_error, log_losses, one_hot, rps

SPLITS = {"validation": ("2019-20", "2022-23"), "test": ("2023-24", "2025-26")}
FIRST_TRAIN_SEASON = "2017-18"  # 2016-17 only warms up the as-of features


def load_dataset(root: Path) -> pd.DataFrame:
    """features_match + result + de-vigged market probabilities (close_*, open_*) for every match."""
    with sqlite3.connect(Path(root) / "data/football.sqlite") as connection:
        features = pd.read_sql("SELECT * FROM features_match", connection)
        results = pd.read_sql("SELECT match_id, fthg, ftag, ftr FROM matches", connection)
        odds = pd.read_sql("SELECT match_id, bookmaker, market, is_closing, selection, price FROM odds WHERE market = '1x2'",
                           connection)
    data = features.merge(results, on="match_id", how="left")
    for kind in ("close", "open"):
        data = data.merge(market_probabilities(odds, kind), on="match_id", how="left")
    return data


def weekly_folds(data: pd.DataFrame, league: str, split: str):
    """Yield (train, test) per week. train: played matches (all leagues) before the week's Monday."""
    first, last = SPLITS[split]
    played = data[data.played.astype(bool) & (data.season >= FIRST_TRAIN_SEASON)]
    dates = pd.to_datetime(played.date)
    target = played[(played.league == league) & played.season.between(first, last)]
    mondays = (pd.to_datetime(target.date) - pd.to_timedelta(pd.to_datetime(target.date).dt.weekday, unit="D")).dt.normalize()
    for monday, test in target.groupby(mondays):
        train = played[dates < monday]
        if len(train):
            yield train, test


def walk_forward(model, data: pd.DataFrame, league: str, split: str, train_leagues: list[str] | None = None) -> pd.DataFrame:
    predictions = []
    for train, test in weekly_folds(data, league, split):
        if train_leagues is not None:
            train = train[train.league.isin(train_leagues)]
        model.fit(train)
        p = np.asarray(model.predict(test), dtype=float)
        if p.shape != (len(test), 3) or not np.isfinite(p).all():
            raise ValueError(f"{type(model).__name__} returned invalid probabilities for week of {test.date.min()}")
        predictions.append(test[["match_id", "date", "season", "home_team", "away_team", "ftr"]
                                + [f"{k}_{o}" for k in ("close", "open") for o in OUTCOMES]].assign(
            p_H=p[:, 0], p_D=p[:, 1], p_A=p[:, 2]))
    return pd.concat(predictions, ignore_index=True)


def score(predictions: pd.DataFrame) -> dict:
    """Model metrics, both market benchmarks on the same matches, paired log-loss difference and the blend test."""
    rows = predictions.dropna(subset=[f"close_{o}" for o in OUTCOMES])
    y = one_hot(rows.ftr)
    p = rows[["p_H", "p_D", "p_A"]].to_numpy()
    close = rows[[f"close_{o}" for o in OUTCOMES]].to_numpy()
    model_ll, close_ll = log_losses(p, y), log_losses(close, y)
    diff = model_ll - close_ll
    out = {"n": len(rows), "logloss": model_ll.mean(), "rps": rps(p, y).mean(), "brier": brier(p, y).mean(),
           "ece": calibration_error(p, y), "close_logloss": close_ll.mean(), "close_rps": rps(close, y).mean(),
           "ll_minus_close": diff.mean(), "ll_minus_close_se": diff.std(ddof=1) / np.sqrt(len(diff))}
    with_open = rows.dropna(subset=[f"open_{o}" for o in OUTCOMES])
    if len(with_open):
        open_p = with_open[[f"open_{o}" for o in OUTCOMES]].to_numpy()
        out["open_logloss_same_rows"] = log_losses(open_p, one_hot(with_open.ftr)).mean()
        out["n_open"] = len(with_open)
    out.update(blend_test(close, p, y))
    return {k: round(float(v), 5) if isinstance(v, (float, np.floating)) else v for k, v in out.items()}


def _git_commit(root: Path) -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "football_ml"], cwd=root, capture_output=True, text=True).stdout
        return commit + ("+dirty" if dirty.strip() else "")
    except OSError:
        return "unknown"


def evaluate(root: Path, model, league: str = "E0", split: str = "validation", train_leagues: list[str] | None = None,
             confirm_test: bool = False) -> dict:
    """Run the harness, append one row to outputs/leaderboard.csv and save the predictions."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    if split == "test" and not confirm_test:
        raise ValueError("the test split is for final candidates only: pass --confirm-test")
    root = Path(root)
    predictions = walk_forward(model, load_dataset(root), league, split, train_leagues)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": _git_commit(root),
              "model": model.name, "params": json.dumps(getattr(model, "params", {}), sort_keys=True), "league": league,
              "split": split, "train_leagues": ",".join(train_leagues) if train_leagues else "all", **score(predictions)}
    out = root / "outputs"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions" / f"{model.name}_{league}_{split}.csv", index=False)
    board = out / "leaderboard.csv"
    row = pd.DataFrame([result])
    if board.exists():
        row = pd.concat([pd.read_csv(board), row], ignore_index=True)
    row.to_csv(board, index=False)
    return result

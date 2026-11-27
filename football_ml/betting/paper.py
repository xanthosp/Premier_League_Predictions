"""Paper trading: backtest of the decision rule, then a forward ledger that is never rewritten.

`backtest`: walk-forward over 2018-19..2022-23 (validation; the test seasons stay untouched). The model weight w of
each season is fitted on the earlier seasons only, so 2018-19 only fits and bets start in 2019-20. Writes the bets
and `policy.json` (w fitted on all five seasons) that the forward run uses.

`run` (every day, after `update`): settle -> lock -> report.
- lock: EPL matches of today/tomorrow in the latest fixtures snapshot that are not locked yet. The model is refit on
  every played match, mixed with the snapshot's open prices, and the forecast and bets are appended to
  outputs/paper/forecasts.csv and bets.csv (append-only: a locked row is never changed).
- settle: bets of matches whose result is in the database get profit and CLV, appended to settlements.csv.
- report: P&L and CLV per week (outputs/paper/report.csv).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate.walkforward import FIRST_TRAIN_SEASON, _git_commit, load_dataset, weekly_folds
from ..ingest import odds_api
from ..ingest.football_data import parse_fixtures
from ..models.ml import PoissonGLM
from . import policy
from .markets import market_matrix, market_prices, market_rates, over_probability, payoff

LEAGUE = "E0"
BACKTEST_SEASONS = ("2018-19", "2022-23")
TEST_SEASONS = ("2023-24", "2025-26")
# Fixed before the test seasons were seen (validation, "Mine" prices): main candidate first.
TEST_CANDIDATES = ("move_clv1", "stack_p10_move")
API_BOOKS = ("pinnacle", "betano")              # from the Odds API, filled automatically on the sheet
USER_BOOKS = ("stoiximan", "superbet", "bet365", "novibet")  # filled by hand on the sheet
FORWARD_STRATEGY = "move_clv1"  # test 2023-26: CLV +0.9% ± 1.1, skill +5.3% (docs/PLAN.md)
PARALLEL_STRATEGY = "consensus_ev1"  # Phase 5 ablation (validation only, test already spent): locked in parallel


def _folder(root: Path) -> Path:
    folder = Path(root) / "outputs/paper"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _model():
    return PoissonGLM(drop="goals")  # Phase 3 candidate


def _odds(root: Path) -> pd.DataFrame:
    with sqlite3.connect(Path(root) / "data/football.sqlite") as connection:
        return pd.read_sql("SELECT match_id, bookmaker, market, is_closing, selection, line, price FROM odds "
                           f"WHERE match_id LIKE '{LEAGUE}_%'", connection)


def _with_market(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach prices and the market's implied expected goals; drops matches without a full 1X2 price."""
    frame = frame.merge(prices, on="match_id", how="inner").dropna(subset=["fair_1x2_H", "fair_1x2_D", "fair_1x2_A"])
    frame = frame.reset_index(drop=True)
    frame["market_lh"], frame["market_la"] = market_rates(frame)
    return frame


def _close_matrix(bets: pd.DataFrame, close: pd.DataFrame) -> np.ndarray:
    """The closing market's exact scoreline matrix for each bet row (NaN where no closing price)."""
    rows = bets[["match_id"]].merge(_with_market(bets[["match_id"]].drop_duplicates(), close), on="match_id", how="left")
    return market_matrix(rows.fillna({"market_lh": 1.0, "market_la": 1.0}))


def _summary(bets: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = bets.groupby(by)
    out = pd.DataFrame({"bets": g.size(), "staked": g.stake.sum(), "profit": g.profit.sum(),
                        "kelly_staked": g.kelly_stake.sum(), "kelly_profit": g.kelly_profit.sum(),
                        "mean_ev": g.ev.mean(), "mean_clv": g.clv.mean(), "clv_positive": g.clv.apply(lambda c: (c > 0).mean())})
    out["roi"] = out.profit / out.staked
    return out.round(4).reset_index()


def _match_frame(data: pd.DataFrame, odds: pd.DataFrame, seasons: tuple[str, str]) -> pd.DataFrame:
    """Walk-forward model rates for every EPL match of `seasons`, with open prices and the closing market's rates."""
    rows = []
    for train, test in weekly_folds(data, LEAGUE, seasons):
        model = _model()
        model.fit(train)
        lh, la = model.rates(test)
        rows.append(test[["match_id", "date", "season", "fthg", "ftag"]].assign(model_lh=lh, model_la=la))
    frame = _with_market(pd.concat(rows, ignore_index=True), market_prices(odds, False, policy.OPEN_BOOKS))
    closing = _with_market(frame[["match_id"]], market_prices(odds, True, policy.CLOSE_BOOKS))
    return frame.merge(closing[["match_id", "market_lh", "market_la"]].rename(
        columns={"market_lh": "close_lh", "market_la": "close_la"}), on="match_id", how="left")


def backtest(root: Path) -> dict:
    root = Path(root)
    data = load_dataset(root)
    odds = _odds(root)
    frame = _match_frame(data, odds, BACKTEST_SEASONS)
    close = market_prices(odds, True, policy.CLOSE_BOOKS)

    weights, all_bets = {}, []
    seasons = sorted(frame.season.unique())
    for season in seasons[1:]:
        past = frame[frame.season < season]
        w = policy.fit_weight((past.model_lh, past.model_la), (past.market_lh, past.market_la), past.fthg, past.ftag)
        weights[season] = w
        current = frame[frame.season == season].reset_index(drop=True)
        bets = policy.decide(current, w).merge(current[["match_id", "date", "season", "fthg", "ftag"]], on="match_id")
        all_bets.append(bets)
    bets = pd.concat(all_bets, ignore_index=True)
    bets = bets.join(policy.settle(bets, bets.fthg, bets.ftag, _close_matrix(bets, close)))

    folder = _folder(root)
    bets.to_csv(folder / "backtest_bets.csv", index=False)
    # Every match (not only bets) with model, open and closing rates: the input of error analysis.
    frame.to_csv(folder / "backtest_matches.csv", index=False)
    w_all = policy.fit_weight((frame.model_lh, frame.model_la), (frame.market_lh, frame.market_la), frame.fthg, frame.ftag)
    by_market = _summary(bets, ["market"])
    by_season = _summary(bets, ["season"])
    by_market.to_csv(folder / "backtest_by_market.csv", index=False)
    return {"matches": int((frame.season > seasons[0]).sum()), "weight_by_season": weights, "w_forward": w_all,
            "total": _summary(bets.assign(all="all"), ["all"]).drop(columns="all").to_dict("records")[0],
            "by_market": by_market.to_dict("records"), "by_season": by_season.to_dict("records")}


def fit_policy(root: Path) -> dict:
    """The forward decision rule (FORWARD_STRATEGY, chosen on validation, checked once on test): fits the movement
    model on every played EPL match since 2018-19 (walk-forward model rates) and writes outputs/paper/policy.json.
    Rerun every few weeks so the new matches enter the movement model; the goal model itself is refit at every lock."""
    from . import strategy
    root = Path(root)
    data, odds = load_dataset(root), _odds(root)
    last = data[data.played.astype(bool) & (data.league == LEAGUE)].season.max()
    frame = _match_frame(data, odds, (BACKTEST_SEASONS[0], last))
    frame = _strategy_inputs(root, frame, odds)[0]
    move = strategy.MoveModel().fit(frame)
    source, min_ev = strategy.STRATEGIES[FORWARD_STRATEGY][:2]
    config = {"strategy": FORWARD_STRATEGY, "source": source, "min_ev": min_ev, "prices": "Mine (max B365/BW 1x2, B365 ou25/ah)",
              "model": _model().name, "move_model": move.to_dict(), "move_rows": int(frame.move_ok.sum()),
              "fitted_on": [BACKTEST_SEASONS[0], last], "last_match": str(frame.date.max()),
              "stake": policy.STAKE, "kelly_fraction": policy.KELLY_FRACTION, "kelly_cap": policy.KELLY_CAP,
              "bankroll": policy.BANKROLL, "open_books": policy.OPEN_BOOKS["1x2"], "commit": _git_commit(root),
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (_folder(root) / "policy.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    return {k: v for k, v in config.items() if k != "move_model"}


def _feature_diffs(data: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away pre-match features used by the movement model."""
    def europe(side):
        return data[f"{side}_prev_europe"].fillna(0) + data[f"{side}_next_europe"].fillna(0)
    return pd.DataFrame({
        "match_id": data.match_id, "elo_diff": data.home_elo - data.away_elo,
        "npxg_for": data.home_npxgf - data.away_npxgf, "npxg_against": data.home_npxga - data.away_npxga,
        "shots_for": data.home_sf - data.away_sf, "shots_against": data.home_sa - data.away_sa,
        "rest_days": data.home_rest_days - data.away_rest_days, "matches_14d": data.home_matches_14d - data.away_matches_14d,
        "europe": europe("home") - europe("away"), "missing_xg": data.home_missing_xg_share - data.away_missing_xg_share,
        "load_8d": data.home_regulars_load_8d - data.away_regulars_load_8d})


def _mine_prices(odds: pd.DataFrame) -> pd.DataFrame:
    """"Mine": the better of Bet365 and bwin (1X2), Bet365 for O/U and AH: a proxy for a few real accounts."""
    b365 = market_prices(odds, False, {m: ["B365"] for m in policy.OPEN_BOOKS})
    bw = market_prices(odds, False, {"1x2": ["BW"], "ou25": [], "ah": []})
    bw = bw[["match_id", *[c for c in ("1x2_H", "1x2_D", "1x2_A") if c in bw]]]
    mine = b365.merge(bw, on="match_id", how="left", suffixes=("", "_bw"))
    for s in ("H", "D", "A"):
        if f"1x2_{s}_bw" in mine:
            mine[f"1x2_{s}"] = mine[[f"1x2_{s}", f"1x2_{s}_bw"]].max(axis=1)
    return mine.drop(columns=[c for c in mine if c.endswith("_bw")])


def _strategy_inputs(root: Path, frame: pd.DataFrame, odds: pd.DataFrame) -> tuple:
    """Features, the closing scoreline matrix and the price frames (Avg, Mine, Max) of the strategy comparison."""
    frame = frame.merge(_feature_diffs(load_dataset(root)), on="match_id", how="left")
    close_rows = frame[["match_id"]].merge(_with_market(frame[["match_id"]], market_prices(odds, True, policy.CLOSE_BOOKS)),
                                           on="match_id", how="left")
    close_matrix = market_matrix(close_rows.fillna({"market_lh": 1.0, "market_la": 1.0}))
    # A price move is only measured where open and close are built from the same markets (1X2 and O/U 2.5).
    frame["move_ok"] = close_rows.fair_ou25_O.notna().to_numpy() & frame.fair_ou25_O.notna().to_numpy()
    best = market_prices(odds, False, {m: ["Max"] for m in policy.OPEN_BOOKS})
    price_cols = [c for c in best.columns if c != "match_id" and not c.startswith("fair_")]
    with_max = frame.drop(columns=[c for c in price_cols if c in frame]).merge(best[["match_id", *price_cols]],
                                                                             on="match_id", how="left")
    mine = _mine_prices(odds)
    mine_cols = [c for c in price_cols if c in mine]
    with_mine = frame.drop(columns=[c for c in mine_cols if c in frame]).merge(mine[["match_id", *mine_cols]],
                                                                              on="match_id", how="left")
    return frame, close_matrix, {"Avg": frame, "Mine": with_mine, "Max": with_max}


def _with_baseline(bets: pd.DataFrame, close_matrix: np.ndarray, price_frames: dict) -> pd.DataFrame:
    """Baseline: CLV of backing every selection at that price. Skill = how much a strategy's picks beat it."""
    from . import strategy
    base = []
    for name, prices in price_frames.items():
        everything = strategy.all_selections(prices)
        everything["clv"] = strategy.ev_under(close_matrix, everything)
        base.append(everything.groupby(["season", "market"]).clv.mean().rename("clv_base").reset_index().assign(prices=name))
    return bets.merge(pd.concat(base), on=["season", "market", "prices"], how="left")


def strategies(root: Path) -> dict:
    """Compare the strategy menu (strategy.STRATEGIES) on the backtest matches, by CLV."""
    from . import strategy
    root = Path(root)
    folder = _folder(root)
    path = folder / "backtest_matches.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m football_ml paper --backtest` first")
    frame, close_matrix, price_frames = _strategy_inputs(root, pd.read_csv(path), _odds(root))
    bets = _with_baseline(strategy.run_strategies(frame, close_matrix, price_frames), close_matrix, price_frames)
    bets.to_csv(folder / "strategy_bets.csv", index=False)
    table = strategy.summarize(bets)
    table.to_csv(folder / "strategy_summary.csv", index=False)
    return {"strategies": table.to_dict("records")}


def confirm_test(root: Path, confirm: bool = False) -> dict:
    """The one-time check of the candidates chosen on validation, on the untouched test seasons.
    Same walk-forward as validation (every season fitted on all earlier ones); only test-season bets are reported."""
    from . import strategy
    root = Path(root)
    folder = _folder(root)
    if not confirm:
        raise ValueError("The test seasons are used once: add --confirm-test")
    if (folder / "test_summary.csv").exists():
        raise ValueError("outputs/paper/test_summary.csv exists: the test check already ran (rule 4: no reuse)")
    odds = _odds(root)
    frame = _match_frame(load_dataset(root), odds, (BACKTEST_SEASONS[0], TEST_SEASONS[1]))
    frame, close_matrix, price_frames = _strategy_inputs(root, frame, odds)
    menu = {name: strategy.STRATEGIES[name] for name in TEST_CANDIDATES}
    prices = {"Mine": price_frames["Mine"]}
    bets = strategy.run_strategies(frame, close_matrix, prices, menu)
    bets = _with_baseline(bets[bets.season >= TEST_SEASONS[0]], close_matrix, prices)
    bets.to_csv(folder / "test_bets.csv", index=False)
    table = strategy.summarize(bets).assign(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                            commit=_git_commit(root))
    table.to_csv(folder / "test_summary.csv", index=False)
    return {"test": table.drop(columns=["created_utc", "commit"]).to_dict("records")}


def _append(path: Path, rows: pd.DataFrame) -> None:
    """Append to a ledger file in the column order of its existing header (a new column would misalign it)."""
    if not len(rows):
        return
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        extra = sorted(set(rows.columns) - set(header))
        if extra:
            raise ValueError(f"{path.name}: new columns {extra}; ledger files keep their header, write them elsewhere")
        rows = rows.reindex(columns=header)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def lock(root: Path, today: date | None = None) -> dict:
    root, today = Path(root), today or date.today()
    folder = _folder(root)
    policy_path = folder / "policy.json"
    if not policy_path.exists():
        raise FileNotFoundError("No outputs/paper/policy.json; run `python -m football_ml paper --fit-policy` first")
    config = json.loads(policy_path.read_text(encoding="utf-8"))
    if config.get("strategy") != FORWARD_STRATEGY:
        raise ValueError("policy.json is from an older decision rule; run `python -m football_ml paper --fit-policy`")
    snapshots = sorted((root / "data/raw/football_data/fixtures").glob("*.csv"))
    if not snapshots:
        return {"locked": 0, "note": "no fixtures snapshot yet (run update)"}
    fixtures, odds = parse_fixtures(snapshots[-1], leagues=(LEAGUE,))
    locked = _read(folder / "forecasts.csv")
    known = set(locked.match_id) if len(locked) else set()
    window = fixtures[(fixtures.date >= today.isoformat()) & (fixtures.date <= (today + timedelta(days=1)).isoformat())
                      & ~fixtures.match_id.isin(known)]
    if window.empty:
        return {"locked": 0, "snapshot": snapshots[-1].name, "note": "no unlocked EPL match today or tomorrow"}

    data = load_dataset(root)
    upcoming = data[~data.played.astype(bool) & (data.league == LEAGUE)]
    # Same id normally; fall back to the pairing (each pairing happens once per season) if the date moved.
    target = window[["match_id"]].merge(upcoming, on="match_id", how="inner")
    moved = window[~window.match_id.isin(target.match_id)]
    if len(moved):
        alt = moved[["match_id", "date", "home_team", "away_team"]].merge(
            upcoming.drop(columns=["match_id", "date"]), on=["home_team", "away_team"], how="inner")
        target = pd.concat([target, alt[target.columns.intersection(alt.columns)]], ignore_index=True)
    missing = sorted(set(window.match_id) - set(target.match_id))
    if target.empty:
        return {"locked": 0, "missing_features": missing}

    train = data[data.played.astype(bool) & (data.season >= FIRST_TRAIN_SEASON)]
    model = _model()
    model.fit(train)
    lh, la = model.rates(target)
    from . import strategy
    target = target.assign(model_lh=lh, model_la=la).merge(_feature_diffs(target), on="match_id", how="left")
    keep = ["match_id", "date", "home_team", "away_team", "model_lh", "model_la", *strategy.MOVE_FEATURES[4:]]
    frame = _with_market(target[keep], market_prices(odds, False, policy.OPEN_BOOKS))
    if frame.empty:
        return {"locked": 0, "no_prices": sorted(target.match_id), "missing_features": missing}
    # Predicted closing market (open + predicted move); bets where our price beats it by min_ev.
    m, d = strategy.move_matrix(frame, strategy.MoveModel.from_dict(config["move_model"]))
    mine = _mine_prices(odds)
    price_cols = [c for c in mine if c != "match_id" and not c.startswith("fair_")]
    prices = frame.drop(columns=[c for c in price_cols if c in frame]).merge(mine[["match_id", *price_cols]],
                                                                         on="match_id", how="left")
    diff = np.subtract.outer(np.arange(m.shape[1]), np.arange(m.shape[2]))
    p = np.column_stack([(m * (diff > 0)).sum(axis=(1, 2)), (m * (diff == 0)).sum(axis=(1, 2)), (m * (diff < 0)).sum(axis=(1, 2))])
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    forecasts = frame[["match_id", "date", "home_team", "away_team", "model_lh", "model_la", "market_lh", "market_la"]].assign(
        pred_d_sup=d[:, 0], pred_d_tot=d[:, 1], p_H=p[:, 0], p_D=p[:, 1], p_A=p[:, 2], p_over25=over_probability(m),
        strategy=config["strategy"], locked_utc=stamp, snapshot=snapshots[-1].name, commit=_git_commit(root)).round(4)
    # Two rules in parallel ledgers: the forward rule (predicted close) and the consensus rule (de-vigged open,
    # Phase 5: CLV +4.7% vs +2.9% on validation, untested). Same prices and stakes; bets.csv has a strategy column.
    matrices = {config["strategy"]: m, PARALLEL_STRATEGY: market_matrix(frame)}
    bets, cards = [], {}
    for name, matrix in matrices.items():
        min_ev = config["min_ev"] if name == config["strategy"] else strategy.STRATEGIES[name][1]
        cand = strategy.candidates(prices, matrix)
        cand = cand[cand.ev >= min_ev].drop(columns="row").reset_index(drop=True)
        b = cand.assign(strategy=name, book=cand.market.map({"1x2": "B365/BW", "ou25": "B365", "ah": "B365"}),
                        stake=config["stake"], locked_utc=stamp,
                        kelly_stake=np.minimum(config["kelly_fraction"] * cand.ev.clip(lower=0) / (cand.price - 1)
                                               * config["bankroll"], config["kelly_cap"]).round(2))
        b.insert(0, "bet_id", name + "|" + b.match_id + "|" + b.market + "|" + b.selection)
        bets.append(b)
        # Price card: fair and minimum acceptable price of every selection, to compare with any bookmaker by hand.
        c = strategy.price_card(prices, matrix, min_ev).round(3)
        c["bet"] = (name + "|" + c.match_id + "|" + c.market + "|" + c.selection).isin(b.bet_id)
        cards[name] = frame[["match_id", "date", "home_team", "away_team"]].merge(c, on="match_id").assign(locked_utc=stamp)
    bets = pd.concat(bets, ignore_index=True)
    card, consensus = cards[config["strategy"]], cards[PARALLEL_STRATEGY]
    # Pinnacle / Betano UK (Stoiximan's group) at lock time, from today's Odds API snapshot when there is one.
    api = odds_api.latest(root)
    if len(api) and api.snapshot.iloc[0][:8] == datetime.now(timezone.utc).strftime("%Y%m%d"):
        card = card.merge(odds_api.wide_for_card(api), on=["match_id", "market", "selection"], how="left")
    for book in API_BOOKS:
        if book not in card:
            card[book] = np.nan
    _append(folder / "forecasts.csv", forecasts)
    _append(folder / "bets.csv", bets)
    _append(folder / "cards.csv", card)
    _append(folder / "cards_consensus.csv", consensus)
    key = ["match_id", "market", "selection"]
    card = card.merge(consensus[key + ["fair_price", "min_price", "bet"]].rename(
        columns={"fair_price": "consensus_fair", "min_price": "consensus_min", "bet": "bet_consensus"}), on=key, how="left")
    _write_sheet(folder, card, stamp)
    show = card.assign(match=card.home_team + " - " + card.away_team)
    return {"locked": len(forecasts),
            "bets": bets[["strategy", "match_id", "market", "selection", "line", "price", "ev"]].round(3).to_dict("records"),
            "card": show[["match", "market", "selection", "line", "fair_price", "min_price", "consensus_min", "our_price",
                          "bet", "bet_consensus"]].to_dict("records"),
            "no_prices": sorted(set(target.match_id) - set(frame.match_id)), "missing_features": missing}


def _write_sheet(folder: Path, card: pd.DataFrame, stamp: str) -> Path:
    """A sheet for the user: fill the prices seen at your bookmakers (and placed_book/stake if you bet).
    `my_prices` reads every sheet back and measures each bookmaker against the predicted and the real close."""
    sheet = card.assign(match=card.home_team + " - " + card.away_team)[
        ["match_id", "date", "match", "market", "selection", "line", "fair_price", "min_price", "consensus_fair",
         "consensus_min", "our_price", *API_BOOKS, "bet", "bet_consensus"]]
    sheet = sheet.rename(columns={"our_price": "b365_bw"}).assign(**{b: np.nan for b in USER_BOOKS}, placed_book="", stake=np.nan)
    path = folder / "sheets" / f"{stamp[:16].replace(':', '').replace('-', '')}.csv"
    path.parent.mkdir(exist_ok=True)
    sheet.to_csv(path, index=False)
    return path


def my_prices(root: Path) -> dict:
    """Every price filled in outputs/paper/sheets/*.csv (plus our B365/bwin, Pinnacle and Betano columns for
    comparison): expected profit against the predicted close, and once the match is played, CLV against the real
    close and the profit of what was placed. Per bookmaker: does it beat the minimum price, and the close?"""
    root = Path(root)
    folder = _folder(root)
    sheets = sorted((folder / "sheets").glob("*.csv"))
    if not sheets:
        return {"prices": 0}
    rows = pd.concat([pd.read_csv(f).assign(sheet=f.name) for f in sheets], ignore_index=True)
    books = [b for b in ("b365_bw", *API_BOOKS, *USER_BOOKS) if b in rows]
    long = rows.melt(id_vars=["match_id", "match", "market", "selection", "line", "fair_price", "min_price",
                              "placed_book", "stake", "sheet"], value_vars=books, var_name="bookmaker", value_name="price")
    long = long.dropna(subset=["price"]).drop_duplicates(["match_id", "market", "selection", "bookmaker"], keep="last")
    if long.empty:
        return {"prices": 0}
    long = long.reset_index(drop=True)
    # Expected profit is linear in the price: 0 at the fair price, min_ev at the minimum price.
    long["ev_vs_predicted_close"] = 0.01 * (long.price - long.fair_price) / (long.min_price - long.fair_price)
    long["beats_min"] = long.price >= long.min_price
    if {"consensus_fair", "consensus_min"} <= set(rows):  # sheets from before the parallel rule lack these
        cons = rows.drop_duplicates(["match_id", "market", "selection"], keep="last")[
            ["match_id", "market", "selection", "consensus_fair", "consensus_min"]]
        long = long.merge(cons, on=["match_id", "market", "selection"], how="left")
        long["ev_vs_consensus"] = 0.01 * (long.price - long.consensus_fair) / (long.consensus_min - long.consensus_fair)
        long = long.drop(columns=["consensus_fair", "consensus_min"])
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        results = pd.read_sql("SELECT match_id, fthg, ftag FROM matches", connection)
    played = long.merge(results, on="match_id", how="inner").reset_index(drop=True)
    long["clv"], long["profit"] = np.nan, np.nan
    if len(played):
        close = market_prices(_odds(root), True, policy.CLOSE_BOOKS)
        clv = policy.evaluate_bets(_close_matrix(played, close), played)
        placed = (played.placed_book.astype(str).str.lower() == played.bookmaker) & played.stake.notna()
        unit = np.array([float(payoff(r.market, r.selection, r.line, r.price, r.fthg, r.ftag)) for r in played.itertuples()])
        key = ["match_id", "market", "selection", "bookmaker"]
        extra = played[key].assign(clv=clv, profit=np.where(placed, unit * played.stake.fillna(0), np.nan))
        long = long.drop(columns=["clv", "profit"]).merge(extra, on=key, how="left")
    long.to_csv(folder / "my_prices_report.csv", index=False)
    g = long.groupby("bookmaker")
    table = pd.DataFrame({"prices": g.size(), "beats_min": g.beats_min.mean(), "ev_vs_predicted_close": g.ev_vs_predicted_close.mean(),
                          "settled": g.clv.count(), "clv": g.clv.mean(), "placed": g.profit.count(), "profit": g.profit.sum()})
    if "ev_vs_consensus" in long:
        table.insert(3, "ev_vs_consensus", g.ev_vs_consensus.mean())
    return {"by_bookmaker": table.round(4).reset_index().to_dict("records")}


def settle_open(root: Path) -> dict:
    root = Path(root)
    folder = _folder(root)
    bets, done = _read(folder / "bets.csv"), _read(folder / "settlements.csv")
    if bets.empty:
        return {"settled": 0}
    open_bets = bets[~bets.bet_id.isin(done.bet_id)] if len(done) else bets
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        results = pd.read_sql("SELECT match_id, fthg, ftag FROM matches", connection)
    ready = open_bets.merge(results, on="match_id", how="inner").reset_index(drop=True)
    if ready.empty:
        return {"settled": 0, "open": len(open_bets)}
    close = market_prices(_odds(root), True, policy.CLOSE_BOOKS)
    settled = policy.settle(ready, ready.fthg, ready.ftag, _close_matrix(ready, close))
    rows = ready[["bet_id", "fthg", "ftag"]].join(settled.round(4)).assign(
        settled_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _append(folder / "settlements.csv", rows)
    return {"settled": len(rows), "open": len(open_bets) - len(rows)}


def report(root: Path) -> dict:
    folder = _folder(Path(root))
    bets, done = _read(folder / "bets.csv"), _read(folder / "settlements.csv")
    if bets.empty or done.empty:
        return {"settled_bets": 0, "open_bets": len(bets)}
    merged = bets.merge(done, on="bet_id")
    merged["strategy"] = merged.strategy.fillna(FORWARD_STRATEGY) if "strategy" in merged else FORWARD_STRATEGY
    match_dates = pd.to_datetime(merged.match_id.str.split("_").str[1], format="%Y%m%d")
    merged["week"] = (match_dates - pd.to_timedelta(match_dates.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    weekly = _summary(merged, ["strategy", "week"])
    weekly["cum_profit"] = weekly.groupby("strategy").profit.cumsum()
    weekly["cum_kelly_profit"] = weekly.groupby("strategy").kelly_profit.cumsum()
    weekly.to_csv(folder / "report.csv", index=False)
    return {"by_strategy": _summary(merged, ["strategy"]).to_dict("records"),
            "by_market": _summary(merged, ["strategy", "market"]).to_dict("records"),
            "last_weeks": weekly.tail(4).to_dict("records"), "open_bets": len(bets) - len(merged)}


def run(root: Path) -> dict:
    return {"settle": settle_open(root), "lock": lock(root), "report": report(root), "my_prices": my_prices(root)}

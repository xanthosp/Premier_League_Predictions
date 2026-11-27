# Football_ML: guide for Claude

Personal research project: pre-match probabilities for the Premier League, with the goal of finding a
measurable betting edge (value). The user writes in Greek; answer in Greek. The user has a limited
Claude usage budget: be economical (read only what the task needs, no long reports, and have the user run
long jobs locally).

## Map

| Path | What it is |
|---|---|
| `football_ml/` | The only active code. Subpackages: `ingest` → `features` → `models` → `evaluate` → `betting`. |
| `data/raw/` | Immutable downloads. **Never edit.** Sources are listed below. |
| `data/processed/`, `data/football.sqlite` | Derived and regenerable. |
| `outputs/` | Results of new runs: `leaderboard.csv` (one row per evaluation) and `predictions/`. |
| `tests/` | pytest suite for the active code. |
| `docs/PLAN.md` | Phased plan and current status. **Read this first when starting new work.** |
| `docs/DATA_SOURCES.md`, `docs/FREE_API_FINDINGS.md` | Earlier research on free sources. |
| `archive/` | Frozen experiments from 26/10–14/11/2026. **Do not read unless asked.** Index: `archive/INDEX.md`. Paths in the scripts there are broken. |

## Raw data (`data/raw/`)

- `football_data/<DIV>/<SSSS>_<DIV>.csv`: football-data.co.uk, 6 leagues (E0 EPL, E1 Championship, SP1, D1, I1, F1),
  2016-17 onward. `update` downloads only missing seasons and the current one. Parsed into the SQLite tables:
  - `matches` (24k rows): results, shots, corners, cards. xG only from 2026-27.
  - `odds` (long format): bookmaker × market (1x2/ou25/ah) × is_closing × selection × line.
    Sharp closing: **PS** (Pinnacle) up to mid-2025-26, **BFE** (Betfair Exchange) from 2024-25 onward.
    Avg/Max closing from 2019-20 onward. Coverage: `outputs/coverage_football_data.csv`.
- `pannadata/` (github peteowen1/pannadata, Opta): `fixtures.parquet` (all ~100 competitions, also upcoming),
  `shot_events.parquet` (per-shot xG, our 6 leagues, 2015-16 onward), `player_match_shots.parquet` (shooters only), `lineups.parquet` (full matchday squads with minutes for our
  leagues + European club competitions + domestic cups).
  Built into `match_xg` (xG, npxG, shots, big chances per side; 23.7k matches linked, 0 score mismatches) and
  `team_names_opta` (Opta team id → football-data name, paired by date + score).
- `football_data/fixtures/<UTC stamp>.csv`: fixtures.csv snapshots (upcoming matches with pre-match prices), saved by
  `update` when changed. Paper bets are locked at these prices.
- Paper ledger `outputs/paper/` (append-only, not regenerable, never rewrite): `forecasts.csv`, `bets.csv`, `settlements.csv`,
  `cards.csv` (fair and minimum acceptable price of every selection, for manual comparison with Stoiximan/Superbet),
  `sheets/<stamp>.csv` (one per lock; the user fills stoiximan/superbet/bet365/novibet prices, placed_book, stake;
  `my_prices` in `paper` scores them -> `my_prices_report.csv`),
  plus `policy.json` (strategy move_clv1, movement-model coefficients, stakes), `report.csv`.
  Decision rule: `football_ml/betting/strategy.py` (move_clv1: bet where the Mine price beats the predicted close by ≥1%).
  In parallel (same ledger, `strategy` column in bets.csv; card in `cards_consensus.csv`, consensus_* columns on sheets):
  consensus_ev1 = Mine price ≥1% above the de-vigged Avg open (Phase 5 ablation; validation only). Ledger files keep
  their header: `_append` refuses new columns.
- `odds_api/<UTC stamp>.json`: The Odds API (`football_ml/ingest/odds_api.py`), EPL h2h + totals, regions eu+uk
  (Pinnacle, Betfair, Betano UK ...; no bet365/Stoiximan/Superbet). 4 credits per call of 500/month; only called
  when an EPL match is within 2 days. Key only in `.env`.
- `fpl_api/<UTC stamp>/`: one snapshot of the official FPL API per `update` (injury status, news, chance of playing).
  Built into `fpl_availability` (history of all snapshots) and `fpl_fixtures` (latest, incl. upcoming, with team names).
  Also built into `team_schedule` (every match of our clubs in any competition, incl. scheduled) and `player_minutes`.
- Derived: `features_match` (football_ml/features): one row per match incl. upcoming E0 fixtures, as-of Elo, decayed form,
  fixture congestion (all competitions) and player availability/load (missing regulars, missing xG share).
- `fpl_history/`: FPL per-player per-gameweek data, 2019-20..2025-26 (minutes, xG, xA, starts).
- `transfermarkt/`: player valuations and appearances (Kaggle dcaribou dataset, see its LICENSE).
- `openfootball/`: EPL results 2022-23..2025-26 (used only by the old baseline).

## Commands

```powershell
python -m football_ml update     # all sources + rebuild (--offline: rebuild only, --skip-shots: no pannadata download)
python -m football_ml features   # rebuild features_match only (update also does it)
python -m football_ml evaluate --model elo_ordered [--league E1] [--split test --confirm-test]
python -m football_ml daily      # scheduled task (10:00): update + Odds API snapshot + paper; log in outputs/paper/daily_log/
python -m football_ml paper      # daily: settle finished bets, lock today's/tomorrow's EPL bets, report (outputs/paper/)
python -m football_ml paper --fit-policy # fit the forward rule (move_clv1: movement model) -> outputs/paper/policy.json; rerun every few weeks
python -m football_ml paper --backtest   # original blend-rule backtest 2018-19..2022-23 (writes backtest_matches.csv)
python -m football_ml paper --strategies # compare the strategy menu (betting/strategy.py) by CLV, after --backtest
python -m football_ml paper --test --confirm-test  # ONE-TIME check on 2023-26 (already used on 26/11/2026; refuses to rerun)
python -m football_ml prepare    # openfootball -> data/processed + sqlite (old baseline)
python -m football_ml baseline   # simple walk-forward Poisson
python -m pytest -q
python -m ruff check football_ml tests
```

Python 3.12. Libraries available: pandas, numpy, scipy, scikit-learn, xgboost, pyarrow.

## Non-negotiable methodological rules

1. **As-of**: every feature uses only information known before the prediction time. No random splits.
2. **Walk-forward** with periodic refits (weekly). Never a single frozen cutoff for months.
3. **Benchmark = the market**: de-vigged closing odds. Metrics: log loss, RPS, calibration, plus a blend test
   `logit(p) = a*logit(market) + b*logit(model)`. Main forward KPI: **CLV**, not ROI.
4. Never choose a model or threshold by ROI. Don't reuse the same test set for dozens of variants.
5. Locked forecasts are never rewritten after results are known.
6. Features must exist for **all 20 teams** at prediction time. Otherwise they are not usable.
7. Few, well-chosen features over hundreds of correlated ones.

## Work conventions

- Don't print or read large CSV/parquet files; summarize them with pandas (shape, head, describe).
- A new model is a module in `football_ml/models/` with `name`, `params`, `fit(train)`, `predict(test) -> (n,3) H/D/A`,
  registered in `models/simple.py:MODELS`; `evaluate` runs it walk-forward (football_ml/evaluate) and adds a leaderboard row.
  No new folder per experiment.
- Short reports. Commit only when the user asks.

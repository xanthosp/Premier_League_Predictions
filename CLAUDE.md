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
| `outputs/` | Results of new runs. A single `outputs/leaderboard.csv` (to be built in Phase 2). |
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
  `shot_events.parquet` (per-shot xG, our 6 leagues, 2015-16 onward), `player_match_shots.parquet` (player shots/SOT/minutes).
  Built into `match_xg` (xG, npxG, shots, big chances per side; 23.7k matches linked, 0 score mismatches) and
  `team_names_opta` (Opta team id → football-data name, paired by date + score).
- `fpl_api/<UTC stamp>/`: one snapshot of the official FPL API per `update` (injury status, news, chance of playing).
  Built into `fpl_availability` (history of all snapshots) and `fpl_fixtures` (latest, incl. upcoming, with team names).
- `fpl_history/`: FPL per-player per-gameweek data, 2019-20..2025-26 (minutes, xG, xA, starts).
- `transfermarkt/`: player valuations and appearances (Kaggle dcaribou dataset, see its LICENSE).
- `openfootball/`: EPL results 2022-23..2025-26 (used only by the old baseline).

## Commands

```powershell
python -m football_ml update     # all sources + rebuild (--offline: rebuild only, --skip-shots: no pannadata download)
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
- A new model is a module in `football_ml/models/`, runs through the shared harness, and adds a row to the leaderboard.
  No new folder per experiment.
- Short reports. Commit only when the user asks.

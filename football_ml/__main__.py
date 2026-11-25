import argparse
import json
from pathlib import Path

from .evaluate.walkforward import SPLITS, evaluate
from .features import match_features
from .ingest import football_data, fpl, pannadata
from .ingest.api_clients import fetch_api, load_keys
from .ingest.openfootball import prepare
from .models.baseline_poisson import run_baseline
from .models.simple import MODELS


def run_update(root: Path, offline: bool = False, skip_shots: bool = False) -> dict:
    """Download every source (unless offline), then rebuild the SQLite tables in dependency order."""
    result = {}
    if not offline:
        log = football_data.download(root)
        result["football_data_changed_or_failed"] = [e for e in log if e["status"] != "unchanged"]
        if not skip_shots:
            result["pannadata_downloaded"] = pannadata.download(root)
        result["fpl_snapshot"] = fpl.snapshot(root)
    result["matches_odds"] = football_data.build(root)   # must come first: the others link to `matches`
    result["match_xg"] = pannadata.build(root)
    result["fpl"] = fpl.build(root)
    result["features"] = match_features.build(root)
    return result


def main():
    parser = argparse.ArgumentParser(description="Premier League data preparation and first goal baseline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("update", help="Download all sources (football-data, pannadata, FPL) and rebuild the database")
    update.add_argument("--offline", action="store_true", help="Skip downloads; only rebuild from data/raw")
    update.add_argument("--skip-shots", action="store_true", help="Do not re-download the ~120MB pannadata files")
    commands.add_parser("features", help="Rebuild the as-of table features_match from the database")
    ev = commands.add_parser("evaluate", help="Walk-forward evaluation of one model; appends a row to outputs/leaderboard.csv")
    ev.add_argument("--model", choices=sorted(MODELS), required=True)
    ev.add_argument("--league", default="E0")
    ev.add_argument("--split", choices=sorted(SPLITS), default="validation")
    ev.add_argument("--train-leagues", help="Comma-separated leagues to train on (default: the evaluated league)")
    ev.add_argument("--confirm-test", action="store_true", help="Required for the test split (final candidates only)")
    commands.add_parser("prepare", help="Normalize local OpenFootball results into CSV and SQLite")
    commands.add_parser("baseline", help="Run the fixed independent-Poisson baseline chronologically")
    commands.add_parser("status", help="Show dataset and key availability without exposing keys")
    api = commands.add_parser("api", help="One bounded GET call; no automatic pagination or retries")
    api.add_argument("provider", choices=["api-football", "highlightly", "odds", "balldontlie"])
    api.add_argument("operation")
    api.add_argument("--season", type=int, default=2024)
    api.add_argument("--league-id", type=int)
    api.add_argument("--league-name")
    api.add_argument("--fixture-id", type=int)
    api.add_argument("--offset", type=int, default=0)
    api.add_argument("--regions", choices=["uk", "us", "eu"], default="uk")
    args = parser.parse_args()
    try:
        if args.command == "update":
            result = run_update(args.root, offline=args.offline, skip_shots=args.skip_shots)
        elif args.command == "features":
            result = match_features.build(args.root)
        elif args.command == "evaluate":
            leagues = args.train_leagues.split(",") if args.train_leagues else [args.league]
            result = evaluate(args.root, MODELS[args.model](), args.league, args.split, leagues, args.confirm_test)
        elif args.command == "prepare":
            result = prepare(args.root)
            result = {key: result[key] for key in ("rows", "seasons", "teams_by_season", "duplicate_match_keys")}
        elif args.command == "baseline":
            result = run_baseline(args.root)
        elif args.command == "status":
            result = {"root": str(args.root), "prepared_dataset": (args.root / "data/processed/matches.csv").exists(),
                      "keys_present": {key: bool(value) for key, value in load_keys(args.root).items()}}
        else:
            result = fetch_api(args.root, args.provider, args.operation, vars(args))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

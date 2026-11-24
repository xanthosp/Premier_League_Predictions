import argparse
import json
from pathlib import Path

from .ingest.api_clients import fetch_api, load_keys
from .ingest.openfootball import prepare
from .models.baseline_poisson import run_baseline


def main():
    parser = argparse.ArgumentParser(description="Premier League data preparation and first goal baseline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
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
        if args.command == "prepare":
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

"""The Odds API (free tier, 500 credits/month): current EPL prices of Pinnacle, Betfair and Betano UK (the group of
Stoiximan), which football-data.co.uk does not give us in its fixtures file.

`snapshot` makes one call (h2h + totals, regions eu and uk = 4 credits) only when an EPL match kicks off in the next
two days, and saves the response under data/raw/odds_api/<UTC stamp>.json (the key is never written).
`parse` turns a snapshot into the long odds format with football-data match ids.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from .api_clients import endpoint, load_keys, scrub
from .football_data import make_match_ids, parse_fixtures

REGIONS = "eu,uk"
BOOKS = {"pinnacle": "Pinnacle", "betfair_ex_eu": "Betfair", "betfair_ex_uk": "Betfair", "betano_uk": "Betano",
         "williamhill": "WilliamHill", "unibet": "Unibet", "marathonbet": "Marathon"}
# Odds API team name -> football-data name (only where they differ).
TEAMS = {"Brighton and Hove Albion": "Brighton", "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
         "Leeds United": "Leeds", "Leicester City": "Leicester", "Luton Town": "Luton", "Manchester City": "Man City",
         "Manchester United": "Man United", "Newcastle United": "Newcastle", "Norwich City": "Norwich",
         "Nottingham Forest": "Nott'm Forest", "Sheffield United": "Sheffield United", "Stoke City": "Stoke",
         "Swansea City": "Swansea", "Tottenham Hotspur": "Tottenham", "West Bromwich Albion": "West Brom",
         "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves", "Cardiff City": "Cardiff",
         "Huddersfield Town": "Huddersfield", "AFC Bournemouth": "Bournemouth"}


def _folder(root: Path) -> Path:
    return Path(root) / "data/raw/odds_api"


def _match_soon(root: Path, days: int) -> bool:
    snapshots = sorted((Path(root) / "data/raw/football_data/fixtures").glob("*.csv"))
    if not snapshots:
        return True  # no fixtures file to check against: let the call decide
    fixtures = parse_fixtures(snapshots[-1])[0]
    today = datetime.now(timezone.utc).date()
    return bool(fixtures.date.between(today.isoformat(), (today + timedelta(days=days)).isoformat()).any())


def snapshot(root: Path, days: int = 2) -> dict:
    root = Path(root)
    keys = load_keys(root)
    if not keys.get("ODDS_API_KEY"):
        return {"status": "skipped: no ODDS_API_KEY in .env"}
    if not _match_soon(root, days):
        return {"status": f"skipped: no EPL match in the next {days} days (no credits spent)"}
    _, request, secret = endpoint("odds", "odds", {"regions": REGIONS}, keys)
    try:
        with urlopen(request, timeout=30) as response:
            body, remaining = response.read(5_000_001), response.headers.get("x-requests-remaining")
    except HTTPError as exc:
        return {"status": f"error: HTTP {exc.code}"}
    except (URLError, TimeoutError, OSError):
        return {"status": "error: network"}
    data = scrub(json.loads(body), secret)
    folder = _folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}.json"
    target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"status": "new", "file": target.name, "events": len(data), "credits_remaining": remaining}


def parse(path: Path) -> pd.DataFrame:
    """Long format: match_id, bookmaker, market (1x2 / ou), selection, line, price, last_update."""
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(events, list):  # an error message instead of a list of events
        return pd.DataFrame()
    rows = []
    for e in events:
        home, away = TEAMS.get(e["home_team"], e["home_team"]), TEAMS.get(e["away_team"], e["away_team"])
        side = {e["home_team"]: "H", e["away_team"]: "A", "Draw": "D", "Over": "O", "Under": "U"}
        for b in e.get("bookmakers", []):
            book = BOOKS.get(b["key"], b["key"])
            for m in b.get("markets", []):
                market = {"h2h": "1x2", "totals": "ou"}.get(m["key"])
                if market is None:
                    continue
                for o in m["outcomes"]:
                    rows.append({"home_team": home, "away_team": away, "commence_utc": e["commence_time"],
                                 "bookmaker": book, "market": market, "selection": side.get(o["name"]),
                                 "line": o.get("point"), "price": o["price"], "last_update": m.get("last_update")})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # football-data dates are UK dates; EPL kickoffs are between 11:00 and 21:00 UTC, so the UTC date is the same.
    dates = pd.to_datetime(frame.commence_utc).dt.tz_localize(None).dt.normalize()
    frame.insert(0, "match_id", make_match_ids("E0", dates, frame.home_team, frame.away_team))
    frame["snapshot"] = Path(path).name
    return frame.drop_duplicates(["match_id", "bookmaker", "market", "selection", "line"])


def latest(root: Path) -> pd.DataFrame:
    files = sorted(_folder(root).glob("*.json"))
    return parse(files[-1]) if files else pd.DataFrame()


def wide_for_card(odds: pd.DataFrame, books=("Pinnacle", "Betano")) -> pd.DataFrame:
    """One row per (match_id, market, selection) with each book's price; totals only at the 2.5 line."""
    if odds.empty:
        return pd.DataFrame(columns=["match_id", "market", "selection", *[b.lower() for b in books]])
    sub = odds[odds.bookmaker.isin(books) & ((odds.market == "1x2") | (odds.line == 2.5))].copy()
    sub["market"] = sub.market.replace({"ou": "ou25"})
    wide = sub.pivot_table(index=["match_id", "market", "selection"], columns="bookmaker", values="price", aggfunc="first")
    wide.columns = [c.lower() for c in wide.columns]
    return wide.reset_index()

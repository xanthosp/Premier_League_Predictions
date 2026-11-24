"""football-data.co.uk: results, match stats, xG and bookmaker odds (opening and closing).

Raw files live in data/raw/football_data/<DIV>/<SSSS>_<DIV>.csv (e.g. 2627_E0.csv).
Completed seasons are downloaded once and never rewritten. Only the current season is re-downloaded.
`build()` normalizes every file into two SQLite tables:

- matches: one row per completed match (results, stats, xG)
- odds:    long format, one row per (match, bookmaker, market, opening/closing, selection)
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
LEAGUES = {"E0": "Premier League", "E1": "Championship", "SP1": "La Liga", "D1": "Bundesliga", "I1": "Serie A", "F1": "Ligue 1"}
FIRST_SEASON = 2016

# Bookmaker column prefixes, longest first so that e.g. "BFD" is tried before "BF" and "PS" before "P".
# Aliases merge codes that mean the same source across seasons.
BOOKMAKERS = sorted(["1XB", "Avg", "B365", "BF", "BFD", "BFE", "BMGM", "BV", "BW", "BbAv", "BbMx", "CL", "IW", "LB",
                     "Max", "P", "PP", "PS", "SKB", "VC", "WH"], key=len, reverse=True)
ALIASES = {"P": "PS", "BbMx": "Max", "BbAv": "Avg"}  # P/PS = Pinnacle, Bb* = old BetBrain max/average
SELECTION_PATTERNS = [
    (re.compile(r"^(C?)([HDA])$"), "1x2"),
    (re.compile(r"^(C?)([<>])2\.5$"), "ou25"),
    (re.compile(r"^(C?)AH([HA])$"), "ah"),
]
STAT_COLUMNS = {"HTHG": "hthg", "HTAG": "htag", "HTR": "htr", "Referee": "referee", "HS": "hs", "AS": "as_", "HST": "hst",
                "AST": "ast", "HF": "hf", "AF": "af", "HC": "hc", "AC": "ac", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
                "HxG": "hxg", "AxG": "axg"}
MATCH_FIELDS = ["match_id", "league", "season", "date", "time", "home_team", "away_team", "fthg", "ftag", "ftr",
                *STAT_COLUMNS.values(), "source_file"]


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(code: str) -> str:
    return f"20{code[:2]}-{code[2:]}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def parse_odds_column(column: str) -> tuple[str, str, bool, str] | None:
    """Return (bookmaker, market, is_closing, selection) for a price column, or None if it is not one."""
    for prefix in BOOKMAKERS:
        if not column.startswith(prefix):
            continue
        rest = column[len(prefix):]
        for pattern, market in SELECTION_PATTERNS:
            found = pattern.match(rest)
            if found:
                selection = {"<": "U", ">": "O"}.get(found[2], found[2])
                return ALIASES.get(prefix, prefix), market, found[1] == "C", selection
    return None


def _parse_dates(values: pd.Series) -> pd.Series:
    long = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(values, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


def read_raw(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    frame = pd.read_csv(io.StringIO(text), on_bad_lines="warn")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    return frame.dropna(subset=["HomeTeam", "AwayTeam"], how="any")


def parse_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize one raw season file into (matches, odds). Unplayed rows are dropped."""
    frame = read_raw(path)
    code, div = path.stem.split("_", 1)
    frame = frame.dropna(subset=["FTHG", "FTAG"]).copy()
    dates = _parse_dates(frame["Date"])
    if dates.isna().any():
        raise ValueError(f"{path.name}: unparseable dates {frame.loc[dates.isna(), 'Date'].tolist()[:3]}")
    matches = pd.DataFrame({
        "league": div, "season": season_label(code), "date": dates.dt.strftime("%Y-%m-%d"),
        "time": frame.get("Time"), "home_team": frame["HomeTeam"].str.strip(), "away_team": frame["AwayTeam"].str.strip(),
        "fthg": frame["FTHG"].astype(int), "ftag": frame["FTAG"].astype(int), "ftr": frame["FTR"],
    })
    for source, target in STAT_COLUMNS.items():
        matches[target] = frame[source] if source in frame else None
    matches["source_file"] = path.name
    matches.insert(0, "match_id", div + "_" + dates.dt.strftime("%Y%m%d") + "_"
                   + matches.home_team.str.replace(" ", "") + "_" + matches.away_team.str.replace(" ", ""))
    if matches.match_id.duplicated().any():
        raise ValueError(f"{path.name}: duplicate match ids {matches.match_id[matches.match_id.duplicated()].tolist()[:3]}")

    pieces = []
    lines = {False: frame.get("AHh", frame.get("BbAHh")), True: frame.get("AHCh")}
    for column in frame.columns:
        parsed = parse_odds_column(str(column))
        if parsed is None:
            continue
        bookmaker, market, closing, selection = parsed
        prices = pd.to_numeric(frame[column], errors="coerce")
        if market == "ah":
            line = pd.to_numeric(lines[closing], errors="coerce") if lines[closing] is not None else pd.Series(float("nan"), index=frame.index)
        else:
            line = pd.Series(2.5 if market == "ou25" else float("nan"), index=frame.index)
        piece = pd.DataFrame({"match_id": matches.match_id, "bookmaker": bookmaker, "market": market,
                              "is_closing": int(closing), "selection": selection, "line": line, "price": prices})
        pieces.append(piece[piece.price > 1.0])
    odds = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    # A bookmaker alias (e.g. P and PS) can appear twice for the same cell; keep the first.
    odds = odds.drop_duplicates(["match_id", "bookmaker", "market", "is_closing", "selection"])
    return matches[MATCH_FIELDS].reset_index(drop=True), odds


def _manifest_path(root: Path) -> Path:
    return root / "data/raw/football_data/manifest.json"


def download(root: Path, leagues=LEAGUES, first_season: int = FIRST_SEASON, today: date | None = None,
             pause: float = 1.0) -> list[dict]:
    """Fetch missing seasons and refresh the current one. Returns a log of what happened per file."""
    root = Path(root)
    manifest_path = _manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    current = current_season_start(today)
    log = []
    for div in leagues:
        for start in range(first_season, current + 1):
            code = season_code(start)
            target = root / f"data/raw/football_data/{div}/{code}_{div}.csv"
            if target.exists() and start != current:
                continue
            url = BASE_URL.format(code=code, div=div)
            request = Request(url, headers={"User-Agent": "football-ml-research/0.2"})
            try:
                with urlopen(request, timeout=60) as response:
                    payload = response.read()
            except Exception as exc:  # network errors are reported, never fatal for other files
                log.append({"file": target.name, "status": f"error: {exc}"})
                continue
            digest = hashlib.sha256(payload).hexdigest()
            previous = manifest.get(target.name, {}).get("sha256")
            target.parent.mkdir(parents=True, exist_ok=True)
            if digest != previous or not target.exists():
                target.write_bytes(payload)
            manifest[target.name] = {"url": url, "sha256": digest, "bytes": len(payload),
                                     "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            log.append({"file": target.name, "status": "unchanged" if digest == previous else "updated"})
            time.sleep(pause)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1), encoding="utf-8")
    return log


def build(root: Path) -> dict:
    """Parse every raw file into the matches/odds tables and write a coverage report."""
    root = Path(root)
    files = sorted((root / "data/raw/football_data").glob("*/*_*.csv"))
    if not files:
        raise FileNotFoundError("No football-data files; run `python -m football_ml update` first")
    parsed = [parse_file(path) for path in files]
    matches = pd.concat([m for m, _ in parsed], ignore_index=True).sort_values(["date", "league", "match_id"])
    odds = pd.concat([o for _, o in parsed], ignore_index=True)
    if matches.match_id.duplicated().any():
        raise ValueError("Duplicate match ids across files")
    with sqlite3.connect(root / "data/football.sqlite") as connection:
        matches.to_sql("matches", connection, if_exists="replace", index=False)
        odds.to_sql("odds", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_matches_id ON matches(match_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS ix_odds_match ON odds(match_id, market, is_closing)")
    coverage = coverage_report(matches, odds)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    coverage.to_csv(out / "coverage_football_data.csv", index=False)
    return {"files": len(files), "matches": len(matches), "odds_rows": len(odds),
            "date_range": [matches.date.min(), matches.date.max()], "coverage_csv": "outputs/coverage_football_data.csv"}


def coverage_report(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Per league-season: match count and share of matches with key fields and sharp/closing prices."""
    def has(market: str, closing: int, bookmakers: tuple[str, ...]) -> pd.Series:
        subset = odds[(odds.market == market) & (odds.is_closing == closing) & odds.bookmaker.isin(bookmakers)]
        return matches.match_id.isin(subset.match_id)

    flags = pd.DataFrame({
        "league": matches.league, "season": matches.season,
        "shots": matches.hs.notna(), "xg": matches.hxg.notna(),
        "1x2_open_avg": has("1x2", 0, ("Avg",)), "1x2_close_avg": has("1x2", 1, ("Avg",)),
        "1x2_close_pinnacle": has("1x2", 1, ("PS",)), "1x2_close_bf_exchange": has("1x2", 1, ("BFE",)),
        "ou25_close_avg": has("ou25", 1, ("Avg",)), "ah_close_avg": has("ah", 1, ("Avg",)),
    })
    grouped = flags.groupby(["league", "season"])
    report = grouped.mean().round(3)
    report.insert(0, "matches", grouped.size())
    return report.reset_index()

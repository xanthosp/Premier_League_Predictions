"""Bounded GET clients for documented APIs. No signups, bets or paid upgrades."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

KEY_NAMES = ("API_FOOTBALL_KEY", "HIGHLIGHTLY_API_KEY", "ODDS_API_KEY")


def load_keys(root):
    values = {}
    path = Path(root) / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() in KEY_NAMES:
                values[name.strip()] = value.strip().strip("\"'")
    return {name: os.getenv(name) or values.get(name, "") for name in KEY_NAMES}


def endpoint(provider, operation, options, keys):
    """Return a sanitized public URL separately from the authenticated request."""
    params, headers = {}, {"Accept": "application/json", "User-Agent": "FootballML-personal-research/0.1"}
    secret = None
    if provider == "balldontlie" and operation == "plans":
        base = "https://api.balldontlie.io/account/v1/plans/epl"
    elif provider == "api-football":
        secret = keys.get("API_FOOTBALL_KEY")
        if not secret:
            raise ValueError("Missing API_FOOTBALL_KEY; configure .env or the environment")
        headers["x-apisports-key"] = secret
        if operation in ("coverage", "fixtures"):
            path = "leagues" if operation == "coverage" else "fixtures"
            params = {"id" if operation == "coverage" else "league": options.get("league_id") or 39,
                      "season": options["season"]}
        elif operation == "players" and options.get("fixture_id"):
            path, params = "fixtures/players", {"fixture": options["fixture_id"]}
        else:
            raise ValueError("API-Football operations: coverage, fixtures, players (--fixture-id)")
        base = "https://v3.football.api-sports.io/" + path
    elif provider == "highlightly":
        secret = keys.get("HIGHLIGHTLY_API_KEY")
        if not secret:
            raise ValueError("Missing HIGHLIGHTLY_API_KEY; configure .env or the environment")
        headers["x-rapidapi-key"] = secret
        if operation == "leagues":
            path = "leagues"
            params = {"leagueName": options.get("league_name") or "Premier League", "season": options["season"], "limit": 20, "offset": options.get("offset", 0)}
        elif operation == "matches" and options.get("league_id"):
            path = "matches"
            params = {"leagueId": options["league_id"], "season": options["season"], "limit": 20, "offset": options.get("offset", 0)}
        elif operation == "players" and options.get("fixture_id"):
            path = f"box-score/{int(options['fixture_id'])}"
        else:
            raise ValueError("Highlightly operations: leagues, matches (--league-id), players (--fixture-id)")
        base = "https://soccer.highlightly.net/" + path
    elif provider == "odds":
        secret = keys.get("ODDS_API_KEY")
        if not secret:
            raise ValueError("Missing ODDS_API_KEY; configure .env or the environment")
        if operation == "sports":
            base = "https://api.the-odds-api.com/v4/sports"
        elif operation in ("events", "odds"):
            base = f"https://api.the-odds-api.com/v4/sports/soccer_epl/{operation}"
            if operation == "odds":
                params = {"regions": options.get("regions") or "uk", "markets": "h2h,totals", "oddsFormat": "decimal"}
        else:
            raise ValueError("Odds operations: sports, events, odds (current h2h and totals only)")
    else:
        raise ValueError("Supported providers: api-football, highlightly, odds, balldontlie")
    public_url = base + ("?" + urlencode(params) if params else "")
    request_params = {**params, **({"apiKey": secret} if provider == "odds" else {})}
    request_url = base + ("?" + urlencode(request_params) if request_params else "")
    return public_url, Request(request_url, headers=headers), secret


def scrub(value, secret):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if k.lower() in ("apikey", "api_key", "authorization", "token", "access_token") else scrub(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secret) for v in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value


def fetch_api(root, provider, operation, options):
    public_url, request, secret = endpoint(provider, operation, options, load_keys(root))
    try:
        with urlopen(request, timeout=25) as response:
            body = response.read(5_000_001)
            status = response.status
            quota = {k: v for k, v in response.headers.items() if k.lower() in (
                "x-requests-remaining", "x-requests-used", "x-requests-last",
                "x-ratelimit-requests-remaining", "x-ratelimit-requests-limit")}
    except HTTPError as exc:
        raise ValueError(f"{provider}: HTTP {exc.code}; stopped without retry or plan changes") from None
    except (URLError, TimeoutError, OSError):
        raise ValueError(f"{provider}: network request failed; stopped without retry") from None
    if len(body) > 5_000_000:
        raise ValueError("Response exceeded the 5 MB sample limit; not saved")
    try:
        data = scrub(json.loads(body), secret)
    except (ValueError, UnicodeDecodeError):
        raise ValueError(f"{provider}: response was not JSON") from None
    timestamp = datetime.now(timezone.utc)
    label = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "_" + hashlib.sha256(public_url.encode()).hexdigest()[:10]
    directory = Path(root) / "data/api_samples" / provider
    directory.mkdir(parents=True, exist_ok=True)
    payload_file = directory / (label + ".json")
    payload_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {"provider": provider, "operation": operation, "public_request_url": public_url,
                "captured_at": timestamp.isoformat(), "http_status": status, "quota": quota,
                "payload_file": payload_file.name, "credential_material_redacted": True,
                "list_lengths": {k: len(v) for k, v in data.items() if isinstance(v, list)} if isinstance(data, dict) else {"items": len(data) if isinstance(data, list) else None},
                "pagination": data.get("pagination") if isinstance(data, dict) else None,
                "plan": data.get("plan") if isinstance(data, dict) else None,
                "provider_errors": data.get("errors") if isinstance(data, dict) else None,
                "training_use": "pending source/coverage review; not automatically joined"}
    (directory / (label + ".metadata.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

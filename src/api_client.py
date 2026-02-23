"""Cliente para la API de API-Football (v3.football.api-sports.io)."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://v3.football.api-sports.io"

def _headers():
    return {"x-apisports-key": os.getenv("API_FOOTBALL_KEY")}

def get(endpoint: str, params: dict = None) -> dict:
    """Hace un GET a la API y retorna el JSON de respuesta."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_league_seasons(league_id: int) -> dict:
    return get("leagues", {"id": league_id})

def get_fixtures(league_id: int, season: int, last: int = 5) -> dict:
    return get("fixtures", {"league": league_id, "season": season, "last": last})

def get_fixture_stats(fixture_id: int) -> dict:
    return get("fixtures/statistics", {"fixture": fixture_id})

def get_fixture_lineups(fixture_id: int) -> dict:
    return get("fixtures/lineups", {"fixture": fixture_id})

def get_fixture_players(fixture_id: int) -> dict:
    return get("fixtures/players", {"fixture": fixture_id})

def get_fixture_events(fixture_id: int) -> dict:
    return get("fixtures/events", {"fixture": fixture_id})

def get_standings(league_id: int, season: int) -> dict:
    return get("standings", {"league": league_id, "season": season})

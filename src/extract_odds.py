"""
Extractor de odds de Pinnacle/Bet365 via API-Football.
Descarga odds por temporada con paginacion y cache.

Uso:
    python src/extract_odds.py --season 2024
    python src/extract_odds.py --season 2025
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.api_client import get

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────
LEAGUE_ID = 262
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
ODDS_CACHE_DIR = os.path.join(RAW_DIR, 'odds_cache')
DELAY_BETWEEN_CALLS = 6.5
PREFERRED_BOOKMAKERS = ["Pinnacle", "Bet365"]


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ODDS_CACHE_DIR, exist_ok=True)


# ── Download season odds with pagination ─────────────────────────────
def download_season_odds(season: int) -> list[dict]:
    """Pagina /odds?league=262&season=X&page=N, cachea cada pagina, retorna lista combinada."""
    ensure_dirs()
    all_odds = []
    page = 1

    while True:
        cache_file = os.path.join(ODDS_CACHE_DIR, f'odds_{season}_page_{page}.json')

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  Pagina {page}: cache (odds_{season}_page_{page}.json)")
        else:
            print(f"  Pagina {page}: descargando...", end=' ', flush=True)
            data = get('odds', {
                'league': LEAGUE_ID,
                'season': season,
                'page': page,
            })

            # Check for rate limit
            if data.get('errors') and any('limit' in str(v).lower() for v in data['errors'].values()):
                print("RATE LIMIT. Re-ejecuta despues.")
                break

            with open(cache_file, 'w') as f:
                json.dump(data, f)

            print(f"OK ({len(data.get('response', []))} fixtures)")

            if page < data.get('paging', {}).get('total', 1):
                time.sleep(DELAY_BETWEEN_CALLS)

        all_odds.extend(data.get('response', []))

        total_pages = data.get('paging', {}).get('total', 1)
        if page >= total_pages:
            break
        page += 1

    print(f"  Total: {len(all_odds)} fixtures con odds")
    return all_odds


# ── Extract implied probabilities ────────────────────────────────────
def extract_implied_probs(fixture_odds_entry: dict) -> dict | None:
    """
    Extrae odds de Pinnacle (fallback Bet365), calcula probabilidades implicitas.
    Retorna {fixture_id, bookmaker, prob_home, prob_draw, prob_away} o None.
    """
    fixture_id = fixture_odds_entry.get('fixture', {}).get('id')
    bookmakers = fixture_odds_entry.get('bookmakers', [])

    if not fixture_id or not bookmakers:
        return None

    # Find preferred bookmaker
    selected_book = None
    selected_name = None

    for preferred in PREFERRED_BOOKMAKERS:
        for book in bookmakers:
            if book.get('name', '').lower() == preferred.lower():
                selected_book = book
                selected_name = preferred
                break
        if selected_book:
            break

    if not selected_book:
        return None

    # Find "Match Winner" bet (id=1 typically)
    match_winner = None
    for bet in selected_book.get('bets', []):
        if bet.get('name') == 'Match Winner':
            match_winner = bet
            break

    if not match_winner:
        return None

    # Extract odds values
    odds_map = {}
    for val in match_winner.get('values', []):
        odds_map[val['value']] = float(val['odd'])

    if 'Home' not in odds_map or 'Draw' not in odds_map or 'Away' not in odds_map:
        return None

    # Calculate implied probabilities with normalization
    raw_home = 1.0 / odds_map['Home']
    raw_draw = 1.0 / odds_map['Draw']
    raw_away = 1.0 / odds_map['Away']
    total = raw_home + raw_draw + raw_away

    return {
        'fixture_id': fixture_id,
        'bookmaker': selected_name,
        'prob_home': round(raw_home / total, 4),
        'prob_draw': round(raw_draw / total, 4),
        'prob_away': round(raw_away / total, 4),
    }


# ── Build odds CSV ───────────────────────────────────────────────────
def build_odds_csv(season: int) -> str:
    """Construye y guarda data/raw/odds_{season}.csv. Retorna path."""
    all_odds = download_season_odds(season)

    rows = []
    skipped = 0
    for entry in all_odds:
        result = extract_implied_probs(entry)
        if result:
            rows.append(result)
        else:
            skipped += 1

    if not rows:
        print(f"  No se encontraron odds de {PREFERRED_BOOKMAKERS} para season {season}")
        return None

    df = pd.DataFrame(rows)
    output_path = os.path.join(RAW_DIR, f'odds_{season}.csv')
    df.to_csv(output_path, index=False)
    print(f"\n  Guardado: {output_path}")
    print(f"  {len(rows)} fixtures con odds, {skipped} sin bookmaker preferido")
    print(f"  Bookmakers: {df['bookmaker'].value_counts().to_dict()}")

    return output_path


# ── Fetch individual fixture odds (for live/jornada use) ─────────────
def fetch_fixture_odds(fixture_id: int) -> dict | None:
    """
    Descarga odds de un fixture individual, cachea en odds_cache/{fixture_id}_odds.json.
    Retorna dict con probs o None.
    """
    ensure_dirs()
    cache_file = os.path.join(ODDS_CACHE_DIR, f'{fixture_id}_odds.json')

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
    else:
        data = get('odds', {'fixture': fixture_id})
        if data.get('errors') and any('limit' in str(v).lower() for v in data['errors'].values()):
            return None
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    responses = data.get('response', [])
    if not responses:
        return None

    return extract_implied_probs(responses[0])


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Extractor de odds Liga MX')
    parser.add_argument('--season', type=int, required=True,
                        help='Temporada (e.g. 2024, 2025)')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Descargando odds Liga MX — Season {args.season}")
    print(f"{'='*50}\n")

    path = build_odds_csv(args.season)
    if path:
        df = pd.read_csv(path)
        print(f"\n  Preview:")
        print(df.head().to_string(index=False))


if __name__ == '__main__':
    main()

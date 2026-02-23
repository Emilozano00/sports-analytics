"""
Extractor general de datos de Liga MX por torneo.
- Descarga partidos con stats de equipo y jugador.
- Guarda progreso incremental (se puede resumir).
- Respeta rate limits de API-Football.

Nota: API season N corresponde a la temporada N/N+1 de Liga MX.
      Los archivos se nombran con el año oficial del torneo (season + 1).
      Ej: --season 2024 genera apertura2025_*, clausura2025_*

Uso:
    python src/extract_season.py Clausura                  # descarga Clausura 2025 (API season 2024)
    python src/extract_season.py Clausura --season 2025    # descarga Clausura 2026
    python src/extract_season.py Apertura                  # descarga Apertura 2025
    python src/extract_season.py Clausura --status         # solo muestra progreso
    python src/extract_season.py all                       # descarga ambos
    python src/extract_season.py all --build-only          # solo construir CSVs
"""

import sys
import os
import json
import time
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.api_client import get

import pandas as pd

# --- Config ---
LEAGUE_ID = 262
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CACHE_DIR = os.path.join(RAW_DIR, 'fixtures_cache')
DELAY_BETWEEN_CALLS = 6.5


def official_year(season: int) -> int:
    """API season N -> official Liga MX tournament year N+1.

    API season 2024 covers the 2024-25 Liga MX season:
      Apertura 2025 (played Jul-Dec 2024) + Clausura 2025 (played Jan-May 2025).
    """
    return season + 1


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_fixture_list(tournament: str, season: int) -> list[dict]:
    """Obtiene la lista de fixtures de un torneo (Apertura/Clausura). Usa cache."""
    year = official_year(season)
    cache_file = os.path.join(CACHE_DIR, f'{tournament.lower()}_{year}_fixtures_list.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    # Buscar si ya existe la lista completa de la temporada en cache
    full_cache = os.path.join(CACHE_DIR, f'full_season_{season}_fixtures.json')
    if os.path.exists(full_cache):
        with open(full_cache) as f:
            all_fixtures = json.load(f)
    else:
        print(f"Descargando lista completa de fixtures {season}...")
        data = get('fixtures', {'league': LEAGUE_ID, 'season': season})
        all_fixtures = data['response']
        with open(full_cache, 'w') as f:
            json.dump(all_fixtures, f)

    fixtures = []
    for f in all_fixtures:
        if tournament in f['league']['round']:
            fixtures.append({
                'fixture_id': f['fixture']['id'],
                'date': f['fixture']['date'],
                'round': f['league']['round'],
                'home': f['teams']['home']['name'],
                'away': f['teams']['away']['name'],
                'status': f['fixture']['status']['short'],
            })

    fixtures.sort(key=lambda x: x['date'])
    with open(cache_file, 'w') as f:
        json.dump(fixtures, f, indent=2)
    print(f"  → {tournament}: {len(fixtures)} partidos encontrados")
    return fixtures


def fixture_is_cached(fixture_id: int) -> bool:
    return os.path.exists(os.path.join(CACHE_DIR, f'{fixture_id}.json'))


def download_fixture(fixture_id: int) -> dict:
    """Descarga un fixture completo. Detecta rate limit."""
    cache_file = os.path.join(CACHE_DIR, f'{fixture_id}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    data = get('fixtures', {'id': fixture_id})
    if data.get('errors') and any('limit' in str(v).lower() for v in data['errors'].values()):
        raise RuntimeError("RATE LIMIT alcanzado. Re-ejecuta mañana.")
    if data['response']:
        fixture = data['response'][0]
        has_data = len(fixture.get('statistics', [])) > 0 or len(fixture.get('players', [])) > 0
        if has_data:
            with open(cache_file, 'w') as f:
                json.dump(fixture, f)
            return fixture
    return None


def build_team_stats(fixtures_data: list[dict]) -> pd.DataFrame:
    rows = []
    for f in fixtures_data:
        base = {
            'fixture_id': f['fixture']['id'],
            'fecha': f['fixture']['date'][:10],
            'torneo': 'Clausura' if 'Clausura' in f['league']['round'] else 'Apertura',
            'ronda': f['league']['round'],
            'arbitro': f['fixture']['referee'],
            'estadio': f['fixture']['venue']['name'],
        }
        for side in ['home', 'away']:
            other = 'away' if side == 'home' else 'home'
            row = {
                **base,
                'side': side,
                'equipo': f['teams'][side]['name'],
                'equipo_id': f['teams'][side]['id'],
                'oponente': f['teams'][other]['name'],
                'goles': f['goals'][side],
                'goles_rival': f['goals'][other],
                'goles_ht': f['score']['halftime'][side],
                'resultado': 'W' if f['teams'][side].get('winner') is True
                             else ('L' if f['teams'][side].get('winner') is False else 'D'),
            }
            for stat_block in f.get('statistics', []):
                if stat_block['team']['id'] == f['teams'][side]['id']:
                    for s in stat_block['statistics']:
                        key = s['type'].lower().replace(' ', '_')
                        row[key] = s['value']
                    break
            rows.append(row)
    return pd.DataFrame(rows)


def build_player_stats(fixtures_data: list[dict]) -> pd.DataFrame:
    rows = []
    for f in fixtures_data:
        fixture_id = f['fixture']['id']
        fecha = f['fixture']['date'][:10]
        ronda = f['league']['round']
        torneo = 'Clausura' if 'Clausura' in ronda else 'Apertura'

        for team_block in f.get('players', []):
            team_name = team_block['team']['name']
            team_id = team_block['team']['id']
            for p in team_block['players']:
                pinfo = p['player']
                s = p['statistics'][0]
                row = {
                    'fixture_id': fixture_id,
                    'fecha': fecha,
                    'torneo': torneo,
                    'ronda': ronda,
                    'equipo': team_name,
                    'equipo_id': team_id,
                    'jugador_id': pinfo['id'],
                    'jugador': pinfo['name'],
                    'minutos': s['games']['minutes'],
                    'numero': s['games']['number'],
                    'posicion': s['games']['position'],
                    'rating': s['games']['rating'],
                    'capitan': s['games']['captain'],
                    'titular': not s['games']['substitute'],
                    'tiros_total': s['shots']['total'],
                    'tiros_a_gol': s['shots']['on'],
                    'goles': s['goals']['total'],
                    'goles_recibidos': s['goals']['conceded'],
                    'asistencias': s['goals']['assists'],
                    'atajadas': s['goals']['saves'],
                    'pases_total': s['passes']['total'],
                    'pases_clave': s['passes']['key'],
                    'pases_precision': s['passes']['accuracy'],
                    'tackles': s['tackles']['total'],
                    'bloqueos': s['tackles']['blocks'],
                    'intercepciones': s['tackles']['interceptions'],
                    'duelos_total': s['duels']['total'],
                    'duelos_ganados': s['duels']['won'],
                    'regates_intentos': s['dribbles']['attempts'],
                    'regates_exitosos': s['dribbles']['success'],
                    'regateado': s['dribbles']['past'],
                    'faltas_recibidas': s['fouls']['drawn'],
                    'faltas_cometidas': s['fouls']['committed'],
                    'amarillas': s['cards']['yellow'],
                    'rojas': s['cards']['red'],
                    'penal_ganado': s['penalty']['won'],
                    'penal_cometido': s['penalty']['commited'],
                    'penal_anotado': s['penalty']['scored'],
                    'penal_fallado': s['penalty']['missed'],
                    'penal_atajado': s['penalty']['saved'],
                    'fuera_de_juego': s['offsides'],
                }
                rows.append(row)
    return pd.DataFrame(rows)


def download_tournament(tournament: str, fixtures_list: list[dict], season: int):
    """Descarga fixtures pendientes de un torneo."""
    year = official_year(season)
    pending = [f for f in fixtures_list if not fixture_is_cached(f['fixture_id'])]
    total = len(fixtures_list)
    cached = total - len(pending)
    print(f"\n{tournament} {year}: {total} partidos | {cached} en cache | {len(pending)} por descargar")

    if not pending:
        print("  ✓ Todo descargado")
        return

    for i, fx in enumerate(pending):
        fid = fx['fixture_id']
        print(f"  [{cached + i + 1}/{total}] {fx['date'][:10]} {fx['home']} vs {fx['away']} (ID:{fid})...", end=' ', flush=True)
        try:
            result = download_fixture(fid)
            if result:
                has_stats = len(result.get('statistics', [])) > 0
                has_players = len(result.get('players', [])) > 0
                print(f"OK (stats:{'Y' if has_stats else 'N'} players:{'Y' if has_players else 'N'})")
            else:
                print("EMPTY")
        except RuntimeError as e:
            print(f"\n  ✗ {e}")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            break

        if i < len(pending) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)


def build_csvs(tournaments: list[str], season: int):
    """Construye CSVs con datos descargados de los torneos especificados."""
    all_fixtures_data = []
    total_expected = 0

    for t in tournaments:
        fixtures_list = get_fixture_list(t, season)
        total_expected += len(fixtures_list)
        for fx in fixtures_list:
            cache_file = os.path.join(CACHE_DIR, f'{fx["fixture_id"]}.json')
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    all_fixtures_data.append(json.load(f))

    if not all_fixtures_data:
        print("No hay datos descargados aún.")
        return

    df_teams = build_team_stats(all_fixtures_data)
    df_players = build_player_stats(all_fixtures_data)

    # Guardar por torneo y combinado (usando año oficial)
    year = official_year(season)
    for t in tournaments:
        t_lower = t.lower()
        df_t = df_teams[df_teams['torneo'] == t]
        df_p = df_players[df_players['torneo'] == t]
        if len(df_t) > 0:
            df_t.to_csv(os.path.join(RAW_DIR, f'{t_lower}{year}_team_stats.csv'), index=False)
            df_p.to_csv(os.path.join(RAW_DIR, f'{t_lower}{year}_player_stats.csv'), index=False)

    # Combinado si son ambos torneos
    if len(tournaments) > 1:
        df_teams.to_csv(os.path.join(RAW_DIR, f'ligamx{year}_team_stats.csv'), index=False)
        df_players.to_csv(os.path.join(RAW_DIR, f'ligamx{year}_player_stats.csv'), index=False)

    # Resumen
    n_fixtures = df_teams['fixture_id'].nunique()
    print(f"\n{'='*70}")
    print(f"RESUMEN DE EXTRACCIÓN - Liga MX {year} (API season {season})")
    print(f"{'='*70}")
    for t in tournaments:
        df_t = df_teams[df_teams['torneo'] == t]
        fl = get_fixture_list(t, season)
        n = df_t['fixture_id'].nunique()
        print(f"  {t} {year}: {n}/{len(fl)} partidos ({n/len(fl)*100:.0f}%)")
    print(f"  TOTAL: {n_fixtures}/{total_expected} partidos")
    print(f"  Equipos: {df_teams['equipo'].nunique()} | Jugadores: {df_players['jugador_id'].nunique()}")
    print(f"  Team stats: {len(df_teams)} filas × {len(df_teams.columns)} cols")
    print(f"  Player stats: {len(df_players)} filas × {len(df_players.columns)} cols")

    print(f"\nArchivos guardados:")
    for f in sorted(os.listdir(RAW_DIR)):
        if f.endswith('.csv') and str(year) in f:
            size = os.path.getsize(os.path.join(RAW_DIR, f)) / 1024
            print(f"  → data/raw/{f} ({size:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Extractor Liga MX')
    parser.add_argument('tournament', choices=['Apertura', 'Clausura', 'all'],
                        help='Torneo a descargar')
    parser.add_argument('--season', type=int, default=2024,
                        help='Temporada API-Football (default: 2024)')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()

    season = args.season
    ensure_dirs()

    if args.tournament == 'all':
        tournaments = ['Clausura', 'Apertura']
    else:
        tournaments = [args.tournament]

    # Obtener listas
    all_lists = {}
    for t in tournaments:
        all_lists[t] = get_fixture_list(t, season)

    if args.status:
        for t in tournaments:
            cached = sum(1 for f in all_lists[t] if fixture_is_cached(f['fixture_id']))
            total = len(all_lists[t])
            pending = total - cached
            print(f"{t}: {cached}/{total} ({cached/total*100:.0f}%) | faltan {pending} (~{pending*DELAY_BETWEEN_CALLS/60:.0f} min)")
        return

    # Descargar
    if not args.build_only:
        for t in tournaments:
            download_tournament(t, all_lists[t], season)

    # Construir CSVs
    print("\nConstruyendo CSVs...")
    build_csvs(tournaments, season)


if __name__ == '__main__':
    main()

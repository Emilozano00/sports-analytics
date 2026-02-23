"""
Extractor de datos del Apertura 2024 de Liga MX.
- Descarga todos los partidos con stats de equipo y jugador.
- Guarda progreso incremental (se puede resumir).
- Respeta rate limits de API-Football.

Uso:
    python src/extract_apertura2024.py          # descarga datos
    python src/extract_apertura2024.py --status # solo muestra progreso
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.api_client import get

import pandas as pd

# --- Config ---
LEAGUE_ID = 262
SEASON = 2024
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CACHE_DIR = os.path.join(RAW_DIR, 'fixtures_cache')
DELAY_BETWEEN_CALLS = 6.5  # segundos entre llamadas (free plan: ~10 req/min)


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_apertura_fixture_ids() -> list[dict]:
    """Obtiene la lista de fixtures del Apertura 2024 (usa cache si existe)."""
    cache_file = os.path.join(CACHE_DIR, 'apertura_fixtures_list.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    print("Descargando lista de fixtures Apertura 2024...")
    data = get('fixtures', {'league': LEAGUE_ID, 'season': SEASON})
    fixtures = []
    for f in data['response']:
        if 'Apertura' in f['league']['round']:
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
    print(f"  → {len(fixtures)} partidos encontrados")
    return fixtures


def fixture_is_cached(fixture_id: int) -> bool:
    return os.path.exists(os.path.join(CACHE_DIR, f'{fixture_id}.json'))


def download_fixture(fixture_id: int) -> dict:
    """Descarga un fixture completo (con stats, jugadores, eventos, lineups)."""
    cache_file = os.path.join(CACHE_DIR, f'{fixture_id}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    data = get('fixtures', {'id': fixture_id})
    # Detectar rate limit (API regresa errors o response vacío)
    if data.get('errors') and any('limit' in str(v).lower() for v in data['errors'].values()):
        raise RuntimeError("RATE LIMIT alcanzado. Re-ejecuta mañana.")
    if data['response']:
        fixture = data['response'][0]
        # Solo guardar si tiene datos reales (stats y jugadores)
        has_data = len(fixture.get('statistics', [])) > 0 or len(fixture.get('players', [])) > 0
        if has_data:
            with open(cache_file, 'w') as f:
                json.dump(fixture, f)
            return fixture
    return None


def build_team_stats(fixtures_data: list[dict]) -> pd.DataFrame:
    """Construye DataFrame de estadísticas por equipo por partido."""
    rows = []
    for f in fixtures_data:
        base = {
            'fixture_id': f['fixture']['id'],
            'fecha': f['fixture']['date'][:10],
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
            # Stats del equipo
            for stat_block in f.get('statistics', []):
                if stat_block['team']['id'] == f['teams'][side]['id']:
                    for s in stat_block['statistics']:
                        key = s['type'].lower().replace(' ', '_')
                        row[key] = s['value']
                    break
            rows.append(row)
    return pd.DataFrame(rows)


def build_player_stats(fixtures_data: list[dict]) -> pd.DataFrame:
    """Construye DataFrame de estadísticas por jugador por partido."""
    rows = []
    for f in fixtures_data:
        fixture_id = f['fixture']['id']
        fecha = f['fixture']['date'][:10]
        ronda = f['league']['round']

        for team_block in f.get('players', []):
            team_name = team_block['team']['name']
            team_id = team_block['team']['id']

            for p in team_block['players']:
                pinfo = p['player']
                s = p['statistics'][0]

                row = {
                    'fixture_id': fixture_id,
                    'fecha': fecha,
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


def print_status(fixtures_list):
    """Muestra el estado actual de la descarga."""
    cached = sum(1 for f in fixtures_list if fixture_is_cached(f['fixture_id']))
    total = len(fixtures_list)
    print(f"\nProgreso: {cached}/{total} partidos descargados ({cached/total*100:.0f}%)")
    print(f"Faltan: {total - cached} partidos (~{(total-cached)*DELAY_BETWEEN_CALLS/60:.0f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true', help='Solo mostrar progreso')
    parser.add_argument('--build-only', action='store_true', help='Solo construir CSVs de datos ya descargados')
    args = parser.parse_args()

    ensure_dirs()
    fixtures_list = get_apertura_fixture_ids()

    if args.status:
        print_status(fixtures_list)
        return

    # --- Descarga ---
    if not args.build_only:
        pending = [f for f in fixtures_list if not fixture_is_cached(f['fixture_id'])]
        total = len(fixtures_list)
        cached = total - len(pending)
        print(f"\nApertura 2024: {total} partidos | {cached} en cache | {len(pending)} por descargar")

        for i, fx in enumerate(pending):
            fid = fx['fixture_id']
            print(f"  [{cached + i + 1}/{total}] {fx['date'][:10]} {fx['home']} vs {fx['away']} (ID:{fid})...", end=' ')
            try:
                result = download_fixture(fid)
                if result:
                    has_stats = len(result.get('statistics', [])) > 0
                    has_players = len(result.get('players', [])) > 0
                    print(f"OK (stats:{'Y' if has_stats else 'N'} players:{'Y' if has_players else 'N'})")
                else:
                    print("EMPTY")
            except Exception as e:
                print(f"ERROR: {e}")
                print(f"  ↳ Descarga interrumpida. Re-ejecuta el script para continuar.")
                break

            if i < len(pending) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    # --- Construir CSVs ---
    print("\nConstruyendo DataFrames con datos descargados...")
    fixtures_data = []
    for fx in fixtures_list:
        cache_file = os.path.join(CACHE_DIR, f'{fx["fixture_id"]}.json')
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                fixtures_data.append(json.load(f))

    if not fixtures_data:
        print("No hay datos descargados aún.")
        return

    df_teams = build_team_stats(fixtures_data)
    df_players = build_player_stats(fixtures_data)

    teams_path = os.path.join(RAW_DIR, 'apertura2024_team_stats.csv')
    players_path = os.path.join(RAW_DIR, 'apertura2024_player_stats.csv')
    df_teams.to_csv(teams_path, index=False)
    df_players.to_csv(players_path, index=False)

    # --- Resumen ---
    n_fixtures = df_teams['fixture_id'].nunique()
    n_total = len(fixtures_list)
    print(f"\n{'='*70}")
    print(f"RESUMEN DE EXTRACCIÓN - Apertura 2024 Liga MX")
    print(f"{'='*70}")
    print(f"Partidos descargados:    {n_fixtures}/{n_total} ({n_fixtures/n_total*100:.0f}%)")
    print(f"Equipos únicos:          {df_teams['equipo'].nunique()}")
    print(f"Jugadores únicos:        {df_players['jugador_id'].nunique()}")
    print(f"Registros equipo:        {len(df_teams)} filas × {len(df_teams.columns)} cols")
    print(f"Registros jugador:       {len(df_players)} filas × {len(df_players.columns)} cols")

    # Completitud de datos
    print(f"\n--- Completitud: Stats por equipo ---")
    for col in df_teams.columns:
        if col in ('fixture_id', 'fecha', 'ronda', 'equipo', 'equipo_id', 'oponente', 'side',
                    'goles', 'goles_rival', 'goles_ht', 'resultado', 'arbitro', 'estadio'):
            continue
        pct = df_teams[col].notna().mean() * 100
        non_null = df_teams[col].notna().sum()
        print(f"  {col:<25} {non_null:>5}/{len(df_teams)} ({pct:5.1f}%)")

    print(f"\n--- Completitud: Stats por jugador ---")
    for col in df_players.columns:
        if col in ('fixture_id', 'fecha', 'ronda', 'equipo', 'equipo_id',
                    'jugador_id', 'jugador', 'numero', 'posicion', 'capitan', 'titular'):
            continue
        pct = df_players[col].notna().mean() * 100
        non_null = df_players[col].notna().sum()
        print(f"  {col:<25} {non_null:>5}/{len(df_players)} ({pct:5.1f}%)")

    print(f"\nArchivos guardados:")
    print(f"  → {teams_path}")
    print(f"  → {players_path}")


if __name__ == '__main__':
    main()

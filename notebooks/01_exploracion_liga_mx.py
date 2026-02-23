"""
Exploración de datos de Liga MX via API-Football
Partido: Pachuca 6 - 2 Necaxa (Apertura 2024, Jornada 15)
Fixture ID: 1206136
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.api_client import *
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 30)

FIXTURE_ID = 1206136

# %% [1] Datos generales del partido
print("=" * 80)
print("1. DATOS GENERALES DEL PARTIDO")
print("=" * 80)

fixture_data = get('fixtures', {'id': FIXTURE_ID})['response'][0]

fixture_info = {
    'fixture_id': fixture_data['fixture']['id'],
    'fecha': fixture_data['fixture']['date'],
    'arbitro': fixture_data['fixture']['referee'],
    'estadio': fixture_data['fixture']['venue']['name'],
    'ciudad': fixture_data['fixture']['venue']['city'],
    'liga': fixture_data['league']['name'],
    'temporada': fixture_data['league']['season'],
    'ronda': fixture_data['league']['round'],
    'equipo_local': fixture_data['teams']['home']['name'],
    'equipo_visitante': fixture_data['teams']['away']['name'],
    'goles_local': fixture_data['goals']['home'],
    'goles_visitante': fixture_data['goals']['away'],
    'goles_ht_local': fixture_data['score']['halftime']['home'],
    'goles_ht_visitante': fixture_data['score']['halftime']['away'],
    'status': fixture_data['fixture']['status']['long'],
}

df_fixture = pd.DataFrame([fixture_info])
print(df_fixture.T.to_string(header=False))

# %% [2] Estadísticas por equipo
print("\n" + "=" * 80)
print("2. ESTADÍSTICAS POR EQUIPO (TEAM-LEVEL)")
print("=" * 80)

stats_data = get_fixture_stats(FIXTURE_ID)['response']

rows = []
for team_block in stats_data:
    row = {'equipo': team_block['team']['name']}
    for stat in team_block['statistics']:
        row[stat['type']] = stat['value']
    rows.append(row)

df_team_stats = pd.DataFrame(rows)
print(df_team_stats.T.to_string())
print(f"\n→ {len(df_team_stats.columns) - 1} estadísticas por equipo")

# %% [3] Eventos del partido
print("\n" + "=" * 80)
print("3. EVENTOS DEL PARTIDO")
print("=" * 80)

events_data = get_fixture_events(FIXTURE_ID)['response']

df_events = pd.DataFrame([{
    'minuto': e['time']['elapsed'],
    'minuto_extra': e['time']['extra'],
    'equipo': e['team']['name'],
    'jugador': e['player']['name'],
    'jugador_id': e['player']['id'],
    'asistencia': e['assist']['name'] if e['assist'] else None,
    'tipo': e['type'],
    'detalle': e['detail'],
    'comentarios': e['comments'],
} for e in events_data])

print(df_events.to_string(index=False))
print(f"\n→ Columnas de eventos: {list(df_events.columns)}")

# %% [4] Alineaciones
print("\n" + "=" * 80)
print("4. ALINEACIONES")
print("=" * 80)

lineups_data = get_fixture_lineups(FIXTURE_ID)['response']

lineup_rows = []
for team_lu in lineups_data:
    team_name = team_lu['team']['name']
    formation = team_lu['formation']
    coach = team_lu['coach']['name']
    for p in team_lu['startXI']:
        lineup_rows.append({
            'equipo': team_name,
            'formacion': formation,
            'dt': coach,
            'titular': True,
            'jugador_id': p['player']['id'],
            'jugador': p['player']['name'],
            'numero': p['player']['number'],
            'posicion': p['player']['pos'],
        })
    for p in team_lu['substitutes']:
        lineup_rows.append({
            'equipo': team_name,
            'formacion': formation,
            'dt': coach,
            'titular': False,
            'jugador_id': p['player']['id'],
            'jugador': p['player']['name'],
            'numero': p['player']['number'],
            'posicion': p['player']['pos'],
        })

df_lineups = pd.DataFrame(lineup_rows)
print(df_lineups[df_lineups['titular'] == True][['equipo', 'formacion', 'numero', 'jugador', 'posicion']].to_string(index=False))
print(f"\n→ Columnas de alineación: {list(df_lineups.columns)}")

# %% [5] Estadísticas por jugador (el plato fuerte)
print("\n" + "=" * 80)
print("5. ESTADÍSTICAS POR JUGADOR (PLAYER-LEVEL)")
print("=" * 80)

players_data = get_fixture_players(FIXTURE_ID)['response']

player_rows = []
for team_block in players_data:
    team_name = team_block['team']['name']
    for p in team_block['players']:
        pinfo = p['player']
        s = p['statistics'][0]  # siempre 1 stat block por fixture
        row = {
            'equipo': team_name,
            'jugador_id': pinfo['id'],
            'jugador': pinfo['name'],
            # games
            'minutos': s['games']['minutes'],
            'numero': s['games']['number'],
            'posicion': s['games']['position'],
            'rating': s['games']['rating'],
            'capitan': s['games']['captain'],
            'suplente': s['games']['substitute'],
            # shots
            'tiros_total': s['shots']['total'],
            'tiros_a_gol': s['shots']['on'],
            # goals
            'goles': s['goals']['total'],
            'goles_recibidos': s['goals']['conceded'],
            'asistencias': s['goals']['assists'],
            'atajadas': s['goals']['saves'],
            # passes
            'pases_total': s['passes']['total'],
            'pases_clave': s['passes']['key'],
            'pases_precision': s['passes']['accuracy'],
            # tackles
            'tackles': s['tackles']['total'],
            'bloqueos': s['tackles']['blocks'],
            'intercepciones': s['tackles']['interceptions'],
            # duels
            'duelos_total': s['duels']['total'],
            'duelos_ganados': s['duels']['won'],
            # dribbles
            'regates_intentos': s['dribbles']['attempts'],
            'regates_exitosos': s['dribbles']['success'],
            'regateado': s['dribbles']['past'],
            # fouls
            'faltas_recibidas': s['fouls']['drawn'],
            'faltas_cometidas': s['fouls']['committed'],
            # cards
            'amarillas': s['cards']['yellow'],
            'rojas': s['cards']['red'],
            # penalty
            'penal_ganado': s['penalty']['won'],
            'penal_cometido': s['penalty']['commited'],
            'penal_anotado': s['penalty']['scored'],
            'penal_fallado': s['penalty']['missed'],
            'penal_atajado': s['penalty']['saved'],
            # offsides
            'fuera_de_juego': s['offsides'],
        }
        player_rows.append(row)

df_players = pd.DataFrame(player_rows)

# Mostrar resumen ofensivo
print("\n--- Top jugadores ofensivos ---")
cols_ofensivos = ['equipo', 'jugador', 'posicion', 'minutos', 'rating',
                  'goles', 'asistencias', 'tiros_total', 'tiros_a_gol',
                  'pases_clave', 'regates_exitosos']
print(df_players[cols_ofensivos].sort_values('rating', ascending=False).head(10).to_string(index=False))

print("\n--- Top jugadores defensivos ---")
cols_defensivos = ['equipo', 'jugador', 'posicion', 'minutos', 'rating',
                   'tackles', 'intercepciones', 'bloqueos', 'duelos_total', 'duelos_ganados']
print(df_players[cols_defensivos].sort_values('duelos_ganados', ascending=False).head(10).to_string(index=False))

print(f"\n→ TOTAL: {len(df_players.columns)} variables por jugador por partido")
print(f"→ Variables disponibles:\n{list(df_players.columns)}")

# %% [6] Resumen de dimensiones disponibles
print("\n" + "=" * 80)
print("RESUMEN: DIMENSIONES DISPONIBLES EN API-FOOTBALL")
print("=" * 80)
print(f"""
ENDPOINT                  | VARIABLES | DESCRIPCIÓN
--------------------------|-----------|------------------------------------------
fixtures (general)        | {len(df_fixture.columns):>3}      | Info del partido, equipos, marcador, venue
fixtures/statistics       | {len(df_team_stats.columns)-1:>3}      | Estadísticas agregadas por equipo
fixtures/events           | {len(df_events.columns):>3}      | Goles, tarjetas, sustituciones
fixtures/lineups          | {len(df_lineups.columns):>3}      | Formación, titulares, suplentes, DT
fixtures/players          | {len(df_players.columns)-3:>3}      | Stats individuales (sin equipo/id/nombre)
""")

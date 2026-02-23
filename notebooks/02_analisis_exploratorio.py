"""
Análisis exploratorio - Apertura 2024 Liga MX
1. Jugadores con más tiros por partido
2. Mejor ratio de regates exitosos
3. Posesión promedio por equipo
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# --- Config visual ---
mpl.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#1a1d23',
    'axes.edgecolor': '#2d3139',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'grid.color': '#2d3139',
    'font.size': 11,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
})

ACCENT = '#00d4aa'
ACCENT2 = '#ff6b6b'
ACCENT3 = '#4ecdc4'
PALETTE = ['#00d4aa', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
           '#dfe6e9', '#fab1a0', '#ff6b6b', '#a29bfe', '#fd79a8',
           '#e17055', '#00b894', '#6c5ce7', '#fdcb6e', '#e84393',
           '#0984e3', '#636e72', '#2d3436']

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# --- Cargar datos ---
df_p = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'apertura2024_player_stats.csv'))
df_t = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'apertura2024_team_stats.csv'))

MIN_PARTIDOS = 5  # mínimo de partidos para incluir en rankings

# =====================================================================
# 1. JUGADORES CON MÁS TIROS POR PARTIDO
# =====================================================================
print("=" * 60)
print("1. TOP JUGADORES: TIROS POR PARTIDO")
print("=" * 60)

tiros = df_p.groupby(['jugador_id', 'jugador', 'equipo', 'posicion']).agg(
    tiros_total=('tiros_total', 'sum'),
    tiros_a_gol=('tiros_a_gol', 'sum'),
    partidos=('fixture_id', 'nunique'),
    minutos=('minutos', 'sum'),
).reset_index()

tiros = tiros[tiros['partidos'] >= MIN_PARTIDOS].copy()
tiros['tiros_por_partido'] = tiros['tiros_total'] / tiros['partidos']
tiros['tiros_gol_por_partido'] = tiros['tiros_a_gol'] / tiros['partidos']
tiros['precision_tiro'] = (tiros['tiros_a_gol'] / tiros['tiros_total'] * 100).round(1)
tiros = tiros.sort_values('tiros_por_partido', ascending=False)

top_tiros = tiros.head(15).copy()
top_tiros['label'] = top_tiros['jugador'] + ' (' + top_tiros['equipo'].str[:3].str.upper() + ')'

print(top_tiros[['jugador', 'equipo', 'posicion', 'partidos', 'tiros_total',
                  'tiros_por_partido', 'tiros_gol_por_partido', 'precision_tiro']].to_string(index=False))

# Gráfica
fig, ax = plt.subplots(figsize=(12, 7))
y_pos = range(len(top_tiros) - 1, -1, -1)

bars_total = ax.barh(y_pos, top_tiros['tiros_por_partido'].values,
                     color=ACCENT, alpha=0.3, label='Tiros desviados/bloqueados', height=0.6)
bars_on = ax.barh(y_pos, top_tiros['tiros_gol_por_partido'].values,
                  color=ACCENT, alpha=0.9, label='Tiros a gol', height=0.6)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(top_tiros['label'].values)
ax.set_xlabel('Tiros por partido')
ax.set_title('Top 15 Jugadores: Tiros por Partido — Apertura 2024 Liga MX', fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3)
ax.grid(axis='x', alpha=0.3)

for i, (total, on_target, pct) in enumerate(zip(
    top_tiros['tiros_por_partido'].values,
    top_tiros['tiros_gol_por_partido'].values,
    top_tiros['precision_tiro'].values
)):
    ax.text(total + 0.08, list(y_pos)[i], f'{pct:.0f}%', va='center', fontsize=9, color='#a0a0a0')

ax.text(0.98, 0.02, '% = precisión de tiro', transform=ax.transAxes,
        ha='right', fontsize=9, color='#666', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'top_tiros_por_partido.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n→ Guardado: data/processed/top_tiros_por_partido.png")


# =====================================================================
# 2. MEJOR RATIO DE REGATES EXITOSOS
# =====================================================================
print("\n" + "=" * 60)
print("2. TOP JUGADORES: RATIO DE REGATES EXITOSOS")
print("=" * 60)

regates = df_p.groupby(['jugador_id', 'jugador', 'equipo', 'posicion']).agg(
    regates_intentos=('regates_intentos', 'sum'),
    regates_exitosos=('regates_exitosos', 'sum'),
    partidos=('fixture_id', 'nunique'),
    minutos=('minutos', 'sum'),
).reset_index()

MIN_REGATES = 15  # mínimo intentos para ser significativo
regates = regates[(regates['partidos'] >= MIN_PARTIDOS) & (regates['regates_intentos'] >= MIN_REGATES)].copy()
regates['ratio_exito'] = (regates['regates_exitosos'] / regates['regates_intentos'] * 100).round(1)
regates['regates_por_partido'] = (regates['regates_exitosos'] / regates['partidos']).round(2)
regates = regates.sort_values('ratio_exito', ascending=False)

top_regates = regates.head(15).copy()
top_regates['label'] = top_regates['jugador'] + ' (' + top_regates['equipo'].str[:3].str.upper() + ')'

print(top_regates[['jugador', 'equipo', 'posicion', 'partidos',
                    'regates_intentos', 'regates_exitosos', 'ratio_exito', 'regates_por_partido']].to_string(index=False))

# Gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.2, 1]})

# Panel izquierdo: ratio de éxito
y_pos = range(len(top_regates) - 1, -1, -1)
colors = [ACCENT if r >= 60 else ACCENT3 if r >= 50 else ACCENT2 for r in top_regates['ratio_exito'].values]

bars = ax1.barh(y_pos, top_regates['ratio_exito'].values, color=colors, alpha=0.85, height=0.6)
ax1.set_yticks(list(y_pos))
ax1.set_yticklabels(top_regates['label'].values)
ax1.set_xlabel('Tasa de éxito (%)')
ax1.set_title('Ratio de Regates Exitosos', fontweight='bold', pad=15)
ax1.set_xlim(0, 100)
ax1.axvline(x=50, color='#555', linestyle='--', alpha=0.5)
ax1.grid(axis='x', alpha=0.3)

for i, (ratio, intentos) in enumerate(zip(top_regates['ratio_exito'].values, top_regates['regates_intentos'].values)):
    ax1.text(ratio + 1.5, list(y_pos)[i], f'{ratio:.0f}% ({int(intentos)})', va='center', fontsize=9, color='#a0a0a0')

# Panel derecho: regates exitosos por partido (volumen)
top_vol = regates.sort_values('regates_por_partido', ascending=False).head(15).copy()
top_vol['label'] = top_vol['jugador'] + ' (' + top_vol['equipo'].str[:3].str.upper() + ')'
y_pos2 = range(len(top_vol) - 1, -1, -1)

ax2.barh(y_pos2, top_vol['regates_por_partido'].values, color=ACCENT3, alpha=0.85, height=0.6)
ax2.set_yticks(list(y_pos2))
ax2.set_yticklabels(top_vol['label'].values)
ax2.set_xlabel('Regates exitosos por partido')
ax2.set_title('Volumen de Regates por Partido', fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

for i, (rpg, ratio) in enumerate(zip(top_vol['regates_por_partido'].values, top_vol['ratio_exito'].values)):
    ax2.text(rpg + 0.05, list(y_pos2)[i], f'{ratio:.0f}%', va='center', fontsize=9, color='#a0a0a0')

ax2.text(0.98, 0.02, '% = tasa de éxito', transform=ax2.transAxes,
         ha='right', fontsize=9, color='#666', style='italic')

fig.suptitle('Regates — Apertura 2024 Liga MX', fontweight='bold', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'top_regates.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n→ Guardado: data/processed/top_regates.png")


# =====================================================================
# 3. POSESIÓN PROMEDIO POR EQUIPO
# =====================================================================
print("\n" + "=" * 60)
print("3. POSESIÓN PROMEDIO POR EQUIPO")
print("=" * 60)

df_t['posesion_num'] = df_t['ball_possession'].str.replace('%', '').astype(float)

posesion = df_t.groupby('equipo').agg(
    posesion_promedio=('posesion_num', 'mean'),
    posesion_local=('posesion_num', lambda x: x[df_t.loc[x.index, 'side'] == 'home'].mean()),
    posesion_visita=('posesion_num', lambda x: x[df_t.loc[x.index, 'side'] == 'away'].mean()),
    partidos=('fixture_id', 'nunique'),
    pases_promedio=('total_passes', 'mean'),
    precision_pases=('passes_accurate', lambda x: (x.sum() / df_t.loc[x.index, 'total_passes'].sum() * 100)),
).reset_index()
posesion = posesion.sort_values('posesion_promedio', ascending=False)
posesion['posesion_promedio'] = posesion['posesion_promedio'].round(1)
posesion['posesion_local'] = posesion['posesion_local'].round(1)
posesion['posesion_visita'] = posesion['posesion_visita'].round(1)
posesion['pases_promedio'] = posesion['pases_promedio'].round(0)
posesion['precision_pases'] = posesion['precision_pases'].round(1)

print(posesion[['equipo', 'partidos', 'posesion_promedio', 'posesion_local',
                 'posesion_visita', 'pases_promedio', 'precision_pases']].to_string(index=False))

# Gráfica
fig, ax = plt.subplots(figsize=(14, 8))

x = range(len(posesion))
width = 0.35

bars_home = ax.bar([i - width/2 for i in x], posesion['posesion_local'].values,
                   width, label='Local', color=ACCENT, alpha=0.85)
bars_away = ax.bar([i + width/2 for i in x], posesion['posesion_visita'].values,
                   width, label='Visitante', color=ACCENT3, alpha=0.85)

# Línea de posesión promedio global
ax.axhline(y=50, color='#ff6b6b', linestyle='--', alpha=0.5, linewidth=1)
ax.text(len(posesion) - 0.5, 50.5, '50%', color='#ff6b6b', fontsize=9, alpha=0.7)

ax.set_xticks(list(x))
ax.set_xticklabels(posesion['equipo'].values, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Posesión (%)')
ax.set_title('Posesión Promedio por Equipo (Local vs Visitante) — Apertura 2024 Liga MX',
             fontweight='bold', pad=15)
ax.legend(framealpha=0.3)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(30, 70)

# Anotar promedio general encima de cada equipo
for i, prom in enumerate(posesion['posesion_promedio'].values):
    ax.text(i, max(posesion['posesion_local'].values[i], posesion['posesion_visita'].values[i]) + 1,
            f'{prom}%', ha='center', fontsize=8, color='#ccc', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'posesion_por_equipo.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n→ Guardado: data/processed/posesion_por_equipo.png")

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS EN data/processed/:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f} ({size:.0f} KB)")
print("=" * 60)

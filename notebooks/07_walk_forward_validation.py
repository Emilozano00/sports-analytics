"""
Modelo v5: Walk-Forward Validation — simula predicción en torneo en curso.

Escenario simulado:
  - Torneo anterior: Apertura 2024 (jul-oct 2024)
  - Torneo actual:   Clausura 2024 (ene-mar 2025) → simula "Clausura 2025"

Experimento:
  1. Train = Apertura 2024 completo + primeras K jornadas del Clausura
  2. Test  = jornadas restantes del Clausura
  3. Iterar K = 0, 1, 2, ... para ver cómo mejora con más datos del torneo actual

Uso:
    python notebooks/07_walk_forward_validation.py
"""

import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
})

ACCENT = '#00d4aa'
ACCENT2 = '#ff6b6b'
ACCENT3 = '#4ecdc4'
ORANGE = '#f0a500'
PURPLE = '#a855f7'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(OUT_DIR, exist_ok=True)

ROLLING_N = 5
STREAK_N = 3


# =====================================================================
# 1. CARGAR DATOS
# =====================================================================
print("=" * 70)
print("1. CARGA DE DATOS")
print("=" * 70)

df_apertura = pd.read_csv(os.path.join(RAW_DIR, 'apertura2024_team_stats.csv'))
df_clausura = pd.read_csv(os.path.join(RAW_DIR, 'clausura2024_team_stats.csv'))

if 'torneo' not in df_apertura.columns:
    df_apertura['torneo'] = 'Apertura'
if 'torneo' not in df_clausura.columns:
    df_clausura['torneo'] = 'Clausura'

# Full combined (chronological)
df_all = pd.concat([df_apertura, df_clausura], ignore_index=True)

print(f"  Apertura 2024 (torneo anterior): {df_apertura['fixture_id'].nunique()} partidos")
print(f"  Clausura 2024 (simula actual):   {df_clausura['fixture_id'].nunique()} partidos")

# Identify Clausura rounds in chronological order
df_clausura_tmp = df_clausura.copy()
df_clausura_tmp['fecha'] = pd.to_datetime(df_clausura_tmp['fecha'])
round_dates = df_clausura_tmp.groupby('ronda')['fecha'].min().sort_values()
clausura_rounds = round_dates.index.tolist()
print(f"\n  Jornadas del Clausura (orden cronológico):")
for i, r in enumerate(clausura_rounds):
    n_matches = df_clausura_tmp[df_clausura_tmp['ronda'] == r]['fixture_id'].nunique()
    date = round_dates[r].strftime('%Y-%m-%d')
    print(f"    J{i+1:2d}: {r:<30} ({date}, {n_matches} partidos)")


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
def build_features(df):
    """Construye features y devuelve matches con labels."""
    df = df.copy()
    df['posesion'] = df['ball_possession'].str.replace('%', '').astype(float)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['fecha', 'fixture_id']).reset_index(drop=True)
    df['puntos'] = df['resultado'].map({'W': 3, 'D': 1, 'L': 0})

    feat_map = {}
    for team in df['equipo'].unique():
        tdf = df[df['equipo'] == team].sort_values('fecha').copy()

        tdf['goles_rolling'] = tdf['goles'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        tdf['goles_rec_rolling'] = tdf['goles_rival'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        tdf['posesion_rolling'] = tdf['posesion'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        tdf['tiros_rolling'] = tdf['total_shots'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        tdf['tiros_gol_rolling'] = tdf['shots_on_goal'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        tdf['racha_puntos'] = tdf['puntos'].rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)
        tdf['racha_victorias'] = (tdf['resultado'] == 'W').astype(int).rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)
        tdf['racha_derrotas'] = (tdf['resultado'] == 'L').astype(int).rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)

        tdf['goles_hist_local'] = np.nan
        tdf['goles_hist_visita'] = np.nan
        home_idx = tdf[tdf['side'] == 'home'].index
        away_idx = tdf[tdf['side'] == 'away'].index
        if len(home_idx) > 0:
            tdf.loc[home_idx, 'goles_hist_local'] = tdf.loc[home_idx, 'goles'].expanding(min_periods=1).mean().shift(1)
        if len(away_idx) > 0:
            tdf.loc[away_idx, 'goles_hist_visita'] = tdf.loc[away_idx, 'goles'].expanding(min_periods=1).mean().shift(1)
        tdf['goles_hist_local'] = tdf['goles_hist_local'].ffill()
        tdf['goles_hist_visita'] = tdf['goles_hist_visita'].ffill()

        tdf['puntos_acum'] = tdf['puntos'].cumsum().shift(1)
        tdf['gd_acum'] = (tdf['goles'] - tdf['goles_rival']).cumsum().shift(1)

        for _, row in tdf.iterrows():
            feat_map[(row['fixture_id'], team)] = {
                'goles_rolling': row['goles_rolling'],
                'goles_rec_rolling': row['goles_rec_rolling'],
                'posesion_rolling': row['posesion_rolling'],
                'tiros_rolling': row['tiros_rolling'],
                'tiros_gol_rolling': row['tiros_gol_rolling'],
                'racha_puntos': row['racha_puntos'],
                'racha_victorias': row['racha_victorias'],
                'racha_derrotas': row['racha_derrotas'],
                'goles_hist_local': row['goles_hist_local'],
                'goles_hist_visita': row['goles_hist_visita'],
                'puntos_acum': row['puntos_acum'],
                'gd_acum': row['gd_acum'],
            }

    df_h = df[df['side'] == 'home'][['fixture_id', 'fecha', 'equipo', 'goles', 'torneo', 'ronda']].copy()
    df_h.columns = ['fixture_id', 'fecha', 'home', 'goles_home', 'torneo', 'ronda']
    df_a = df[df['side'] == 'away'][['fixture_id', 'equipo', 'goles']].copy()
    df_a.columns = ['fixture_id', 'away', 'goles_away']
    matches = df_h.merge(df_a, on='fixture_id')
    matches['total_goles'] = matches['goles_home'] + matches['goles_away']
    matches['over25'] = (matches['total_goles'] > 2.5).astype(int)

    all_feats = ['goles_rolling', 'goles_rec_rolling', 'posesion_rolling',
                 'tiros_rolling', 'tiros_gol_rolling', 'racha_puntos',
                 'racha_victorias', 'racha_derrotas', 'goles_hist_local',
                 'goles_hist_visita', 'puntos_acum', 'gd_acum']

    for prefix, col in [('home', 'home'), ('away', 'away')]:
        for feat in all_feats:
            matches[f'{prefix}_{feat}'] = matches.apply(
                lambda row: feat_map.get((row['fixture_id'], row[col]), {}).get(feat), axis=1)

    matches['diff_puntos_tabla'] = matches['home_puntos_acum'] - matches['away_puntos_acum']
    matches['diff_gd'] = matches['home_gd_acum'] - matches['away_gd_acum']
    matches['diff_racha'] = matches['home_racha_puntos'] - matches['away_racha_puntos']
    matches['sum_goles_rolling'] = matches['home_goles_rolling'] + matches['away_goles_rolling']
    matches['sum_goles_rec_rolling'] = matches['home_goles_rec_rolling'] + matches['away_goles_rec_rolling']

    return matches


FEATURES = [
    'home_goles_rolling', 'away_goles_rolling',
    'home_goles_rec_rolling', 'away_goles_rec_rolling',
    'home_posesion_rolling', 'away_posesion_rolling',
    'home_tiros_rolling', 'away_tiros_rolling',
    'home_tiros_gol_rolling', 'away_tiros_gol_rolling',
    'home_racha_puntos', 'away_racha_puntos',
    'home_racha_victorias', 'away_racha_victorias',
    'home_racha_derrotas', 'away_racha_derrotas',
    'home_goles_hist_local', 'away_goles_hist_visita',
    'diff_puntos_tabla', 'diff_gd', 'diff_racha',
    'sum_goles_rolling', 'sum_goles_rec_rolling',
]

FEATURE_LABELS = [
    'Goles local (avg 5)', 'Goles visitante (avg 5)',
    'Goles rec. local (avg 5)', 'Goles rec. visit. (avg 5)',
    'Posesión local (avg 5)', 'Posesión visit. (avg 5)',
    'Tiros local (avg 5)', 'Tiros visit. (avg 5)',
    'Tiros a gol local (avg 5)', 'Tiros a gol visit. (avg 5)',
    'Racha puntos local (3)', 'Racha puntos visit. (3)',
    'Rachas W local (3)', 'Rachas W visit. (3)',
    'Rachas L local (3)', 'Rachas L visit. (3)',
    'Goles hist. como local', 'Goles hist. como visita',
    'Δ Posición tabla', 'Δ Diferencia goles',
    'Δ Racha', 'Suma goles rolling', 'Suma goles rec. rolling',
]


# =====================================================================
# 3. BUILD FEATURES ON FULL TIMELINE
# =====================================================================
print("\n" + "=" * 70)
print("2. CONSTRUYENDO FEATURES (timeline completa)")
print("=" * 70)

# Build on the full combined dataset so rolling features carry over
matches = build_features(df_all)
matches_clean = matches.dropna(subset=FEATURES).copy()

# Identify Clausura matches by round
matches_clean['fecha'] = pd.to_datetime(matches_clean['fecha'])
mask_apertura = matches_clean['torneo'] == 'Apertura'
mask_clausura = matches_clean['torneo'] == 'Clausura'

# Map Clausura rounds to chronological order
round_order = {r: i for i, r in enumerate(clausura_rounds)}
matches_clean.loc[mask_clausura, 'round_idx'] = matches_clean.loc[mask_clausura, 'ronda'].map(round_order)

n_apertura = mask_apertura.sum()
n_clausura = mask_clausura.sum()

print(f"  Partidos con features completas:")
print(f"    Apertura (train base): {n_apertura}")
print(f"    Clausura (simulado):   {n_clausura}")
print(f"    Total:                 {len(matches_clean)}")

# Over/Under distribution
for label, mask in [('Apertura', mask_apertura), ('Clausura', mask_clausura)]:
    y = matches_clean.loc[mask, 'over25']
    print(f"  {label}: Over={y.sum()} ({y.mean():.1%}) Under={len(y)-y.sum()} ({1-y.mean():.1%})")


# =====================================================================
# 4. WALK-FORWARD: Train on Apertura + K jornadas → Test on rest
# =====================================================================
print("\n" + "=" * 70)
print("3. WALK-FORWARD VALIDATION")
print("=" * 70)

def get_models():
    return {
        'LogReg': LogisticRegression(random_state=42, max_iter=1000),
        'LogReg L1': LogisticRegression(random_state=42, max_iter=1000, C=0.5, l1_ratio=1.0, solver='saga', penalty='elasticnet'),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=5, max_features='sqrt', random_state=42),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, learning_rate=0.1, random_state=42),
    }

apertura_matches = matches_clean[mask_apertura]
clausura_matches = matches_clean[mask_clausura].sort_values('round_idx')

# Test from round cut_after+1 onwards; need at least 2 rounds to test
max_train_rounds = len(clausura_rounds) - 2  # leave at least 2 rounds for testing

walk_forward_results = []

print(f"\n  {'K jornadas':>12} | {'Train':>6} | {'Test':>5} | {'Test Over%':>10} | ", end='')
for m in get_models():
    print(f" {m:>10}", end='')
print(f" | {'Baseline':>8}")
print(f"  {'-'*100}")

for k in range(0, max_train_rounds + 1):
    # Train: all Apertura + first K rounds of Clausura
    if k == 0:
        train_df = apertura_matches
    else:
        clau_train_rounds = clausura_rounds[:k]
        clau_train = clausura_matches[clausura_matches['ronda'].isin(clau_train_rounds)]
        train_df = pd.concat([apertura_matches, clau_train])

    # Test: remaining Clausura rounds
    clau_test_rounds = clausura_rounds[k:]
    test_df = clausura_matches[clausura_matches['ronda'].isin(clau_test_rounds)]

    if len(test_df) < 5:
        continue

    X_train = train_df[FEATURES].values
    y_train = train_df['over25'].values
    X_test = test_df[FEATURES].values
    y_test = test_df['over25'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline = max(y_test.mean(), 1 - y_test.mean())

    label = f"Aper + J1-J{k}" if k > 0 else "Solo Apertura"
    print(f"  {label:>12} | {len(y_train):>6} | {len(y_test):>5} | {y_test.mean():>9.1%} | ", end='')

    for model_name, model in get_models().items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

        walk_forward_results.append({
            'k': k,
            'label': label,
            'model': model_name,
            'train_n': len(y_train),
            'test_n': len(y_test),
            'accuracy': acc,
            'auc': auc,
            'baseline': baseline,
            'lift': acc - baseline,
            'over_rate': y_test.mean(),
        })

        marker = '+' if acc > baseline else '-'
        print(f" {acc:>9.1%}{marker}", end='')

    print(f" | {baseline:>7.1%}")

df_wf = pd.DataFrame(walk_forward_results)


# =====================================================================
# 5. ANÁLISIS: ¿Cuántas jornadas necesita el modelo?
# =====================================================================
print("\n" + "=" * 70)
print("4. ANÁLISIS: ¿Cuántas jornadas del torneo actual necesita el modelo?")
print("=" * 70)

# Best model per K
print(f"\n  {'K':>3} {'Train':>12} {'Mejor modelo':>15} {'Acc':>7} {'Baseline':>9} {'Lift':>7} {'AUC':>7}")
print(f"  {'-'*70}")

best_per_k = []
for k in df_wf['k'].unique():
    k_data = df_wf[df_wf['k'] == k]
    best = k_data.sort_values('accuracy', ascending=False).iloc[0]
    best_per_k.append(best)
    marker = "✓" if best['lift'] > 0 else "✗"
    print(f"  {int(k):>3} {best['label']:>12} {best['model']:>15} {best['accuracy']:>6.1%} {best['baseline']:>8.1%} {best['lift']:>+6.1%} {best['auc']:>6.3f} {marker}")

df_best_k = pd.DataFrame(best_per_k)


# =====================================================================
# 6. COMPARATIVA: Solo Clausura CV vs Walk-Forward
# =====================================================================
print("\n" + "=" * 70)
print("5. COMPARATIVA: Clausura CV interno vs Walk-Forward")
print("=" * 70)

# CV on full Clausura
X_clau = clausura_matches[FEATURES].values
y_clau = clausura_matches['over25'].values
scaler_clau = StandardScaler()
X_clau_s = scaler_clau.fit_transform(X_clau)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Método':<35} {'Modelo':>15} {'Acc':>7} {'AUC':>7} {'vs Baseline':>12}")
print(f"  {'-'*80}")

baseline_clau = max(y_clau.mean(), 1 - y_clau.mean())

for model_name, model in get_models().items():
    cv_acc = cross_val_score(model, X_clau_s, y_clau, cv=cv, scoring='accuracy').mean()
    cv_auc = cross_val_score(model, X_clau_s, y_clau, cv=cv, scoring='roc_auc').mean()
    lift = cv_acc - baseline_clau
    marker = "✓" if lift > 0 else "✗"
    print(f"  {'Clausura CV (5-fold)':<35} {model_name:>15} {cv_acc:>6.1%} {cv_auc:>6.3f} {lift:>+11.1%} {marker}")

# Walk-forward with best K
print()
for _, row in df_best_k.iterrows():
    marker = "✓" if row['lift'] > 0 else "✗"
    print(f"  {'Walk-fwd: ' + row['label']:<35} {row['model']:>15} {row['accuracy']:>6.1%} {row['auc']:>6.3f} {row['lift']:>+11.1%} {marker}")


# =====================================================================
# 7. DETALLE: Predicciones del mejor walk-forward por partido
# =====================================================================
print("\n" + "=" * 70)
print("6. PREDICCIONES PARTIDO A PARTIDO (mejor configuración)")
print("=" * 70)

# Find the K with best lift
best_overall = df_wf.sort_values('lift', ascending=False).iloc[0]
best_k = int(best_overall['k'])
best_model_name = best_overall['model']

print(f"\n  Mejor: {best_overall['label']} con {best_model_name}")
print(f"  Accuracy: {best_overall['accuracy']:.1%} | AUC: {best_overall['auc']:.3f} | Lift: {best_overall['lift']:+.1%}")

# Re-train best model
if best_k == 0:
    train_df = apertura_matches
else:
    clau_train = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[:best_k])]
    train_df = pd.concat([apertura_matches, clau_train])

clau_test = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[best_k:])]

X_tr = train_df[FEATURES].values
y_tr = train_df['over25'].values
X_te = clau_test[FEATURES].values
y_te = clau_test['over25'].values

scaler_best = StandardScaler()
X_tr_s = scaler_best.fit_transform(X_tr)
X_te_s = scaler_best.transform(X_te)

model_best = get_models()[best_model_name]
model_best.fit(X_tr_s, y_tr)
y_pred = model_best.predict(X_te_s)
y_prob = model_best.predict_proba(X_te_s)[:, 1]

clau_test = clau_test.copy()
clau_test['pred'] = y_pred
clau_test['prob_over'] = y_prob
clau_test['correct'] = (clau_test['pred'] == clau_test['over25']).astype(int)

print(f"\n  {'Fecha':<12} {'Local':<18} {'Visita':<18} {'Goles':>5} {'Real':>6} {'Pred':>6} {'Prob':>6} {'':>3}")
print(f"  {'-'*85}")

for _, row in clau_test.sort_values('fecha').iterrows():
    real = 'Over' if row['over25'] == 1 else 'Under'
    pred = 'Over' if row['pred'] == 1 else 'Under'
    ok = '✓' if row['correct'] else '✗'
    print(f"  {str(row['fecha'])[:10]:<12} {row['home']:<18} {row['away']:<18} {row['total_goles']:>5.0f} {real:>6} {pred:>6} {row['prob_over']:>5.0%} {ok:>3}")

# Accuracy by round
print(f"\n  Accuracy por jornada:")
for r in clausura_rounds[best_k:]:
    r_data = clau_test[clau_test['ronda'] == r]
    if len(r_data) > 0:
        r_acc = r_data['correct'].mean()
        print(f"    {r:<30} {r_acc:.0%} ({int(r_data['correct'].sum())}/{len(r_data)})")


# =====================================================================
# 8. VISUALIZACIONES
# =====================================================================
print("\n" + "=" * 70)
print("7. GENERANDO VISUALIZACIONES")
print("=" * 70)

# --- 8a. Walk-forward accuracy vs K jornadas (heatmap) ---
fig, ax = plt.subplots(figsize=(14, 6))

pivot = df_wf.pivot_table(index='model', columns='k', values='accuracy')
# Order models by max accuracy
model_order = df_wf.groupby('model')['accuracy'].max().sort_values(ascending=True).index
pivot = pivot.loc[model_order]

im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.35, vmax=0.75)
ax.set_xticks(range(pivot.shape[1]))
x_labels = [f'J0\n(solo Aper)'] + [f'J1-J{k}' for k in range(1, pivot.shape[1])]
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)
ax.set_xlabel('Jornadas del Clausura incluidas en training')

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = '#1a1d23' if val > 0.6 else '#e0e0e0'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

# Add baseline line annotation
ax.set_title('Walk-Forward: Accuracy al agregar jornadas del torneo actual al training',
             fontweight='bold', pad=15)
fig.colorbar(im, ax=ax, label='Accuracy', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v5_walkforward_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v5_walkforward_heatmap.png")

# --- 8b. Accuracy + AUC curves by K for each model ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors_model = {'LogReg': '#636e72', 'LogReg L1': ACCENT, 'RF': ACCENT3, 'GBM': ORANGE}

for model_name in get_models():
    m_data = df_wf[df_wf['model'] == model_name].sort_values('k')
    ax1.plot(m_data['k'], m_data['accuracy'], 'o-', color=colors_model[model_name],
             linewidth=2, markersize=6, label=model_name)
    ax2.plot(m_data['k'], m_data['auc'], 'o-', color=colors_model[model_name],
             linewidth=2, markersize=6, label=model_name)

# Baseline (varies with K because test set changes)
baselines = df_wf.groupby('k')['baseline'].first()
ax1.plot(baselines.index, baselines.values, '--', color=ACCENT2, linewidth=1.5, alpha=0.7, label='Baseline')
ax2.axhline(y=0.5, color=ACCENT2, linestyle='--', linewidth=1, alpha=0.5, label='Random')

ax1.set_xlabel('Jornadas del Clausura en training')
ax1.set_ylabel('Accuracy en jornadas restantes')
ax1.set_title('Accuracy vs jornadas incluidas', fontweight='bold')
ax1.legend(framealpha=0.3, fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylim(0.30, 0.80)

ax2.set_xlabel('Jornadas del Clausura en training')
ax2.set_ylabel('AUC en jornadas restantes')
ax2.set_title('AUC vs jornadas incluidas', fontweight='bold')
ax2.legend(framealpha=0.3, fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0.25, 0.80)

plt.suptitle('Walk-Forward: ¿Cuántas jornadas necesita el modelo para generalizarse?',
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v5_walkforward_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v5_walkforward_curves.png")

# --- 8c. ROC curves for best K ---
fig, ax = plt.subplots(figsize=(8, 8))

# Re-run all models for best K
if best_k == 0:
    train_roc = apertura_matches
else:
    clau_train_roc = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[:best_k])]
    train_roc = pd.concat([apertura_matches, clau_train_roc])

test_roc = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[best_k:])]
X_tr_roc = train_roc[FEATURES].values
y_tr_roc = train_roc['over25'].values
X_te_roc = test_roc[FEATURES].values
y_te_roc = test_roc['over25'].values

scaler_roc = StandardScaler()
X_tr_roc_s = scaler_roc.fit_transform(X_tr_roc)
X_te_roc_s = scaler_roc.transform(X_te_roc)

for (model_name, model), color in zip(get_models().items(), [c for c in colors_model.values()]):
    model.fit(X_tr_roc_s, y_tr_roc)
    y_prob_roc = model.predict_proba(X_te_roc_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_te_roc, y_prob_roc)
    auc_val = roc_auc_score(y_te_roc, y_prob_roc)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{model_name} (AUC:{auc_val:.3f})')

ax.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1, label='Random (0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC — Walk-Forward (Train: {best_overall["label"]}, Test: J{best_k+1}+)',
             fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3, fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v5_roc_walkforward.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v5_roc_walkforward.png")

# --- 8d. Match-level prediction visualization ---
test_sorted = clau_test.sort_values('fecha')
fig, ax = plt.subplots(figsize=(16, 5))

x = range(len(test_sorted))
colors = [ACCENT if c else ACCENT2 for c in test_sorted['correct']]
bars = ax.bar(x, test_sorted['prob_over'], color=colors, alpha=0.8, width=0.8)

# Horizontal line at 0.5
ax.axhline(y=0.5, color='#888', linestyle='--', linewidth=1, alpha=0.5)

# Mark actual overs
for i, (_, row) in enumerate(test_sorted.iterrows()):
    marker = '●' if row['over25'] == 1 else '○'
    ax.text(i, 1.02, marker, ha='center', va='bottom', fontsize=8,
            color=ACCENT if row['over25'] == 1 else '#888')

ax.set_xticks(list(x))
labels = [f"{str(r['fecha'])[:5]}\n{r['home'][:3]}v{r['away'][:3]}" for _, r in test_sorted.iterrows()]
ax.set_xticklabels(labels, fontsize=5, rotation=90)
ax.set_ylabel('P(Over 2.5)')
ax.set_title(f'Predicciones partido a partido — {best_model_name} ({best_overall["label"]})\n'
             f'Verde=acierto, Rojo=error | ●=Over real, ○=Under real',
             fontweight='bold', pad=15)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# Add accuracy text
acc_text = f'Accuracy: {best_overall["accuracy"]:.1%} ({int(test_sorted["correct"].sum())}/{len(test_sorted)})'
ax.text(0.98, 0.95, acc_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=11, fontweight='bold', color=ACCENT,
        bbox=dict(boxstyle='round', facecolor='#1a1d23', edgecolor=ACCENT, alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v5_match_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v5_match_predictions.png")


# =====================================================================
# RESUMEN FINAL
# =====================================================================
print(f"\n{'='*70}")
print("RESUMEN FINAL")
print(f"{'='*70}")

print(f"\n  Escenario: Apertura 2024 (anterior) + Clausura 2024 (simula actual)")
print(f"  {n_apertura} partidos Apertura + {n_clausura} partidos Clausura")

print(f"\n  Walk-Forward — Accuracy por jornadas incluidas:")
for _, row in df_best_k.iterrows():
    k = int(row['k'])
    marker = "✓" if row['lift'] > 0 else "✗"
    print(f"    K={k:>2} ({row['label']:<15}) → {row['model']:>10}: {row['accuracy']:.1%} (baseline:{row['baseline']:.1%}, lift:{row['lift']:+.1%}) {marker}")

# Find inflection point
beats_baseline = df_best_k[df_best_k['lift'] > 0]
if len(beats_baseline) > 0:
    first_beat = beats_baseline.iloc[0]
    print(f"\n  ➜ El modelo supera baseline desde K={int(first_beat['k'])} jornadas")
    print(f"    ({first_beat['model']}: {first_beat['accuracy']:.1%} vs {first_beat['baseline']:.1%})")
else:
    print(f"\n  ➜ El modelo no supera baseline en ningún punto del walk-forward")

best_wf = df_best_k.sort_values('lift', ascending=False).iloc[0]
print(f"\n  ➜ Mejor configuración: K={int(best_wf['k'])} jornadas")
print(f"    {best_wf['model']}: {best_wf['accuracy']:.1%} (AUC:{best_wf['auc']:.3f}, lift:{best_wf['lift']:+.1%})")

print(f"\n  Visualizaciones en data/processed/:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('v5_'):
        print(f"    → {f}")

"""
Modelo v4: Over/Under 2.5 goles - Feature de torneo + validación cruzada temporal
- Agrega 'torneo' como feature categórica al modelo combinado
- Entrena modelos separados por torneo
- Valida generalización: Train Clausura 2024 → Test Apertura 2024 (forward temporal)

Uso:
    python notebooks/06_modelo_v4_torneo_feature.py
"""

import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

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

apertura_path = os.path.join(RAW_DIR, 'apertura2024_team_stats.csv')
clausura_path = os.path.join(RAW_DIR, 'clausura2024_team_stats.csv')

df_apertura = pd.read_csv(apertura_path)
df_clausura = pd.read_csv(clausura_path)

if 'torneo' not in df_apertura.columns:
    df_apertura['torneo'] = 'Apertura'
if 'torneo' not in df_clausura.columns:
    df_clausura['torneo'] = 'Clausura'

df_all = pd.concat([df_clausura, df_apertura], ignore_index=True)

print(f"  Clausura 2024: {df_clausura['fixture_id'].nunique()} partidos")
print(f"  Apertura 2024: {df_apertura['fixture_id'].nunique()} partidos")
print(f"  Combinado:     {df_all['fixture_id'].nunique()} partidos")


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
def build_features(df, add_torneo=False):
    """Construye features para modelado. add_torneo agrega feature categórica."""
    df = df.copy()
    df['posesion'] = df['ball_possession'].str.replace('%', '').astype(float)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['fecha', 'fixture_id']).reset_index(drop=True)
    df['puntos'] = df['resultado'].map({'W': 3, 'D': 1, 'L': 0})

    # Rolling features por equipo
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

    # Construir matches
    df_h = df[df['side'] == 'home'][['fixture_id', 'fecha', 'equipo', 'goles', 'torneo']].copy()
    df_h.columns = ['fixture_id', 'fecha', 'home', 'goles_home', 'torneo']
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

    if add_torneo:
        matches['is_apertura'] = (matches['torneo'] == 'Apertura').astype(int)

    return matches


BASE_FEATURES = [
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

TORNEO_FEATURES = BASE_FEATURES + ['is_apertura']

BASE_LABELS = [
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

TORNEO_LABELS = BASE_LABELS + ['Es Apertura (torneo)']


def get_models():
    return {
        'LogReg (baseline)': LogisticRegression(random_state=42, max_iter=1000),
        'LogReg + L1 (C=0.5)': LogisticRegression(random_state=42, max_iter=1000, C=0.5, l1_ratio=1.0, solver='saga', penalty='elasticnet'),
        'RF (depth=3, leaf=5)': RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=5, max_features='sqrt', random_state=42),
        'GBM (depth=2, n=100)': GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, learning_rate=0.1, random_state=42),
    }


# =====================================================================
# 3. PARTE A: COMBINADO SIN vs CON FEATURE DE TORNEO
# =====================================================================
print("\n" + "=" * 70)
print("2. COMBINADO: SIN vs CON FEATURE DE TORNEO")
print("=" * 70)

matches_no_torneo = build_features(df_all, add_torneo=False)
matches_with_torneo = build_features(df_all, add_torneo=True)

results_torneo_feat = []

for label, matches, feat_names in [
    ('Sin torneo (23 features)', matches_no_torneo, BASE_FEATURES),
    ('Con torneo (24 features)', matches_with_torneo, TORNEO_FEATURES),
]:
    clean = matches.dropna(subset=feat_names).copy()
    X = clean[feat_names].values
    y = clean['over25'].values
    n = len(y)
    baseline = max(y.mean(), 1 - y.mean())

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  --- {label} (n={n}) ---")
    print(f"  Over: {y.sum()} ({y.mean():.1%}) | Under: {n-y.sum()} ({1-y.mean():.1%}) | Baseline: {baseline:.1%}")

    for model_name, model in get_models().items():
        cv_acc = cross_val_score(model, X_s, y, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(model, X_s, y, cv=cv, scoring='roc_auc')
        results_torneo_feat.append({
            'config': label,
            'model': model_name,
            'cv_acc': cv_acc.mean(),
            'cv_std': cv_acc.std(),
            'cv_auc': cv_auc.mean(),
            'baseline': baseline,
        })
        print(f"  {model_name:<25} CV:{cv_acc.mean():.1%}±{cv_acc.std():.1%}  AUC:{cv_auc.mean():.3f}")

df_tf = pd.DataFrame(results_torneo_feat)

# Mostrar mejora
print(f"\n  --- Δ Mejora al agregar torneo ---")
for m in get_models():
    sin = df_tf[(df_tf['config'].str.startswith('Sin')) & (df_tf['model'] == m)].iloc[0]
    con = df_tf[(df_tf['config'].str.startswith('Con')) & (df_tf['model'] == m)].iloc[0]
    delta = con['cv_acc'] - sin['cv_acc']
    print(f"  {m:<25} {delta:+.1%} ({sin['cv_acc']:.1%} → {con['cv_acc']:.1%})")


# =====================================================================
# 4. PARTE B: MODELOS SEPARADOS POR TORNEO
# =====================================================================
print("\n" + "=" * 70)
print("3. MODELOS SEPARADOS POR TORNEO")
print("=" * 70)

results_per_torneo = []

for torneo_name, df_torneo in [('Clausura 2024', df_clausura), ('Apertura 2024', df_apertura)]:
    matches = build_features(df_torneo, add_torneo=False)
    clean = matches.dropna(subset=BASE_FEATURES).copy()
    X = clean[BASE_FEATURES].values
    y = clean['over25'].values
    n = len(y)
    baseline = max(y.mean(), 1 - y.mean())

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  --- {torneo_name} (n={n}) ---")
    print(f"  Over: {y.sum()} ({y.mean():.1%}) | Under: {n-y.sum()} ({1-y.mean():.1%}) | Baseline: {baseline:.1%}")

    for model_name, model in get_models().items():
        cv_acc = cross_val_score(model, X_s, y, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(model, X_s, y, cv=cv, scoring='roc_auc')
        results_per_torneo.append({
            'torneo': torneo_name,
            'model': model_name,
            'cv_acc': cv_acc.mean(),
            'cv_std': cv_acc.std(),
            'cv_auc': cv_auc.mean(),
            'baseline': baseline,
            'n': n,
        })
        print(f"  {model_name:<25} CV:{cv_acc.mean():.1%}±{cv_acc.std():.1%}  AUC:{cv_auc.mean():.3f}")
    print(f"  {'Baseline':<25} {baseline:.1%}")


# =====================================================================
# 5. PARTE C: VALIDACIÓN CRUZADA TEMPORAL (Train→Test entre torneos)
# =====================================================================
print("\n" + "=" * 70)
print("4. VALIDACIÓN CRUZADA TEMPORAL (generalización entre torneos)")
print("=" * 70)

# Build features on the combined dataset so rolling stats carry across tournaments
matches_all = build_features(df_all, add_torneo=True)
matches_all_clean = matches_all.dropna(subset=BASE_FEATURES).copy()

# Split by tournament
mask_clausura = matches_all_clean['torneo'] == 'Clausura'
mask_apertura = matches_all_clean['torneo'] == 'Apertura'

df_train_clau = matches_all_clean[mask_clausura]
df_test_aper = matches_all_clean[mask_apertura]
df_train_aper = matches_all_clean[mask_apertura]
df_test_clau = matches_all_clean[mask_clausura]

temporal_results = []

for direction, df_train, df_test, train_name, test_name in [
    ('Clausura → Apertura', df_train_clau, df_test_aper, 'Clausura', 'Apertura'),
    ('Apertura → Clausura', df_train_aper, df_test_clau, 'Apertura', 'Clausura'),
]:
    X_train = df_train[BASE_FEATURES].values
    y_train = df_train['over25'].values
    X_test = df_test[BASE_FEATURES].values
    y_test = df_test['over25'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline_test = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n  --- {direction} ---")
    print(f"  Train: {len(y_train)} partidos ({train_name}) | Test: {len(y_test)} partidos ({test_name})")
    print(f"  Test Over: {y_test.sum()} ({y_test.mean():.1%}) | Test Baseline: {baseline_test:.1%}")

    for model_name, model in get_models().items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        lift = acc - baseline_test

        temporal_results.append({
            'direction': direction,
            'model': model_name,
            'train_n': len(y_train),
            'test_n': len(y_test),
            'accuracy': acc,
            'auc': auc,
            'baseline': baseline_test,
            'lift': lift,
            'y_test': y_test,
            'y_prob': y_prob,
            'y_pred': y_pred,
        })

        marker = "✓" if lift > 0 else "✗"
        print(f"  {marker} {model_name:<25} Acc:{acc:.1%}  AUC:{auc:.3f}  Lift:{lift:+.1%}")

    print(f"  {'Always majority':<27} Acc:{baseline_test:.1%}")

df_temporal = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_prob', 'y_pred']} for r in temporal_results])


# =====================================================================
# 6. ANÁLISIS DETALLADO: Clausura → Apertura (forward temporal)
# =====================================================================
print("\n" + "=" * 70)
print("5. ANÁLISIS DETALLADO: Clausura → Apertura (simulación forward)")
print("=" * 70)

# Best model for Clausura → Apertura
forward_results = [r for r in temporal_results if r['direction'] == 'Clausura → Apertura']
best_forward = max(forward_results, key=lambda x: x['accuracy'])

print(f"\n  Mejor modelo: {best_forward['model']}")
print(f"  Accuracy: {best_forward['accuracy']:.1%} (baseline: {best_forward['baseline']:.1%}, lift: {best_forward['lift']:+.1%})")
print(f"  AUC: {best_forward['auc']:.3f}")

# Classification report for best model
print(f"\n  Classification Report ({best_forward['model']}):")
report = classification_report(best_forward['y_test'], best_forward['y_pred'],
                               target_names=['Under 2.5', 'Over 2.5'])
for line in report.split('\n'):
    print(f"  {line}")

# Feature importance: train GBM on Clausura, show what matters
print(f"\n  --- Feature Importance (GBM entrenado en Clausura) ---")
X_train_clau = df_train_clau[BASE_FEATURES].values
y_train_clau = df_train_clau['over25'].values
scaler_clau = StandardScaler()
X_train_clau_s = scaler_clau.fit_transform(X_train_clau)

gbm_forward = GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, learning_rate=0.1, random_state=42)
gbm_forward.fit(X_train_clau_s, y_train_clau)

importances = pd.DataFrame({
    'feature': BASE_LABELS,
    'importance': gbm_forward.feature_importances_,
}).sort_values('importance', ascending=False)

print(f"  {'Feature':<30} {'Imp':>8}")
print(f"  {'-'*42}")
for _, row in importances.head(10).iterrows():
    bar = '█' * int(row['importance'] * 60)
    print(f"  {row['feature']:<30} {row['importance']:>7.3f}  {bar}")


# =====================================================================
# 7. ANÁLISIS POR JORNADA: ¿Mejora con más jornadas del Apertura?
# =====================================================================
print("\n" + "=" * 70)
print("6. CURVA DE GENERALIZACIÓN: Acc por jornada del Apertura")
print("=" * 70)

# Sort test data by date and evaluate cumulatively
test_sorted = df_test_aper.sort_values('fecha').copy()
X_test_all = scaler_clau.transform(test_sorted[BASE_FEATURES].values)
y_test_all = test_sorted['over25'].values
y_prob_all = gbm_forward.predict_proba(X_test_all)[:, 1]
y_pred_all = gbm_forward.predict(X_test_all)

# Cumulative accuracy as we go through more Apertura matches
cumulative_acc = []
cumulative_n = []
for i in range(5, len(y_test_all) + 1):
    acc_i = accuracy_score(y_test_all[:i], y_pred_all[:i])
    cumulative_acc.append(acc_i)
    cumulative_n.append(i)

print(f"  Primeros 10 partidos: {accuracy_score(y_test_all[:10], y_pred_all[:10]):.1%}")
print(f"  Primeros 20 partidos: {accuracy_score(y_test_all[:20], y_pred_all[:20]):.1%}")
if len(y_test_all) >= 30:
    print(f"  Primeros 30 partidos: {accuracy_score(y_test_all[:30], y_pred_all[:30]):.1%}")
print(f"  Todos ({len(y_test_all)} partidos): {accuracy_score(y_test_all, y_pred_all):.1%}")


# =====================================================================
# 8. VISUALIZACIONES
# =====================================================================
print("\n" + "=" * 70)
print("7. GENERANDO VISUALIZACIONES")
print("=" * 70)

# --- 8a. Sin vs Con torneo (bar chart comparativo) ---
fig, ax = plt.subplots(figsize=(12, 6))
models_list = list(get_models().keys())
x = np.arange(len(models_list))
width = 0.35

sin_vals = [df_tf[(df_tf['config'].str.startswith('Sin')) & (df_tf['model'] == m)].iloc[0]['cv_acc'] for m in models_list]
con_vals = [df_tf[(df_tf['config'].str.startswith('Con')) & (df_tf['model'] == m)].iloc[0]['cv_acc'] for m in models_list]
sin_std = [df_tf[(df_tf['config'].str.startswith('Sin')) & (df_tf['model'] == m)].iloc[0]['cv_std'] for m in models_list]
con_std = [df_tf[(df_tf['config'].str.startswith('Con')) & (df_tf['model'] == m)].iloc[0]['cv_std'] for m in models_list]

bars1 = ax.bar(x - width/2, sin_vals, width, yerr=sin_std, capsize=4,
               label='Sin torneo (23 feat)', color=ACCENT3, alpha=0.85)
bars2 = ax.bar(x + width/2, con_vals, width, yerr=con_std, capsize=4,
               label='Con torneo (24 feat)', color=ACCENT, alpha=0.85)

baseline_val = df_tf.iloc[0]['baseline']
ax.axhline(y=baseline_val, color=ACCENT2, linestyle='--', alpha=0.6, label=f'Baseline ({baseline_val:.1%})')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('CV Accuracy')
ax.set_title('Efecto de agregar "torneo" como feature al modelo combinado', fontweight='bold', pad=15)
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.40, 0.75)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.1%}',
                ha='center', va='bottom', fontsize=8, color='#ccc')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v4_torneo_feature_effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v4_torneo_feature_effect.png")

# --- 8b. Validación temporal (Clausura→Apertura vs Apertura→Clausura) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, direction in enumerate(['Clausura → Apertura', 'Apertura → Clausura']):
    ax = axes[idx]
    dir_data = df_temporal[df_temporal['direction'] == direction]
    models_names = dir_data['model'].values
    accs = dir_data['accuracy'].values
    aucs = dir_data['auc'].values
    baseline = dir_data['baseline'].iloc[0]

    x = np.arange(len(models_names))
    colors = [ACCENT if a > baseline else ACCENT2 for a in accs]

    ax.barh(x, accs, color=colors, alpha=0.85, height=0.6)
    ax.axvline(x=baseline, color='#888', linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.1%})')
    ax.axvline(x=0.5, color=ACCENT2, linestyle=':', alpha=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(models_names, fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Train {direction.split("→")[0].strip()} → Test {direction.split("→")[1].strip()}',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0.30, 0.80)

    for i, (acc, auc) in enumerate(zip(accs, aucs)):
        ax.text(acc + 0.01, i, f'{acc:.1%} (AUC:{auc:.3f})', va='center', fontsize=8, color='#ccc')

plt.suptitle('Validación Temporal: ¿Generaliza el modelo entre torneos?',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v4_temporal_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v4_temporal_validation.png")

# --- 8c. ROC curves para Clausura → Apertura (forward) ---
fig, ax = plt.subplots(figsize=(8, 8))
colors_roc = ['#636e72', ACCENT, ACCENT3, ORANGE]

for r, color in zip(forward_results, colors_roc):
    fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{r['model']} (AUC:{r['auc']:.3f})")

ax.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1, label='Random (0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC — Clausura 2024 → Apertura 2024 (forward)', fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3, fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v4_roc_forward.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v4_roc_forward.png")

# --- 8d. Curva de generalización (accuracy acumulativa en Apertura) ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cumulative_n, cumulative_acc, color=ACCENT, linewidth=2, label='GBM (Clausura→Apertura)')
ax.axhline(y=best_forward['baseline'], color=ACCENT2, linestyle='--', alpha=0.6,
           label=f'Baseline ({best_forward["baseline"]:.1%})')
ax.axhline(y=0.5, color='#555', linestyle=':', alpha=0.4)
ax.fill_between(cumulative_n, [best_forward['baseline']]*len(cumulative_n), cumulative_acc,
                where=[a > best_forward['baseline'] for a in cumulative_acc],
                alpha=0.15, color=ACCENT)
ax.fill_between(cumulative_n, [best_forward['baseline']]*len(cumulative_n), cumulative_acc,
                where=[a <= best_forward['baseline'] for a in cumulative_acc],
                alpha=0.15, color=ACCENT2)
ax.set_xlabel('Partidos del Apertura evaluados (acumulativo)')
ax.set_ylabel('Accuracy acumulativa')
ax.set_title('Curva de generalización: GBM entrenado en Clausura, evaluado en Apertura',
             fontweight='bold', pad=15)
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.30, 0.80)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v4_generalization_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v4_generalization_curve.png")

# --- 8e. Feature importance comparativa (Clausura-trained vs Apertura-trained) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, (train_df, train_name) in enumerate([(df_train_clau, 'Clausura'), (df_train_aper, 'Apertura')]):
    ax = axes[idx]
    X_t = train_df[BASE_FEATURES].values
    y_t = train_df['over25'].values
    sc = StandardScaler()
    X_t_s = sc.fit_transform(X_t)

    gbm_t = GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, learning_rate=0.1, random_state=42)
    gbm_t.fit(X_t_s, y_t)

    imp = pd.DataFrame({'feature': BASE_LABELS, 'importance': gbm_t.feature_importances_})
    imp = imp.sort_values('importance', ascending=True).tail(12)

    colors = [ACCENT if train_name == 'Clausura' else ORANGE for _ in range(len(imp))]
    ax.barh(range(len(imp)), imp['importance'].values, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['feature'].values, fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title(f'GBM entrenado en {train_name}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Feature Importance: ¿Qué features importan en cada torneo?',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v4_feature_importance_by_torneo.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v4_feature_importance_by_torneo.png")


# =====================================================================
# RESUMEN FINAL
# =====================================================================
print(f"\n{'='*70}")
print("RESUMEN FINAL")
print(f"{'='*70}")

print(f"\n  A) Feature de torneo en modelo combinado:")
best_sin = df_tf[df_tf['config'].str.startswith('Sin')].sort_values('cv_acc', ascending=False).iloc[0]
best_con = df_tf[df_tf['config'].str.startswith('Con')].sort_values('cv_acc', ascending=False).iloc[0]
print(f"     Sin torneo: {best_sin['model']} → {best_sin['cv_acc']:.1%}")
print(f"     Con torneo: {best_con['model']} → {best_con['cv_acc']:.1%}")
delta_torneo = best_con['cv_acc'] - best_sin['cv_acc']
print(f"     Δ: {delta_torneo:+.1%}")

print(f"\n  B) Modelos por torneo:")
for t in ['Clausura 2024', 'Apertura 2024']:
    df_t = pd.DataFrame([r for r in results_per_torneo if r['torneo'] == t])
    best_t = df_t.sort_values('cv_acc', ascending=False).iloc[0]
    print(f"     {t}: {best_t['model']} → {best_t['cv_acc']:.1%} (AUC:{best_t['cv_auc']:.3f}, baseline:{best_t['baseline']:.1%})")

print(f"\n  C) Generalización temporal:")
for direction in ['Clausura → Apertura', 'Apertura → Clausura']:
    dir_data = df_temporal[df_temporal['direction'] == direction]
    best_d = dir_data.sort_values('accuracy', ascending=False).iloc[0]
    print(f"     {direction}: {best_d['model']} → {best_d['accuracy']:.1%} (AUC:{best_d['auc']:.3f}, baseline:{best_d['baseline']:.1%}, lift:{best_d['lift']:+.1%})")

print(f"\n  Nota: API-Football Free no tiene datos 2025.")
print(f"  La validación Clausura→Apertura simula la generalización forward.")

print(f"\n  Visualizaciones en data/processed/:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('v4_'):
        print(f"    → {f}")

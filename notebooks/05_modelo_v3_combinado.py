"""
Modelo v3: Over/Under 2.5 goles - Liga MX 2024 COMPLETA (Clausura + Apertura)
Compara rendimiento con dataset pequeño (solo Apertura) vs combinado.

Uso:
    python notebooks/05_modelo_v3_combinado.py
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
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

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

# Intentar cargar combinado, si no existe usar solo Apertura
combined_path = os.path.join(RAW_DIR, 'ligamx2024_team_stats.csv')
apertura_path = os.path.join(RAW_DIR, 'apertura2024_team_stats.csv')
clausura_path = os.path.join(RAW_DIR, 'clausura2024_team_stats.csv')

datasets = {}

if os.path.exists(combined_path):
    df_combined = pd.read_csv(combined_path)
    datasets['Liga MX 2024 (combinado)'] = df_combined
    print(f"  ✓ Combinado: {df_combined['fixture_id'].nunique()} partidos")

if os.path.exists(apertura_path):
    df_apertura = pd.read_csv(apertura_path)
    # Add torneo column if missing (backwards compat)
    if 'torneo' not in df_apertura.columns:
        df_apertura['torneo'] = 'Apertura'
    datasets['Apertura 2024'] = df_apertura
    print(f"  ✓ Apertura: {df_apertura['fixture_id'].nunique()} partidos")

if os.path.exists(clausura_path):
    df_clausura = pd.read_csv(clausura_path)
    datasets['Clausura 2024'] = df_clausura
    print(f"  ✓ Clausura: {df_clausura['fixture_id'].nunique()} partidos")

# Si tenemos ambos pero no combinado, combinamos manualmente
if 'Liga MX 2024 (combinado)' not in datasets and 'Clausura 2024' in datasets:
    df_combined = pd.concat([datasets['Clausura 2024'], datasets['Apertura 2024']], ignore_index=True)
    datasets['Liga MX 2024 (combinado)'] = df_combined
    print(f"  ✓ Combinado (manual): {df_combined['fixture_id'].nunique()} partidos")

if len(datasets) == 0:
    print("ERROR: No hay datos. Ejecuta primero el extractor.")
    sys.exit(1)


# =====================================================================
# 2. FEATURE ENGINEERING (función reutilizable)
# =====================================================================
def build_features(df):
    """Construye features y devuelve dataset de partidos listo para modelar."""
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
    df_h = df[df['side'] == 'home'][['fixture_id', 'fecha', 'equipo', 'goles']].copy()
    df_h.columns = ['fixture_id', 'fecha', 'home', 'goles_home']
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


FEATURE_NAMES = [
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
# 3. EVALUAR MODELOS EN CADA DATASET
# =====================================================================
print("\n" + "=" * 70)
print("2. EVALUACIÓN DE MODELOS POR DATASET")
print("=" * 70)

def get_models():
    return {
        'LogReg (baseline)': LogisticRegression(random_state=42, max_iter=1000),
        'LogReg + L1 (C=0.5)': LogisticRegression(random_state=42, max_iter=1000, C=0.5, l1_ratio=1.0, solver='saga', penalty='elasticnet'),
        'RF (depth=3, leaf=5)': RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=5, max_features='sqrt', random_state=42),
        'RF (depth=4, leaf=4)': RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=4, max_features='sqrt', random_state=42),
        'GBM (depth=2, n=100)': GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, learning_rate=0.1, random_state=42),
    }

all_results = []

for ds_name, df_raw in datasets.items():
    print(f"\n--- {ds_name} ---")
    matches = build_features(df_raw)
    matches_clean = matches.dropna(subset=FEATURE_NAMES).copy()

    X = matches_clean[FEATURE_NAMES].values
    y = matches_clean['over25'].values
    n = len(y)
    baseline = max(y.mean(), 1 - y.mean())

    print(f"  Partidos usables: {n} | Over: {y.sum()} ({y.mean():.1%}) | Under: {n-y.sum()} ({1-y.mean():.1%})")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use LOO only for small datasets
    use_loo = n <= 80

    for model_name, model in get_models().items():
        cv_acc = cross_val_score(model, X_s, y, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(model, X_s, y, cv=cv, scoring='roc_auc')

        result = {
            'dataset': ds_name,
            'model': model_name,
            'n_partidos': n,
            'cv_acc_mean': cv_acc.mean(),
            'cv_acc_std': cv_acc.std(),
            'cv_auc_mean': cv_auc.mean(),
            'cv_auc_std': cv_auc.std(),
            'baseline': baseline,
        }

        if use_loo:
            loo_acc = cross_val_score(model, X_s, y, cv=LeaveOneOut(), scoring='accuracy')
            result['loo_acc'] = loo_acc.mean()

        all_results.append(result)
        loo_str = f" LOO:{result.get('loo_acc', 0):.1%}" if use_loo else ""
        print(f"  {model_name:<25} CV:{cv_acc.mean():.1%}±{cv_acc.std():.1%}  AUC:{cv_auc.mean():.3f}{loo_str}")

    print(f"  {'Baseline':<25} {baseline:.1%}")

df_results = pd.DataFrame(all_results)

# =====================================================================
# 4. ANÁLISIS DEL MEJOR MODELO (en el dataset más grande)
# =====================================================================
print("\n" + "=" * 70)
print("3. MEJOR MODELO - ANÁLISIS DETALLADO")
print("=" * 70)

# Seleccionar el dataset más grande
largest_ds = max(datasets.keys(), key=lambda k: datasets[k]['fixture_id'].nunique())
df_best = datasets[largest_ds]
matches_best = build_features(df_best)
matches_best = matches_best.dropna(subset=FEATURE_NAMES).copy()

X_best = matches_best[FEATURE_NAMES].values
y_best = matches_best['over25'].values
scaler_best = StandardScaler()
X_best_s = scaler_best.fit_transform(X_best)
cv_best = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Encontrar mejor modelo para este dataset
best_row = df_results[df_results['dataset'] == largest_ds].sort_values('cv_acc_mean', ascending=False).iloc[0]
print(f"\nDataset: {largest_ds} ({len(y_best)} partidos)")
print(f"Mejor modelo: {best_row['model']} (CV Acc: {best_row['cv_acc_mean']:.1%})")

# Entrenar mejor modelo y RF para feature importance
best_models = get_models()

# Feature importance con RF
rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=4, max_features='sqrt', random_state=42)
rf.fit(X_best_s, y_best)

importances = pd.DataFrame({
    'feature': FEATURE_LABELS,
    'importance': rf.feature_importances_,
}).sort_values('importance', ascending=False)

print(f"\n--- Feature Importance (Random Forest) ---")
print(f"{'Feature':<30} {'Importance':>12}")
print("-" * 45)
for _, row in importances.head(12).iterrows():
    bar = '█' * int(row['importance'] * 80)
    print(f"{row['feature']:<30} {row['importance']:>10.3f}  {bar}")

# =====================================================================
# 5. VISUALIZACIONES
# =====================================================================
print("\n" + "=" * 70)
print("4. GENERANDO VISUALIZACIONES")
print("=" * 70)

# --- 5a. Comparación datasets x modelos (heatmap) ---
pivot = df_results.pivot_table(index='model', columns='dataset', values='cv_acc_mean')

fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 4), 6))
models_ordered = df_results.groupby('model')['cv_acc_mean'].max().sort_values(ascending=True).index
pivot = pivot.loc[models_ordered]

im = ax.imshow(pivot.values, cmap='YlGn', aspect='auto', vmin=0.4, vmax=0.75)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = '#1a1d23' if val > 0.6 else '#e0e0e0'
            ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=13, fontweight='bold', color=color)

ax.set_title('CV Accuracy por Modelo y Dataset', fontweight='bold', pad=15)
fig.colorbar(im, ax=ax, label='Accuracy', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v3_heatmap_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v3_heatmap_modelos.png")

# --- 5b. Feature Importance del mejor dataset ---
fig, ax = plt.subplots(figsize=(12, 8))
imp_plot = importances.head(15).sort_values('importance', ascending=True)
y_pos = range(len(imp_plot))
colors = [ACCENT if i >= len(imp_plot) - 5 else ACCENT3 for i in range(len(imp_plot))]
ax.barh(y_pos, imp_plot['importance'].values, color=colors, alpha=0.85, height=0.6)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(imp_plot['feature'].values, fontsize=10)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top 15 Features — Random Forest ({largest_ds})', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(imp_plot['importance'].values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9, color='#ccc')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v3_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v3_feature_importance.png")

# --- 5c. ROC Comparativa (mejor dataset, todos los modelos) ---
fig, ax = plt.subplots(figsize=(8, 8))
colors_roc = ['#636e72', ACCENT, ACCENT3, ORANGE, ACCENT2]

for (model_name, model), color in zip(get_models().items(), colors_roc):
    y_prob_cv = cross_val_predict(model, X_best_s, y_best, cv=cv_best, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y_best, y_prob_cv)
    auc_val = roc_auc_score(y_best, y_prob_cv)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{model_name} ({auc_val:.3f})')

ax.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1, label='Random (0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Curva ROC — {largest_ds}', fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3, fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v3_roc_comparativa.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v3_roc_comparativa.png")

# --- 5d. Improvement chart (Apertura vs Combinado if both exist) ---
if len(datasets) > 1:
    fig, ax = plt.subplots(figsize=(12, 6))
    ds_names = list(datasets.keys())
    n_models = len(get_models())
    x = np.arange(n_models)
    width = 0.8 / len(ds_names)

    for i, ds in enumerate(ds_names):
        ds_data = df_results[df_results['dataset'] == ds].set_index('model').loc[list(get_models().keys())]
        offset = (i - len(ds_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, ds_data['cv_acc_mean'].values, width,
                      yerr=ds_data['cv_acc_std'].values, capsize=3,
                      label=f"{ds} (n={ds_data['n_partidos'].iloc[0]})",
                      alpha=0.85)

    ax.axhline(y=0.5, color=ACCENT2, linestyle='--', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(list(get_models().keys()), rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('CV Accuracy')
    ax.set_title('Efecto del Tamaño del Dataset en Performance', fontweight='bold', pad=15)
    ax.legend(framealpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.35, 0.80)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'v3_dataset_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → v3_dataset_comparison.png")

# =====================================================================
# RESUMEN FINAL
# =====================================================================
print(f"\n{'='*70}")
print("RESUMEN FINAL")
print(f"{'='*70}")

for ds in datasets:
    ds_res = df_results[df_results['dataset'] == ds]
    best = ds_res.sort_values('cv_acc_mean', ascending=False).iloc[0]
    bl = best['baseline']
    lift = best['cv_acc_mean'] - bl
    print(f"\n{ds} ({int(best['n_partidos'])} partidos):")
    print(f"  Mejor: {best['model']}")
    print(f"  CV Acc: {best['cv_acc_mean']:.1%} ± {best['cv_acc_std']:.1%}  |  AUC: {best['cv_auc_mean']:.3f}")
    print(f"  Baseline: {bl:.1%}  |  Lift: {lift:+.1%}")

print(f"\nVisualizaciones en data/processed/:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('v3_'):
        print(f"  → {f}")

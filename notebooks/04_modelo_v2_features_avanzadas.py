"""
Modelo v2: Over/Under 2.5 goles - Apertura 2024 Liga MX
Features avanzadas + Logistic Regression vs Random Forest

Nuevas features:
- Racha de victorias/derrotas (últimos 3 partidos)
- Promedio de goles como local / visitante (histórico en el torneo)
- Diferencia de posición en tabla acumulada
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)

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
})

ACCENT = '#00d4aa'
ACCENT2 = '#ff6b6b'
ACCENT3 = '#4ecdc4'
ORANGE = '#f0a500'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# 1. CARGAR Y PREPARAR DATOS
# =====================================================================
print("=" * 70)
print("1. PREPARACIÓN DE DATOS")
print("=" * 70)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'apertura2024_team_stats.csv'))
df['posesion'] = df['ball_possession'].str.replace('%', '').astype(float)
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values(['fecha', 'fixture_id']).reset_index(drop=True)

# Puntos por resultado (para tabla acumulada)
df['puntos'] = df['resultado'].map({'W': 3, 'D': 1, 'L': 0})

# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

ROLLING_N = 5
STREAK_N = 3

def compute_all_features(df):
    """Calcula todas las features por equipo antes de cada partido."""
    features = {}

    for team in df['equipo'].unique():
        team_df = df[df['equipo'] == team].sort_values('fecha').copy()

        # --- Rolling averages (últimos 5) ---
        team_df['goles_rolling'] = team_df['goles'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        team_df['goles_rec_rolling'] = team_df['goles_rival'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        team_df['posesion_rolling'] = team_df['posesion'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        team_df['tiros_rolling'] = team_df['total_shots'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        team_df['tiros_gol_rolling'] = team_df['shots_on_goal'].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)

        # --- Racha últimos 3 partidos ---
        # Puntos acumulados en últimos 3 (0-9 escala, 9 = 3 victorias seguidas)
        team_df['racha_puntos'] = team_df['puntos'].rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)
        # Victorias en últimos 3
        team_df['racha_victorias'] = (team_df['resultado'] == 'W').astype(int).rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)
        # Derrotas en últimos 3
        team_df['racha_derrotas'] = (team_df['resultado'] == 'L').astype(int).rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)

        # --- Goles como local vs visitante (promedio acumulado en el torneo) ---
        team_home = team_df[team_df['side'] == 'home']['goles']
        team_away = team_df[team_df['side'] == 'away']['goles']
        # Expanding mean shifted (solo data previa)
        team_df['goles_hist_local'] = np.nan
        team_df['goles_hist_visita'] = np.nan

        home_idx = team_df[team_df['side'] == 'home'].index
        away_idx = team_df[team_df['side'] == 'away'].index

        if len(home_idx) > 0:
            home_expanding = team_df.loc[home_idx, 'goles'].expanding(min_periods=1).mean().shift(1)
            team_df.loc[home_idx, 'goles_hist_local'] = home_expanding
        if len(away_idx) > 0:
            away_expanding = team_df.loc[away_idx, 'goles'].expanding(min_periods=1).mean().shift(1)
            team_df.loc[away_idx, 'goles_hist_visita'] = away_expanding

        # Forward-fill para que la última media conocida esté disponible
        team_df['goles_hist_local'] = team_df['goles_hist_local'].ffill()
        team_df['goles_hist_visita'] = team_df['goles_hist_visita'].ffill()

        # --- Posición en tabla acumulada (puntos totales + GD) ---
        team_df['puntos_acum'] = team_df['puntos'].cumsum().shift(1)
        team_df['gd_acum'] = (team_df['goles'] - team_df['goles_rival']).cumsum().shift(1)

        for _, row in team_df.iterrows():
            features[(row['fixture_id'], team)] = {
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
    return features

feat_map = compute_all_features(df)

# =====================================================================
# 3. CONSTRUIR DATASET DE PARTIDOS
# =====================================================================
df_home = df[df['side'] == 'home'][['fixture_id', 'fecha', 'ronda', 'equipo', 'goles']].copy()
df_home.columns = ['fixture_id', 'fecha', 'ronda', 'home', 'goles_home']

df_away = df[df['side'] == 'away'][['fixture_id', 'equipo', 'goles']].copy()
df_away.columns = ['fixture_id', 'away', 'goles_away']

matches = df_home.merge(df_away, on='fixture_id')
matches['total_goles'] = matches['goles_home'] + matches['goles_away']
matches['over25'] = (matches['total_goles'] > 2.5).astype(int)

# Agregar todas las features
all_feats = ['goles_rolling', 'goles_rec_rolling', 'posesion_rolling',
             'tiros_rolling', 'tiros_gol_rolling',
             'racha_puntos', 'racha_victorias', 'racha_derrotas',
             'goles_hist_local', 'goles_hist_visita',
             'puntos_acum', 'gd_acum']

for prefix, team_col in [('home', 'home'), ('away', 'away')]:
    for feat in all_feats:
        matches[f'{prefix}_{feat}'] = matches.apply(
            lambda row: feat_map.get((row['fixture_id'], row[team_col]), {}).get(feat), axis=1
        )

# Features derivadas: diferencias entre equipos
matches['diff_puntos_tabla'] = matches['home_puntos_acum'] - matches['away_puntos_acum']
matches['diff_gd'] = matches['home_gd_acum'] - matches['away_gd_acum']
matches['diff_racha'] = matches['home_racha_puntos'] - matches['away_racha_puntos']
matches['sum_goles_rolling'] = matches['home_goles_rolling'] + matches['away_goles_rolling']
matches['sum_goles_rec_rolling'] = matches['home_goles_rec_rolling'] + matches['away_goles_rec_rolling']

# --- Definición de features para los modelos ---
# v1: features originales (para comparar)
v1_features = [
    'home_goles_rolling', 'away_goles_rolling',
    'home_goles_rec_rolling', 'away_goles_rec_rolling',
    'home_posesion_rolling', 'away_posesion_rolling',
    'home_tiros_rolling', 'away_tiros_rolling',
    'home_tiros_gol_rolling', 'away_tiros_gol_rolling',
]

# v2: todas las features
v2_features = v1_features + [
    'home_racha_puntos', 'away_racha_puntos',
    'home_racha_victorias', 'away_racha_victorias',
    'home_racha_derrotas', 'away_racha_derrotas',
    'home_goles_hist_local', 'away_goles_hist_visita',
    'diff_puntos_tabla', 'diff_gd', 'diff_racha',
    'sum_goles_rolling', 'sum_goles_rec_rolling',
]

v2_labels = [
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

# Filtrar partidos con features completas
required = v2_features
matches_model = matches.dropna(subset=required).copy()

print(f"Partidos totales:     {len(matches)}")
print(f"Partidos usables:     {len(matches_model)}")
print(f"Features v1 (orig):   {len(v1_features)}")
print(f"Features v2 (nuevas): {len(v2_features)}")
print(f"\nDistribución target:")
print(f"  Over 2.5:  {matches_model['over25'].sum()} ({matches_model['over25'].mean()*100:.1f}%)")
print(f"  Under 2.5: {(~matches_model['over25'].astype(bool)).sum()} ({(1-matches_model['over25'].mean())*100:.1f}%)")

# Preview de features nuevas
print(f"\n--- Preview: Features nuevas (últimos 5 partidos) ---")
preview_cols = ['home', 'away', 'home_racha_puntos', 'away_racha_puntos',
                'home_goles_hist_local', 'away_goles_hist_visita',
                'diff_puntos_tabla', 'over25']
print(matches_model[preview_cols].tail(8).to_string(index=False))

# =====================================================================
# 4. COMPARACIÓN DE MODELOS
# =====================================================================
print("\n" + "=" * 70)
print("3. COMPARACIÓN DE MODELOS")
print("=" * 70)

X_v1 = matches_model[v1_features].values
X_v2 = matches_model[v2_features].values
y = matches_model['over25'].values

scaler_v1 = StandardScaler()
scaler_v2 = StandardScaler()
X_v1_s = scaler_v1.fit_transform(X_v1)
X_v2_s = scaler_v2.fit_transform(X_v2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
loo = LeaveOneOut()

models = {
    'LogReg v1 (10 feat)': (LogisticRegression(random_state=42, max_iter=1000, C=1.0), X_v1_s),
    'LogReg v2 (23 feat)': (LogisticRegression(random_state=42, max_iter=1000, C=1.0), X_v2_s),
    'LogReg v2 + L1 (C=0.5)': (LogisticRegression(random_state=42, max_iter=1000, C=0.5, penalty='l1', solver='liblinear'), X_v2_s),
    'LogReg v2 + L1 (C=0.1)': (LogisticRegression(random_state=42, max_iter=1000, C=0.1, penalty='l1', solver='liblinear'), X_v2_s),
    'RF (max_depth=3, n=100)': (RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, max_features='sqrt', random_state=42), X_v2_s),
    'RF (max_depth=2, n=200)': (RandomForestClassifier(n_estimators=200, max_depth=2, min_samples_leaf=6, max_features='sqrt', random_state=42), X_v2_s),
    'RF (max_depth=3, n=200, leaf=8)': (RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features='sqrt', random_state=42), X_v2_s),
}

results = []
print(f"\n{'Modelo':<35} {'CV Acc':>8} {'±':>5} {'CV AUC':>8} {'±':>5} {'LOO Acc':>8}")
print("-" * 75)

for name, (model, X_data) in models.items():
    cv_acc = cross_val_score(model, X_data, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(model, X_data, y, cv=cv, scoring='roc_auc')
    loo_acc = cross_val_score(model, X_data, y, cv=loo, scoring='accuracy')

    results.append({
        'name': name,
        'cv_acc_mean': cv_acc.mean(),
        'cv_acc_std': cv_acc.std(),
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'loo_acc': loo_acc.mean(),
    })
    print(f"{name:<35} {cv_acc.mean():>7.1%} {cv_acc.std():>5.1%} {cv_auc.mean():>8.3f} {cv_auc.std():>5.3f} {loo_acc.mean():>7.1%}")

baseline = y.mean()
print(f"\n{'Baseline (siempre Over)':<35} {baseline:>7.1%}")
print(f"{'Baseline (siempre Under)':<35} {1-baseline:>7.1%}")

df_results = pd.DataFrame(results)
best = df_results.loc[df_results['cv_acc_mean'].idxmax()]
print(f"\n★ Mejor modelo: {best['name']} (CV Acc: {best['cv_acc_mean']:.1%}, LOO: {best['loo_acc']:.1%})")

# =====================================================================
# 5. ANÁLISIS DEL MEJOR MODELO
# =====================================================================
print("\n" + "=" * 70)
print("4. ANÁLISIS DEL MEJOR MODELO")
print("=" * 70)

# Entrenar el mejor LogReg con L1 para interpretar coeficientes
best_lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1, penalty='l1', solver='liblinear')
best_lr.fit(X_v2_s, y)

coefs = pd.DataFrame({
    'feature': v2_labels,
    'coef': best_lr.coef_[0],
    'abs_coef': np.abs(best_lr.coef_[0]),
}).sort_values('abs_coef', ascending=False)

print("\n--- LogReg L1 (C=0.1): Features seleccionadas ---")
active = coefs[coefs['abs_coef'] > 0.001]
inactive = coefs[coefs['abs_coef'] <= 0.001]
print(f"Features activas: {len(active)}/{len(coefs)}")
print(f"Features eliminadas por L1: {len(inactive)}")
print(f"\n{'Feature':<30} {'Coef':>8} {'Efecto':>10}")
print("-" * 52)
for _, row in active.iterrows():
    efecto = "→ OVER" if row['coef'] > 0 else "→ UNDER"
    print(f"{row['feature']:<30} {row['coef']:>+8.3f} {efecto:>10}")
if len(inactive) > 0:
    print(f"\nEliminadas (coef ≈ 0): {', '.join(inactive['feature'].tolist())}")

# Entrenar mejor RF para feature importance
best_rf = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features='sqrt', random_state=42)
best_rf.fit(X_v2_s, y)

importances = pd.DataFrame({
    'feature': v2_labels,
    'importance': best_rf.feature_importances_,
}).sort_values('importance', ascending=False)

print(f"\n--- Random Forest: Feature Importance ---")
print(f"\n{'Feature':<30} {'Importance':>12}")
print("-" * 45)
for _, row in importances.head(15).iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"{row['feature']:<30} {row['importance']:>10.3f}  {bar}")

# =====================================================================
# 6. VISUALIZACIONES
# =====================================================================

# --- 6a. Comparación de modelos ---
fig, ax = plt.subplots(figsize=(14, 6))

df_r = df_results.sort_values('cv_acc_mean', ascending=True)
y_pos = range(len(df_r))

# LOO bars (background)
ax.barh(y_pos, df_r['loo_acc'].values, color='#2d3139', height=0.5, label='LOO Accuracy', alpha=0.7)
# CV bars (foreground)
bars = ax.barh(y_pos, df_r['cv_acc_mean'].values, color=ACCENT, height=0.5, label='CV 5-Fold Accuracy', alpha=0.85)
# Error bars
ax.errorbar(df_r['cv_acc_mean'].values, list(y_pos), xerr=df_r['cv_acc_std'].values,
            fmt='none', ecolor='#aaa', capsize=3, linewidth=1.5)

ax.axvline(x=max(baseline, 1 - baseline), color=ACCENT2, linestyle='--', alpha=0.6, linewidth=1.5)
ax.text(max(baseline, 1 - baseline) + 0.005, len(df_r) - 0.5, f'Baseline {max(baseline,1-baseline):.0%}',
        color=ACCENT2, fontsize=9, alpha=0.7)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(df_r['name'].values)
ax.set_xlabel('Accuracy')
ax.set_title('Comparación de Modelos — Over/Under 2.5 Goles', fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0.35, 0.85)

for i, (cv_acc, loo_acc, auc) in enumerate(zip(df_r['cv_acc_mean'], df_r['loo_acc'], df_r['cv_auc_mean'])):
    ax.text(max(cv_acc, loo_acc) + 0.01, i, f'AUC:{auc:.2f}', va='center', fontsize=9, color='#aaa')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v2_comparacion_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 6b. Feature importance: LogReg L1 vs Random Forest ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1]})

# Panel izquierdo: LogReg L1 coeficientes
coefs_plot = coefs[coefs['abs_coef'] > 0.001].sort_values('coef')
colors_lr = [ACCENT if c > 0 else ACCENT2 for c in coefs_plot['coef'].values]
y1 = range(len(coefs_plot))
ax1.barh(y1, coefs_plot['coef'].values, color=colors_lr, alpha=0.85, height=0.6)
ax1.set_yticks(list(y1))
ax1.set_yticklabels(coefs_plot['feature'].values, fontsize=10)
ax1.axvline(x=0, color='#555', linewidth=1)
ax1.set_xlabel('Coeficiente (estandarizado)')
ax1.set_title('Logistic Regression L1 (C=0.1)', fontweight='bold', pad=12)
ax1.grid(axis='x', alpha=0.3)
for i, c in enumerate(coefs_plot['coef'].values):
    ax1.text(c + (0.02 if c > 0 else -0.02), i, f'{c:+.2f}',
             va='center', ha='left' if c > 0 else 'right', fontsize=9, color='#ccc')

ax1.text(0.02, 0.98, '← UNDER', transform=ax1.transAxes, ha='left', va='top', fontsize=10, color=ACCENT2, alpha=0.7)
ax1.text(0.98, 0.98, 'OVER →', transform=ax1.transAxes, ha='right', va='top', fontsize=10, color=ACCENT, alpha=0.7)

# Panel derecho: RF importance
imp_plot = importances.head(15).sort_values('importance', ascending=True)
y2 = range(len(imp_plot))
ax2.barh(y2, imp_plot['importance'].values, color=ACCENT3, alpha=0.85, height=0.6)
ax2.set_yticks(list(y2))
ax2.set_yticklabels(imp_plot['feature'].values, fontsize=10)
ax2.set_xlabel('Feature Importance')
ax2.set_title('Random Forest (depth=3, leaf=8)', fontweight='bold', pad=12)
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(imp_plot['importance'].values):
    ax2.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9, color='#ccc')

fig.suptitle('Importancia de Variables — Modelo v2', fontweight='bold', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v2_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 6c. Curva ROC comparativa ---
fig, ax = plt.subplots(figsize=(8, 8))

model_roc = [
    ('LogReg v1', LogisticRegression(random_state=42, max_iter=1000), X_v1_s, '#636e72'),
    ('LogReg v2 + L1', LogisticRegression(random_state=42, max_iter=1000, C=0.1, penalty='l1', solver='liblinear'), X_v2_s, ACCENT),
    ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features='sqrt', random_state=42), X_v2_s, ORANGE),
]

for label, mdl, X_data, color in model_roc:
    y_prob_cv = cross_val_predict(mdl, X_data, y, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob_cv)
    auc_val = roc_auc_score(y, y_prob_cv)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{label} (AUC = {auc_val:.3f})')

ax.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC Comparativa (Cross-Validated)', fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v2_roc_comparativa.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 6d. Confusion matrices side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, (label, mdl, X_data) in zip([ax1, ax2], [
    ('LogReg v2 + L1 (C=0.1)', LogisticRegression(random_state=42, max_iter=1000, C=0.1, penalty='l1', solver='liblinear'), X_v2_s),
    ('Random Forest (depth=3)', RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features='sqrt', random_state=42), X_v2_s),
]):
    y_pred_cv = cross_val_predict(mdl, X_data, y, cv=loo)
    cm = confusion_matrix(y, y_pred_cv)
    im = ax.imshow(cm, cmap='YlGn', alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Under', 'Over'])
    ax.set_yticklabels(['Under', 'Over'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'{label}\nLOO Acc: {accuracy_score(y, y_pred_cv):.1%}', fontweight='bold', pad=10)
    for i in range(2):
        for j in range(2):
            text_color = '#1a1d23' if cm[i, j] > cm.max() / 2 else '#e0e0e0'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=22,
                    fontweight='bold', color=text_color)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v2_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# RESUMEN FINAL
# =====================================================================
print(f"\n{'='*70}")
print("RESUMEN COMPARATIVO v1 → v2")
print(f"{'='*70}")

v1_result = df_results[df_results['name'] == 'LogReg v1 (10 feat)'].iloc[0]
print(f"\n{'Métrica':<25} {'v1 (LogReg 10 feat)':>22} {'Mejor v2':>22}")
print("-" * 72)
print(f"{'Features':<25} {'10':>22} {best['name'].split('(')[0].strip():>22}")
print(f"{'CV Accuracy':<25} {v1_result['cv_acc_mean']:>21.1%} {best['cv_acc_mean']:>21.1%}")
print(f"{'CV AUC':<25} {v1_result['cv_auc_mean']:>22.3f} {best['cv_auc_mean']:>22.3f}")
print(f"{'LOO Accuracy':<25} {v1_result['loo_acc']:>21.1%} {best['loo_acc']:>21.1%}")
print(f"{'Baseline':<25} {max(baseline, 1-baseline):>21.1%} {max(baseline, 1-baseline):>21.1%}")

improvement_cv = best['cv_acc_mean'] - v1_result['cv_acc_mean']
improvement_loo = best['loo_acc'] - v1_result['loo_acc']
print(f"\n{'Mejora CV Accuracy:':<25} {improvement_cv:>+21.1%}")
print(f"{'Mejora LOO Accuracy:':<25} {improvement_loo:>+21.1%}")

print(f"\nArchivos guardados:")
for f in ['v2_comparacion_modelos.png', 'v2_feature_importance.png',
          'v2_roc_comparativa.png', 'v2_confusion_matrices.png']:
    print(f"  → data/processed/{f}")

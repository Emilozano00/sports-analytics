"""
Modelo predictivo: Over/Under 2.5 goles - Apertura 2024 Liga MX
Regresión logística con features de promedios rolling (últimos 5 partidos).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
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
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# 1. CARGAR Y PREPARAR DATOS
# =====================================================================
print("=" * 65)
print("1. PREPARACIÓN DE DATOS")
print("=" * 65)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'apertura2024_team_stats.csv'))

# Limpiar posesión
df['posesion'] = df['ball_possession'].str.replace('%', '').astype(float)

# Ordenar por fecha
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values(['fecha', 'fixture_id']).reset_index(drop=True)

# =====================================================================
# 2. CONSTRUIR FEATURES ROLLING POR EQUIPO
# =====================================================================
print("\n" + "=" * 65)
print("2. FEATURES ROLLING (ÚLTIMOS 5 PARTIDOS POR EQUIPO)")
print("=" * 65)

ROLLING_N = 5

# Para cada equipo, calcular promedios rolling de sus últimos N partidos
# (sin importar si fue local o visitante)
def compute_rolling_features(df, team_col='equipo', n=ROLLING_N):
    """Calcula promedios rolling por equipo antes de cada partido."""
    features = {}

    for team in df[team_col].unique():
        team_df = df[df[team_col] == team].sort_values('fecha').copy()
        team_df['goles_rolling'] = team_df['goles'].rolling(n, min_periods=n).mean().shift(1)
        team_df['goles_recibidos_rolling'] = team_df['goles_rival'].rolling(n, min_periods=n).mean().shift(1)
        team_df['posesion_rolling'] = team_df['posesion'].rolling(n, min_periods=n).mean().shift(1)
        team_df['tiros_rolling'] = team_df['total_shots'].rolling(n, min_periods=n).mean().shift(1)
        team_df['tiros_gol_rolling'] = team_df['shots_on_goal'].rolling(n, min_periods=n).mean().shift(1)
        team_df['pases_rolling'] = team_df['total_passes'].rolling(n, min_periods=n).mean().shift(1)

        for _, row in team_df.iterrows():
            features[(row['fixture_id'], team)] = {
                'goles_rolling': row['goles_rolling'],
                'goles_recibidos_rolling': row['goles_recibidos_rolling'],
                'posesion_rolling': row['posesion_rolling'],
                'tiros_rolling': row['tiros_rolling'],
                'tiros_gol_rolling': row['tiros_gol_rolling'],
                'pases_rolling': row['pases_rolling'],
            }
    return features

rolling = compute_rolling_features(df)

# =====================================================================
# 3. CONSTRUIR DATASET DE PARTIDOS CON FEATURES
# =====================================================================
df_home = df[df['side'] == 'home'][['fixture_id', 'fecha', 'ronda', 'equipo', 'goles']].copy()
df_home.columns = ['fixture_id', 'fecha', 'ronda', 'home', 'goles_home']

df_away = df[df['side'] == 'away'][['fixture_id', 'equipo', 'goles']].copy()
df_away.columns = ['fixture_id', 'away', 'goles_away']

matches = df_home.merge(df_away, on='fixture_id')
matches['total_goles'] = matches['goles_home'] + matches['goles_away']
matches['over25'] = (matches['total_goles'] > 2.5).astype(int)

# Agregar features rolling
for prefix, team_col in [('home', 'home'), ('away', 'away')]:
    for feat in ['goles_rolling', 'goles_recibidos_rolling', 'posesion_rolling',
                 'tiros_rolling', 'tiros_gol_rolling', 'pases_rolling']:
        matches[f'{prefix}_{feat}'] = matches.apply(
            lambda row: rolling.get((row['fixture_id'], row[team_col]), {}).get(feat), axis=1
        )

# Eliminar partidos sin suficiente historial
matches_model = matches.dropna(subset=[
    'home_goles_rolling', 'away_goles_rolling',
    'home_posesion_rolling', 'away_posesion_rolling',
    'home_tiros_rolling', 'away_tiros_rolling',
]).copy()

print(f"Partidos totales: {len(matches)}")
print(f"Partidos con features completas (≥{ROLLING_N} previos por equipo): {len(matches_model)}")
print(f"Partidos descartados (historial insuficiente): {len(matches) - len(matches_model)}")
print(f"\nDistribución target (partidos usables):")
print(f"  Over 2.5:  {matches_model['over25'].sum()} ({matches_model['over25'].mean()*100:.1f}%)")
print(f"  Under 2.5: {(1 - matches_model['over25']).sum()} ({(1-matches_model['over25'].mean())*100:.1f}%)")

# =====================================================================
# 4. DEFINIR FEATURES Y ENTRENAR MODELO
# =====================================================================
print("\n" + "=" * 65)
print("3. ENTRENAMIENTO DEL MODELO")
print("=" * 65)

feature_names = [
    'home_goles_rolling',
    'away_goles_rolling',
    'home_goles_recibidos_rolling',
    'away_goles_recibidos_rolling',
    'home_posesion_rolling',
    'away_posesion_rolling',
    'home_tiros_rolling',
    'away_tiros_rolling',
    'home_tiros_gol_rolling',
    'away_tiros_gol_rolling',
]

feature_labels = [
    'Goles local (avg 5)',
    'Goles visitante (avg 5)',
    'Goles recibidos local (avg 5)',
    'Goles recibidos visit. (avg 5)',
    'Posesión local (avg 5)',
    'Posesión visitante (avg 5)',
    'Tiros local (avg 5)',
    'Tiros visitante (avg 5)',
    'Tiros a gol local (avg 5)',
    'Tiros a gol visit. (avg 5)',
]

X = matches_model[feature_names].values
y = matches_model['over25'].values

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo
model = LogisticRegression(random_state=42, max_iter=1000)

# --- Cross-validation (StratifiedKFold) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
cv_auc = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')

print(f"\n--- Cross-Validation (5-Fold Stratified) ---")
print(f"Accuracy por fold: {[f'{s:.3f}' for s in cv_scores]}")
print(f"Accuracy promedio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"AUC promedio:      {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# --- Leave-One-Out (más robusto con datasets pequeños) ---
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X_scaled, y, cv=loo, scoring='accuracy')
print(f"\n--- Leave-One-Out Cross-Validation ---")
print(f"Accuracy LOO: {loo_scores.mean():.3f} ({loo_scores.sum():.0f}/{len(loo_scores)} correctos)")

# --- Entrenar modelo final en todos los datos ---
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

print(f"\n--- Modelo entrenado en todos los datos ---")
print(f"Accuracy (in-sample): {accuracy_score(y, y_pred):.3f}")
print(f"AUC (in-sample):      {roc_auc_score(y, y_prob):.3f}")

print(f"\n--- Classification Report ---")
print(classification_report(y, y_pred, target_names=['Under 2.5', 'Over 2.5']))

# =====================================================================
# 5. IMPORTANCIA DE VARIABLES (COEFICIENTES)
# =====================================================================
print("=" * 65)
print("4. IMPORTANCIA DE VARIABLES")
print("=" * 65)

coefs = pd.DataFrame({
    'feature': feature_labels,
    'coef': model.coef_[0],
    'abs_coef': np.abs(model.coef_[0]),
    'odds_ratio': np.exp(model.coef_[0]),
}).sort_values('abs_coef', ascending=False)

print("\nCoeficientes (ordenados por importancia absoluta):")
print(f"{'Variable':<35} {'Coef':>8} {'Odds Ratio':>12} {'Efecto':>10}")
print("-" * 70)
for _, row in coefs.iterrows():
    efecto = "→ OVER" if row['coef'] > 0 else "→ UNDER"
    print(f"{row['feature']:<35} {row['coef']:>+8.3f} {row['odds_ratio']:>12.3f} {efecto:>10}")

# =====================================================================
# 6. VISUALIZACIONES
# =====================================================================

# --- 6a. Coeficientes del modelo ---
fig, ax = plt.subplots(figsize=(12, 6))
coefs_sorted = coefs.sort_values('coef')
colors = [ACCENT if c > 0 else ACCENT2 for c in coefs_sorted['coef'].values]
y_pos = range(len(coefs_sorted))

ax.barh(y_pos, coefs_sorted['coef'].values, color=colors, alpha=0.85, height=0.65)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(coefs_sorted['feature'].values)
ax.axvline(x=0, color='#555', linewidth=1)
ax.set_xlabel('Coeficiente (estandarizado)')
ax.set_title('Importancia de Variables — Modelo Over/Under 2.5 Goles', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Anotaciones
for i, (coef, feat) in enumerate(zip(coefs_sorted['coef'].values, coefs_sorted['feature'].values)):
    side = 'OVER' if coef > 0 else 'UNDER'
    ax.text(coef + (0.03 if coef > 0 else -0.03), i,
            f'{coef:+.2f}', va='center', ha='left' if coef > 0 else 'right',
            fontsize=9, color='#ccc')

ax.text(0.02, 0.98, '← Predice UNDER', transform=ax.transAxes,
        ha='left', va='top', fontsize=10, color=ACCENT2, alpha=0.7)
ax.text(0.98, 0.98, 'Predice OVER →', transform=ax.transAxes,
        ha='right', va='top', fontsize=10, color=ACCENT, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'modelo_coeficientes.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 6b. Curva ROC ---
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f'Logistic Regression (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1, label='Random (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC — Over/Under 2.5 Goles', fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.3)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'modelo_roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 6c. Confusion Matrix ---
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap='YlGn', alpha=0.8)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Under 2.5', 'Over 2.5'])
ax.set_yticklabels(['Under 2.5', 'Over 2.5'])
ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusión', fontweight='bold', pad=15)

for i in range(2):
    for j in range(2):
        text_color = '#1a1d23' if cm[i, j] > cm.max() / 2 else '#e0e0e0'
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=20,
                fontweight='bold', color=text_color)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'modelo_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 6d. Vista de predicciones por partido ---
print("\n" + "=" * 65)
print("5. MUESTRA DE PREDICCIONES")
print("=" * 65)

matches_model = matches_model.copy()
matches_model['pred_prob_over'] = y_prob
matches_model['pred_class'] = y_pred
matches_model['correcto'] = (y_pred == y)

sample = matches_model[['fecha', 'home', 'away', 'goles_home', 'goles_away',
                         'total_goles', 'over25', 'pred_prob_over', 'pred_class', 'correcto']].copy()
sample['fecha'] = sample['fecha'].dt.strftime('%Y-%m-%d')
sample = sample.rename(columns={
    'pred_prob_over': 'P(Over)',
    'pred_class': 'pred',
    'over25': 'real',
})
sample['P(Over)'] = sample['P(Over)'].round(3)
print(sample.sort_values('fecha').to_string(index=False))

# Resumen final
print(f"\n{'='*65}")
print(f"RESUMEN DEL MODELO")
print(f"{'='*65}")
print(f"Target:              Over/Under 2.5 goles")
print(f"Partidos usados:     {len(matches_model)}")
print(f"Features:            {len(feature_names)} (rolling avg últimos 5 partidos)")
print(f"Accuracy (CV 5-fold): {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"Accuracy (LOO):       {loo_scores.mean():.1%}")
print(f"AUC (CV 5-fold):      {cv_auc.mean():.3f}")
print(f"Baseline (siempre Over): {y.mean():.1%}")
print(f"\nArchivos guardados en data/processed/:")
for f in ['modelo_coeficientes.png', 'modelo_roc_curve.png', 'modelo_confusion_matrix.png']:
    print(f"  → {f}")

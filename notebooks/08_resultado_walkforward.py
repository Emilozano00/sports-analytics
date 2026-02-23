"""
Modelo v6: Predicción de resultado (Local / Empate / Visitante)
Walk-forward validation: Apertura 2024 + K jornadas Clausura → test restantes.

Uso:
    python notebooks/08_resultado_walkforward.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

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

LABEL_MAP = {0: 'Local', 1: 'Empate', 2: 'Visitante'}
LABEL_COLORS = {0: ACCENT, 1: ORANGE, 2: ACCENT2}


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

df_all = pd.concat([df_apertura, df_clausura], ignore_index=True)

print(f"  Apertura 2024 (anterior): {df_apertura['fixture_id'].nunique()} partidos")
print(f"  Clausura 2024 (actual):   {df_clausura['fixture_id'].nunique()} partidos")

# Clausura rounds in chronological order
df_clausura_tmp = df_clausura.copy()
df_clausura_tmp['fecha'] = pd.to_datetime(df_clausura_tmp['fecha'])
round_dates = df_clausura_tmp.groupby('ronda')['fecha'].min().sort_values()
clausura_rounds = round_dates.index.tolist()


# =====================================================================
# 2. FEATURE ENGINEERING (mismas features v3)
# =====================================================================
def build_features(df):
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

    # Build match-level rows
    df_h = df[df['side'] == 'home'][['fixture_id', 'fecha', 'equipo', 'goles', 'torneo', 'ronda', 'resultado']].copy()
    df_h.columns = ['fixture_id', 'fecha', 'home', 'goles_home', 'torneo', 'ronda', 'resultado_home']
    df_a = df[df['side'] == 'away'][['fixture_id', 'equipo', 'goles']].copy()
    df_a.columns = ['fixture_id', 'away', 'goles_away']
    matches = df_h.merge(df_a, on='fixture_id')

    # Target: 0=Local, 1=Empate, 2=Visitante
    matches['resultado'] = matches['resultado_home'].map({'W': 0, 'D': 1, 'L': 2})

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


def get_models():
    return {
        'LogReg': LogisticRegression(random_state=42, max_iter=1000),
        'LogReg L1': LogisticRegression(random_state=42, max_iter=1000, C=0.5,
                                        penalty='l1', solver='saga'),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=5,
                                     max_features='sqrt', random_state=42),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5,
                                          learning_rate=0.1, random_state=42),
    }


# =====================================================================
# 3. BUILD FEATURES
# =====================================================================
print("\n" + "=" * 70)
print("2. CONSTRUYENDO FEATURES")
print("=" * 70)

matches = build_features(df_all)
matches_clean = matches.dropna(subset=FEATURES).copy()
matches_clean['fecha'] = pd.to_datetime(matches_clean['fecha'])

mask_apertura = matches_clean['torneo'] == 'Apertura'
mask_clausura = matches_clean['torneo'] == 'Clausura'

round_order = {r: i for i, r in enumerate(clausura_rounds)}
matches_clean.loc[mask_clausura, 'round_idx'] = matches_clean.loc[mask_clausura, 'ronda'].map(round_order)

apertura_matches = matches_clean[mask_apertura]
clausura_matches = matches_clean[mask_clausura].sort_values('round_idx')

n_aper = len(apertura_matches)
n_clau = len(clausura_matches)

print(f"  Apertura (train base): {n_aper} partidos")
print(f"  Clausura (simulado):   {n_clau} partidos")

# Distribution
for label, df_sub in [('Apertura', apertura_matches), ('Clausura', clausura_matches), ('Combinado', matches_clean)]:
    y = df_sub['resultado']
    total = len(y)
    counts = y.value_counts().sort_index()
    print(f"  {label:>10}: Local={counts.get(0,0)} ({counts.get(0,0)/total:.0%})  "
          f"Empate={counts.get(1,0)} ({counts.get(1,0)/total:.0%})  "
          f"Visita={counts.get(2,0)} ({counts.get(2,0)/total:.0%})")


# =====================================================================
# 4. CV INTERNO POR TORNEO
# =====================================================================
print("\n" + "=" * 70)
print("3. CV INTERNO POR TORNEO (5-fold)")
print("=" * 70)

cv_results = []

for label, df_sub in [('Apertura', apertura_matches), ('Clausura', clausura_matches),
                       ('Combinado', matches_clean)]:
    X = df_sub[FEATURES].values
    y = df_sub['resultado'].values
    n = len(y)
    baseline = pd.Series(y).value_counts(normalize=True).max()

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  --- {label} (n={n}, baseline={baseline:.1%}) ---")

    for model_name, model in get_models().items():
        acc = cross_val_score(model, X_s, y, cv=cv, scoring='accuracy')
        f1 = cross_val_score(model, X_s, y, cv=cv, scoring='f1_macro')

        cv_results.append({
            'dataset': label, 'model': model_name,
            'cv_acc': acc.mean(), 'cv_std': acc.std(),
            'cv_f1': f1.mean(), 'baseline': baseline, 'n': n,
        })

        lift = acc.mean() - baseline
        marker = "✓" if lift > 0 else "✗"
        print(f"  {marker} {model_name:<15} Acc:{acc.mean():.1%}±{acc.std():.1%}  F1-macro:{f1.mean():.3f}  Lift:{lift:+.1%}")

df_cv = pd.DataFrame(cv_results)


# =====================================================================
# 5. WALK-FORWARD VALIDATION
# =====================================================================
print("\n" + "=" * 70)
print("4. WALK-FORWARD: Apertura + K jornadas → Test restantes")
print("=" * 70)

max_train_rounds = len(clausura_rounds) - 2

wf_results = []

header_models = '  '.join(f'{m:>10}' for m in get_models())
print(f"\n  {'Config':>15} | {'Train':>5} | {'Test':>4} | {header_models} | {'Base':>5}")
print(f"  {'-'*105}")

for k in range(0, max_train_rounds + 1):
    if k == 0:
        train_df = apertura_matches
    else:
        clau_train = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[:k])]
        train_df = pd.concat([apertura_matches, clau_train])

    test_df = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[k:])]
    if len(test_df) < 5:
        continue

    X_train = train_df[FEATURES].values
    y_train = train_df['resultado'].values
    X_test = test_df[FEATURES].values
    y_test = test_df['resultado'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline = pd.Series(y_test).value_counts(normalize=True).max()
    majority_class = pd.Series(y_test).value_counts().idxmax()

    label = f"Aper+J1-J{k}" if k > 0 else "Solo Aper"
    print(f"  {label:>15} | {len(y_train):>5} | {len(y_test):>4} | ", end='')

    for model_name, model in get_models().items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        wf_results.append({
            'k': k, 'label': label, 'model': model_name,
            'train_n': len(y_train), 'test_n': len(y_test),
            'accuracy': acc, 'f1_macro': f1,
            'baseline': baseline, 'lift': acc - baseline,
            'majority_class': majority_class,
            'y_test': y_test, 'y_pred': y_pred,
        })

        marker = '+' if acc > baseline else '-'
        print(f'{acc:>9.1%}{marker}', end='')

    print(f" | {baseline:>4.1%}")

df_wf = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred']} for r in wf_results])


# =====================================================================
# 6. ANÁLISIS: MEJOR POR K
# =====================================================================
print("\n" + "=" * 70)
print("5. MEJOR MODELO POR K JORNADAS")
print("=" * 70)

print(f"\n  {'K':>3} {'Config':>15} {'Modelo':>12} {'Acc':>6} {'F1':>6} {'Base':>6} {'Lift':>7}")
print(f"  {'-'*65}")

best_per_k = []
for k in sorted(df_wf['k'].unique()):
    k_data = df_wf[df_wf['k'] == k]
    best = k_data.sort_values('accuracy', ascending=False).iloc[0]
    best_per_k.append(best)
    marker = "✓" if best['lift'] > 0 else "✗"
    print(f"  {int(k):>3} {best['label']:>15} {best['model']:>12} "
          f"{best['accuracy']:>5.1%} {best['f1_macro']:>5.3f} "
          f"{best['baseline']:>5.1%} {best['lift']:>+6.1%} {marker}")

df_best_k = pd.DataFrame(best_per_k)


# =====================================================================
# 7. DETALLE: MEJOR CONFIGURACIÓN
# =====================================================================
print("\n" + "=" * 70)
print("6. DETALLE DE LA MEJOR CONFIGURACIÓN")
print("=" * 70)

best_overall = max(wf_results, key=lambda x: x['accuracy'])
best_k = best_overall['k']
best_model_name = best_overall['model']

print(f"\n  Mejor: K={best_k} ({best_overall['label']}) con {best_model_name}")
print(f"  Accuracy: {best_overall['accuracy']:.1%} | F1-macro: {best_overall['f1_macro']:.3f} | "
      f"Baseline: {best_overall['baseline']:.1%} | Lift: {best_overall['lift']:+.1%}")

# Re-train for classification report
if best_k == 0:
    train_det = apertura_matches
else:
    clau_train_det = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[:best_k])]
    train_det = pd.concat([apertura_matches, clau_train_det])

test_det = clausura_matches[clausura_matches['ronda'].isin(clausura_rounds[best_k:])]

X_tr = train_det[FEATURES].values
y_tr = train_det['resultado'].values
X_te = test_det[FEATURES].values
y_te = test_det['resultado'].values

scaler_det = StandardScaler()
X_tr_s = scaler_det.fit_transform(X_tr)
X_te_s = scaler_det.transform(X_te)

model_det = get_models()[best_model_name]
model_det.fit(X_tr_s, y_tr)
y_pred_det = model_det.predict(X_te_s)

print(f"\n  Classification Report:")
report = classification_report(y_te, y_pred_det, target_names=['Local', 'Empate', 'Visitante'])
for line in report.split('\n'):
    print(f"  {line}")

cm = confusion_matrix(y_te, y_pred_det)
print(f"  Confusion Matrix:")
print(f"  {'':>12} Pred:Local  Pred:Empate  Pred:Visita")
for i, label in enumerate(['Real:Local', 'Real:Empate', 'Real:Visita']):
    print(f"  {label:>12}  {cm[i,0]:>8}  {cm[i,1]:>10}  {cm[i,2]:>10}")

# Feature importance (GBM for multiclass)
gbm_det = GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5,
                                      learning_rate=0.1, random_state=42)
gbm_det.fit(X_tr_s, y_tr)

importances = pd.DataFrame({
    'feature': FEATURE_LABELS,
    'importance': gbm_det.feature_importances_,
}).sort_values('importance', ascending=False)

print(f"\n  Feature Importance (GBM):")
print(f"  {'Feature':<30} {'Imp':>8}")
print(f"  {'-'*42}")
for _, row in importances.head(10).iterrows():
    bar = '█' * int(row['importance'] * 60)
    print(f"  {row['feature']:<30} {row['importance']:>7.3f}  {bar}")


# =====================================================================
# 8. PREDICCIONES PARTIDO A PARTIDO
# =====================================================================
print("\n" + "=" * 70)
print("7. PREDICCIONES PARTIDO A PARTIDO")
print("=" * 70)

test_detail = test_det.copy()
test_detail['pred'] = y_pred_det
test_detail['correct'] = (test_detail['pred'] == test_detail['resultado']).astype(int)

result_labels = {0: 'Local', 1: 'Empate', 2: 'Visita'}

print(f"\n  {'Fecha':<11} {'Local':<17} {'Visita':<17} {'Score':>5} {'Real':>8} {'Pred':>8} {'':>2}")
print(f"  {'-'*80}")

for _, row in test_detail.sort_values('fecha').iterrows():
    real = result_labels[row['resultado']]
    pred = result_labels[row['pred']]
    ok = '✓' if row['correct'] else '✗'
    score = f"{int(row['goles_home'])}-{int(row['goles_away'])}"
    print(f"  {str(row['fecha'])[:10]:<11} {row['home']:<17} {row['away']:<17} {score:>5} {real:>8} {pred:>8} {ok:>2}")

# Accuracy by round
print(f"\n  Accuracy por jornada:")
for r in clausura_rounds[best_k:]:
    r_data = test_detail[test_detail['ronda'] == r]
    if len(r_data) > 0:
        r_acc = r_data['correct'].mean()
        n_correct = int(r_data['correct'].sum())
        print(f"    {r:<30} {r_acc:.0%} ({n_correct}/{len(r_data)})")

total_acc = test_detail['correct'].mean()
print(f"    {'TOTAL':<30} {total_acc:.0%} ({int(test_detail['correct'].sum())}/{len(test_detail)})")


# =====================================================================
# 9. COMPARACIÓN VS OVER/UNDER (contexto)
# =====================================================================
print("\n" + "=" * 70)
print("8. CONTEXTO: 3 clases vs Over/Under")
print("=" * 70)

# Random baseline for 3 classes = 33.3%
print(f"\n  Baseline aleatoria (3 clases): 33.3%")
print(f"  Baseline mayoría (test set):   {best_overall['baseline']:.1%}")
print(f"  Nuestro mejor modelo:          {best_overall['accuracy']:.1%}")
print(f"  Lift vs aleatoria:            {best_overall['accuracy'] - 1/3:+.1%}")
print(f"  Lift vs mayoría:              {best_overall['lift']:+.1%}")

# How does each class do?
for cls in [0, 1, 2]:
    mask_cls = y_te == cls
    if mask_cls.sum() > 0:
        cls_acc = (y_pred_det[mask_cls] == cls).mean()
        print(f"  {result_labels[cls]:>8}: recall={cls_acc:.0%} ({(y_pred_det[mask_cls] == cls).sum()}/{mask_cls.sum()} detectados)")


# =====================================================================
# 10. VISUALIZACIONES
# =====================================================================
print("\n" + "=" * 70)
print("9. GENERANDO VISUALIZACIONES")
print("=" * 70)

# --- 10a. Walk-forward heatmap ---
fig, ax = plt.subplots(figsize=(14, 6))

pivot = df_wf.pivot_table(index='model', columns='k', values='accuracy')
model_order = df_wf.groupby('model')['accuracy'].max().sort_values(ascending=True).index
pivot = pivot.loc[model_order]

im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.25, vmax=0.60)
ax.set_xticks(range(pivot.shape[1]))
x_labels = ['J0\n(solo Aper)'] + [f'J1-J{k}' for k in range(1, pivot.shape[1])]
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)
ax.set_xlabel('Jornadas del Clausura en training')

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = '#1a1d23' if val > 0.45 else '#e0e0e0'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

ax.set_title('Walk-Forward: Accuracy predicción de resultado (L/E/V)',
             fontweight='bold', pad=15)
fig.colorbar(im, ax=ax, label='Accuracy', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v6_resultado_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v6_resultado_heatmap.png")

# --- 10b. Accuracy curves by K ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors_model = {'LogReg': '#636e72', 'LogReg L1': ACCENT, 'RF': ACCENT3, 'GBM': ORANGE}

for model_name in get_models():
    m_data = df_wf[df_wf['model'] == model_name].sort_values('k')
    ax1.plot(m_data['k'], m_data['accuracy'], 'o-', color=colors_model[model_name],
             linewidth=2, markersize=6, label=model_name)
    ax1.plot(m_data['k'], m_data['f1_macro'], 's--', color=colors_model[model_name],
             linewidth=1, markersize=4, alpha=0.5)

baselines = df_wf.groupby('k')['baseline'].first()
ax1.plot(baselines.index, baselines.values, '--', color=ACCENT2, linewidth=1.5, alpha=0.7, label='Baseline (mayoría)')
ax1.axhline(y=1/3, color='#555', linestyle=':', alpha=0.4, label='Random (33%)')

ax1.set_xlabel('Jornadas del Clausura en training')
ax1.set_ylabel('Score')
ax1.set_title('Accuracy (línea) y F1-macro (punteada)', fontweight='bold')
ax1.legend(framealpha=0.3, fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylim(0.20, 0.65)

# Lift over baseline
for model_name in get_models():
    m_data = df_wf[df_wf['model'] == model_name].sort_values('k')
    ax2.plot(m_data['k'], m_data['lift'], 'o-', color=colors_model[model_name],
             linewidth=2, markersize=6, label=model_name)

ax2.axhline(y=0, color=ACCENT2, linestyle='--', linewidth=1.5, alpha=0.7)
ax2.fill_between(range(max_train_rounds + 1), 0, -0.3, alpha=0.05, color=ACCENT2)
ax2.fill_between(range(max_train_rounds + 1), 0, 0.3, alpha=0.05, color=ACCENT)
ax2.set_xlabel('Jornadas del Clausura en training')
ax2.set_ylabel('Lift vs Baseline')
ax2.set_title('Lift sobre baseline (mayoría)', fontweight='bold')
ax2.legend(framealpha=0.3, fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.25, 0.20)

plt.suptitle('Walk-Forward: Predicción de Resultado (Local / Empate / Visitante)',
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v6_resultado_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v6_resultado_curves.png")

# --- 10c. Confusion matrix for best config ---
fig, ax = plt.subplots(figsize=(7, 6))

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap='YlGn', vmin=0, vmax=0.7)
labels_cm = ['Local', 'Empate', 'Visitante']
ax.set_xticks(range(3))
ax.set_xticklabels([f'Pred: {l}' for l in labels_cm], fontsize=10)
ax.set_yticks(range(3))
ax.set_yticklabels([f'Real: {l}' for l in labels_cm], fontsize=10)

for i in range(3):
    for j in range(3):
        color = '#1a1d23' if cm_norm[i, j] > 0.4 else '#e0e0e0'
        ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

ax.set_title(f'Confusion Matrix — {best_model_name} (K={best_k})\n'
             f'Acc: {best_overall["accuracy"]:.1%}',
             fontweight='bold', pad=15)
fig.colorbar(im, ax=ax, label='Proporción', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v6_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v6_confusion_matrix.png")

# --- 10d. Feature importance ---
fig, ax = plt.subplots(figsize=(12, 8))
imp_plot = importances.head(15).sort_values('importance', ascending=True)
y_pos = range(len(imp_plot))
colors_fi = [ACCENT if i >= len(imp_plot) - 5 else ACCENT3 for i in range(len(imp_plot))]
ax.barh(y_pos, imp_plot['importance'].values, color=colors_fi, alpha=0.85, height=0.6)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(imp_plot['feature'].values, fontsize=10)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top 15 Features — GBM para Resultado (L/E/V)', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(imp_plot['importance'].values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9, color='#ccc')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v6_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v6_feature_importance.png")

# --- 10e. Match predictions bar chart ---
test_sorted = test_detail.sort_values('fecha')
fig, ax = plt.subplots(figsize=(16, 5))

x = range(len(test_sorted))
colors_bar = [ACCENT if c else ACCENT2 for c in test_sorted['correct']]
ax.bar(x, [1]*len(test_sorted), color=colors_bar, alpha=0.7, width=0.8)

# Add result labels
for i, (_, row) in enumerate(test_sorted.iterrows()):
    real_lbl = result_labels[row['resultado']][0]  # L, E, V
    pred_lbl = result_labels[row['pred']][0]
    ax.text(i, 0.7, real_lbl, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax.text(i, 0.3, pred_lbl, ha='center', va='center', fontsize=8, color='#aaa')

ax.set_xticks(list(x))
labels_x = [f"{str(r['fecha'])[:5]}\n{r['home'][:3]}v{r['away'][:3]}" for _, r in test_sorted.iterrows()]
ax.set_xticklabels(labels_x, fontsize=5, rotation=90)
ax.set_ylabel('')
ax.set_yticks([0.3, 0.7])
ax.set_yticklabels(['Predicción', 'Real'], fontsize=9)
ax.set_title(f'Predicciones partido a partido — {best_model_name} (K={best_k})\n'
             f'Verde=acierto, Rojo=error | L=Local E=Empate V=Visitante',
             fontweight='bold', pad=15)
ax.set_ylim(0, 1.05)

acc_text = f'Accuracy: {best_overall["accuracy"]:.1%} ({int(test_detail["correct"].sum())}/{len(test_detail)})'
ax.text(0.98, 0.95, acc_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=11, fontweight='bold', color=ACCENT,
        bbox=dict(boxstyle='round', facecolor='#1a1d23', edgecolor=ACCENT, alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v6_match_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → v6_match_predictions.png")


# =====================================================================
# RESUMEN FINAL
# =====================================================================
print(f"\n{'='*70}")
print("RESUMEN FINAL — Predicción de Resultado (L/E/V)")
print(f"{'='*70}")

print(f"\n  Target: 3 clases (Local / Empate / Visitante)")
print(f"  Baseline aleatoria: 33.3%")

print(f"\n  A) CV interno por torneo:")
for ds in ['Apertura', 'Clausura', 'Combinado']:
    ds_data = df_cv[df_cv['dataset'] == ds]
    best_ds = ds_data.sort_values('cv_acc', ascending=False).iloc[0]
    lift = best_ds['cv_acc'] - best_ds['baseline']
    marker = "✓" if lift > 0 else "✗"
    print(f"     {ds:>10}: {best_ds['model']:>12} → {best_ds['cv_acc']:.1%} "
          f"(F1:{best_ds['cv_f1']:.3f}, base:{best_ds['baseline']:.1%}, lift:{lift:+.1%}) {marker}")

print(f"\n  B) Walk-Forward (Apertura + K jornadas Clausura):")
beats = df_best_k[df_best_k['lift'] > 0]
if len(beats) > 0:
    first = beats.iloc[0]
    print(f"     Supera baseline desde K={int(first['k'])} jornadas")
    print(f"     {first['model']}: {first['accuracy']:.1%} vs {first['baseline']:.1%}")
else:
    print(f"     No supera baseline de mayoría en ningún K")

best_wf = df_best_k.sort_values('lift', ascending=False).iloc[0]
print(f"\n     Mejor config: K={int(best_wf['k'])} → {best_wf['model']}: "
      f"{best_wf['accuracy']:.1%} (base:{best_wf['baseline']:.1%}, lift:{best_wf['lift']:+.1%})")
print(f"     Lift vs random (33%): {best_wf['accuracy'] - 1/3:+.1%}")

print(f"\n  Visualizaciones:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('v6_'):
        print(f"    → data/processed/{f}")

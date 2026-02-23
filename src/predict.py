"""
Predicción de resultado Liga MX — CLI
Basado en modelo v6 (Random Forest, walk-forward validated).

Uso:
    python src/predict.py "Cruz Azul" "Club America"
    python src/predict.py "america" "tigres"        # fuzzy match
    python src/predict.py --teams                    # lista equipos
    python src/predict.py --all                      # todos vs todos (muestra)
"""

import sys
import os
import argparse
import warnings
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

ROLLING_N = 5
STREAK_N = 3

FEATURES = [
    "home_goles_rolling",
    "away_goles_rolling",
    "home_goles_rec_rolling",
    "away_goles_rec_rolling",
    "home_posesion_rolling",
    "away_posesion_rolling",
    "home_tiros_rolling",
    "away_tiros_rolling",
    "home_tiros_gol_rolling",
    "away_tiros_gol_rolling",
    "home_racha_puntos",
    "away_racha_puntos",
    "home_racha_victorias",
    "away_racha_victorias",
    "home_racha_derrotas",
    "away_racha_derrotas",
    "home_goles_hist_local",
    "away_goles_hist_visita",
    "diff_puntos_tabla",
    "diff_gd",
    "diff_racha",
    "sum_goles_rolling",
    "sum_goles_rec_rolling",
]

ODDS_FEATURES = ["odds_prob_home", "odds_prob_draw", "odds_prob_away"]
FEATURES_WITH_ODDS = FEATURES + ODDS_FEATURES  # 26 features


# ── Data loading ─────────────────────────────────────────────────────
def load_odds_data():
    """Glob odds_*.csv en data/raw/, retorna DataFrame combinado o None."""
    import glob
    pattern = os.path.join(RAW_DIR, "odds_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_data():
    import glob
    pattern = os.path.join(RAW_DIR, "*_team_stats.csv")
    files = [f for f in sorted(glob.glob(pattern))
             if not os.path.basename(f).startswith("ligamx")]
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos *_team_stats.csv en {RAW_DIR}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        # Inferir torneo del nombre si falta la columna
        basename = os.path.basename(fp).lower()
        if "torneo" not in df.columns:
            if "apertura" in basename:
                df["torneo"] = "Apertura"
            elif "clausura" in basename:
                df["torneo"] = "Clausura"
        dfs.append(df)
        print(f"  Cargado: {os.path.basename(fp)} ({len(df)} filas)")
    return pd.concat(dfs, ignore_index=True)


# ── Feature engineering (adapted from notebook 08) ───────────────────
def build_features(df, odds_df=None):
    df = df.copy()
    df["posesion"] = df["ball_possession"].str.replace("%", "").astype(float)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)
    df["puntos"] = df["resultado"].map({"W": 3, "D": 1, "L": 0})

    feat_map = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()

        tdf["goles_rolling"] = (
            tdf["goles"].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["goles_rec_rolling"] = (
            tdf["goles_rival"].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["posesion_rolling"] = (
            tdf["posesion"].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["tiros_rolling"] = (
            tdf["total_shots"].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["tiros_gol_rolling"] = (
            tdf["shots_on_goal"].rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["racha_puntos"] = (
            tdf["puntos"].rolling(STREAK_N, min_periods=STREAK_N).sum().shift(1)
        )
        tdf["racha_victorias"] = (
            (tdf["resultado"] == "W")
            .astype(int)
            .rolling(STREAK_N, min_periods=STREAK_N)
            .sum()
            .shift(1)
        )
        tdf["racha_derrotas"] = (
            (tdf["resultado"] == "L")
            .astype(int)
            .rolling(STREAK_N, min_periods=STREAK_N)
            .sum()
            .shift(1)
        )

        tdf["goles_hist_local"] = np.nan
        tdf["goles_hist_visita"] = np.nan
        home_idx = tdf[tdf["side"] == "home"].index
        away_idx = tdf[tdf["side"] == "away"].index
        if len(home_idx) > 0:
            tdf.loc[home_idx, "goles_hist_local"] = (
                tdf.loc[home_idx, "goles"].expanding(min_periods=1).mean().shift(1)
            )
        if len(away_idx) > 0:
            tdf.loc[away_idx, "goles_hist_visita"] = (
                tdf.loc[away_idx, "goles"].expanding(min_periods=1).mean().shift(1)
            )
        tdf["goles_hist_local"] = tdf["goles_hist_local"].ffill()
        tdf["goles_hist_visita"] = tdf["goles_hist_visita"].ffill()

        tdf["puntos_acum"] = tdf["puntos"].cumsum().shift(1)
        tdf["gd_acum"] = (tdf["goles"] - tdf["goles_rival"]).cumsum().shift(1)

        for _, row in tdf.iterrows():
            feat_map[(row["fixture_id"], team)] = {
                "goles_rolling": row["goles_rolling"],
                "goles_rec_rolling": row["goles_rec_rolling"],
                "posesion_rolling": row["posesion_rolling"],
                "tiros_rolling": row["tiros_rolling"],
                "tiros_gol_rolling": row["tiros_gol_rolling"],
                "racha_puntos": row["racha_puntos"],
                "racha_victorias": row["racha_victorias"],
                "racha_derrotas": row["racha_derrotas"],
                "goles_hist_local": row["goles_hist_local"],
                "goles_hist_visita": row["goles_hist_visita"],
                "puntos_acum": row["puntos_acum"],
                "gd_acum": row["gd_acum"],
            }

    # Build match-level rows
    df_h = df[df["side"] == "home"][
        ["fixture_id", "fecha", "equipo", "goles", "torneo", "ronda", "resultado"]
    ].copy()
    df_h.columns = [
        "fixture_id", "fecha", "home", "goles_home", "torneo", "ronda", "resultado_home",
    ]
    df_a = df[df["side"] == "away"][["fixture_id", "equipo", "goles"]].copy()
    df_a.columns = ["fixture_id", "away", "goles_away"]
    matches = df_h.merge(df_a, on="fixture_id")

    matches["resultado"] = matches["resultado_home"].map({"W": 0, "D": 1, "L": 2})

    all_feats = [
        "goles_rolling", "goles_rec_rolling", "posesion_rolling",
        "tiros_rolling", "tiros_gol_rolling", "racha_puntos",
        "racha_victorias", "racha_derrotas", "goles_hist_local",
        "goles_hist_visita", "puntos_acum", "gd_acum",
    ]

    for prefix, col in [("home", "home"), ("away", "away")]:
        for feat in all_feats:
            matches[f"{prefix}_{feat}"] = matches.apply(
                lambda row, f=feat, c=col: feat_map.get(
                    (row["fixture_id"], row[c]), {}
                ).get(f),
                axis=1,
            )

    matches["diff_puntos_tabla"] = matches["home_puntos_acum"] - matches["away_puntos_acum"]
    matches["diff_gd"] = matches["home_gd_acum"] - matches["away_gd_acum"]
    matches["diff_racha"] = matches["home_racha_puntos"] - matches["away_racha_puntos"]
    matches["sum_goles_rolling"] = matches["home_goles_rolling"] + matches["away_goles_rolling"]
    matches["sum_goles_rec_rolling"] = (
        matches["home_goles_rec_rolling"] + matches["away_goles_rec_rolling"]
    )

    # Merge odds if provided
    if odds_df is not None:
        matches = matches.merge(
            odds_df[['fixture_id', 'prob_home', 'prob_draw', 'prob_away']],
            on='fixture_id', how='left'
        ).rename(columns={
            'prob_home': 'odds_prob_home',
            'prob_draw': 'odds_prob_draw',
            'prob_away': 'odds_prob_away',
        })

    return matches, feat_map


def get_latest_features(df):
    """Get the most recent feature snapshot for every team."""
    df = df.copy()
    df["posesion"] = df["ball_possession"].str.replace("%", "").astype(float)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)
    df["puntos"] = df["resultado"].map({"W": 3, "D": 1, "L": 0})

    latest = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()

        tdf["goles_rolling"] = tdf["goles"].rolling(ROLLING_N, min_periods=ROLLING_N).mean()
        tdf["goles_rec_rolling"] = tdf["goles_rival"].rolling(ROLLING_N, min_periods=ROLLING_N).mean()
        tdf["posesion_rolling"] = tdf["posesion"].rolling(ROLLING_N, min_periods=ROLLING_N).mean()
        tdf["tiros_rolling"] = tdf["total_shots"].rolling(ROLLING_N, min_periods=ROLLING_N).mean()
        tdf["tiros_gol_rolling"] = tdf["shots_on_goal"].rolling(ROLLING_N, min_periods=ROLLING_N).mean()
        tdf["racha_puntos"] = tdf["puntos"].rolling(STREAK_N, min_periods=STREAK_N).sum()
        tdf["racha_victorias"] = (
            (tdf["resultado"] == "W").astype(int).rolling(STREAK_N, min_periods=STREAK_N).sum()
        )
        tdf["racha_derrotas"] = (
            (tdf["resultado"] == "L").astype(int).rolling(STREAK_N, min_periods=STREAK_N).sum()
        )

        home_games = tdf[tdf["side"] == "home"]
        away_games = tdf[tdf["side"] == "away"]
        goles_hist_local = home_games["goles"].expanding(min_periods=1).mean().iloc[-1] if len(home_games) > 0 else np.nan
        goles_hist_visita = away_games["goles"].expanding(min_periods=1).mean().iloc[-1] if len(away_games) > 0 else np.nan

        tdf["puntos_acum"] = tdf["puntos"].cumsum()
        tdf["gd_acum"] = (tdf["goles"] - tdf["goles_rival"]).cumsum()

        last = tdf.iloc[-1]
        n_games = len(tdf)

        latest[team] = {
            "goles_rolling": last["goles_rolling"],
            "goles_rec_rolling": last["goles_rec_rolling"],
            "posesion_rolling": last["posesion_rolling"],
            "tiros_rolling": last["tiros_rolling"],
            "tiros_gol_rolling": last["tiros_gol_rolling"],
            "racha_puntos": last["racha_puntos"],
            "racha_victorias": last["racha_victorias"],
            "racha_derrotas": last["racha_derrotas"],
            "goles_hist_local": goles_hist_local,
            "goles_hist_visita": goles_hist_visita,
            "puntos_acum": last["puntos_acum"],
            "gd_acum": last["gd_acum"],
            "n_games": n_games,
        }

    return latest


def build_matchup_vector(home_feats, away_feats, odds=None):
    """Combine home + away features into a single feature vector.

    Args:
        odds: optional dict with keys prob_home, prob_draw, prob_away.
              If provided, returns 26-dim vector using FEATURES_WITH_ODDS.
    """
    row = {}
    all_feats = [
        "goles_rolling", "goles_rec_rolling", "posesion_rolling",
        "tiros_rolling", "tiros_gol_rolling", "racha_puntos",
        "racha_victorias", "racha_derrotas", "goles_hist_local",
        "goles_hist_visita", "puntos_acum", "gd_acum",
    ]
    for feat in all_feats:
        row[f"home_{feat}"] = home_feats[feat]
        row[f"away_{feat}"] = away_feats[feat]

    row["diff_puntos_tabla"] = home_feats["puntos_acum"] - away_feats["puntos_acum"]
    row["diff_gd"] = home_feats["gd_acum"] - away_feats["gd_acum"]
    row["diff_racha"] = home_feats["racha_puntos"] - away_feats["racha_puntos"]
    row["sum_goles_rolling"] = home_feats["goles_rolling"] + away_feats["goles_rolling"]
    row["sum_goles_rec_rolling"] = home_feats["goles_rec_rolling"] + away_feats["goles_rec_rolling"]

    if odds is not None:
        row["odds_prob_home"] = odds["prob_home"]
        row["odds_prob_draw"] = odds["prob_draw"]
        row["odds_prob_away"] = odds["prob_away"]
        return np.array([[row[f] for f in FEATURES_WITH_ODDS]])

    return np.array([[row[f] for f in FEATURES]])


# ── Fuzzy matching ───────────────────────────────────────────────────
def fuzzy_match(name, valid_teams, threshold=0.5):
    name_lower = name.lower().strip()
    # Exact match (case-insensitive)
    for t in valid_teams:
        if t.lower() == name_lower:
            return t, 1.0
    # Substring match
    for t in valid_teams:
        if name_lower in t.lower() or t.lower() in name_lower:
            return t, 0.9
    # SequenceMatcher
    best_team, best_score = None, 0
    for t in valid_teams:
        score = SequenceMatcher(None, name_lower, t.lower()).ratio()
        if score > best_score:
            best_team, best_score = t, score
    if best_score >= threshold:
        return best_team, best_score
    return best_team, best_score


def resolve_team(name, valid_teams):
    team, score = fuzzy_match(name, valid_teams)
    if score >= 0.5:
        if score < 0.9:
            print(f'  (!) "{name}" -> interpretado como "{team}" (similitud {score:.0%})')
        return team
    print(f'\n  Error: No se encontró el equipo "{name}".')
    print(f"  Equipos disponibles:")
    for t in sorted(valid_teams):
        print(f"    - {t}")
    suggestion = team
    if suggestion:
        print(f'\n  Quizás quisiste decir: "{suggestion}"?')
    return None


# ── Model training ───────────────────────────────────────────────────
def train_model(matches_clean, features=None):
    if features is None:
        features = FEATURES
    X = matches_clean[features].values
    y = matches_clean["resultado"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=5,
        max_features="sqrt", random_state=42,
    )
    model.fit(X_scaled, y)

    # CV accuracy estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    cv_acc = cv_scores.mean()

    return model, scaler, cv_acc


# ── Display helpers ──────────────────────────────────────────────────
def confidence_label(max_prob):
    if max_prob > 0.55:
        return "ALTA", "la señal del modelo es clara"
    elif max_prob >= 0.45:
        return "MEDIA", "hay una tendencia pero no es concluyente"
    else:
        return "BAJA", "partido muy parejo, cualquier resultado es posible"


def prob_bar(prob, width=20):
    filled = int(round(prob * width))
    return "\u2588" * filled + "\u258c" * min(1, width - filled) if filled < width else "\u2588" * filled


def print_prediction(home, away, probs, cv_acc, home_feats, away_feats):
    pred_idx = np.argmax(probs)
    pred_labels = {0: f"Victoria de {home} (local)", 1: "Empate", 2: f"Victoria de {away} (visita)"}
    conf_level, conf_text = confidence_label(probs[pred_idx])

    w = 50
    print()
    print("\u2550" * w)
    print(f"  PREDICCION - Liga MX")
    print(f"  {home} (local) vs {away} (visita)")
    print("\u2550" * w)
    print()
    print(f"  {'Victoria local':<22} {probs[0]:>4.0%}  {prob_bar(probs[0])}")
    print(f"  {'Empate':<22} {probs[1]:>4.0%}  {prob_bar(probs[1])}")
    print(f"  {'Victoria visitante':<22} {probs[2]:>4.0%}  {prob_bar(probs[2])}")
    print()
    print(f"  -> Prediccion: {pred_labels[pred_idx]}")
    print(f"  -> Confianza: {conf_level} ({conf_text})")
    print(f"     El modelo acierta ~{cv_acc:.0%} en validacion cruzada.")
    print()
    print(f"  CONTEXTO DEL PARTIDO")
    print(f"  " + "\u2500" * 30)

    h_racha = home_feats["racha_puntos"]
    a_racha = away_feats["racha_puntos"]
    h_goles = home_feats["goles_rolling"]
    a_goles = away_feats["goles_rolling"]
    diff_pts = home_feats["puntos_acum"] - away_feats["puntos_acum"]

    h_racha_str = f"{int(h_racha)}/9 pts" if not np.isnan(h_racha) else "sin datos"
    a_racha_str = f"{int(a_racha)}/9 pts" if not np.isnan(a_racha) else "sin datos"
    h_goles_str = f"{h_goles:.1f} goles/partido" if not np.isnan(h_goles) else "sin datos"
    a_goles_str = f"{a_goles:.1f} goles/partido" if not np.isnan(a_goles) else "sin datos"

    print(f"  {home}: racha {h_racha_str}, {h_goles_str}")
    print(f"  {away}: racha {a_racha_str}, {a_goles_str}")

    if not np.isnan(diff_pts):
        if diff_pts > 0:
            print(f"  D tabla: {home} +{int(diff_pts)} pts arriba")
        elif diff_pts < 0:
            print(f"  D tabla: {away} +{int(abs(diff_pts))} pts arriba")
        else:
            print(f"  D tabla: Equipos igualados en puntos")

    print("\u2550" * w)
    print()


def print_all_predictions(teams, model, scaler, cv_acc, latest_feats):
    """Print a compact table of all possible matchups (sample jornada)."""
    valid = [t for t in teams if not np.isnan(latest_feats[t].get("goles_rolling", np.nan))]
    if len(valid) < 2:
        print("  No hay suficientes equipos con datos completos.")
        return

    # Generate 9 random matchups as a sample jornada
    rng = np.random.RandomState(42)
    shuffled = list(valid)
    rng.shuffle(shuffled)
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, min(18, len(shuffled)) - 1, 2)]

    w = 70
    print()
    print("\u2550" * w)
    print(f"  PREDICCIONES - Jornada ejemplo ({len(pairs)} partidos)")
    print("\u2550" * w)
    print()
    print(f"  {'Local':<18} {'Visita':<18} {'L':>5} {'E':>5} {'V':>5}  {'Pred':<10}")
    print(f"  " + "\u2500" * 64)

    for home, away in pairs:
        X_new = build_matchup_vector(latest_feats[home], latest_feats[away])
        if np.any(np.isnan(X_new)):
            continue
        X_new_s = scaler.transform(X_new)
        probs = model.predict_proba(X_new_s)[0]
        pred_idx = np.argmax(probs)
        pred_str = {0: "Local", 1: "Empate", 2: "Visita"}[pred_idx]
        print(f"  {home:<18} {away:<18} {probs[0]:>4.0%} {probs[1]:>4.0%} {probs[2]:>4.0%}  {pred_str:<10}")

    print()
    print(f"  Accuracy historico del modelo (CV): ~{cv_acc:.0%}")
    print("\u2550" * w)
    print()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prediccion de resultado Liga MX (Local/Empate/Visitante)",
    )
    parser.add_argument("home", nargs="?", help="Equipo local")
    parser.add_argument("away", nargs="?", help="Equipo visitante")
    parser.add_argument("--all", action="store_true", help="Mostrar jornada ejemplo con 9 partidos")
    parser.add_argument("--teams", action="store_true", help="Listar equipos disponibles")
    parser.add_argument("--compare-odds", action="store_true", help="Comparar modelo base vs modelo + odds")
    args = parser.parse_args()

    # Load data
    df = load_data()
    valid_teams = sorted(df["equipo"].unique())

    # --teams: just list and exit
    if args.teams:
        print(f"\n  Equipos disponibles ({len(valid_teams)}):")
        for t in valid_teams:
            print(f"    - {t}")
        print()
        return

    # --compare-odds: train both models on same subset, compare
    if args.compare_odds:
        odds_df = load_odds_data()
        if odds_df is None:
            print("\n  Error: No se encontraron archivos odds_*.csv en data/raw/")
            print("  Ejecuta primero: python src/extract_odds.py --season 2024")
            return

        matches_with_odds, _ = build_features(df, odds_df=odds_df)
        # Only keep matches that have odds (the common subset)
        common = matches_with_odds.dropna(subset=FEATURES_WITH_ODDS).copy()
        n = len(common)

        if n < 20:
            print(f"\n  Error: Solo {n} partidos con odds y features completos. Necesitas mas datos.")
            return

        # Base model (23 features) on the SAME subset
        _, _, base_cv = train_model(common, features=FEATURES)
        # Odds model (26 features) on the SAME subset
        _, _, odds_cv = train_model(common, features=FEATURES_WITH_ODDS)

        delta = (odds_cv - base_cv) * 100

        w = 55
        print()
        print("=" * w)
        print("  COMPARACION: Modelo base vs Modelo + Odds")
        print("=" * w)
        print(f"\n  Partidos con odds disponibles: N={n}")
        print(f"\n  Modelo base  (23 features): {base_cv:.0%} accuracy (CV)")
        print(f"  Modelo + odds (26 features): {odds_cv:.0%} accuracy (CV)")
        print(f"  Delta: {delta:+.1f} pp")
        print()
        print("=" * w)
        return

    # Build features on all historical data
    matches, _ = build_features(df)
    matches_clean = matches.dropna(subset=FEATURES).copy()
    latest_feats = get_latest_features(df)

    # Check which teams have complete features
    teams_with_data = [t for t in valid_teams if not np.isnan(latest_feats[t].get("goles_rolling", np.nan))]
    teams_low_data = [
        t for t in valid_teams
        if latest_feats[t]["n_games"] < 5
    ]

    # Train model
    model, scaler, cv_acc = train_model(matches_clean)

    # --all mode
    if args.all:
        print_all_predictions(valid_teams, model, scaler, cv_acc, latest_feats)
        return

    # Single prediction mode — need both teams
    if not args.home or not args.away:
        parser.print_help()
        print(f'\n  Ejemplo: python src/predict.py "Cruz Azul" "Club America"')
        return

    home = resolve_team(args.home, valid_teams)
    away = resolve_team(args.away, valid_teams)
    if not home or not away:
        return

    if home == away:
        print(f"\n  Error: Un equipo no puede jugar contra sí mismo.")
        return

    # Warn if low data
    for t in [home, away]:
        if t in teams_low_data:
            print(f"  (!) Advertencia: {t} tiene menos de 5 partidos, la prediccion puede ser imprecisa.")

    # Check features are available
    hf = latest_feats[home]
    af = latest_feats[away]

    X_new = build_matchup_vector(hf, af)
    if np.any(np.isnan(X_new)):
        nan_count = np.isnan(X_new).sum()
        print(f"\n  Error: Faltan {nan_count} features (datos insuficientes).")
        print(f"  Alguno de los equipos no tiene suficientes partidos para calcular rolling/racha.")
        return

    X_new_s = scaler.transform(X_new)
    probs = model.predict_proba(X_new_s)[0]

    print_prediction(home, away, probs, cv_acc, hf, af)


if __name__ == "__main__":
    main()

"""
Predicción de mercados secundarios — Liga MX
Modelos de regresión para Over/Under en tarjetas amarillas, corners, y tiros.

Uso:
    python src/predict_props.py                          # evalua modelo de tarjetas
    python src/predict_props.py --corners                # tambien evalua corners
    python src/predict_props.py --shots                  # tambien evalua tiros totales
    python src/predict_props.py "Cruz Azul" "America"    # prediccion de tarjetas
"""

import sys
import os
import warnings
import argparse
import unicodedata
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

ROLLING_N = 5

TRAIN_SOURCES = {"apertura2025_team_stats.csv", "clausura2025_team_stats.csv"}
TEST_SOURCES = {"clausura2026_team_stats.csv"}

# ── Cards config ─────────────────────────────────────────────────────
CARDS_LINE = 3.5
CARDS_FEATURES = [
    "home_yc_avg",
    "away_yc_avg",
    "ref_yc_avg",
]

# ── Corners config ───────────────────────────────────────────────────
CORNERS_LINE = 9.5
CORNERS_FEATURES = [
    "home_corners_avg",
    "home_corners_against_avg",
    "home_shots_avg",
    "away_corners_avg",
    "away_corners_against_avg",
    "away_shots_avg",
]

# ── Shots config (does not beat baseline) ────────────────────────────
SHOTS_LINE = 24.5
SHOTS_FEATURES = [
    "home_shots_avg",
    "home_shots_against_avg",
    "away_shots_avg",
    "away_shots_against_avg",
]


# ══════════════════════════════════════════════════════════════════════
#  SHARED
# ══════════════════════════════════════════════════════════════════════
def load_data():
    """Load team stats CSVs with source tag for train/test split."""
    import glob

    pattern = os.path.join(RAW_DIR, "*_team_stats.csv")
    files = [
        f
        for f in sorted(glob.glob(pattern))
        if not os.path.basename(f).startswith("ligamx")
    ]
    if not files:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["_source"] = os.path.basename(fp).lower()
        dfs.append(df)
        print(f"  Cargado: {os.path.basename(fp)} ({len(df)} filas)")
    return pd.concat(dfs, ignore_index=True)


def _add_opponent_stat(df, col):
    """Add opponent's value of `col` for the same fixture as `{col}_against`."""
    opp = df[["fixture_id", "equipo", col]].rename(
        columns={"equipo": "oponente", col: f"{col}_against"}
    )
    return df.merge(opp, on=["fixture_id", "oponente"])


def normalize_referee(name):
    """Normalize referee name to 'X. Surname' format, stripping accents."""
    if pd.isna(name):
        return None
    # Strip country suffix ("..., Mexico")
    name = name.split(",")[0].strip()
    # Strip accents
    name = "".join(
        c
        for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    parts = name.split()
    if len(parts) < 2:
        return name
    # Already abbreviated ("D. Quintero", "C. A. Ramos", "L. E. Santander Aguirre")
    if parts[0].endswith("."):
        initial = parts[0][0]
        for p in parts[1:]:
            if not p.endswith("."):
                return f"{initial}. {p}"
        return name
    # Full name: "Ivan Antonio Lopez Sanchez" → "I. Lopez"
    initial = parts[0][0].upper()
    if len(parts) == 2:
        return f"{initial}. {parts[1]}"
    # 3+ words: first initial + second-to-last word (paternal surname)
    return f"{initial}. {parts[-2]}"


def fuzzy_match(name, valid_teams, threshold=0.5):
    name_lower = name.lower().strip()
    for t in valid_teams:
        if t.lower() == name_lower:
            return t, 1.0
    for t in valid_teams:
        if name_lower in t.lower() or t.lower() in name_lower:
            return t, 0.9
    best_team, best_score = None, 0
    for t in valid_teams:
        score = SequenceMatcher(None, name_lower, t.lower()).ratio()
        if score > best_score:
            best_team, best_score = t, score
    return best_team, best_score


# ══════════════════════════════════════════════════════════════════════
#  CARDS MODEL (tarjetas amarillas)
# ══════════════════════════════════════════════════════════════════════
def build_cards_matches(df):
    """Build match-level DataFrame with team yellow card rolling avgs + referee avg."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["yellow_cards"] = pd.to_numeric(df["yellow_cards"], errors="coerce")
    df["ref_norm"] = df["arbitro"].apply(normalize_referee)
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    # ── Per-team rolling avg of yellow cards ──
    team_feat_map = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()
        tdf["yc_avg"] = (
            tdf["yellow_cards"]
            .rolling(ROLLING_N, min_periods=ROLLING_N)
            .mean()
            .shift(1)
        )
        for _, row in tdf.iterrows():
            team_feat_map[(row["fixture_id"], team)] = row["yc_avg"]

    # ── Match-level DataFrame ──
    df_h = df[df["side"] == "home"][
        ["fixture_id", "fecha", "equipo", "yellow_cards", "ref_norm",
         "torneo", "ronda", "_source"]
    ].copy()
    df_h.columns = [
        "fixture_id", "fecha", "home", "yc_home", "ref_norm",
        "torneo", "ronda", "_source",
    ]
    df_a = df[df["side"] == "away"][
        ["fixture_id", "equipo", "yellow_cards"]
    ].copy()
    df_a.columns = ["fixture_id", "away", "yc_away"]

    matches = df_h.merge(df_a, on="fixture_id")
    matches["total_yc"] = matches["yc_home"] + matches["yc_away"]

    # ── Referee expanding average (only prior games) ──
    matches = matches.sort_values("fecha").reset_index(drop=True)
    ref_yc_map = {}
    ref_history = {}

    for _, row in matches.iterrows():
        ref = row["ref_norm"]
        fid = row["fixture_id"]
        total = row["total_yc"]

        # Feature: referee's avg from all prior games
        if ref and ref in ref_history and len(ref_history[ref]) > 0:
            ref_yc_map[fid] = np.mean(ref_history[ref])
        else:
            ref_yc_map[fid] = np.nan

        # Update history AFTER computing feature
        if ref and not pd.isna(total):
            ref_history.setdefault(ref, []).append(total)

    matches["ref_yc_avg"] = matches["fixture_id"].map(ref_yc_map)

    # ── Attach team features ──
    for prefix, col in [("home", "home"), ("away", "away")]:
        matches[f"{prefix}_yc_avg"] = matches.apply(
            lambda row, c=col: team_feat_map.get(
                (row["fixture_id"], row[c]), np.nan
            ),
            axis=1,
        )

    return matches, team_feat_map, ref_history


def train_cards_model():
    """Train yellow cards model on Ap2025+Cl2025, evaluate on Clausura 2026."""
    print("\n" + "=" * 60)
    print("  Modelo de Tarjetas Amarillas — Random Forest Regressor")
    print("=" * 60)

    df = load_data()
    matches, _, ref_history = build_cards_matches(df)

    all_matches = matches.dropna(subset=CARDS_FEATURES + ["total_yc"])
    train = all_matches[all_matches["_source"].isin(TRAIN_SOURCES)]
    test = all_matches[all_matches["_source"].isin(TEST_SOURCES)]

    print(f"\n  Train: {len(train)} partidos (Apertura 2025 + Clausura 2025)")
    print(f"  Test:  {len(test)} partidos (Clausura 2026)")

    # Referee coverage in test set
    test_all = matches[matches["_source"].isin(TEST_SOURCES)]
    test_with_ref = test_all.dropna(subset=["ref_yc_avg"])
    print(f"  Test con arbitro conocido: {len(test_with_ref)}/{len(test_all)} ({len(test_with_ref)/len(test_all):.0%})")

    if len(train) < 10 or len(test) < 5:
        print("  Error: datos insuficientes.")
        return None, None, None

    X_train = train[CARDS_FEATURES].values
    y_train = train["total_yc"].values
    X_test = test[CARDS_FEATURES].values
    y_test = test["total_yc"].values

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Baseline: always predict OVER (accuracy = % of actual overs)
    actual_over = y_test > CARDS_LINE
    baseline_over_acc = actual_over.mean()

    # MAE baseline: always predict training mean
    train_mean = y_train.mean()
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, train_mean))

    # Model O/U accuracy
    pred_over = y_pred > CARDS_LINE
    ou_accuracy = (actual_over == pred_over).mean()
    ou_correct = int(ou_accuracy * len(y_test))

    print(f"\n  {'─' * 50}")
    print(f"  RESULTADOS (Clausura 2026)")
    print(f"  {'─' * 50}")
    print(f"  MAE modelo:   {mae:.2f} tarjetas")
    print(f"  MAE baseline: {baseline_mae:.2f} tarjetas (siempre predecir {train_mean:.1f})")
    print(f"  Mejora MAE:   {baseline_mae - mae:.2f} tarjetas menos de error")
    print()
    print(f"  Over/Under {CARDS_LINE}:")
    print(f"    Modelo:            {ou_accuracy:.0%} ({ou_correct}/{len(y_test)})")
    print(f"    Siempre Over:      {baseline_over_acc:.0%} ({int(baseline_over_acc * len(y_test))}/{len(y_test)})")
    print(f"    Ventaja modelo:    {(ou_accuracy - baseline_over_acc) * 100:+.1f} pp")
    print()
    print(f"  Distribucion real (test):")
    print(f"    Promedio: {y_test.mean():.1f} tarjetas/partido")
    print(f"    Std:      {y_test.std():.1f}")
    print(f"    Rango:    {int(y_test.min())} – {int(y_test.max())}")
    print(f"    Over {CARDS_LINE}: {actual_over.sum()}/{len(y_test)} ({actual_over.mean():.0%})")
    print()
    print(f"  Importancia de features:")
    for feat, imp in sorted(
        zip(CARDS_FEATURES, model.feature_importances_), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<30} {imp:.0%} {bar}")

    # Top referees by avg cards
    print(f"\n  Arbitros mas tarjeteros (min 5 partidos):")
    for ref, history in sorted(ref_history.items(), key=lambda x: -np.mean(x[1])):
        if len(history) >= 5:
            print(f"    {ref:<20} {np.mean(history):.1f} tarjetas/partido ({len(history)} partidos)")

    # Sample predictions
    print(f"\n  {'─' * 50}")
    print(f"  MUESTRA DE PREDICCIONES")
    print(f"  {'─' * 50}")
    print(f"  {'Local':<18} {'Visita':<15} {'Arb':>5} {'Real':>5} {'Pred':>5} {'O/U':>6}")
    print(f"  {'─' * 58}")

    test_reset = test.reset_index(drop=True)
    n_sample = min(15, len(test_reset))
    for i in range(n_sample):
        row = test_reset.iloc[i]
        pred_i = y_pred[i]
        real_i = row["total_yc"]
        ref_avg = row["ref_yc_avg"]
        hit = (real_i > CARDS_LINE) == (pred_i > CARDS_LINE)
        ou_label = "OK" if hit else "MISS"
        print(
            f"  {row['home']:<18} {row['away']:<15} "
            f"{ref_avg:>5.1f} {real_i:>5.0f} {pred_i:>5.1f} {ou_label:>6}"
        )

    print(f"\n{'=' * 60}")
    return model, df, ref_history


# ══════════════════════════════════════════════════════════════════════
#  CORNERS MODEL (does not beat baseline — kept for reference)
# ══════════════════════════════════════════════════════════════════════
def build_corners_matches(df):
    """Build match-level DataFrame with rolling corner + shots features."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["corner_kicks"] = pd.to_numeric(df["corner_kicks"], errors="coerce")
    df["total_shots"] = pd.to_numeric(df["total_shots"], errors="coerce")
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    df = _add_opponent_stat(df, "corner_kicks")

    feat_map = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()
        tdf["corners_avg"] = (
            tdf["corner_kicks"]
            .rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["corners_against_avg"] = (
            tdf["corner_kicks_against"]
            .rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["shots_avg"] = (
            tdf["total_shots"]
            .rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        for _, row in tdf.iterrows():
            feat_map[(row["fixture_id"], team)] = {
                "corners_avg": row["corners_avg"],
                "corners_against_avg": row["corners_against_avg"],
                "shots_avg": row["shots_avg"],
            }

    df_h = df[df["side"] == "home"][
        ["fixture_id", "fecha", "equipo", "corner_kicks", "torneo", "ronda", "_source"]
    ].copy()
    df_h.columns = [
        "fixture_id", "fecha", "home", "corners_home", "torneo", "ronda", "_source",
    ]
    df_a = df[df["side"] == "away"][["fixture_id", "equipo", "corner_kicks"]].copy()
    df_a.columns = ["fixture_id", "away", "corners_away"]

    matches = df_h.merge(df_a, on="fixture_id")
    matches["total_corners"] = matches["corners_home"] + matches["corners_away"]

    for prefix, col in [("home", "home"), ("away", "away")]:
        for feat in ["corners_avg", "corners_against_avg", "shots_avg"]:
            matches[f"{prefix}_{feat}"] = matches.apply(
                lambda row, f=feat, c=col: feat_map.get(
                    (row["fixture_id"], row[c]), {}
                ).get(f),
                axis=1,
            )

    return matches, feat_map


def train_corners_model():
    """Train corners model (does not beat baseline — kept for comparison)."""
    print("\n" + "=" * 60)
    print("  Modelo de Corners Totales — Random Forest Regressor")
    print("  (No supera baseline — solo para referencia)")
    print("=" * 60)

    df = load_data()
    matches, _ = build_corners_matches(df)

    all_matches = matches.dropna(subset=CORNERS_FEATURES + ["total_corners"])
    train = all_matches[all_matches["_source"].isin(TRAIN_SOURCES)]
    test = all_matches[all_matches["_source"].isin(TEST_SOURCES)]

    print(f"\n  Train: {len(train)} partidos | Test: {len(test)} partidos")

    if len(train) < 10 or len(test) < 5:
        print("  Error: datos insuficientes.")
        return

    X_train = train[CORNERS_FEATURES].values
    y_train = train["total_corners"].values
    X_test = test[CORNERS_FEATURES].values
    y_test = test["total_corners"].values

    model = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
    actual_over = y_test > CORNERS_LINE
    ou_accuracy = ((actual_over) == (y_pred > CORNERS_LINE)).mean()
    baseline_ou = actual_over.mean()

    print(f"  MAE modelo: {mae:.2f} | MAE baseline: {baseline_mae:.2f}")
    print(f"  O/U {CORNERS_LINE}: modelo {ou_accuracy:.0%} | siempre over {baseline_ou:.0%}")
    print(f"  Veredicto: {'Supera' if ou_accuracy > baseline_ou else 'No supera'} baseline")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
#  SHOTS MODEL (does not beat baseline — kept for reference)
# ══════════════════════════════════════════════════════════════════════
def build_shots_matches(df):
    """Build match-level DataFrame with rolling shot features."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["total_shots"] = pd.to_numeric(df["total_shots"], errors="coerce")
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    df = _add_opponent_stat(df, "total_shots")

    feat_map = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()
        tdf["shots_avg"] = (
            tdf["total_shots"]
            .rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        tdf["shots_against_avg"] = (
            tdf["total_shots_against"]
            .rolling(ROLLING_N, min_periods=ROLLING_N).mean().shift(1)
        )
        for _, row in tdf.iterrows():
            feat_map[(row["fixture_id"], team)] = {
                "shots_avg": row["shots_avg"],
                "shots_against_avg": row["shots_against_avg"],
            }

    df_h = df[df["side"] == "home"][
        ["fixture_id", "fecha", "equipo", "total_shots", "torneo", "ronda", "_source"]
    ].copy()
    df_h.columns = [
        "fixture_id", "fecha", "home", "shots_home", "torneo", "ronda", "_source",
    ]
    df_a = df[df["side"] == "away"][["fixture_id", "equipo", "total_shots"]].copy()
    df_a.columns = ["fixture_id", "away", "shots_away"]

    matches = df_h.merge(df_a, on="fixture_id")
    matches["total_shots_match"] = matches["shots_home"] + matches["shots_away"]

    for prefix, col in [("home", "home"), ("away", "away")]:
        for feat in ["shots_avg", "shots_against_avg"]:
            matches[f"{prefix}_{feat}"] = matches.apply(
                lambda row, f=feat, c=col: feat_map.get(
                    (row["fixture_id"], row[c]), {}
                ).get(f),
                axis=1,
            )

    return matches, feat_map


def train_shots_model():
    """Train shots model (does not beat baseline — kept for comparison)."""
    print("\n" + "=" * 60)
    print("  Modelo de Tiros Totales — Random Forest Regressor")
    print("  (No supera baseline — solo para referencia)")
    print("=" * 60)

    df = load_data()
    matches, _ = build_shots_matches(df)

    all_matches = matches.dropna(subset=SHOTS_FEATURES + ["total_shots_match"])
    train = all_matches[all_matches["_source"].isin(TRAIN_SOURCES)]
    test = all_matches[all_matches["_source"].isin(TEST_SOURCES)]

    print(f"\n  Train: {len(train)} partidos | Test: {len(test)} partidos")

    if len(train) < 10 or len(test) < 5:
        print("  Error: datos insuficientes.")
        return

    X_train = train[SHOTS_FEATURES].values
    y_train = train["total_shots_match"].values
    X_test = test[SHOTS_FEATURES].values
    y_test = test["total_shots_match"].values

    model = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
    ou_accuracy = ((y_test > SHOTS_LINE) == (y_pred > SHOTS_LINE)).mean()

    print(f"  MAE modelo: {mae:.2f} | MAE baseline: {baseline_mae:.2f}")
    print(f"  O/U {SHOTS_LINE}: {ou_accuracy:.0%}")
    print(f"  Veredicto: {'Supera' if mae < baseline_mae else 'No supera'} baseline")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
#  INDIVIDUAL PREDICTION (cards)
# ══════════════════════════════════════════════════════════════════════
def get_latest_cards_features(df):
    """Get most recent rolling yellow card avg for each team."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["yellow_cards"] = pd.to_numeric(df["yellow_cards"], errors="coerce")
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    latest = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()
        tdf["yc_avg"] = (
            tdf["yellow_cards"].rolling(ROLLING_N, min_periods=ROLLING_N).mean()
        )
        last = tdf.iloc[-1]
        latest[team] = {"yc_avg": last["yc_avg"]}
    return latest


def predict_cards(model, latest_feats, ref_history, home, away, referee=None):
    """Predict total yellow cards for a specific matchup."""
    hf = latest_feats.get(home)
    af = latest_feats.get(away)
    if not hf or not af:
        print("  Error: datos insuficientes para alguno de los equipos.")
        return

    if referee:
        ref_norm = normalize_referee(referee)
        if ref_norm in ref_history:
            ref_avg = np.mean(ref_history[ref_norm])
        else:
            print(f"  Arbitro '{referee}' ({ref_norm}) no encontrado, usando promedio global.")
            all_avgs = [np.mean(v) for v in ref_history.values() if len(v) >= 3]
            ref_avg = np.mean(all_avgs)
    else:
        all_avgs = [np.mean(v) for v in ref_history.values() if len(v) >= 3]
        ref_avg = np.mean(all_avgs)
        print(f"  (Sin arbitro especificado, usando promedio global: {ref_avg:.1f})")

    X = np.array([[hf["yc_avg"], af["yc_avg"], ref_avg]])

    if np.any(np.isnan(X)):
        print(f"  Error: se necesitan al menos {ROLLING_N} partidos por equipo.")
        return

    pred = model.predict(X)[0]
    ou = "OVER" if pred > CARDS_LINE else "UNDER"
    margin = abs(pred - CARDS_LINE)

    print(f"\n  {home} (local) vs {away} (visita)")
    print(f"  Prediccion: {pred:.1f} tarjetas amarillas totales")
    print(f"  Linea {CARDS_LINE}: {ou} (margen: {margin:.1f})")
    print(f"  {home}: promedia {hf['yc_avg']:.1f} tarjetas/partido")
    print(f"  {away}: promedia {af['yc_avg']:.1f} tarjetas/partido")
    print(f"  Arbitro: {ref_avg:.1f} tarjetas/partido historico")
    print()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prediccion de mercados secundarios Liga MX"
    )
    parser.add_argument("home", nargs="?", help="Equipo local")
    parser.add_argument("away", nargs="?", help="Equipo visitante")
    parser.add_argument("--referee", help="Nombre del arbitro (opcional)")
    parser.add_argument(
        "--corners", action="store_true",
        help="Tambien evaluar modelo de corners",
    )
    parser.add_argument(
        "--shots", action="store_true",
        help="Tambien evaluar modelo de tiros totales",
    )
    args = parser.parse_args()

    # Always run cards model
    model, df, ref_history = train_cards_model()
    if model is None:
        return

    # Optionally run other models
    if args.corners:
        train_corners_model()
    if args.shots:
        train_shots_model()

    # Individual prediction
    if args.home and args.away:
        latest_feats = get_latest_cards_features(df)
        valid_teams = sorted(latest_feats.keys())
        home, hs = fuzzy_match(args.home, valid_teams)
        away, as_ = fuzzy_match(args.away, valid_teams)
        if hs < 0.5:
            print(f'  Error: equipo "{args.home}" no encontrado.')
            return
        if as_ < 0.5:
            print(f'  Error: equipo "{args.away}" no encontrado.')
            return
        predict_cards(model, latest_feats, ref_history, home, away, args.referee)


if __name__ == "__main__":
    main()

"""
Liga MX Match Predictor — Streamlit UI
Uso: streamlit run src/app.py
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Reuse prediction logic from predict.py ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import (
    FEATURES,
    FEATURES_WITH_ODDS,
    ODDS_FEATURES,
    build_features,
    build_matchup_vector,
    get_latest_features,
    load_data,
    load_odds_data,
)
from extract_odds import fetch_fixture_odds
from predict_props import normalize_referee, CARDS_LINE

# ── Backtest reliability profile (Clausura J8-J13, walk-forward) ─────
BACKTEST_BY_RESULT = {
    0: (29, 31),   # Local: 94%
    1: (0, 7),     # Empate: 0%
    2: (6, 14),    # Visita: 43%
}
BACKTEST_BY_CONF = {
    "ALTA": (11, 15),   # 73%
    "MEDIA": (14, 21),  # 67%
    "BAJA": (10, 16),   # 62%
}
BACKTEST_TOTAL = (35, 52)  # 67% overall

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Liga MX Predictor", page_icon="⚽", layout="centered")


# ══════════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset & base ─────────────────────────────────────────────── */
.block-container { padding-top: 1rem !important; }

/* ── Header hero ──────────────────────────────────────────────── */
.header-hero {
    background: #1a472a;
    border-bottom: 4px solid #c9a84c;
    border-radius: 12px;
    padding: 2rem 1.5rem 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.header-hero h1 {
    color: #ffffff;
    font-size: 2.2rem;
    margin: 0 0 0.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.header-hero .subtitle {
    color: #c9a84c;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}
.header-hero .model-tag {
    display: inline-block;
    background: rgba(201,168,76,0.18);
    color: #c9a84c;
    font-size: 0.78rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-top: 0.6rem;
    border: 1px solid rgba(201,168,76,0.3);
}

/* (match card section uses native Streamlit components) */

/* ── Probability tricolor bar ─────────────────────────────────── */
.prob-bar-container {
    margin: 1rem 0;
}
.prob-bar {
    display: flex;
    border-radius: 10px;
    overflow: hidden;
    height: 42px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.prob-bar .seg {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 0.85rem;
    transition: width 0.5s ease;
    min-width: 38px;
}
.prob-bar .seg.local  { background: #2d8a4e; }
.prob-bar .seg.empate { background: #f0a500; }
.prob-bar .seg.visita { background: #c0392b; }
.prob-bar-legend {
    display: flex;
    justify-content: space-between;
    margin-top: 0.4rem;
    font-size: 0.75rem;
    color: #666;
}

/* ── Prediction callout ───────────────────────────────────────── */
.pred-callout {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin: 1rem 0;
    color: #fff;
    font-weight: 700;
    font-size: 1.15rem;
}
.pred-callout.local  { background: #2d8a4e; }
.pred-callout.empate { background: #f0a500; color: #333; }
.pred-callout.visita { background: #c0392b; }
.pred-callout .pred-label { font-size: 0.78rem; font-weight: 400; opacity: 0.85; display: block; margin-bottom: 0.2rem; }

/* ── Confidence badge ─────────────────────────────────────────── */
.conf-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.conf-badge.alta  { background: #2d8a4e; color: #fff; }
.conf-badge.media { background: #f0a500; color: #333; }
.conf-badge.baja  { background: #c0392b; color: #fff; }
.conf-desc {
    color: #666;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* ── Reliability inline ───────────────────────────────────────── */
.reliability-inline {
    background: #f0f2f6;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: #444;
    border-left: 4px solid #c9a84c;
}
.reliability-inline strong { color: #1a472a; }
.reliability-inline.warn { border-left-color: #f0a500; }
.reliability-inline.danger { border-left-color: #c0392b; }

/* (context section uses native Streamlit components) */

/* ── Section divider ──────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 1.8rem 0;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1a472a;
    margin: 0.5rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 3px solid #c9a84c;
    display: inline-block;
}

/* (jornada section uses only native Streamlit components) */

/* ── Footer ───────────────────────────────────────────────────── */
.app-footer {
    background: #1a472a;
    border-top: 3px solid #c9a84c;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 2rem;
    color: rgba(255,255,255,0.7);
    font-size: 0.78rem;
}
.app-footer a { color: #c9a84c; text-decoration: none; }

/* ── Mobile responsive ────────────────────────────────────────── */
@media (max-width: 768px) {
    .header-hero h1 { font-size: 1.6rem; }
    .header-hero .subtitle { font-size: 0.95rem; }
    .prob-bar { height: 36px; }
    .prob-bar .seg { font-size: 0.75rem; }
    .pred-callout { font-size: 1rem; }
    /* jornada section is native Streamlit — no custom mobile rules */
}
</style>
""", unsafe_allow_html=True)


# ── Cached model training ───────────────────────────────────────────
def _train_rf(X, y):
    """Train a Random Forest and return model, scaler, cv_acc."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=5,
        max_features="sqrt", random_state=42,
    )
    model.fit(X_s, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_s, y, cv=cv, scoring="accuracy").mean()
    return model, scaler, cv_acc


@st.cache_data(show_spinner="Entrenando modelo...")
def prepare_model():
    df = load_data()
    odds_df = load_odds_data()

    matches, _ = build_features(df, odds_df=odds_df)
    matches_clean = matches.dropna(subset=FEATURES).copy()
    latest = get_latest_features(df)
    teams = sorted(df["equipo"].unique())

    # Base model (23 features)
    X_base = matches_clean[FEATURES].values
    y_base = matches_clean["resultado"].values
    base_model, base_scaler, base_cv = _train_rf(X_base, y_base)

    # Odds model (26 features) — only if odds data available
    odds_model, odds_scaler, odds_cv = None, None, None
    if odds_df is not None and all(f in matches.columns for f in ODDS_FEATURES):
        matches_odds = matches.dropna(subset=FEATURES_WITH_ODDS).copy()
        if len(matches_odds) >= 20:
            X_odds = matches_odds[FEATURES_WITH_ODDS].values
            y_odds = matches_odds["resultado"].values
            odds_model, odds_scaler, odds_cv = _train_rf(X_odds, y_odds)

    return (base_model, base_scaler, base_cv,
            odds_model, odds_scaler, odds_cv,
            latest, teams, odds_df)


@st.cache_data(ttl=3600, show_spinner="Cargando tabla de posiciones...")
def load_standings():
    from api_client import get_standings
    data = get_standings(262, 2025)
    table = data["response"][0]["league"]["standings"][0]
    return {t["team"]["name"]: t for t in table}


@st.cache_data(ttl=3600, show_spinner="Cargando fixtures...")
def load_fixtures_list(api_season, tournament):
    """Carga fixtures desde archivo local o desde la API."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cache_dir = os.path.join(base_dir, "data", "raw", "fixtures_cache")
    year = api_season + 1  # API season 2025 -> Clausura 2026
    path = os.path.join(cache_dir, f"{tournament.lower()}_{year}_fixtures_list.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback: fetch from API
    from api_client import get
    data = get("fixtures", {"league": 262, "season": api_season})
    fixtures = []
    for f in data["response"]:
        if tournament in f["league"]["round"]:
            fixtures.append({
                "fixture_id": f["fixture"]["id"],
                "date": f["fixture"]["date"],
                "round": f["league"]["round"],
                "home": f["teams"]["home"]["name"],
                "away": f["teams"]["away"]["name"],
                "status": f["fixture"]["status"]["short"],
            })
    fixtures.sort(key=lambda x: x["date"])
    return fixtures


@st.cache_data(ttl=3600)
def get_jornada_odds(fixture_ids: tuple) -> dict:
    """Fetch odds por fixture_id, cachea 1 hora. Returns {fixture_id: odds_dict}."""
    results = {}
    for fid in fixture_ids:
        odds = fetch_fixture_odds(fid)
        if odds:
            results[fid] = odds
        time.sleep(0.1)
    return results


def detect_next_jornada(fixtures_list, rounds):
    """Find the next jornada that has unplayed matches (NS/TBD status)."""
    now = datetime.now(timezone.utc)
    for r in rounds:
        round_fx = [f for f in fixtures_list if f["round"] == r]
        # A jornada is "next" if it has any unplayed match
        has_unplayed = any(f["status"] in ("NS", "TBD", "PST", "CANC", "1H", "HT", "2H")
                          for f in round_fx)
        if has_unplayed:
            return r
    # Fallback: last jornada
    return rounds[-1] if rounds else None


@st.cache_data(show_spinner="Cargando análisis de tarjetas...")
def prepare_cards_model():
    """Train yellow cards model, compute team/referee averages."""
    from sklearn.ensemble import RandomForestRegressor

    df = load_data()
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["yellow_cards"] = pd.to_numeric(df["yellow_cards"], errors="coerce")
    df["ref_norm"] = df["arbitro"].apply(normalize_referee)
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    # Latest per-team rolling YC average (not shifted — for live prediction)
    team_yc = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha")
        avgs = tdf["yellow_cards"].rolling(5, min_periods=5).mean()
        valid = avgs.dropna()
        team_yc[team] = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

    # Match-level data
    df_h = df[df["side"] == "home"][
        ["fixture_id", "fecha", "equipo", "yellow_cards", "ref_norm"]
    ].copy()
    df_h.columns = ["fixture_id", "fecha", "home", "yc_home", "ref_norm"]
    df_a = df[df["side"] == "away"][
        ["fixture_id", "equipo", "yellow_cards"]
    ].copy()
    df_a.columns = ["fixture_id", "away", "yc_away"]

    matches = df_h.merge(df_a, on="fixture_id").sort_values("fecha").reset_index(drop=True)
    matches["total_yc"] = matches["yc_home"] + matches["yc_away"]

    # Referee expanding average (prior games only for training features)
    ref_history = {}
    ref_avg_map = {}
    for _, row in matches.iterrows():
        ref = row["ref_norm"]
        fid = row["fixture_id"]
        total = row["total_yc"]
        if ref and ref in ref_history and len(ref_history[ref]) > 0:
            ref_avg_map[fid] = float(np.mean(ref_history[ref]))
        else:
            ref_avg_map[fid] = float("nan")
        if ref and not pd.isna(total):
            ref_history.setdefault(ref, []).append(float(total))

    matches["ref_yc_avg"] = matches["fixture_id"].map(ref_avg_map)

    # Per-team rolling (shifted for training)
    team_feat_shifted = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha").copy()
        tdf["yc_avg"] = tdf["yellow_cards"].rolling(5, min_periods=5).mean().shift(1)
        for _, r in tdf.iterrows():
            team_feat_shifted[(r["fixture_id"], team)] = r["yc_avg"]

    for prefix, col in [("home", "home"), ("away", "away")]:
        matches[f"{prefix}_yc_avg"] = matches.apply(
            lambda row, c=col: team_feat_shifted.get(
                (row["fixture_id"], row[c]), np.nan
            ),
            axis=1,
        )

    features = ["home_yc_avg", "away_yc_avg", "ref_yc_avg"]
    clean = matches.dropna(subset=features + ["total_yc"])

    if len(clean) < 10:
        return None, team_yc, ref_history

    model = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42,
    )
    model.fit(clean[features].values, clean["total_yc"].values)
    return model, team_yc, ref_history


@st.cache_data(ttl=3600)
def load_fixture_referees():
    """Load referee assignments from full season API cache."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cache_path = os.path.join(
        base, "data", "raw", "fixtures_cache", "full_season_2025_fixtures.json"
    )
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path) as f:
        data = json.load(f)
    return {
        fx["fixture"]["id"]: fx["fixture"].get("referee")
        for fx in data
        if fx["fixture"].get("referee")
    }


# ── Load everything ──────────────────────────────────────────────────
(base_model, base_scaler, base_cv,
 odds_model, odds_scaler, odds_cv,
 latest_feats, all_teams, odds_df) = prepare_model()

try:
    standings = load_standings()
except Exception:
    standings = None

# Backward-compat aliases used throughout
model, scaler, cv_acc = base_model, base_scaler, base_cv

# Teams with complete features
teams_ready = [t for t in all_teams if not np.isnan(latest_feats[t].get("goles_rolling", np.nan))]

# Load Clausura 2026 fixtures (API season 2025)
fixtures_list = load_fixtures_list(2025, "Clausura")

# Cards model (for referee analysis)
try:
    cards_model, team_yc, ref_history = prepare_cards_model()
    fixture_referees = load_fixture_referees()
except Exception:
    cards_model, team_yc, ref_history = None, {}, {}
    fixture_referees = {}


# ══════════════════════════════════════════════════════════════════════
# HEADER HERO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-hero">
    <h1>Liga MX Predictor</h1>
    <p class="subtitle">Clausura 2026</p>
    <span class="model-tag">Predicciones basadas en los ultimos 5 partidos de cada equipo</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: JORNADA COMPLETA (primary view)
# ══════════════════════════════════════════════════════════════════════
if fixtures_list:
    # Extract unique jornadas
    rounds = sorted(set(f["round"] for f in fixtures_list),
                    key=lambda r: int(r.split(" - ")[-1]) if " - " in r else 0)

    # Auto-detect next jornada
    next_round = detect_next_jornada(fixtures_list, rounds)
    default_idx = rounds.index(next_round) if next_round in rounds else len(rounds) - 1

    # Jornada number for display
    jornada_num = next_round.split(" - ")[-1] if next_round and " - " in next_round else "?"
    st.markdown(f'<div class="section-title">Jornada {jornada_num} &mdash; Predicciones</div>',
                unsafe_allow_html=True)

    selected_round = st.selectbox("Selecciona jornada", rounds, index=default_idx)

    # Filter fixtures for this round
    round_fixtures = [f for f in fixtures_list if f["round"] == selected_round]

    if round_fixtures:
        # Get odds for this jornada
        fixture_ids = tuple(f["fixture_id"] for f in round_fixtures)
        jornada_odds = get_jornada_odds(fixture_ids)

        # ── Summary table (all matches at a glance) ──────────────
        table_rows = []
        for fx in round_fixtures:
            fid = fx["fixture_id"]
            home_team = fx["home"]
            away_team = fx["away"]

            hf_j = latest_feats.get(home_team)
            af_j = latest_feats.get(away_team)

            has_model = (hf_j is not None and af_j is not None
                         and not np.isnan(hf_j.get("goles_rolling", np.nan))
                         and not np.isnan(af_j.get("goles_rolling", np.nan)))

            has_odds = fid in jornada_odds
            model_probs = None

            if has_model:
                X_j = build_matchup_vector(hf_j, af_j)
                if not np.any(np.isnan(X_j)):
                    X_j_s = scaler.transform(X_j)
                    model_probs = model.predict_proba(X_j_s)[0]
                else:
                    has_model = False

            table_rows.append({
                "fid": fid,
                "home": home_team,
                "away": away_team,
                "has_model": has_model,
                "has_odds": has_odds,
                "model_probs": model_probs,
                "odds": jornada_odds.get(fid),
            })

        # ── Summary: one simple row per match ─────────────────────────
        for r in table_rows:
            if r["has_model"] and r["model_probs"] is not None:
                mp = r["model_probs"]
                j_pred = int(np.argmax(mp))
                j_max = mp[j_pred]

                # Build a human-readable phrase
                if j_pred == 0:
                    if j_max > 0.55:
                        phrase = f"🟢 **Favorito: {r['home']}** (juega de local)"
                    else:
                        phrase = f"🟡 Ligera ventaja para **{r['home']}**, pero puede pasar cualquier cosa"
                elif j_pred == 2:
                    if j_max > 0.55:
                        phrase = f"🔴 **Favorito: {r['away']}** (gana de visitante)"
                    else:
                        phrase = f"🟡 Ligera ventaja para **{r['away']}**, pero el local puede sorprender"
                else:
                    phrase = "🟡 **Partido muy parejo**, dificil de predecir"

                # Odds insight (human-readable)
                odds_phrase = ""
                if r["has_odds"] and r["odds"]:
                    o = r["odds"]
                    diff_home = (mp[0] - o["prob_home"]) * 100
                    diff_away = (mp[2] - o["prob_away"]) * 100
                    if diff_home > 10:
                        odds_phrase = f"El modelo ve mas valor en **{r['home']}** que las casas de apuestas"
                    elif diff_away > 10:
                        odds_phrase = f"El modelo ve mas valor en **{r['away']}** que las casas de apuestas"
                    elif diff_home < -10:
                        odds_phrase = f"Las casas de apuestas favorecen mas a **{r['home']}** que nuestro modelo"
                    elif diff_away < -10:
                        odds_phrase = f"Las casas de apuestas favorecen mas a **{r['away']}** que nuestro modelo"
            else:
                phrase = "Sin datos suficientes para predecir"
                odds_phrase = ""

            st.markdown(f"### {r['home']} vs {r['away']}")
            st.markdown(phrase)
            if odds_phrase:
                st.caption(odds_phrase)
            st.markdown("---")

        # ── Detailed cards (expandable) — native Streamlit ────────
        with st.expander("Ver detalle de probabilidades por partido"):
            for row_start in range(0, len(round_fixtures), 3):
                row_slice = round_fixtures[row_start:row_start + 3]
                cols = st.columns(3)

                for idx, fx in enumerate(row_slice):
                    fid = fx["fixture_id"]
                    home_team = fx["home"]
                    away_team = fx["away"]

                    # Find pre-computed data
                    r = next(rr for rr in table_rows if rr["fid"] == fid)

                    with cols[idx]:
                        st.markdown(f"**{home_team}** vs **{away_team}**")

                        if not r["has_model"] and not r["has_odds"]:
                            st.caption("Sin datos suficientes")
                            st.markdown("---")
                            continue

                        model_probs = r["model_probs"]
                        has_model = r["has_model"]
                        has_odds = r["has_odds"]

                        if has_model and model_probs is not None:
                            j_pred = int(np.argmax(model_probs))
                            j_max = model_probs[j_pred]
                            pred_labels = [home_team, "Empate", away_team]

                            if j_max > 0.55:
                                j_conf = "alta"
                            elif j_max >= 0.45:
                                j_conf = "media"
                            else:
                                j_conf = "baja"

                            st.metric(
                                label="Prediccion",
                                value=pred_labels[j_pred],
                                delta=f"Confianza {j_conf} ({j_max:.0%})",
                            )

                            # Probability bars
                            st.caption(f"Gana {home_team}: {model_probs[0]:.0%}")
                            st.progress(model_probs[0])
                            st.caption(f"Empate: {model_probs[1]:.0%}")
                            st.progress(model_probs[1])
                            st.caption(f"Gana {away_team}: {model_probs[2]:.0%}")
                            st.progress(model_probs[2])
                        else:
                            st.caption("No hay datos suficientes para este partido")

                        if has_odds and r["odds"]:
                            o = r["odds"]
                            bk = o.get("bookmaker", "Casas de apuestas")
                            st.caption(
                                f"{bk}: Local {o['prob_home']:.0%} / "
                                f"Empate {o['prob_draw']:.0%} / "
                                f"Visita {o['prob_away']:.0%}"
                            )

                        st.markdown("---")

    # Show model accuracy comparison if odds model available
    if odds_model is not None:
        with st.expander("Modelo con datos de casas de apuestas"):
            st.markdown(
                f"Tenemos dos versiones del modelo: una que solo usa estadisticas de partidos "
                f"(acierta **{base_cv:.0%}**), y otra que tambien usa informacion de las "
                f"casas de apuestas (acierta **{odds_cv:.0%}**)."
            )
            delta_pp = (odds_cv - base_cv) * 100
            if delta_pp > 0:
                st.markdown(f"Agregar datos de apuestas mejora el modelo en **{delta_pp:.1f} puntos porcentuales**.")
            else:
                st.markdown("Agregar datos de apuestas no mejora significativamente el modelo.")
else:
    st.info("No se encontraron fixtures de Clausura 2026. Ejecuta extract_season.py primero.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PREDICCION INDIVIDUAL (secondary view)
# ══════════════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prediccion individual</div>', unsafe_allow_html=True)

# Build match list from the selected jornada
jornada_match_list = []
try:
    for fx in round_fixtures:
        if fx["home"] in teams_ready and fx["away"] in teams_ready:
            jornada_match_list.append(fx)
except NameError:
    pass

custom_mode = st.checkbox(
    "Partido personalizado (fuera de la jornada)",
    value=len(jornada_match_list) == 0,
)

if not custom_mode and jornada_match_list:
    match_labels = [f"{fx['home']} vs {fx['away']}" for fx in jornada_match_list]
    selected_match = st.selectbox("Selecciona un partido de la jornada", match_labels)
    match_idx = match_labels.index(selected_match)
    home = jornada_match_list[match_idx]["home"]
    away = jornada_match_list[match_idx]["away"]
else:
    col_home, col_vs, col_away = st.columns([5, 2, 5])

    with col_home:
        st.markdown("**LOCAL**")
        home = st.selectbox(
            "Equipo local", teams_ready,
            index=teams_ready.index("Cruz Azul") if "Cruz Azul" in teams_ready else 0,
            label_visibility="collapsed",
        )

    with col_vs:
        st.markdown("")
        st.markdown("### VS")

    with col_away:
        st.markdown("**VISITANTE**")
        default_away = teams_ready.index("Club America") if "Club America" in teams_ready else 1
        away = st.selectbox(
            "Equipo visitante", teams_ready,
            index=default_away,
            label_visibility="collapsed",
        )

# ── Validation ───────────────────────────────────────────────────────
if home == away:
    st.warning("Selecciona dos equipos diferentes.")
    st.stop()

# ── Predict ──────────────────────────────────────────────────────────
hf = latest_feats[home]
af = latest_feats[away]
X_new = build_matchup_vector(hf, af)

if np.any(np.isnan(X_new)):
    st.error("Datos insuficientes para esta combinacion de equipos.")
    st.stop()

X_new_s = scaler.transform(X_new)
probs = model.predict_proba(X_new_s)[0]
pred_idx = int(np.argmax(probs))

labels = ["Victoria local", "Empate", "Victoria visitante"]


# ══════════════════════════════════════════════════════════════════════
# RESULTADO — Tricolor bar + prediction callout + confidence badge
# ══════════════════════════════════════════════════════════════════════

# Tricolor probability bar
pct_l = max(probs[0] * 100, 5)
pct_e = max(probs[1] * 100, 5)
pct_v = max(probs[2] * 100, 5)
total = pct_l + pct_e + pct_v
pct_l_n = pct_l / total * 100
pct_e_n = pct_e / total * 100
pct_v_n = pct_v / total * 100

st.markdown(f"""
<div class="prob-bar-container">
    <div class="prob-bar">
        <div class="seg local" style="width:{pct_l_n:.1f}%">{probs[0]:.0%}</div>
        <div class="seg empate" style="width:{pct_e_n:.1f}%">{probs[1]:.0%}</div>
        <div class="seg visita" style="width:{pct_v_n:.1f}%">{probs[2]:.0%}</div>
    </div>
    <div class="prob-bar-legend">
        <span>{home}</span>
        <span>Empate</span>
        <span>{away}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Prediction callout
pred_class = ["local", "empate", "visita"][pred_idx]
pred_text = {
    0: f"Victoria de {home} (local)",
    1: "Empate",
    2: f"Victoria de {away} (visita)",
}
st.markdown(f"""
<div class="pred-callout {pred_class}">
    <span class="pred-label">Prediccion del modelo</span>
    {pred_text[pred_idx]}
</div>
""", unsafe_allow_html=True)

# Human-readable explanation
max_prob = probs[pred_idx]
if pred_idx == 0:
    if max_prob > 0.55:
        human_phrase = f"El modelo cree que **{home}** tiene buenas posibilidades de ganar en casa."
    else:
        human_phrase = f"El modelo le da una ligera ventaja a **{home}** jugando de local, pero no es seguro."
elif pred_idx == 2:
    if max_prob > 0.55:
        human_phrase = f"El modelo cree que **{away}** puede ganar incluso jugando de visitante."
    else:
        human_phrase = f"El modelo le da una ligera ventaja a **{away}**, pero el local puede complicarle."
else:
    human_phrase = "El modelo no ve un favorito claro. Cualquier resultado es posible."

st.markdown(human_phrase)

# Confidence
if max_prob > 0.55:
    conf_level, conf_css = "ALTA", "alta"
    conf_desc = "El modelo tiene bastante seguridad en esta prediccion."
elif max_prob >= 0.45:
    conf_level, conf_css = "MEDIA", "media"
    conf_desc = "Hay una tendencia, pero no es concluyente."
else:
    conf_level, conf_css = "BAJA", "baja"
    conf_desc = "Partido muy parejo, cualquier resultado es posible."

st.markdown(f"""
<div style="text-align:center;margin:0.8rem 0;">
    <span class="conf-badge {conf_css}">Confianza {conf_level}</span>
    <p class="conf-desc">{conf_desc}</p>
</div>
""", unsafe_allow_html=True)

# ── Reliability (human-readable) ─────────────────────────────────────
result_ok, result_n = BACKTEST_BY_RESULT[pred_idx]
result_acc = result_ok / result_n if result_n > 0 else 0

conf_ok, conf_n = BACKTEST_BY_CONF[conf_level]
conf_acc = conf_ok / conf_n

# Convert to "X de cada 10" format
result_of_10 = round(result_acc * 10)
conf_of_10 = round(conf_acc * 10)

if pred_idx == 1:
    st.warning(
        "**Ojo:** Este modelo casi nunca predice empates, y cuando lo hace, "
        "no suele acertar. Toma esta prediccion con mucha cautela."
    )
else:
    result_labels_es = {0: "victoria local", 1: "empate", 2: "victoria visitante"}
    if result_acc >= 0.7:
        st.info(
            f"**Historial de aciertos:** De cada 10 veces que el modelo predice "
            f"{result_labels_es[pred_idx]}, acierta **{result_of_10}**. "
            f"Con confianza {conf_level.lower()}, acierta **{conf_of_10} de cada 10**."
        )
    elif result_acc >= 0.5:
        st.warning(
            f"**Historial de aciertos:** De cada 10 veces que el modelo predice "
            f"{result_labels_es[pred_idx]}, acierta **{result_of_10}**. "
            f"No es tan seguro — tomalo como orientacion, no como certeza."
        )
    else:
        st.error(
            f"**Historial de aciertos:** De cada 10 veces que el modelo predice "
            f"{result_labels_es[pred_idx]}, solo acierta **{result_of_10}**. "
            f"Esta prediccion es poco confiable."
        )


# ══════════════════════════════════════════════════════════════════════
# CONTEXTO DEL PARTIDO
# ══════════════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Contexto del partido</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

st_home = standings.get(home) if standings else None
st_away = standings.get(away) if standings else None

with c1:
    st.markdown(f"**{home}**")
    racha_h = hf["racha_puntos"]
    goles_h = hf["goles_rolling"]
    if not np.isnan(racha_h):
        st.metric("Racha (ult. 3)", f"{int(racha_h)}/9 pts")
    if not np.isnan(goles_h):
        st.metric("Goles/partido (ult. 5)", f"{goles_h:.1f}")
    if st_home:
        st.metric("Posicion", f"#{st_home['rank']}")
        gd = st_home["goalsDiff"]
        st.metric("Puntos", f"{st_home['points']}", delta=f"GD {gd:+d}")
    else:
        pts_h_fallback = hf["puntos_acum"]
        if not np.isnan(pts_h_fallback):
            st.metric("Puntos acumulados", f"{int(pts_h_fallback)}")

with c2:
    st.markdown(f"**{away}**")
    racha_a = af["racha_puntos"]
    goles_a = af["goles_rolling"]
    if not np.isnan(racha_a):
        st.metric("Racha (ult. 3)", f"{int(racha_a)}/9 pts")
    if not np.isnan(goles_a):
        st.metric("Goles/partido (ult. 5)", f"{goles_a:.1f}")
    if st_away:
        st.metric("Posicion", f"#{st_away['rank']}")
        gd = st_away["goalsDiff"]
        st.metric("Puntos", f"{st_away['points']}", delta=f"GD {gd:+d}")
    else:
        pts_a_fallback = af["puntos_acum"]
        if not np.isnan(pts_a_fallback):
            st.metric("Puntos acumulados", f"{int(pts_a_fallback)}")

# Table comparison
if standings and home in standings and away in standings:
    rank_h = standings[home]["rank"]
    rank_a = standings[away]["rank"]
    pts_h = standings[home]["points"]
    pts_a = standings[away]["points"]
    diff_pts = pts_h - pts_a
    if rank_h < rank_a:
        st.info(f"**{home}** (#{rank_h}) vs **{away}** (#{rank_a}) — diferencia de **{abs(diff_pts)} pts**")
    elif rank_h > rank_a:
        st.info(f"**{away}** (#{rank_a}) vs **{home}** (#{rank_h}) — diferencia de **{abs(diff_pts)} pts**")
    else:
        st.info("Equipos igualados en la tabla")
else:
    diff_pts = hf["puntos_acum"] - af["puntos_acum"]
    if not np.isnan(diff_pts):
        if diff_pts > 0:
            st.info(f"**{home}** va **+{int(diff_pts)} pts** arriba en la tabla")
        elif diff_pts < 0:
            st.info(f"**{away}** va **+{int(abs(diff_pts))} pts** arriba en la tabla")
        else:
            st.info("Equipos igualados en puntos")


# ══════════════════════════════════════════════════════════════════════
# SECTION: ANALISIS DEL ARBITRO
# ══════════════════════════════════════════════════════════════════════
if cards_model is not None:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis del arbitro</div>',
                unsafe_allow_html=True)

    # Find referee for this matchup from upcoming fixtures
    referee_raw = None
    for fx in fixtures_list:
        if fx["home"] == home and fx["away"] == away:
            referee_raw = fixture_referees.get(fx["fixture_id"])
            break

    ref_norm = normalize_referee(referee_raw) if referee_raw else None
    ref_avg = None
    ref_games = 0
    if ref_norm and ref_norm in ref_history:
        ref_avg = np.mean(ref_history[ref_norm])
        ref_games = len(ref_history[ref_norm])

    # Global average as fallback
    global_avgs = [np.mean(v) for v in ref_history.values() if len(v) >= 3]
    global_avg = np.mean(global_avgs) if global_avgs else 4.4

    if referee_raw and ref_avg is not None:
        ref_display = referee_raw.split(",")[0].strip()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Arbitro", ref_display)
        with c2:
            st.metric("Tarjetas/partido", f"{ref_avg:.1f}", delta=f"{ref_games} partidos dirigidos")

        if ref_avg >= 5:
            st.markdown(
                f"**Arbitro de mano dura** — historicamente saca "
                f"{ref_avg:.0f} tarjetas por partido. "
                f"Espera un juego con muchas faltas marcadas."
            )
        elif ref_avg >= 4.2:
            st.markdown(
                f"**Arbitro promedio** — saca alrededor de "
                f"{ref_avg:.0f} tarjetas por partido. Nivel normal de disciplina."
            )
        else:
            st.markdown(
                f"**Arbitro permisivo** — promedia solo "
                f"{ref_avg:.0f} tarjetas por partido. Tiende a dejar jugar."
            )

        pred_ref_avg = ref_avg
    else:
        if referee_raw:
            st.markdown(
                f"Arbitro asignado: **{referee_raw.split(',')[0].strip()}** "
                f"— sin historial suficiente en nuestra base de datos."
            )
        else:
            st.caption(
                "Arbitro no asignado todavia para este partido. "
                "Estimacion con promedio de la liga."
            )
        pred_ref_avg = global_avg

    # Model prediction
    home_yc = team_yc.get(home, np.nan)
    away_yc = team_yc.get(away, np.nan)

    if not np.isnan(home_yc) and not np.isnan(away_yc):
        X_cards = np.array([[home_yc, away_yc, pred_ref_avg]])
        pred_yc = cards_model.predict(X_cards)[0]
        margin = abs(pred_yc - CARDS_LINE)

        if pred_yc > CARDS_LINE:
            st.warning(
                f"Prediccion: **{pred_yc:.1f} tarjetas amarillas** en el partido "
                f"— **Over {CARDS_LINE}** (margen: {margin:.1f})"
            )
        else:
            st.info(
                f"Prediccion: **{pred_yc:.1f} tarjetas amarillas** en el partido "
                f"— **Under {CARDS_LINE}** (margen: {margin:.1f})"
            )
    else:
        st.caption("Datos insuficientes para predecir tarjetas de estos equipos.")


# ══════════════════════════════════════════════════════════════════════
# TABLA DE POSICIONES (expander)
# ══════════════════════════════════════════════════════════════════════
if standings:
    with st.expander("Tabla de posiciones — Clausura 2026"):
        rows = []
        for t in standings.values():
            all_stats = t.get("all", {})
            rows.append({
                "Pos": t["rank"],
                "Equipo": t["team"]["name"],
                "Pts": t["points"],
                "JJ": all_stats.get("played", 0),
                "JG": all_stats.get("win", 0),
                "JE": all_stats.get("draw", 0),
                "JP": all_stats.get("lose", 0),
                "GF": all_stats.get("goals", {}).get("for", 0),
                "GC": all_stats.get("goals", {}).get("against", 0),
                "DG": t["goalsDiff"],
            })
        rows.sort(key=lambda r: r["Pos"])
        df_st = pd.DataFrame(rows)
        st.markdown(df_st.to_markdown(index=False))


# ══════════════════════════════════════════════════════════════════════
# RELIABILITY PROFILE (expander)
# ══════════════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
with st.expander("Historial de aciertos del modelo"):
    bt_ok, bt_n = BACKTEST_TOTAL
    bt_of_10 = round(bt_ok / bt_n * 10)
    st.markdown(
        f"Probamos el modelo con **{bt_n} partidos pasados** del Clausura 2025 "
        f"(jornadas 8 a 13) para ver que tan bien predice."
    )
    st.markdown(f"**En general, acierta {bt_of_10} de cada 10 partidos.**")

    st.markdown("---")
    st.markdown("**Cuando predice que gana el local:**")
    ok_l, n_l = BACKTEST_BY_RESULT[0]
    st.markdown(f"Acierta **{round(ok_l/n_l*10)} de cada 10** veces — es donde mejor funciona.")
    st.progress(ok_l / n_l)

    st.markdown("**Cuando predice que gana el visitante:**")
    ok_v, n_v = BACKTEST_BY_RESULT[2]
    st.markdown(f"Acierta **{round(ok_v/n_v*10)} de cada 10** veces — acierto moderado.")
    st.progress(ok_v / n_v)

    st.markdown("**Cuando predice empate:**")
    ok_e, n_e = BACKTEST_BY_RESULT[1]
    st.markdown("Practicamente **nunca acierta**. El modelo no es bueno para detectar empates.")
    st.progress(0.01)

    st.markdown("---")
    st.markdown("**Segun la confianza del modelo:**")
    for conf_name, conf_desc in [("ALTA", "muy seguro"), ("MEDIA", "algo seguro"), ("BAJA", "poco seguro")]:
        ok_c, n_c = BACKTEST_BY_CONF[conf_name]
        of_10 = round(ok_c / n_c * 10)
        st.markdown(f"Cuando esta {conf_desc} (confianza {conf_name.lower()}): acierta **{of_10} de cada 10**")
        st.progress(ok_c / n_c)

    st.markdown("---")
    st.markdown("**Cuando NO confiar en este modelo:**")
    st.markdown(
        "- **Empates:** El modelo casi nunca los predice, y cuando lo hace, falla. "
        "Si ves prediccion de empate, no te fies mucho.\n"
        "- **Visitante favorito:** Acierta menos de la mitad de las veces. "
        "Si dice que gana el visitante, toma la prediccion con cautela.\n"
        "- **Partidos cerrados:** En juegos donde la diferencia es de 1 gol o menos, "
        "el modelo es casi como lanzar una moneda.\n"
        "- **Goleadas:** Cuando un equipo gana por 2 o mas goles, el modelo "
        "suele haberlo visto venir (acierta 8 de cada 10 veces)."
    )


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    El modelo aprendio patrones de 3 torneos de Liga MX y usa los resultados mas recientes del Clausura 2026 para generar cada prediccion.<br>
    Esto es solo una estimacion — los partidos de futbol son impredecibles por naturaleza.
</div>
""", unsafe_allow_html=True)

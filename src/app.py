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
    EXP8_OVERRIDE_THRESHOLD,
    build_features,
    build_matchup_vector,
    get_latest_features,
    hybrid_predict,
    load_data,
    load_odds_data,
)
from extract_odds import fetch_fixture_odds
from predict_props import normalize_referee, CARDS_LINE

# ── Backtest reliability profile (Clausura J8-J13, walk-forward hybrid) ─
BACKTEST_BY_RESULT = {
    0: (24, 31),   # Local: 77%
    1: (0, 7),     # Empate: 0%
    2: (10, 16),   # Visita: 62%
}
BACKTEST_BY_CONF = {
    "ALTA": (12, 13),   # 92%
    "MEDIA": (14, 25),  # 56%
    "BAJA": (8, 16),    # 50%
}
BACKTEST_TOTAL = (34, 54)  # 63% overall (hybrid Variant D)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Liga MX Predictor", page_icon="⚽", layout="centered")


# ══════════════════════════════════════════════════════════════════════
# CSS GLOBAL — Dark theme
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset & base ─────────────────────────────────────────── */
.block-container {
    padding-top: 1rem !important;
    max-width: 720px !important;
}

/* ── Header hero ──────────────────────────────────────────── */
.header-hero {
    background: linear-gradient(135deg, #0E1117 0%, #161B22 100%);
    border-bottom: 3px solid #C4A43F;
    border-radius: 0 0 16px 16px;
    padding: 2rem 1.5rem 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.header-hero h1 {
    color: #FAFAFA;
    font-size: 2rem;
    margin: 0 0 0.25rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.header-hero .subtitle {
    color: #C4A43F;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.6rem;
}
.header-hero .hero-stat {
    color: rgba(250,250,250,0.55);
    font-size: 0.82rem;
    margin: 0;
}
.header-hero .hero-stat strong {
    color: #00843D;
    font-size: 0.95rem;
}

/* ── Section titles ───────────────────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FAFAFA;
    margin: 1.2rem 0 0.8rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #C4A43F;
    display: inline-block;
}

/* ── Match card ───────────────────────────────────────────── */
.match-card {
    background: #1A1D24;
    border: 1px solid #2A2D35;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    margin-bottom: 0.8rem;
}
.match-card.high-conf {
    border-color: #C4A43F;
    border-width: 2px;
}
.match-card .match-teams {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.match-card .team-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FAFAFA;
    flex: 1;
}
.match-card .team-name.away {
    text-align: right;
}
.match-card .vs-label {
    color: rgba(250,250,250,0.3);
    font-size: 0.8rem;
    font-weight: 400;
    padding: 0 0.6rem;
}
.match-card .pred-result {
    text-align: center;
    font-size: 0.9rem;
    font-weight: 700;
    margin: 0.4rem 0 0;
    color: #FAFAFA;
}

/* ── Confidence badges ────────────────────────────────────── */
.conf-badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    vertical-align: middle;
}
.conf-badge.alta  { background: #00843D; color: #fff; }
.conf-badge.media { background: #C4A43F; color: #1A1D24; }
.conf-badge.baja  { background: #C0392B; color: #fff; }

/* ── Override badge ───────────────────────────────────────── */
.override-badge {
    display: inline-block;
    background: rgba(196,164,63,0.12);
    color: #C4A43F;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 12px;
    border: 1px solid rgba(196,164,63,0.25);
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* ── Tricolor probability bar ─────────────────────────────── */
.prob-bar-container { margin: 0.6rem 0 0.3rem; }
.prob-bar {
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    height: 34px;
}
.prob-bar .seg {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 0.78rem;
    min-width: 34px;
}
.prob-bar .seg.local  { background: #00843D; }
.prob-bar .seg.empate { background: #4A4D55; }
.prob-bar .seg.visita { background: #2563EB; }
.prob-bar-legend {
    display: flex;
    justify-content: space-between;
    margin-top: 0.25rem;
    font-size: 0.68rem;
    color: rgba(250,250,250,0.4);
}

/* ── Prediction callout (individual section) ──────────────── */
.pred-callout {
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    margin: 0.8rem 0;
    color: #fff;
    font-weight: 700;
    font-size: 1.05rem;
}
.pred-callout.local  { background: #00843D; }
.pred-callout.empate { background: #4A4D55; }
.pred-callout.visita { background: #2563EB; }
.pred-callout .pred-label {
    font-size: 0.72rem;
    font-weight: 400;
    opacity: 0.8;
    display: block;
    margin-bottom: 0.15rem;
}

/* ── Footer ───────────────────────────────────────────────── */
.app-footer {
    background: #1A1D24;
    border-top: 2px solid #2A2D35;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
    margin-top: 2rem;
    color: rgba(250,250,250,0.35);
    font-size: 0.72rem;
}

/* ── Mobile responsive ────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { max-width: 100% !important; }
    .header-hero h1 { font-size: 1.5rem; }
    .header-hero .subtitle { font-size: 0.9rem; }
    .match-card .team-name { font-size: 0.92rem; }
    .prob-bar { height: 28px; }
    .prob-bar .seg { font-size: 0.68rem; min-width: 28px; }
    .pred-callout { font-size: 0.95rem; }
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


def _train_rf_exp8(X, y):
    """Train Exp8 model: deeper RF with balanced classes for away detection."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced", random_state=42,
    )
    model.fit(X_s, y)
    return model, scaler


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

    # Exp8 model (same features, different hyperparams for hybrid logic)
    exp8_model, exp8_scaler = _train_rf_exp8(X_base, y_base)

    # Odds model (26 features) — only if odds data available
    odds_model, odds_scaler, odds_cv = None, None, None
    if odds_df is not None and all(f in matches.columns for f in ODDS_FEATURES):
        matches_odds = matches.dropna(subset=FEATURES_WITH_ODDS).copy()
        if len(matches_odds) >= 20:
            X_odds = matches_odds[FEATURES_WITH_ODDS].values
            y_odds = matches_odds["resultado"].values
            odds_model, odds_scaler, odds_cv = _train_rf(X_odds, y_odds)

    return (base_model, base_scaler, base_cv,
            exp8_model, exp8_scaler,
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
    year = api_season + 1
    path = os.path.join(cache_dir, f"{tournament.lower()}_{year}_fixtures_list.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
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
    """Fetch odds por fixture_id, cachea 1 hora."""
    results = {}
    for fid in fixture_ids:
        odds = fetch_fixture_odds(fid)
        if odds:
            results[fid] = odds
        time.sleep(0.1)
    return results


def detect_next_jornada(fixtures_list, rounds):
    """Find the next jornada that has unplayed matches."""
    for r in rounds:
        round_fx = [f for f in fixtures_list if f["round"] == r]
        has_unplayed = any(f["status"] in ("NS", "TBD", "PST", "CANC", "1H", "HT", "2H")
                          for f in round_fx)
        if has_unplayed:
            return r
    return rounds[-1] if rounds else None


@st.cache_data(show_spinner="Cargando analisis de tarjetas...")
def prepare_cards_model():
    """Train yellow cards model, compute team/referee averages."""
    from sklearn.ensemble import RandomForestRegressor

    df = load_data()
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["yellow_cards"] = pd.to_numeric(df["yellow_cards"], errors="coerce")
    df["ref_norm"] = df["arbitro"].apply(normalize_referee)
    df = df.sort_values(["fecha", "fixture_id"]).reset_index(drop=True)

    team_yc = {}
    for team in df["equipo"].unique():
        tdf = df[df["equipo"] == team].sort_values("fecha")
        avgs = tdf["yellow_cards"].rolling(5, min_periods=5).mean()
        valid = avgs.dropna()
        team_yc[team] = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

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


@st.cache_data(show_spinner=False)
def get_recent_form(team_name, n=5):
    """Last n results as emoji string (W/D/L)."""
    df = load_data()
    df["fecha"] = pd.to_datetime(df["fecha"])
    tdf = df[df["equipo"] == team_name].sort_values("fecha").tail(n)
    icons = {"W": "🟢", "D": "🟡", "L": "🔴"}
    return " ".join(icons.get(r, "⚪") for r in tdf["resultado"])


# ── Load everything ──────────────────────────────────────────────────
(base_model, base_scaler, base_cv,
 exp8_model, exp8_scaler,
 odds_model, odds_scaler, odds_cv,
 latest_feats, all_teams, odds_df) = prepare_model()

try:
    standings = load_standings()
except Exception:
    standings = None

model, scaler, cv_acc = base_model, base_scaler, base_cv

teams_ready = [t for t in all_teams if not np.isnan(latest_feats[t].get("goles_rolling", np.nan))]

fixtures_list = load_fixtures_list(2025, "Clausura")

try:
    cards_model, team_yc, ref_history = prepare_cards_model()
    fixture_referees = load_fixture_referees()
except Exception:
    cards_model, team_yc, ref_history = None, {}, {}
    fixture_referees = {}

# ── Auto-detect jornada (before header for dynamic subtitle) ─────────
if fixtures_list:
    _rounds = sorted(set(f["round"] for f in fixtures_list),
                     key=lambda r: int(r.split(" - ")[-1]) if " - " in r else 0)
    _auto_round = detect_next_jornada(fixtures_list, _rounds)
    _jornada_num = _auto_round.split(" - ")[-1] if _auto_round and " - " in _auto_round else "?"
else:
    _rounds = []
    _auto_round = None
    _jornada_num = "?"


# ══════════════════════════════════════════════════════════════════════
# HEADER HERO
# ══════════════════════════════════════════════════════════════════════
bt_alta_ok, bt_alta_n = BACKTEST_BY_CONF["ALTA"]
bt_alta_pct = round(bt_alta_ok / bt_alta_n * 100)

st.markdown(f"""
<div class="header-hero">
    <h1>Liga MX Predictor</h1>
    <p class="subtitle">Clausura 2026 &mdash; Jornada {_jornada_num}</p>
    <p class="hero-stat"><strong>{bt_alta_pct}%</strong> de acierto en predicciones de alta confianza</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: JORNADA — Auto-detected, card-based
# ══════════════════════════════════════════════════════════════════════
if fixtures_list and _auto_round:
    st.markdown(f'<div class="section-title">Jornada {_jornada_num}</div>',
                unsafe_allow_html=True)

    selected_round = _auto_round
    round_fixtures = [f for f in fixtures_list if f["round"] == selected_round]

    if round_fixtures:
        # Get odds for this jornada
        fixture_ids = tuple(f["fixture_id"] for f in round_fixtures)
        jornada_odds = get_jornada_odds(fixture_ids)

        # ── Compute predictions for all matches ──────────────
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
            overridden = False

            if has_model:
                X_j = build_matchup_vector(hf_j, af_j)
                if not np.any(np.isnan(X_j)):
                    X_j_s = scaler.transform(X_j)
                    model_probs = model.predict_proba(X_j_s)[0]
                    X_j_s_exp8 = exp8_scaler.transform(X_j)
                    exp8_probs = exp8_model.predict_proba(X_j_s_exp8)[0]
                    _, _, overridden = hybrid_predict(model_probs, exp8_probs)
                else:
                    has_model = False

            table_rows.append({
                "fid": fid,
                "home": home_team,
                "away": away_team,
                "has_model": has_model,
                "has_odds": has_odds,
                "model_probs": model_probs,
                "overridden": overridden,
                "odds": jornada_odds.get(fid),
            })

        # ── Render match cards ───────────────────────────────
        for r in table_rows:
            if not r["has_model"] or r["model_probs"] is None:
                st.markdown(f"""
                <div class="match-card">
                    <div class="match-teams">
                        <span class="team-name">{r['home']}</span>
                        <span class="vs-label">vs</span>
                        <span class="team-name away">{r['away']}</span>
                    </div>
                    <div class="pred-result" style="color:rgba(250,250,250,0.35);">Sin datos suficientes</div>
                </div>
                """, unsafe_allow_html=True)
                continue

            mp = r["model_probs"]
            is_override = r.get("overridden", False)

            if is_override:
                j_pred = 2
            else:
                j_pred = int(np.argmax(mp))
            j_max = mp[j_pred]

            if j_max > 0.55:
                conf_css = "alta"
            elif j_max >= 0.45:
                conf_css = "media"
            else:
                conf_css = "baja"

            card_class = "match-card high-conf" if conf_css == "alta" else "match-card"

            pred_labels = {0: r["home"], 1: "Empate", 2: r["away"]}
            pred_text = pred_labels[j_pred]

            override_html = '<span class="override-badge">&#x1f504; Segunda opinion</span>' if is_override else ''

            # Tricolor bar widths
            pct_l = max(mp[0] * 100, 5)
            pct_e = max(mp[1] * 100, 5)
            pct_v = max(mp[2] * 100, 5)
            total = pct_l + pct_e + pct_v
            pct_l_n = pct_l / total * 100
            pct_e_n = pct_e / total * 100
            pct_v_n = pct_v / total * 100

            st.markdown(f"""
            <div class="{card_class}">
                <div class="match-teams">
                    <span class="team-name">{r['home']}</span>
                    <span class="vs-label">vs</span>
                    <span class="team-name away">{r['away']}</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar">
                        <div class="seg local" style="width:{pct_l_n:.1f}%">{mp[0]:.0%}</div>
                        <div class="seg empate" style="width:{pct_e_n:.1f}%">{mp[1]:.0%}</div>
                        <div class="seg visita" style="width:{pct_v_n:.1f}%">{mp[2]:.0%}</div>
                    </div>
                    <div class="prob-bar-legend">
                        <span>{r['home']}</span>
                        <span>Empate</span>
                        <span>{r['away']}</span>
                    </div>
                </div>
                <div class="pred-result">
                    {pred_text} &nbsp;
                    <span class="conf-badge {conf_css}">{conf_css.upper()}</span>
                    {override_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("No se encontraron fixtures de Clausura 2026.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PREDICCION INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Prediccion individual</div>', unsafe_allow_html=True)

# Build match list from the auto-detected jornada
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

# Hybrid Variant D
X_new_s_exp8 = exp8_scaler.transform(X_new)
exp8_probs = exp8_model.predict_proba(X_new_s_exp8)[0]
pred_idx, _, individual_overridden = hybrid_predict(probs, exp8_probs)


# ══════════════════════════════════════════════════════════════════════
# RESULTADO — Tricolor bar + prediction callout + confidence
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

if individual_overridden:
    st.caption(
        "🔄 Segunda opinion: modelo secundario detecta ventaja visitante."
    )

# Confidence
max_prob = probs[pred_idx]
if max_prob > 0.55:
    conf_level, conf_css = "ALTA", "alta"
elif max_prob >= 0.45:
    conf_level, conf_css = "MEDIA", "media"
else:
    conf_level, conf_css = "BAJA", "baja"

st.markdown(f"""
<div style="text-align:center;margin:0.6rem 0;">
    <span class="conf-badge {conf_css}">Confianza {conf_level}</span>
</div>
""", unsafe_allow_html=True)

# Human-readable explanation
if pred_idx == 0:
    if max_prob > 0.55:
        human_phrase = f"El modelo cree que **{home}** tiene buenas posibilidades de ganar en casa."
    else:
        human_phrase = f"Ligera ventaja para **{home}** jugando de local, pero no es seguro."
elif pred_idx == 2:
    if max_prob > 0.55:
        human_phrase = f"El modelo cree que **{away}** puede ganar incluso de visitante."
    else:
        human_phrase = f"Ligera ventaja para **{away}**, pero el local puede complicarle."
else:
    human_phrase = "El modelo no ve un favorito claro. Cualquier resultado es posible."

st.markdown(human_phrase)

# ── Reliability (per-prediction) ─────────────────────────────────────
result_ok, result_n = BACKTEST_BY_RESULT[pred_idx]
result_acc = result_ok / result_n if result_n > 0 else 0

if pred_idx == 1:
    st.warning(
        "**Ojo:** Este modelo casi nunca predice empates, y cuando lo hace, "
        "no suele acertar."
    )
else:
    result_of_10 = round(result_acc * 10)
    result_labels_es = {0: "victoria local", 1: "empate", 2: "victoria visitante"}
    if result_acc >= 0.7:
        st.info(
            f"Cuando predice {result_labels_es[pred_idx]}, acierta "
            f"**{result_of_10} de cada 10** veces."
        )
    elif result_acc >= 0.5:
        st.warning(
            f"Cuando predice {result_labels_es[pred_idx]}, acierta "
            f"**{result_of_10} de cada 10**. Tomalo como orientacion."
        )
    else:
        st.error(
            f"Cuando predice {result_labels_es[pred_idx]}, solo acierta "
            f"**{result_of_10} de cada 10**. Poco confiable."
        )


# ══════════════════════════════════════════════════════════════════════
# CONTEXTO DEL PARTIDO
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Contexto del partido</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

st_home = standings.get(home) if standings else None
st_away = standings.get(away) if standings else None

with c1:
    st.markdown(f"**{home}**")
    st.caption(f"Ultimos 5: {get_recent_form(home)}")
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
    st.caption(f"Ultimos 5: {get_recent_form(away)}")
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
    pts_diff = abs(standings[home]["points"] - standings[away]["points"])
    st.caption(f"#{rank_h} vs #{rank_a} en la tabla ({pts_diff} pts de diferencia)")
else:
    diff_pts = hf["puntos_acum"] - af["puntos_acum"]
    if not np.isnan(diff_pts) and diff_pts != 0:
        leader = home if diff_pts > 0 else away
        st.caption(f"**{leader}** va +{int(abs(diff_pts))} pts arriba en la tabla")


# ══════════════════════════════════════════════════════════════════════
# ANALISIS DEL ARBITRO (collapsible)
# ══════════════════════════════════════════════════════════════════════
if cards_model is not None:
    st.markdown("---")
    with st.expander("Analisis del arbitro y tarjetas"):
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

        global_avgs = [np.mean(v) for v in ref_history.values() if len(v) >= 3]
        global_avg = np.mean(global_avgs) if global_avgs else 4.4

        if referee_raw and ref_avg is not None:
            ref_display = referee_raw.split(",")[0].strip()

            rc1, rc2 = st.columns(2)
            with rc1:
                st.metric("Arbitro", ref_display)
            with rc2:
                st.metric("Tarjetas/partido", f"{ref_avg:.1f}", delta=f"{ref_games} partidos")

            if ref_avg >= 5:
                st.markdown(f"**Mano dura** — promedia {ref_avg:.0f} tarjetas por partido.")
            elif ref_avg >= 4.2:
                st.markdown(f"**Promedio** — ~{ref_avg:.0f} tarjetas por partido.")
            else:
                st.markdown(f"**Permisivo** — solo {ref_avg:.0f} tarjetas por partido.")

            pred_ref_avg = ref_avg
        else:
            if referee_raw:
                st.markdown(
                    f"Arbitro: **{referee_raw.split(',')[0].strip()}** "
                    f"— sin historial suficiente."
                )
            else:
                st.caption("Arbitro no asignado. Estimacion con promedio de la liga.")
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
                    f"Prediccion: **{pred_yc:.1f} tarjetas** — "
                    f"**Over {CARDS_LINE}** (margen: {margin:.1f})"
                )
            else:
                st.info(
                    f"Prediccion: **{pred_yc:.1f} tarjetas** — "
                    f"**Under {CARDS_LINE}** (margen: {margin:.1f})"
                )


# ══════════════════════════════════════════════════════════════════════
# TABLA DE POSICIONES (collapsible)
# ══════════════════════════════════════════════════════════════════════
if standings:
    with st.expander("Tabla de posiciones — Clausura 2026"):
        rows = []
        for t in standings.values():
            all_stats = t.get("all", {})
            team_name = t["team"]["name"]
            # Bold the teams involved in the current matchup
            display_name = f"**{team_name}**" if team_name in (home, away) else team_name
            rows.append({
                "Pos": t["rank"],
                "Equipo": display_name,
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
# HISTORIAL DE ACIERTOS — KPIs + collapsible detail
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Historial de aciertos</div>',
            unsafe_allow_html=True)

bt_ok, bt_n = BACKTEST_TOTAL
ok_l, n_l = BACKTEST_BY_RESULT[0]
ok_v, n_v = BACKTEST_BY_RESULT[2]
ok_alta, n_alta = BACKTEST_BY_CONF["ALTA"]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("General", f"{bt_ok/bt_n:.0%}", delta=f"{bt_n} partidos", delta_color="off")
with kpi2:
    st.metric("Alta conf.", f"{ok_alta/n_alta:.0%}", delta=f"{n_alta} pred.", delta_color="off")
with kpi3:
    st.metric("Local", f"{ok_l/n_l:.0%}", delta=f"{n_l} pred.", delta_color="off")
with kpi4:
    st.metric("Visita", f"{ok_v/n_v:.0%}", delta=f"{n_v} pred.", delta_color="off")

with st.expander("Ver detalle"):
    st.markdown(
        f"Evaluado con **{bt_n} partidos** del Clausura 2025 (jornadas 8-13)."
    )
    st.markdown("**Por tipo de prediccion:**")
    st.markdown(f"- Local: **{ok_l}/{n_l}** ({ok_l/n_l:.0%})")
    st.markdown(f"- Visita: **{ok_v}/{n_v}** ({ok_v/n_v:.0%})")
    st.markdown("- Empate: no confiable (el modelo rara vez predice empates)")

    st.markdown("**Por nivel de confianza:**")
    for conf_name in ["ALTA", "MEDIA", "BAJA"]:
        ok_c, n_c = BACKTEST_BY_CONF[conf_name]
        st.markdown(f"- {conf_name}: **{ok_c}/{n_c}** ({ok_c/n_c:.0%})")


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    Esto es solo una estimacion &mdash; los partidos de futbol son impredecibles por naturaleza.
</div>
""", unsafe_allow_html=True)

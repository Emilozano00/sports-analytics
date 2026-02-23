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

/* ── Match card ───────────────────────────────────────────────── */
.match-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}
.match-card-header {
    text-align: center;
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
    font-weight: 600;
}
.vs-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: #c9a84c;
    color: #ffffff;
    font-weight: 800;
    font-size: 1rem;
    margin: 0 auto;
    box-shadow: 0 2px 8px rgba(201,168,76,0.3);
}
.team-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
}
.team-label.home { color: #2d8a4e; }
.team-label.away { color: #c0392b; }

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

/* ── Context card ─────────────────────────────────────────────── */
.ctx-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin: 1rem 0;
}
.ctx-header {
    padding: 0.6rem 1rem;
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
}
.ctx-header.home { background: #2d8a4e; }
.ctx-header.away { background: #c0392b; }
.ctx-body { padding: 0.8rem 1rem; }
.ctx-stat {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid #f0f2f6;
    font-size: 0.88rem;
}
.ctx-stat:last-child { border-bottom: none; }
.ctx-stat .label { color: #888; }
.ctx-stat .value { font-weight: 700; color: #1a472a; }

/* ── Table comparison bar ─────────────────────────────────────── */
.table-compare {
    background: #f0f2f6;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    font-size: 0.9rem;
    color: #444;
    margin: 0.5rem 0 1rem;
}

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

/* ── Jornada card ─────────────────────────────────────────────── */
.jornada-card {
    background: #fff;
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    padding: 1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.15s ease;
}
.jornada-card:hover { transform: translateY(-2px); }
.jornada-card.border-local  { border-color: #2d8a4e; }
.jornada-card.border-empate { border-color: #f0a500; }
.jornada-card.border-visita { border-color: #c0392b; }
.jornada-card .teams {
    font-weight: 700;
    font-size: 0.95rem;
    color: #1a472a;
    margin-bottom: 0.5rem;
    text-align: center;
}
.jornada-card .mini-bar {
    display: flex;
    border-radius: 6px;
    overflow: hidden;
    height: 22px;
    margin-bottom: 0.5rem;
}
.jornada-card .mini-bar .seg {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    min-width: 28px;
}
.jornada-card .mini-bar .seg.local  { background: #2d8a4e; }
.jornada-card .mini-bar .seg.empate { background: #f0a500; }
.jornada-card .mini-bar .seg.visita { background: #c0392b; }
.jornada-card .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.jornada-card .mini-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
}
.jornada-card .mini-badge.alta  { background: #2d8a4e; color: #fff; }
.jornada-card .mini-badge.media { background: #f0a500; color: #333; }
.jornada-card .mini-badge.baja  { background: #c0392b; color: #fff; }
.jornada-card .pred-text {
    font-size: 0.8rem;
    color: #666;
    font-weight: 600;
}

/* ── Odds comparison in jornada ───────────────────────────────── */
.odds-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #888;
    padding: 0.2rem 0;
}
.odds-row .diff-warn { color: #c0392b; font-weight: 700; }

/* ── Jornada summary table ────────────────────────────────────── */
.jornada-summary {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
.jornada-summary table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.jornada-summary thead th {
    background: #1a472a;
    color: #fff;
    padding: 0.7rem 0.6rem;
    text-align: center;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.jornada-summary thead th:first-child { text-align: left; padding-left: 1rem; }
.jornada-summary tbody td {
    padding: 0.6rem;
    text-align: center;
    border-bottom: 1px solid #f0f2f6;
}
.jornada-summary tbody td:first-child { text-align: left; padding-left: 1rem; font-weight: 600; color: #1a472a; }
.jornada-summary tbody tr:last-child td { border-bottom: none; }
.jornada-summary tbody tr:hover { background: #f8f9fa; }
.jornada-summary .pred-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #fff;
}
.jornada-summary .pred-pill.local  { background: #2d8a4e; }
.jornada-summary .pred-pill.empate { background: #f0a500; color: #333; }
.jornada-summary .pred-pill.visita { background: #c0392b; }
.jornada-summary .conf-pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.68rem;
    font-weight: 700;
}
.jornada-summary .conf-pill.alta  { background: rgba(45,138,78,0.15); color: #2d8a4e; }
.jornada-summary .conf-pill.media { background: rgba(240,165,0,0.15); color: #b8860b; }
.jornada-summary .conf-pill.baja  { background: rgba(192,57,43,0.15); color: #c0392b; }

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
    .jornada-card .teams { font-size: 0.85rem; }
    .jornada-summary { font-size: 0.82rem; }
    .jornada-summary thead th { font-size: 0.72rem; padding: 0.5rem 0.3rem; }
    .jornada-summary tbody td { padding: 0.5rem 0.3rem; font-size: 0.8rem; }
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


# ══════════════════════════════════════════════════════════════════════
# HEADER HERO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-hero">
    <h1>Liga MX Predictor</h1>
    <p class="subtitle">Clausura 2026</p>
    <span class="model-tag">Random Forest v6 &middot; 200 arboles</span>
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

        # Build HTML summary table
        html_rows = ""
        for r in table_rows:
            matchup = f"{r['home']} vs {r['away']}"

            if r["has_model"] and r["model_probs"] is not None:
                mp = r["model_probs"]
                j_pred = int(np.argmax(mp))
                j_max = mp[j_pred]

                pred_labels = {0: "L", 1: "E", 2: "V"}
                pred_css = ["local", "empate", "visita"][j_pred]
                pred_full = {0: r["home"], 1: "Empate", 2: r["away"]}

                if j_max > 0.55:
                    conf_lbl, conf_css = "ALTA", "alta"
                elif j_max >= 0.45:
                    conf_lbl, conf_css = "MEDIA", "media"
                else:
                    conf_lbl, conf_css = "BAJA", "baja"

                probs_cell = f"{mp[0]:.0%} / {mp[1]:.0%} / {mp[2]:.0%}"
                pred_cell = f'<span class="pred-pill {pred_css}">{pred_labels[j_pred]}</span>'
                conf_cell = f'<span class="conf-pill {conf_css}">{conf_lbl}</span>'

                # Odds comparison
                odds_cell = "—"
                if r["has_odds"] and r["odds"]:
                    o = r["odds"]
                    odds_cell = f"{o['prob_home']:.0%} / {o['prob_draw']:.0%} / {o['prob_away']:.0%}"
            else:
                probs_cell = "—"
                pred_cell = "—"
                conf_cell = "—"
                odds_cell = "—"
                if r["has_odds"] and r["odds"]:
                    o = r["odds"]
                    odds_cell = f"{o['prob_home']:.0%} / {o['prob_draw']:.0%} / {o['prob_away']:.0%}"

            html_rows += f"""
            <tr>
                <td>{matchup}</td>
                <td>{probs_cell}</td>
                <td>{pred_cell}</td>
                <td>{conf_cell}</td>
                <td>{odds_cell}</td>
            </tr>
            """

        st.markdown(f"""
        <div class="jornada-summary">
            <table>
                <thead>
                    <tr>
                        <th>Partido</th>
                        <th>L / E / V</th>
                        <th>Pred</th>
                        <th>Conf</th>
                        <th>Mercado</th>
                    </tr>
                </thead>
                <tbody>
                    {html_rows}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Detailed cards (expandable) ──────────────────────────
        with st.expander("Ver tarjetas detalladas por partido"):
            for row_start in range(0, len(round_fixtures), 3):
                row_fixtures = round_fixtures[row_start:row_start + 3]
                cols = st.columns(3)

                for idx, fx in enumerate(row_fixtures):
                    fid = fx["fixture_id"]
                    home_team = fx["home"]
                    away_team = fx["away"]

                    # Find pre-computed data
                    r = next(rr for rr in table_rows if rr["fid"] == fid)

                    with cols[idx]:
                        if not r["has_model"] and not r["has_odds"]:
                            st.markdown(f"""
                            <div class="jornada-card">
                                <div class="teams">{home_team} vs {away_team}</div>
                                <div style="text-align:center;color:#888;font-size:0.8rem;">Sin datos</div>
                            </div>
                            """, unsafe_allow_html=True)
                            continue

                        model_probs = r["model_probs"]
                        has_model = r["has_model"]
                        has_odds = r["has_odds"]

                        if has_model and model_probs is not None:
                            j_pred = int(np.argmax(model_probs))
                            j_max = model_probs[j_pred]
                            border_cls = ["border-local", "border-empate", "border-visita"][j_pred]
                            pred_labels = [f"{home_team}", "Empate", f"{away_team}"]
                            pred_short = pred_labels[j_pred]

                            if j_max > 0.55:
                                j_conf, j_conf_css = "ALTA", "alta"
                            elif j_max >= 0.45:
                                j_conf, j_conf_css = "MEDIA", "media"
                            else:
                                j_conf, j_conf_css = "BAJA", "baja"

                            ml = max(model_probs[0] * 100, 5)
                            me = max(model_probs[1] * 100, 5)
                            mv = max(model_probs[2] * 100, 5)
                            mt = ml + me + mv
                            ml_n = ml / mt * 100
                            me_n = me / mt * 100
                            mv_n = mv / mt * 100

                            mini_bar = f"""
                            <div class="mini-bar">
                                <div class="seg local" style="width:{ml_n:.1f}%">{model_probs[0]:.0%}</div>
                                <div class="seg empate" style="width:{me_n:.1f}%">{model_probs[1]:.0%}</div>
                                <div class="seg visita" style="width:{mv_n:.1f}%">{model_probs[2]:.0%}</div>
                            </div>
                            """
                        else:
                            border_cls = ""
                            mini_bar = ""
                            pred_short = "—"
                            j_conf_css = ""
                            j_conf = ""

                        odds_html = ""
                        if has_odds and has_model and model_probs is not None:
                            o = jornada_odds[fid]
                            bk = o.get("bookmaker", "?")
                            diffs = [
                                (model_probs[0] - o['prob_home']) * 100,
                                (model_probs[1] - o['prob_draw']) * 100,
                                (model_probs[2] - o['prob_away']) * 100,
                            ]
                            diff_parts = []
                            for d, lbl in zip(diffs, ["L", "E", "V"]):
                                cls = ' class="diff-warn"' if abs(d) > 10 else ''
                                diff_parts.append(f"<span{cls}>{lbl} {d:+.0f}pp</span>")
                            odds_html = f"""
                            <div class="odds-row">
                                <span style="font-weight:600;">{bk}</span>
                                {" ".join(diff_parts)}
                            </div>
                            """
                        elif has_odds:
                            o = jornada_odds[fid]
                            bk = o.get("bookmaker", "?")
                            odds_html = f"""
                            <div class="odds-row">
                                <span>{bk}: L {o['prob_home']:.0%} E {o['prob_draw']:.0%} V {o['prob_away']:.0%}</span>
                            </div>
                            """

                        card_footer = ""
                        if has_model:
                            card_footer = f"""
                            <div class="card-footer">
                                <span class="pred-text">{pred_short}</span>
                                <span class="mini-badge {j_conf_css}">{j_conf}</span>
                            </div>
                            """

                        st.markdown(f"""
                        <div class="jornada-card {border_cls}">
                            <div class="teams">{home_team} vs {away_team}</div>
                            {mini_bar}
                            {odds_html}
                            {card_footer}
                        </div>
                        """, unsafe_allow_html=True)

    # Show model accuracy comparison if odds model available
    if odds_model is not None:
        with st.expander("Comparacion de modelos (base vs + odds)"):
            st.markdown(f"**Modelo base** (23 features): {base_cv:.0%} accuracy (CV)")
            st.markdown(f"**Modelo + odds** (26 features): {odds_cv:.0%} accuracy (CV)")
            delta_pp = (odds_cv - base_cv) * 100
            st.markdown(f"**Delta:** {delta_pp:+.1f} pp")
else:
    st.info("No se encontraron fixtures de Clausura 2026. Ejecuta extract_season.py primero.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: PREDICCION INDIVIDUAL (secondary view)
# ══════════════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prediccion individual</div>', unsafe_allow_html=True)

st.markdown('<div class="match-card">', unsafe_allow_html=True)
st.markdown('<div class="match-card-header">Selecciona los equipos</div>', unsafe_allow_html=True)

col_home, col_vs, col_away = st.columns([5, 2, 5])

with col_home:
    st.markdown('<p class="team-label home">LOCAL</p>', unsafe_allow_html=True)
    home = st.selectbox("Equipo local", teams_ready,
                        index=teams_ready.index("Cruz Azul"),
                        label_visibility="collapsed")

with col_vs:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:1.2rem;">
        <div class="vs-circle">VS</div>
    </div>
    """, unsafe_allow_html=True)

with col_away:
    st.markdown('<p class="team-label away">VISITANTE</p>', unsafe_allow_html=True)
    default_away = teams_ready.index("Club America") if "Club America" in teams_ready else 1
    away = st.selectbox("Equipo visitante", teams_ready,
                        index=default_away,
                        label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

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

# Confidence
max_prob = probs[pred_idx]
if max_prob > 0.55:
    conf_level, conf_css = "ALTA", "alta"
    conf_desc = "La senal del modelo es clara."
elif max_prob >= 0.45:
    conf_level, conf_css = "MEDIA", "media"
    conf_desc = "Hay una tendencia pero no es concluyente."
else:
    conf_level, conf_css = "BAJA", "baja"
    conf_desc = "Partido muy parejo, cualquier resultado es posible."

st.markdown(f"""
<div style="text-align:center;margin:0.8rem 0;">
    <span class="conf-badge {conf_css}">{conf_level}</span>
    <p class="conf-desc">{conf_desc}</p>
</div>
""", unsafe_allow_html=True)

# ── Reliability inline ───────────────────────────────────────────────
result_ok, result_n = BACKTEST_BY_RESULT[pred_idx]
result_acc = result_ok / result_n if result_n > 0 else 0
result_labels_es = {0: "victoria local", 1: "empate", 2: "victoria visitante"}

conf_ok, conf_n = BACKTEST_BY_CONF[conf_level]
conf_acc = conf_ok / conf_n

if pred_idx == 1:
    st.markdown(f"""
    <div class="reliability-inline danger">
        <strong>Atencion:</strong> En el backtest (52 partidos, Clausura J8-J13), el modelo
        acerto <strong>0% de los empates</strong> ({result_ok}/{result_n}).
        Esta prediccion es poco fiable — el modelo casi nunca predice empates y cuando lo hace, no acierta.
    </div>
    """, unsafe_allow_html=True)
else:
    if result_acc >= 0.7:
        border_class = ""
    elif result_acc >= 0.5:
        border_class = "warn"
    else:
        border_class = "danger"
    st.markdown(f"""
    <div class="reliability-inline {border_class}">
        <strong>Fiabilidad historica:</strong> Cuando predice {result_labels_es[pred_idx]},
        acierta <strong>{result_acc:.0%}</strong> ({result_ok}/{result_n}) en backtest.
        Con confianza {conf_level}, acierta <strong>{conf_acc:.0%}</strong> ({conf_ok}/{conf_n}).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CONTEXTO DEL PARTIDO
# ══════════════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Contexto del partido</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)


def _build_ctx_html(team_name, feats, standing, is_home=True):
    """Build HTML for a context card side."""
    side = "home" if is_home else "away"
    rows = ""

    racha = feats["racha_puntos"]
    if not np.isnan(racha):
        rows += f'<div class="ctx-stat"><span class="label">Racha (ult. 3)</span><span class="value">{int(racha)}/9 pts</span></div>'

    goles = feats["goles_rolling"]
    if not np.isnan(goles):
        rows += f'<div class="ctx-stat"><span class="label">Goles/partido (ult. 5)</span><span class="value">{goles:.1f}</span></div>'

    if standing:
        rows += f'<div class="ctx-stat"><span class="label">Posicion</span><span class="value">#{standing["rank"]}</span></div>'
        gd = standing["goalsDiff"]
        gd_str = f"+{gd}" if gd > 0 else str(gd)
        rows += f'<div class="ctx-stat"><span class="label">Puntos</span><span class="value">{standing["points"]} (GD {gd_str})</span></div>'
    else:
        pts = feats["puntos_acum"]
        if not np.isnan(pts):
            rows += f'<div class="ctx-stat"><span class="label">Puntos acum.</span><span class="value">{int(pts)}</span></div>'

    return f"""
    <div class="ctx-card">
        <div class="ctx-header {side}">{team_name}</div>
        <div class="ctx-body">{rows}</div>
    </div>
    """


st_home = standings.get(home) if standings else None
st_away = standings.get(away) if standings else None

with c1:
    st.markdown(_build_ctx_html(home, hf, st_home, is_home=True), unsafe_allow_html=True)

with c2:
    st.markdown(_build_ctx_html(away, af, st_away, is_home=False), unsafe_allow_html=True)

# Table comparison bar
if standings and home in standings and away in standings:
    rank_h = standings[home]["rank"]
    rank_a = standings[away]["rank"]
    pts_h = standings[home]["points"]
    pts_a = standings[away]["points"]
    diff_pts = pts_h - pts_a
    if rank_h < rank_a:
        compare_txt = f"<strong>{home}</strong> (#{rank_h}) vs <strong>{away}</strong> (#{rank_a}) &mdash; diferencia de <strong>{abs(diff_pts)} pts</strong>"
    elif rank_h > rank_a:
        compare_txt = f"<strong>{away}</strong> (#{rank_a}) vs <strong>{home}</strong> (#{rank_h}) &mdash; diferencia de <strong>{abs(diff_pts)} pts</strong>"
    else:
        compare_txt = "Equipos igualados en la tabla"
    st.markdown(f'<div class="table-compare">{compare_txt}</div>', unsafe_allow_html=True)
else:
    diff_pts = hf["puntos_acum"] - af["puntos_acum"]
    if not np.isnan(diff_pts):
        if diff_pts > 0:
            compare_txt = f"<strong>{home}</strong> va <strong>+{int(diff_pts)} pts</strong> arriba en la tabla"
        elif diff_pts < 0:
            compare_txt = f"<strong>{away}</strong> va <strong>+{int(abs(diff_pts))} pts</strong> arriba en la tabla"
        else:
            compare_txt = "Equipos igualados en puntos"
        st.markdown(f'<div class="table-compare">{compare_txt}</div>', unsafe_allow_html=True)


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
with st.expander("Perfil de confiabilidad del modelo (backtest)"):
    st.markdown("**Backtest walk-forward:** Clausura 2025, Jornadas 8-13 (52 partidos)")
    bt_ok, bt_n = BACKTEST_TOTAL
    st.markdown(f"**Accuracy general:** {bt_ok}/{bt_n} ({bt_ok/bt_n:.0%})")

    st.markdown("**Por tipo de resultado predicho:**")
    res_names = {0: "Victoria local", 1: "Empate", 2: "Victoria visitante"}
    for cls in [0, 1, 2]:
        ok, n = BACKTEST_BY_RESULT[cls]
        acc = ok / n if n > 0 else 0
        bar_w = max(acc * 100, 2)
        bar_color = ["#2d8a4e", "#f0a500", "#c0392b"][cls]
        st.markdown(f"""
        <div style="margin:0.4rem 0;">
            <span style="font-weight:600;">{res_names[cls]}: {ok}/{n} ({acc:.0%})</span>
            <div style="background:#e0e0e0;border-radius:6px;height:14px;margin-top:3px;">
                <div style="width:{bar_w}%;background:{bar_color};height:100%;border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Por nivel de confianza:**")
    for conf in ["ALTA", "MEDIA", "BAJA"]:
        ok, n = BACKTEST_BY_CONF[conf]
        acc = ok / n
        bar_w = max(acc * 100, 2)
        bar_color = {"ALTA": "#2d8a4e", "MEDIA": "#f0a500", "BAJA": "#c0392b"}[conf]
        st.markdown(f"""
        <div style="margin:0.4rem 0;">
            <span style="font-weight:600;">{conf}: {ok}/{n} ({acc:.0%})</span>
            <div style="background:#e0e0e0;border-radius:6px;height:14px;margin-top:3px;">
                <div style="width:{bar_w}%;background:{bar_color};height:100%;border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "**Puntos ciegos del modelo:**\n"
        "- Los empates son practicamente indetectables (0% acierto)\n"
        "- Victorias visitantes: acierto moderado (43%)\n"
        "- Partidos cerrados (diferencia <= 1 gol): 52% accuracy\n"
        "- Goleadas (diferencia >= 2 goles): 84% accuracy"
    )


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    Modelo: Random Forest (200 arboles, depth=3) entrenado con Apertura 2025 + Clausura 2025.<br>
    Los resultados son estimaciones estadisticas, no garantias. Usa esta informacion de forma responsable.
</div>
""", unsafe_allow_html=True)

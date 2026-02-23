"""
Liga MX Match Predictor — Streamlit UI
Uso: streamlit run src/app.py
"""

import os
import sys
import json
import time
import warnings

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
# Result type: {pred_class: (aciertos, total)}
BACKTEST_BY_RESULT = {
    0: (29, 31),   # Local: 94%
    1: (0, 7),     # Empate: 0%
    2: (6, 14),    # Visita: 43%
}
# Confidence level: {conf: (aciertos, total)}
BACKTEST_BY_CONF = {
    "ALTA": (11, 15),   # 73%
    "MEDIA": (14, 21),  # 67%
    "BAJA": (10, 16),   # 62%
}
BACKTEST_TOTAL = (35, 52)  # 67% overall

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Liga MX Predictor", page_icon="⚽", layout="centered")


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

# ── Header ───────────────────────────────────────────────────────────
st.markdown("# ⚽ Liga MX Predictor")
st.markdown("Prediccion de resultado basada en Random Forest (modelo v6)")
st.markdown("---")

# ── Team selectors ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏠 Local")
    home = st.selectbox("Equipo local", teams_ready, index=teams_ready.index("Cruz Azul"), label_visibility="collapsed")

with col2:
    st.markdown("### ✈️ Visitante")
    default_away = teams_ready.index("Club America") if "Club America" in teams_ready else 1
    away = st.selectbox("Equipo visitante", teams_ready, index=default_away, label_visibility="collapsed")

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
colors = ["#00d4aa", "#f0a500", "#ff6b6b"]
icons = ["🏠", "🤝", "✈️"]

# ── Results ──────────────────────────────────────────────────────────
st.markdown("---")

# Probability bars
for i in range(3):
    col_icon, col_bar, col_pct = st.columns([0.8, 8, 1.5])
    with col_icon:
        st.markdown(f"### {icons[i]}")
    with col_bar:
        st.markdown(f"**{labels[i]}**")
        st.progress(probs[i])
    with col_pct:
        st.markdown(f"### {probs[i]:.0%}")

st.markdown("")

# Prediction callout
pred_text = {
    0: f"**Victoria de {home}** (local)",
    1: "**Empate**",
    2: f"**Victoria de {away}** (visita)",
}

max_prob = probs[pred_idx]
if max_prob > 0.55:
    conf_level, conf_color = "ALTA", "green"
    conf_desc = "La señal del modelo es clara."
elif max_prob >= 0.45:
    conf_level, conf_color = "MEDIA", "orange"
    conf_desc = "Hay una tendencia pero no es concluyente."
else:
    conf_level, conf_color = "BAJA", "red"
    conf_desc = "Partido muy parejo, cualquier resultado es posible."

st.success(f"**Prediccion:** {pred_text[pred_idx]}")
st.markdown(f"**Confianza:** :{conf_color}[{conf_level}] — {conf_desc}")

# ── Reliability profile for this specific prediction ─────────────
result_ok, result_n = BACKTEST_BY_RESULT[pred_idx]
result_acc = result_ok / result_n if result_n > 0 else 0
result_labels_es = {0: "victoria local", 1: "empate", 2: "victoria visitante"}

conf_ok, conf_n = BACKTEST_BY_CONF[conf_level]
conf_acc = conf_ok / conf_n

if pred_idx == 1:
    st.error(
        f"**Atencion:** En el backtest (52 partidos, Clausura J8-J13), el modelo "
        f"acerto **0% de los empates** ({result_ok}/{result_n}). "
        f"Esta prediccion es poco fiable — el modelo casi nunca predice empates "
        f"y cuando lo hace, no acierta."
    )
else:
    if result_acc >= 0.7:
        st.info(
            f"**Fiabilidad historica:** Cuando predice {result_labels_es[pred_idx]}, "
            f"acierta **{result_acc:.0%}** ({result_ok}/{result_n}) en backtest. "
            f"Con confianza {conf_level}, acierta **{conf_acc:.0%}** ({conf_ok}/{conf_n})."
        )
    elif result_acc >= 0.5:
        st.warning(
            f"**Fiabilidad historica:** Cuando predice {result_labels_es[pred_idx]}, "
            f"acierta **{result_acc:.0%}** ({result_ok}/{result_n}) en backtest. "
            f"Con confianza {conf_level}, acierta **{conf_acc:.0%}** ({conf_ok}/{conf_n})."
        )
    else:
        st.error(
            f"**Fiabilidad historica:** Cuando predice {result_labels_es[pred_idx]}, "
            f"solo acierta **{result_acc:.0%}** ({result_ok}/{result_n}) en backtest. "
            f"Con confianza {conf_level}, acierta **{conf_acc:.0%}** ({conf_ok}/{conf_n}). "
            f"Tomar con precaucion."
        )

# ── Match context ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Contexto del partido")

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**🏠 {home}**")
    racha_h = hf["racha_puntos"]
    goles_h = hf["goles_rolling"]
    if not np.isnan(racha_h):
        st.metric("Racha (ult. 3)", f"{int(racha_h)}/9 pts")
    if not np.isnan(goles_h):
        st.metric("Goles/partido (ult. 5)", f"{goles_h:.1f}")
    if standings and home in standings:
        sh = standings[home]
        rank_h = sh["rank"]
        pts_h = sh["points"]
        gd_h = sh["goalsDiff"]
        st.metric("Posicion", f"#{rank_h}")
        st.metric("Puntos", f"{pts_h}", delta=f"GD {gd_h:+d}")
    else:
        pts_h_fallback = hf["puntos_acum"]
        if not np.isnan(pts_h_fallback):
            st.metric("Puntos acumulados", f"{int(pts_h_fallback)}")

with c2:
    st.markdown(f"**✈️ {away}**")
    racha_a = af["racha_puntos"]
    goles_a = af["goles_rolling"]
    if not np.isnan(racha_a):
        st.metric("Racha (ult. 3)", f"{int(racha_a)}/9 pts")
    if not np.isnan(goles_a):
        st.metric("Goles/partido (ult. 5)", f"{goles_a:.1f}")
    if standings and away in standings:
        sa = standings[away]
        rank_a = sa["rank"]
        pts_a = sa["points"]
        gd_a = sa["goalsDiff"]
        st.metric("Posicion", f"#{rank_a}")
        st.metric("Puntos", f"{pts_a}", delta=f"GD {gd_a:+d}")
    else:
        pts_a_fallback = af["puntos_acum"]
        if not np.isnan(pts_a_fallback):
            st.metric("Puntos acumulados", f"{int(pts_a_fallback)}")

# Delta tabla
if standings and home in standings and away in standings:
    rank_h = standings[home]["rank"]
    rank_a = standings[away]["rank"]
    pts_h = standings[home]["points"]
    pts_a = standings[away]["points"]
    diff_pts = pts_h - pts_a
    if rank_h < rank_a:
        st.info(f"📊 **{home}** (#{rank_h}) vs **{away}** (#{rank_a}) — diferencia de **{abs(diff_pts)} pts**")
    elif rank_h > rank_a:
        st.info(f"📊 **{away}** (#{rank_a}) vs **{home}** (#{rank_h}) — diferencia de **{abs(diff_pts)} pts**")
    else:
        st.info("📊 Equipos igualados en la tabla")
else:
    diff_pts = hf["puntos_acum"] - af["puntos_acum"]
    if not np.isnan(diff_pts):
        if diff_pts > 0:
            st.info(f"📊 **{home}** va **+{int(diff_pts)} pts** arriba en la tabla")
        elif diff_pts < 0:
            st.info(f"📊 **{away}** va **+{int(abs(diff_pts))} pts** arriba en la tabla")
        else:
            st.info("📊 Equipos igualados en puntos")

# ── Standings table (expandable) ──────────────────────────────────────
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
        st.dataframe(pd.DataFrame(rows).astype(str))

# ── Reliability profile (expandable) ─────────────────────────────────
st.markdown("---")
with st.expander("Perfil de confiabilidad del modelo (backtest)"):
    st.markdown("**Backtest walk-forward:** Clausura 2025, Jornadas 8-13 (52 partidos)")
    bt_ok, bt_n = BACKTEST_TOTAL
    st.markdown(f"**Accuracy general:** {bt_ok}/{bt_n} ({bt_ok/bt_n:.0%})")

    st.markdown("**Por tipo de resultado predicho:**")
    res_names = {0: "Victoria local", 1: "Empate", 2: "Victoria visitante"}
    res_icons = {0: "🏠", 1: "🤝", 2: "✈️"}
    for cls in [0, 1, 2]:
        ok, n = BACKTEST_BY_RESULT[cls]
        acc = ok / n if n > 0 else 0
        st.markdown(f"{res_icons[cls]} **{res_names[cls]}**: {ok}/{n} ({acc:.0%})")
        st.progress(min(acc, 1.0))

    st.markdown("**Por nivel de confianza:**")
    conf_icons = {"ALTA": "🟢", "MEDIA": "🟡", "BAJA": "🔴"}
    for conf in ["ALTA", "MEDIA", "BAJA"]:
        ok, n = BACKTEST_BY_CONF[conf]
        acc = ok / n
        st.markdown(f"{conf_icons[conf]} **{conf}**: {ok}/{n} ({acc:.0%})")
        st.progress(acc)

    st.markdown(
        "**Puntos ciegos del modelo:**\n"
        "- Los empates son practicamente indetectables (0% acierto)\n"
        "- Victorias visitantes: acierto moderado (43%)\n"
        "- Partidos cerrados (diferencia <= 1 gol): 52% accuracy\n"
        "- Goleadas (diferencia >= 2 goles): 84% accuracy"
    )

# ── Jornada — Modelo vs Mercado ───────────────────────────────────────
st.markdown("---")
st.markdown("## Jornada — Modelo vs Mercado")


def load_fixtures_list(season, tournament):
    """Lee fixtures_cache/{tournament}_{season}_fixtures_list.json"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cache_dir = os.path.join(base_dir, "data", "raw", "fixtures_cache")
    path = os.path.join(cache_dir, f"{tournament.lower()}_{season}_fixtures_list.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def get_jornada_odds(fixture_ids: tuple) -> dict:
    """Fetch odds por fixture_id, cachea 1 hora. Returns {fixture_id: odds_dict}."""
    results = {}
    for fid in fixture_ids:
        odds = fetch_fixture_odds(fid)
        if odds:
            results[fid] = odds
        time.sleep(0.1)  # small delay between cache reads
    return results


# Load Clausura 2026 fixtures
fixtures_list = load_fixtures_list(2026, "clausura")

if fixtures_list:
    # Extract unique jornadas
    rounds = sorted(set(f["round"] for f in fixtures_list),
                    key=lambda r: int(r.split(" - ")[-1]) if " - " in r else 0)

    selected_round = st.selectbox("Selecciona jornada", rounds,
                                  index=len(rounds) - 1)

    # Filter fixtures for this round
    round_fixtures = [f for f in fixtures_list if f["round"] == selected_round]

    if round_fixtures:
        # Get odds for this jornada
        fixture_ids = tuple(f["fixture_id"] for f in round_fixtures)
        jornada_odds = get_jornada_odds(fixture_ids)

        # Build comparison table
        for fx in round_fixtures:
            fid = fx["fixture_id"]
            home_team = fx["home"]
            away_team = fx["away"]

            st.markdown(f"#### {home_team} vs {away_team}")

            # Model prediction
            hf_j = latest_feats.get(home_team)
            af_j = latest_feats.get(away_team)

            has_model = (hf_j is not None and af_j is not None
                         and not np.isnan(hf_j.get("goles_rolling", np.nan))
                         and not np.isnan(af_j.get("goles_rolling", np.nan)))

            has_odds = fid in jornada_odds

            if has_model:
                X_j = build_matchup_vector(hf_j, af_j)
                if not np.any(np.isnan(X_j)):
                    X_j_s = scaler.transform(X_j)
                    model_probs = model.predict_proba(X_j_s)[0]
                else:
                    has_model = False

            if not has_model and not has_odds:
                st.caption("Sin datos suficientes para este partido.")
                continue

            # Display comparison
            col_label, col_modelo, col_mercado, col_delta = st.columns([2.5, 2.5, 2.5, 2.5])

            with col_label:
                st.markdown("**Resultado**")
                st.markdown("Local (L)")
                st.markdown("Empate (E)")
                st.markdown("Visita (V)")

            with col_modelo:
                st.markdown("**Modelo**")
                if has_model:
                    st.markdown(f"{model_probs[0]:.0%}")
                    st.markdown(f"{model_probs[1]:.0%}")
                    st.markdown(f"{model_probs[2]:.0%}")
                else:
                    st.markdown("—")
                    st.markdown("—")
                    st.markdown("—")

            odds_probs = None
            with col_mercado:
                bookmaker_name = "—"
                if has_odds:
                    odds_probs = jornada_odds[fid]
                    bookmaker_name = odds_probs.get("bookmaker", "?")
                    st.markdown(f"**{bookmaker_name}**")
                    st.markdown(f"{odds_probs['prob_home']:.0%}")
                    st.markdown(f"{odds_probs['prob_draw']:.0%}")
                    st.markdown(f"{odds_probs['prob_away']:.0%}")
                else:
                    st.markdown("**Mercado**")
                    st.markdown("—")
                    st.markdown("—")
                    st.markdown("—")

            with col_delta:
                st.markdown("**Diferencia**")
                if has_model and has_odds:
                    for i, label in enumerate(["L", "E", "V"]):
                        m_prob = model_probs[i]
                        o_prob = [odds_probs['prob_home'], odds_probs['prob_draw'], odds_probs['prob_away']][i]
                        diff_pp = (m_prob - o_prob) * 100
                        diff_str = f"{diff_pp:+.0f}pp"
                        if abs(diff_pp) > 10:
                            st.markdown(f"**:red[{diff_str}]** :warning:")
                        else:
                            st.markdown(diff_str)
                else:
                    st.markdown("—")
                    st.markdown("—")
                    st.markdown("—")

            # Flag significant differences
            if has_model and has_odds:
                diffs = [
                    abs(model_probs[0] - odds_probs['prob_home']),
                    abs(model_probs[1] - odds_probs['prob_draw']),
                    abs(model_probs[2] - odds_probs['prob_away']),
                ]
                if any(d > 0.10 for d in diffs):
                    st.warning("Diferencia significativa (>10pp) entre modelo y mercado")

            st.markdown("---")

    # Show model accuracy comparison if odds model available
    if odds_model is not None:
        with st.expander("Comparacion de modelos (base vs + odds)"):
            st.markdown(f"**Modelo base** (23 features): {base_cv:.0%} accuracy (CV)")
            st.markdown(f"**Modelo + odds** (26 features): {odds_cv:.0%} accuracy (CV)")
            delta_pp = (odds_cv - base_cv) * 100
            st.markdown(f"**Delta:** {delta_pp:+.1f} pp")
else:
    st.info("No se encontraron fixtures de Clausura 2026. Ejecuta extract_season.py primero.")

st.caption("Modelo: Random Forest (200 arboles, depth=3) entrenado con Apertura 2025 + Clausura 2025.")

# streamlit_app/dashboard.py
# Dashboard refactorisé (tabs) + Neutral basé sur confidence (recommandé)
# ✅ Fonctionne même si les dates created_at sont anciennes (ex: 2009) ou partiellement non parsables
# ✅ Ajoute: Model Health + Comparaison modèles + Trends + Explore + Diagnostics

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# --- Optional IA (Groq) ---
try:
    from groq import Groq
except Exception:
    Groq = None


# ----------------------------
# Paths / Imports projet
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # streamlit_app/
project_root = os.path.dirname(current_dir)               # racine projet

if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv(override=True)

try:
    from src.database.db_manager import DatabaseManager
except Exception as e:
    st.error(f"❌ Import DatabaseManager impossible: {e}")
    st.info("Vérifie la structure du projet et lance Streamlit depuis la racine (ou utilise ce fichier).")
    st.stop()


# ----------------------------
# Page config + CSS
# ----------------------------
st.set_page_config(
    page_title="📈 Twitter Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{
  --primary:#007AFF;
  --pos:#28A745;
  --neg:#DC3545;
  --neu:#FFC107;
  --ai:#A653F7;
  --border:#E6E6E6;
  --card:#FFFFFF;
}
.main-title{
  font-size: 2.4rem;
  font-weight: 300;
  color: var(--primary);
  border-bottom: 1px solid var(--border);
  padding-bottom: .6rem;
  margin-bottom: .2rem;
}
.section-title{
  border-left: 5px solid var(--primary);
  padding-left: 10px;
  margin-top: 1.2rem;
}
.ai-card{
  background:#F8F9FA;
  border-left:5px solid var(--ai);
  padding:1rem 1.2rem;
  border-radius:10px;
  box-shadow:0 3px 8px rgba(0,0,0,.05);
}
.small-muted{
  color:#6c757d;
  font-size:.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📈 Dashboard d\'Analyse de Sentiment (Twitter / Sentiment140)</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Neutral = incertitude (confidence &lt; seuil) • Comparaison modèles • Trends • Exploration</div>', unsafe_allow_html=True)
st.markdown("---")


# ----------------------------
# DB init
# ----------------------------
@st.cache_resource
def init_db():
    return DatabaseManager()

st.sidebar.write(f"DEBUG: Shape du DataFrame chargé: {df.shape}")
st.sidebar.write(f"DEBUG: Colonnes: {list(df.columns)}")
st.sidebar.write(f"DEBUG: 3 premières lignes:", df.head(3) if not df.empty else "DataFrame vide")

try:
    db = init_db()
except Exception as e:
    st.error(f"❌ Impossible de se connecter à la DB: {e}")
    st.info(
        "Vérifie PostgreSQL + .env + DB_CONFIG, et que la table 'tweets' existe.\n"
        "Puis relance Streamlit."
    )
    st.stop()


# ----------------------------
# Helpers
# ----------------------------
SENTIMENT_COLOR_MAP = {
    "positive": "#3BFB4B",
    "neutral":  "#2243E7",
    "negative": "#EC2626",
}

def safe_parse_datetime(series: pd.Series) -> pd.Series:
    """Parse robuste (même si timezone texte). UTC pour homogénéité."""
    return pd.to_datetime(series.astype(str), errors="coerce", utc=True)

def generate_ai_insight(query, total, pos, neu, neg):
    """IA  (Groq)."""
    api_key = os.getenv("GROQ_API_KEY")

    pos_pct = (pos / total) * 100 if total else 0
    neu_pct = (neu / total) * 100 if total else 0
    neg_pct = (neg / total) * 100 if total else 0

    # IA (si dispo)
    if api_key and Groq:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""
            Agis comme un expert en analyse de données sociales.
            Sujet: "{query if query else 'Global'}"
            Total: {total}
            Positif: {pos_pct:.1f}%
            Neutre (incertain): {neu_pct:.1f}%
            Négatif: {neg_pct:.1f}%

            Donne 3-4 lignes max en français, style pro. Si négatif > 30%: alerte.
            Ne commence pas par "Voici l'analyse".
            """
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            return completion.choices[0].message.content
        except Exception:
            pass

    # Fallback
    else:
        return print("IA non disponible ou Groq non installé.")

# ----------------------------
# Data loader
# ----------------------------
@st.cache_data(ttl=300)
def load_data(limit: int) -> pd.DataFrame:
    try:
        return db.get_tweets(limit=limit)
    except Exception:
        return pd.DataFrame()


# ----------------------------
# Sidebar (contrôles)
# ----------------------------
st.sidebar.header("⚙️ Contrôles")

search_query = st.sidebar.text_input(
    "🔎 Recherche mots-clés",
    placeholder="Ex: bitcoin, bug, support...",
)

tweet_limit = st.sidebar.slider(
    "🔢 Tweets (max chargés depuis DB)",
    min_value=200,
    max_value=100000,
    value=5000,
    step=100,
)

neutral_threshold = st.sidebar.slider(
    "⚪ Seuil Neutral (confidence < seuil)",
    0.0, 1.0, 0.55, 0.01,
    help="Si confidence est faible, on considère la prédiction comme incertaine (neutral).",
)

# Options sentiment affichées (inclut neutral basé sur confiance)
sentiment_filter = st.sidebar.multiselect(
    "🏷️ Filtre de sentiment (affiché)",
    ["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"],
)

min_confidence = st.sidebar.slider(
    "🎯 Confidence minimale (optionnel)",
    0.0, 1.0, 0.0, 0.01,
    help="Filtre dur: retire les tweets sous ce seuil. (Neutral basé sur threshold reste indépendant)",
)

if st.sidebar.button("🔃 Actualiser"):
    st.cache_data.clear()
    st.rerun()


# ----------------------------
# Load + preprocessing
# ----------------------------
tweets_df = load_data(limit=tweet_limit)

st.sidebar.markdown("---")
if tweets_df.empty:
    st.sidebar.warning("❌ DB chargée: 0 tweet")
    st.warning("📭 Aucun tweet chargé depuis la base.")
    st.info("Exécute d’abord `python main.py` pour insérer des tweets dans la base.")
    st.stop()
else:
    st.sidebar.success(f"✅ DB chargée: {len(tweets_df)} tweets")

# Colonnes attendues (tolérant)
for col in ["text", "clean_text", "sentiment_label", "confidence", "created_at", "user_name", "retweet_count", "favorite_count"]:
    if col not in tweets_df.columns:
        tweets_df[col] = np.nan

# Types
tweets_df["created_at"] = safe_parse_datetime(tweets_df["created_at"])
tweets_df["retweet_count"] = pd.to_numeric(tweets_df["retweet_count"], errors="coerce").fillna(0).astype(int)
tweets_df["favorite_count"] = pd.to_numeric(tweets_df["favorite_count"], errors="coerce").fillna(0).astype(int)
tweets_df["confidence"] = pd.to_numeric(tweets_df["confidence"], errors="coerce")

# Neutral basé sur confidence
tweets_df["sentiment_display"] = tweets_df["sentiment_label"].astype(str).str.lower()

# si pas de sentiment_label, on ne peut pas afficher -> stop propre
if tweets_df["sentiment_display"].isna().all():
    st.error("❌ Colonne sentiment_label absente/invalide dans la DB.")
    st.stop()

# Neutralization (incertain)
conf_series = tweets_df["confidence"].fillna(0)
tweets_df.loc[conf_series < neutral_threshold, "sentiment_display"] = "neutral"

# Engagement + time columns
tweets_df["total_engagement"] = tweets_df["retweet_count"] + tweets_df["favorite_count"]
tweets_df["day"] = tweets_df["created_at"].dt.date
tweets_df["hour"] = tweets_df["created_at"].dt.hour

# Détermination période disponible (si dates parsables)
disable_date_filter = False
if tweets_df["created_at"].isna().mean() > 0.7:
    disable_date_filter = True
    st.sidebar.warning("⚠️ created_at non parsable majoritairement → filtre date désactivé.")
else:
    min_date = tweets_df["created_at"].min().date()
    max_date = tweets_df["created_at"].max().date()

    date_range = st.sidebar.date_input(
        "🗓️ Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )


# ----------------------------
# Apply filters
# ----------------------------
mask = tweets_df["sentiment_display"].isin(sentiment_filter)

# filtre confidence "dur" (optionnel)
mask &= (tweets_df["confidence"].fillna(0) >= min_confidence)

# filtre date si possible
if not disable_date_filter:
    mask &= (tweets_df["created_at"].dt.date >= date_range[0]) & (tweets_df["created_at"].dt.date <= date_range[1])

# recherche
if search_query:
    s = search_query.strip()
    if s:
        mask &= (
            tweets_df["text"].astype(str).str.contains(s, case=False, na=False) |
            tweets_df["clean_text"].astype(str).str.contains(s, case=False, na=False)
        )

filtered_df = tweets_df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.info(f"📌 Après filtres: {len(filtered_df)} tweets")

if filtered_df.empty:
    st.warning("Aucun tweet ne correspond aux filtres.")
    with st.expander("🧪 Diagnostic rapide"):
        st.write("- Essaie d’élargir la période, ou remettre le filtre sentiment sur tout.")
        st.write("- Mets min_confidence à 0.0.")
        st.write("- Désactive la recherche mots-clés.")
        st.write("- Augmente tweet_limit.")
    st.stop()


# ----------------------------
# Tabs layout (version pro)
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "📈 Trends", "🧠 Model Quality", "📝 Explore Tweets"])


# ----------------------------
# TAB 1: Overview
# ----------------------------
with tab1:
    st.markdown('<div class="section-title"><h2>Vue d’ensemble</h2></div>', unsafe_allow_html=True)

    total = len(filtered_df)
    pos = int((filtered_df["sentiment_display"] == "positive").sum())
    neu = int((filtered_df["sentiment_display"] == "neutral").sum())
    neg = int((filtered_df["sentiment_display"] == "negative").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total tweets", total)
    c2.metric("👍 Positifs", pos)
    c3.metric("⚪ Neutres (incertains)", neu)
    c4.metric("👎 Négatifs", neg)
    c5.metric("🎯 Positif (%)", f"{(pos/total*100 if total else 0):.1f}%")

    # IA insight (optionnel)
    with st.spinner("💡 Génération d'insights..."):
        insight = generate_ai_insight(search_query, total, pos, neu, neg)
    st.markdown(f'<div class="ai-card"><strong>💡 Insights (IA optionnelle / fallback local)</strong><br>{insight}</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📊 Distribution des sentiments (affiché)")
        counts = filtered_df["sentiment_display"].value_counts()
        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map=SENTIMENT_COLOR_MAP,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        st.subheader("💬 Engagement (RT + Likes) par sentiment")
        tmp = filtered_df.groupby("sentiment_display")["total_engagement"].mean().reset_index()
        tmp = tmp.sort_values("total_engagement", ascending=False)

        fig_eng = px.bar(
            tmp,
            x="sentiment_display",
            y="total_engagement",
            color="sentiment_display",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            labels={"sentiment_display": "Sentiment", "total_engagement": "Engagement moyen"},
        )
        st.plotly_chart(fig_eng, use_container_width=True)


# ----------------------------
# TAB 2: Trends
# ----------------------------
with tab2:
    st.markdown('<div class="section-title"><h2>Tendances & évolution</h2></div>', unsafe_allow_html=True)

    if disable_date_filter and filtered_df["created_at"].isna().all():
        st.info("Les dates sont indisponibles → certaines analyses temporelles ne peuvent pas s’afficher.")
    else:
        # Evolution quotidienne (barres sentiment + ligne total)
        st.subheader("🗓️ Évolution quotidienne (sentiments + total)")
        daily = filtered_df.dropna(subset=["created_at"]).copy()
        daily["d"] = daily["created_at"].dt.date.astype(str)

        daily_sent = daily.groupby(["d", "sentiment_display"]).size().reset_index(name="count")
        daily_total = daily_sent.groupby("d")["count"].sum().reset_index(name="total")

        fig = go.Figure()
        for s, color in SENTIMENT_COLOR_MAP.items():
            ds = daily_sent[daily_sent["sentiment_display"] == s]
            fig.add_trace(go.Bar(x=ds["d"], y=ds["count"], name=s.capitalize(), marker_color=color, opacity=0.85))

        fig.add_trace(go.Scatter(
            x=daily_total["d"], y=daily_total["total"],
            mode="lines+markers", name="Total",
            line=dict(width=3, color="#007AFF"),
            marker=dict(size=7, color="#007AFF"),
            yaxis="y2"
        ))

        fig.update_layout(
            barmode="group",
            xaxis=dict(title="Date", type="category"),
            yaxis=dict(title="Tweets (par sentiment)", showgrid=False),
            yaxis2=dict(title="Total", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap Sentiment x Jour
        st.subheader("🔥 Heatmap (sentiment × jour)")
        heat = daily.groupby([daily["created_at"].dt.date, "sentiment_display"]).size().reset_index(name="count")
        heat_pivot = heat.pivot(index="sentiment_display", columns="created_at", values="count").fillna(0)
        fig_heat = px.imshow(
            heat_pivot,
            aspect="auto",
            labels=dict(x="Jour", y="Sentiment", color="Nb tweets"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Activité par heure
        st.subheader("⏰ Activité par heure")
        h = daily.groupby(daily["created_at"].dt.hour).size().reset_index(name="count")
        h.columns = ["hour", "count"]
        fig_hour = px.bar(h, x="hour", y="count", labels={"hour": "Heure", "count": "Nb tweets"})
        st.plotly_chart(fig_hour, use_container_width=True)

    # Scatter confiance vs engagement
    st.subheader("📌 Relation Confiance vs Engagement")
    sc = filtered_df[["confidence", "total_engagement", "sentiment_display"]].dropna()
    if len(sc) > 6000:
        sc = sc.sample(6000, random_state=42)

    if sc.empty:
        st.info("Pas assez de données pour afficher ce graphe.")
    else:
        fig_sc = px.scatter(
            sc,
            x="confidence",
            y="total_engagement",
            color="sentiment_display",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            opacity=0.6,
            labels={"confidence": "Confidence", "total_engagement": "Engagement total"},
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ----------------------------
# TAB 3: Model Quality
# ----------------------------
with tab3:
    st.markdown('<div class="section-title"><h2>Qualité du modèle</h2></div>', unsafe_allow_html=True)

    # Model Health KPIs
    avg_conf = float(filtered_df["confidence"].dropna().mean()) if filtered_df["confidence"].notna().any() else 0.0
    low_conf_rate = float((filtered_df["confidence"].fillna(0) < neutral_threshold).mean() * 100)
    neu_rate = float((filtered_df["sentiment_display"] == "neutral").mean() * 100)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Avg confidence", f"{avg_conf:.2f}")
    t2.metric("Neutral (incertain) %", f"{neu_rate:.1f}%")
    t3.metric("Low confidence %", f"{low_conf_rate:.1f}%")
    t4.metric("Seuil neutral", f"{neutral_threshold:.2f}")

    # Confidence distributions
    st.subheader("🎯 Distribution de la confiance")
    conf_df = filtered_df[["sentiment_display", "confidence"]].dropna()
    if conf_df.empty:
        st.info("Pas de colonne confidence (ou vide).")
    else:
        colX, colY = st.columns(2)
        with colX:
            fig_hist = px.histogram(
                conf_df,
                x="confidence",
                color="sentiment_display",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                nbins=20,
                histnorm="percent",
                labels={"confidence": "Confidence", "percent": "%"},
            )
            fig_hist.update_layout(barmode="group")
            st.plotly_chart(fig_hist, use_container_width=True)

        with colY:
            fig_box = px.box(
                conf_df,
                x="sentiment_display",
                y="confidence",
                color="sentiment_display",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                labels={"sentiment_display": "Sentiment affiché", "confidence": "Confidence"},
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Model comparison from CSV
    st.subheader("⚖️ Comparaison des modèles (expérimentation)")
    st.caption("Généré par `python -m src.ml.compare_models` → data/models/model_comparison.csv")

    try:
        comp = pd.read_csv(os.path.join(project_root, "data", "models", "model_comparison.csv"))
        if not comp.empty and "model" in comp.columns:
            comp_sorted = comp.sort_values("f1_macro", ascending=False)

            fig_f1 = px.bar(
                comp_sorted,
                x="model",
                y="f1_macro",
                color="model",
                title="Comparaison (F1_macro)",
                labels={"model": "Modèle", "f1_macro": "F1 macro"},
            )
            st.plotly_chart(fig_f1, use_container_width=True)

            # Option temps si présent
            if "train_time_sec" in comp_sorted.columns:
                fig_time = px.bar(
                    comp_sorted,
                    x="model",
                    y="train_time_sec",
                    color="model",
                    title="Temps d'entraînement (s)",
                    labels={"train_time_sec": "Secondes"},
                )
                st.plotly_chart(fig_time, use_container_width=True)

            if "predict_time_sec" in comp_sorted.columns:
                fig_pred = px.bar(
                    comp_sorted,
                    x="model",
                    y="predict_time_sec",
                    color="model",
                    title="Temps de prédiction (s)",
                    labels={"predict_time_sec": "Secondes"},
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            st.dataframe(comp_sorted, use_container_width=True)
        else:
            st.info("model_comparison.csv vide ou colonnes inattendues.")
    except Exception:
        st.info("Pas encore de model_comparison.csv. Lance la comparaison pour l’afficher ici.")

    with st.expander("🧪 Diagnostic & Cohérence"):
        st.write("- Neutral est défini comme **incertitude** (confidence < seuil).")
        st.write("- Le seuil est ajustable et doit être mentionné dans la méthodologie.")
        st.write("- Les graphiques utilisent `sentiment_display` (et non `sentiment_label`) pour refléter neutral.")


# ----------------------------
# TAB 4: Explore Tweets
# ----------------------------
with tab4:
    st.markdown('<div class="section-title"><h2>Exploration des tweets</h2></div>', unsafe_allow_html=True)

    # Colonnes affichables
    options = [
        "created_at", "user_name", "text", "clean_text",
        "sentiment_label", "sentiment_display",
        "confidence", "retweet_count", "favorite_count", "total_engagement"
    ]
    default_cols = ["created_at", "user_name", "clean_text", "sentiment_display", "confidence"]

    display_columns = st.multiselect("Colonnes à afficher", options=options, default=default_cols)

    # Table
    show_n = st.slider("Nombre de lignes", 10, 200, 50, 10)
    df_show = filtered_df[display_columns].head(show_n).copy()

    # Styling sentiment
    def style_row(row):
        s = str(row.get("sentiment_display", ""))
        if s == "positive":
            return ["background-color:#E6F7ED; color:black;"] * len(row)
        if s == "negative":
            return ["background-color:#FEE7E9; color:black;"] * len(row)
        if s == "neutral":
            return ["background-color:#FFF9E6; color:black;"] * len(row)
        return [""] * len(row)

    try:
        st.dataframe(df_show.style.apply(style_row, axis=1), use_container_width=True, height=420)
    except Exception:
        st.dataframe(df_show, use_container_width=True, height=420)

    # Export CSV
    st.markdown("### 📥 Export")
    export_df = filtered_df[display_columns].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger CSV (filtré)",
        data=csv_bytes,
        file_name=f"tweets_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ----------------------------
# Footer + Diagnostics global
# ----------------------------
st.markdown("---")
with st.expander("🧾 Diagnostics techniques "):
    st.write(f"- Tweets chargés DB: {len(tweets_df)}")
    st.write(f"- Tweets après filtres: {len(filtered_df)}")
    st.write(f"- created_at NaT rate: {tweets_df['created_at'].isna().mean():.2%}")
    st.write(f"- Neutral threshold: {neutral_threshold:.2f}")
    st.write(f"- Min confidence (hard filter): {min_confidence:.2f}")
    st.write("Si tu vois 0 tweet après filtres: augmente tweet_limit, enlève la recherche, remets sentiments sur tout.")

st.markdown(
    "<div style='text-align:center;color:gray;'>📈 Twitter Sentiment Dashboard • Streamlit • Neutral = incertitude (confidence)</div>",
    unsafe_allow_html=True,
)
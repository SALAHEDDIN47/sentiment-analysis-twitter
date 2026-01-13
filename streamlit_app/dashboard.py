# streamlit_app/dashboard.py
# Tableau de bord d'analyse de sentiment Twitter
# Version française - étiquettes brutes sans seuil de confiance

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# --- IA optionnelle (Groq) ---
try:
    from groq import Groq
except Exception:
    Groq = None

# --- Chemins et imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv(override=True)

try:
    from src.database.db_manager import DatabaseManager
except Exception as e:
    st.error(f"Import DatabaseManager impossible: {e}")
    st.info("Vérifiez la structure du projet et exécutez Streamlit depuis la racine.")
    st.stop()

# --- Configuration page ---
st.set_page_config(
    page_title="Tableau de bord d'analyse de sentiment Twitter",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --primary: #007AFF;
    --positive: #28A745;
    --negative: #DC3545;
    --neutral: #FFC107;
    --ai: #A653F7;
    --border: #E6E6E6;
    --card: #FFFFFF;
    --background: #F8F9FA;
    --text-primary: #1A1A1A;
    --text-secondary: #6C757D;
    --text-muted: #868E96;
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 
                 'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
}

.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.main-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    font-style: italic;
    font-weight: 400;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

.subsection-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem 0;
}

.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    transition: box-shadow 0.2s ease;
}

.kpi-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.kpi-label {
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

.ai-card {
    background: var(--background);
    border-left: 4px solid var(--ai);
    padding: 1.2rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}

.ai-card strong {
    color: var(--ai);
    font-weight: 600;
    display: block;
    margin-bottom: 0.5rem;
}

.ai-content {
    color: var(--text-primary);
    line-height: 1.6;
}

.small-muted {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.sidebar-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Tableau de bord d\'analyse de sentiment Twitter</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Étiquettes brutes • Comparaison de modèles • Tendances • Exploration</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- Initialisation base de données ---
@st.cache_resource
def init_db():
    return DatabaseManager()

try:
    db = init_db()
except Exception as e:
    st.error(f"Connexion à la base de données impossible: {e}")
    st.info("Vérifiez PostgreSQL + .env + DB_CONFIG, et assurez-vous que la table 'tweets' existe.")
    st.stop()

# --- Couleurs des sentiments ---
SENTIMENT_COLOR_MAP = {
    "positive": "#28A745",
    "neutral":  "#FFC107",
    "negative": "#DC3545",
}

def parse_datetime_safe(series: pd.Series) -> pd.Series:
    """Parsage robuste des dates (même avec fuseau horaire texte)."""
    return pd.to_datetime(series.astype(str), errors="coerce", utc=True)

def generer_insight_ai(requete, total, pos, neu, neg):
    """Génère des insights IA en français."""
    import os
    import streamlit as st

    try:
        from groq import Groq
    except ImportError:
        Groq = None

    api_key = os.getenv("GROQ_API_KEY")

    if Groq is None:
        return None

    if not api_key:
        return None

    prompt = f"""
Tu es un analyste de données.
Analyse les résultats de sentiment Twitter en français.

Requête: "{requete}"
Total tweets: {total}
Positifs: {pos}
Neutres: {neu}
Négatifs: {neg}

Donne un insight court et clair (2-3 phrases) en français.
"""

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        pass

    # Fallback en français
    dominant = "positif" if pos >= neg and pos >= neu else ("négatif" if neg > pos else "neutre")
    sujet = f"sur '{requete}'" if requete else "sur l'ensemble des données"
    pos_pct = (pos/total*100) if total else 0
    neu_pct = (neu/total*100) if total else 0
    neg_pct = (neg/total*100) if total else 0
    
    texte = (
        f"Sur **{total} tweets** {sujet}, la tendance dominante est **{dominant}**. "
        f"Positifs: **{pos_pct:.1f}%**, Neutres: **{neu_pct:.1f}%**, Négatifs: **{neg_pct:.1f}%**. "
    )
    if neg_pct > 30:
        texte += "Niveau de négativité élevé détecté: analyser les causes (mots-clés, périodes, utilisateurs)."
    elif pos_pct > 60:
        texte += "Sentiment global très favorable: bonne réception de la communauté."
    else:
        texte += "Opinions mitigées: interprétation nuancée selon période et sujets."
    return texte

# --- Chargement données ---
@st.cache_data(ttl=300)
def charger_donnees(limit: int) -> pd.DataFrame:
    try:
        return db.get_tweets(limit=limit)
    except Exception:
        return pd.DataFrame()

# --- Barre latérale (contrôles) ---
st.sidebar.markdown('<div class="sidebar-header">Contrôles</div>', unsafe_allow_html=True)

recherche = st.sidebar.text_input(
    "Recherche par mot-clé",
    placeholder="Exemple: bitcoin, bug, support..."
)

limite_tweets = st.sidebar.slider(
    "Tweets (max chargés depuis la base)",
    min_value=200,
    max_value=50000,
    value=5000,
    step=200,
)

filtre_sentiment = st.sidebar.multiselect(
    "Filtre sentiment (affiché)",
    ["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"],
)

confiance_min = st.sidebar.slider(
    "Confiance minimum (optionnel)",
    0.0, 1.0, 0.0, 0.01,
    help="Filtre strict: supprime les tweets en dessous de ce seuil.",
)

if st.sidebar.button("Actualiser"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- Chargement et prétraitement ---
df_tweets = charger_donnees(limit=limite_tweets)

if df_tweets.empty:
    st.sidebar.warning("Base chargée: 0 tweets")
    st.warning("Aucun tweet chargé depuis la base de données.")
    st.info("Exécutez d'abord `python main.py` pour insérer des tweets dans la base.")
    st.stop()
else:
    st.sidebar.success(f"Base chargée: {len(df_tweets)} tweets")

# Colonnes attendues
for col in ["text", "clean_text", "sentiment_label", "confidence", "created_at", "user_name", "retweet_count", "favorite_count"]:
    if col not in df_tweets.columns:
        df_tweets[col] = np.nan

# Types de données
df_tweets["created_at"] = parse_datetime_safe(df_tweets["created_at"])
df_tweets["retweet_count"] = pd.to_numeric(df_tweets["retweet_count"], errors="coerce").fillna(0).astype(int)
df_tweets["favorite_count"] = pd.to_numeric(df_tweets["favorite_count"], errors="coerce").fillna(0).astype(int)
df_tweets["confidence"] = pd.to_numeric(df_tweets["confidence"], errors="coerce")

# Utiliser les étiquettes brutes directement
df_tweets["sentiment_display"] = df_tweets["sentiment_label"].astype(str).str.lower()

if df_tweets["sentiment_display"].isna().all():
    st.error("Colonne sentiment_label manquante/invalide dans la base.")
    st.stop()

# Engagement + colonnes temporelles
df_tweets["total_engagement"] = df_tweets["retweet_count"] + df_tweets["favorite_count"]
df_tweets["day"] = df_tweets["created_at"].dt.date
df_tweets["hour"] = df_tweets["created_at"].dt.hour

# Période disponible
desactiver_filtre_date = False
if df_tweets["created_at"].isna().mean() > 0.7:
    desactiver_filtre_date = True
    st.sidebar.warning("created_at majoritairement non analysable → filtre date désactivé.")
else:
    date_min = df_tweets["created_at"].min().date()
    date_max = df_tweets["created_at"].max().date()
    plage_dates = st.sidebar.date_input(
        "Période",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

# --- Application des filtres ---
masque = df_tweets["sentiment_display"].isin(filtre_sentiment)
masque &= (df_tweets["confidence"].fillna(0) >= confiance_min)

if not desactiver_filtre_date:
    masque &= (df_tweets["created_at"].dt.date >= plage_dates[0]) & (df_tweets["created_at"].dt.date <= plage_dates[1])

if recherche:
    s = recherche.strip()
    if s:
        masque &= (
            df_tweets["text"].astype(str).str.contains(s, case=False, na=False) |
            df_tweets["clean_text"].astype(str).str.contains(s, case=False, na=False)
        )

df_filtre = df_tweets[masque].copy()
# --- 1. Calcul des scores globaux pour tout le dashboard ---
df_filtre = df_tweets[masque].copy()

# Variables globalisées pour le chatbot et les onglets
total = len(df_filtre)
pos = int((df_filtre["sentiment_display"] == "positive").sum()) if not df_filtre.empty else 0
neu = int((df_filtre["sentiment_display"] == "neutral").sum()) if not df_filtre.empty else 0
neg = int((df_filtre["sentiment_display"] == "negative").sum()) if not df_filtre.empty else 0
conf_moy = float(df_filtre["confidence"].dropna().mean()) if not df_filtre.empty and "confidence" in df_filtre.columns else 0.0
taux_neu = (neu / total * 100) if total > 0 else 0
# --- CHATBOT GROQ (Version avec Import Local Sécurisé) ---
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-header">💬 Assistant IA Expert</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.sidebar.chat_input("Question sur les données ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"):
        st.markdown(prompt)

    with st.sidebar.chat_message("assistant"):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                with st.spinner("L'IA réfléchit..."):
                    # IMPORT LOCAL pour éviter le 'NoneType'
                    from groq import Groq
                    
                    # Initialisation explicite
                    client_chat = Groq(api_key=api_key)
                    
                    context_ia = f"""
                    Tu es un analyste expert. Données actuelles : 
                    Sujet: {recherche if recherche else 'Général'}
                    Tweets: {len(df_filtre)} | Pos: {pos} | Neu: {neu} | Neg: {neg}
                    Réponds brièvement en français.
                    """
                    
                    messages_a_envoyer = [{"role": "system", "content": context_ia}]
                    messages_a_envoyer.extend(st.session_state.messages[-4:])

                    completion = client_chat.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages_a_envoyer,
                        temperature=0.3,
                        max_tokens=250
                    )
                    
                    response = completion.choices[0].message.content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                st.error(f"Erreur technique : {str(e)}")
        else:
            st.warning("Clé API introuvable.")

# Bouton de reset
if st.sidebar.button("Réinitialiser le chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.info(f"Après filtres: {len(df_filtre)} tweets")

if df_filtre.empty:
    st.warning("Aucun tweet ne correspond aux filtres.")
    with st.expander("Diagnostic rapide"):
        st.write("- Élargissez la période, ou réinitialisez le filtre de sentiment.")
        st.write("- Réglez la confiance minimum à 0.0.")
        st.write("- Désactivez la recherche par mot-clé.")
        st.write("- Augmentez la limite de tweets.")
    st.stop()

# --- Onglets ---
tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "Tendances", "Qualité du modèle", "Explorer les tweets"])


# --- TAB 1: Vue d'ensemble (Optimisée & Ultra-Robuste) ---
with tab1:
    st.markdown('<div class="section-title">Analyse du Flux Filtré</div>', unsafe_allow_html=True)

    if df_filtre.empty:
        st.warning("⚠️ Aucun tweet ne correspond à vos filtres. Modifiez votre recherche ou les curseurs.")
    else:
        # 1. KPIs DYNAMIQUES
        total = len(df_filtre)
        pos = int((df_filtre["sentiment_display"] == "positive").sum())
        neu = int((df_filtre["sentiment_display"] == "neutral").sum())
        neg = int((df_filtre["sentiment_display"] == "negative").sum())

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total:,}</div><div class="kpi-label">Volume Filtré</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#28a745">{pos:,}</div><div class="kpi-label">Positifs</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#6c757d">{neu:,}</div><div class="kpi-label">Neutres</div></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#dc3545">{neg:,}</div><div class="kpi-label">Négatifs</div></div>', unsafe_allow_html=True)
        with col5:
            pos_pct = (pos/total*100) if total else 0
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{pos_pct:.1f}%</div><div class="kpi-label">Taux Positif</div></div>', unsafe_allow_html=True)

        # 2. INSIGHT GROQ (DIRECT & FACTUEL)
        st.markdown('<div class="subsection-title">Diagnostic Flash de l\'Expert</div>', unsafe_allow_html=True)
        
        prompt_filtre = f"""
        Expert Analyse Twitter (MAX 7 LIGNES) :
        FILTRE : "{recherche if recherche else 'Général'}" | VOLUME : {total} tweets.
        RÉPARTITION : {pos} Pos, {neu} Neu, {neg} Neg.
        
        INSTRUCTIONS : 
        - Pas d'intro ("Voici...", "En conclusion"). Entre directement dans l'analyse.
        - Identifie l'opportunité majeure dans ces chiffres filtrés.
        - Recommande une action concrète basée sur le sentiment dominant.
        Ton pro, factuel, focus opportunités.
        """
        
        with st.spinner("L'IA analyse votre sélection..."):
            api_key = os.getenv("GROQ_API_KEY")
            if Groq and api_key:
                try:
                    client = Groq(api_key=api_key)
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt_filtre}],
                        temperature=0.2
                    )
                    st.markdown(f'<div class="ai-card" style="border-left: 5px solid #6f42c1;"><strong>🔍 Analyse sur Mesure</strong><div class="ai-content" style="line-height: 1.3;">{completion.choices[0].message.content}</div></div>', unsafe_allow_html=True)
                except: st.info("Diagnostic IA non disponible.")

        # 3. VISUALISATION DES FORCES (CORRIGÉ POUR FILTRES FAIBLES)
        colA, colB = st.columns(2)

        with colA:
            st.markdown('<div class="subsection-title">Part de voix (Volume)</div>', unsafe_allow_html=True)
            counts = df_filtre["sentiment_display"].value_counts().reset_index()
            fig_pie = px.pie(
                counts, values='count', names='sentiment_display',
                color='sentiment_display', color_discrete_map=SENTIMENT_COLOR_MAP,
                hole=0.5
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent")
            fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with colB:
            st.markdown('<div class="subsection-title">Impact vs Présence</div>', unsafe_allow_html=True)
            
            # Agrégation sécurisée
            impact_df = df_filtre.groupby("sentiment_display").agg({
                'text': 'count',
                'total_engagement': 'sum'
            }).rename(columns={'text': 'Nb Tweets', 'total_engagement': 'Engagement'}).reset_index()

            # On utilise un graphique à barres groupées : beaucoup plus robuste que les bulles
            fig_impact = px.bar(
                impact_df, 
                x="sentiment_display", 
                y=["Nb Tweets", "Engagement"],
                barmode="group",
                color_discrete_sequence=["#D1D1D1", "#007AFF"], # Gris pour le volume, Bleu pour l'impact
                labels={"value": "Quantité", "sentiment_display": "Sentiment", "variable": "Indicateur"}
            )
            fig_impact.update_layout(
                plot_bgcolor="white",
                xaxis_title=None,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_impact, use_container_width=True)

# --- TAB 2: Tendances & Évolution ---
with tab2:
    st.markdown('<div class="section-title">Analyses Temporelles & Sémantiques</div>', unsafe_allow_html=True)

    if (desactiver_filtre_date and df_filtre["created_at"].isna().all()) or df_filtre.empty:
        st.warning("⚠️ Données temporelles insuffisantes pour cette sélection.")
    else:
        # --- 1. ÉVOLUTION DANS LE TEMPS (Version Simplifiée en Lignes) ---
        st.markdown('<div class="subsection-title">Évolution temporelle des Sentiments</div>', unsafe_allow_html=True)
        
        quotidien = df_filtre.dropna(subset=["created_at"]).copy()
        df_evol = quotidien.groupby([quotidien["created_at"].dt.date, "sentiment_display"]).size().reset_index(name="Volume")
        df_evol.columns = ["Date", "Sentiment", "Volume"]

        # Utilisation de px.line au lieu de px.area pour plus de clarté
        fig_evol = px.line(
            df_evol, 
            x="Date", 
            y="Volume", 
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLOR_MAP,
            markers=True, # Ajoute des points pour chaque date
            title="Volume quotidien par catégorie"
        )
        
        fig_evol.update_layout(
            plot_bgcolor="white", 
            hovermode="x unified",
            xaxis=dict(gridcolor="#F8F9FA", tickangle=-45),
            yaxis=dict(gridcolor="#F8F9FA"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_evol, use_container_width=True)

        # --- 2. HEATMAP D'ACTIVITÉ ---
        st.markdown('<div class="subsection-title">Pics d\'Activité (Heure x Jour)</div>', unsafe_allow_html=True)
        
        jours_fr = {'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mer', 'Thursday': 'Jeu', 'Friday': 'Ven', 'Saturday': 'Sam', 'Sunday': 'Dim'}
        quotidien['jour_fr'] = quotidien['created_at'].dt.day_name().map(jours_fr)
        ordre_jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']

        heat_data = quotidien.groupby(['hour', 'jour_fr']).size().reset_index(name='count')
        heat_pivot = heat_data.pivot(index='jour_fr', columns='hour', values='count').reindex(ordre_jours).fillna(0)

        fig_heat = px.imshow(
            heat_pivot,
            labels=dict(x="Heure (24h)", y="Jour", color="Tweets"),
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_heat.update_layout(xaxis=dict(tickmode='linear', dtick=2))
        st.plotly_chart(fig_heat, use_container_width=True)

        # --- 3. INSIGHT IA TEMPOREL (GROQ) ---
        st.markdown('<div class="subsection-title">Analyse des Pics (Groq)</div>', unsafe_allow_html=True)
        
        peak_hour = heat_data.loc[heat_data['count'].idxmax(), 'hour'] if not heat_data.empty else "N/A"
        
        prompt_trend = f"""
        Analyste de tendances Twitter (MAX 5 LIGNES) :
        Période : {quotidien['created_at'].min().date()} au {quotidien['created_at'].max().date()}.
        Pic d'activité détecté à : {peak_hour}h.
        Sujet de recherche : "{recherche}".
        
        INSTRUCTIONS : 
        - Ne dis pas "Voici...".
        - Explique brièvement si le rythme est cohérent ou s'il y a une anomalie.
        - Recommande le meilleur moment pour poster sur ce sujet précis.
        Ton direct, pro et factuel.
        """
        
        with st.spinner("L'IA analyse le timing..."):
            api_key = os.getenv("GROQ_API_KEY")
            if Groq and api_key:
                try:
                    client = Groq(api_key=api_key)
                    res = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt_trend}],
                        temperature=0.2
                    )
                    st.markdown(f'<div class="ai-card" style="border-left: 5px solid #007AFF;"><strong>⏱️ Optimisation du Timing</strong><div class="ai-content" style="line-height: 1.3;">{res.choices[0].message.content}</div></div>', unsafe_allow_html=True)
                except: st.info("Analyse temporelle indisponible.")

        # --- 4. TOP MOTS-CLÉS DYNAMIQUE ---
        st.markdown('<div class="subsection-title">Sémantique par Sentiment</div>', unsafe_allow_html=True)
        choix_sent = st.radio("Cibler les mots pour le sentiment :", ["positive", "neutral", "negative"], horizontal=True)
        
        stop_words = ["le", "la", "les", "un", "une", "des", "et", "en", "pour", "dans", "est", "sur", "que", "http", "https", "co", "rt", "tout", "avec", "est", "fait", "ses", "aux"]
        df_mots = df_filtre[df_filtre["sentiment_display"] == choix_sent]

        if not df_mots.empty:
            tous_mots = " ".join(df_mots["clean_text"].astype(str)).lower().split()
            mots_filtres = [m for m in tous_mots if m not in stop_words and len(m) > 3]
            
            if mots_filtres:
                top_mots = pd.Series(mots_filtres).value_counts().head(12).reset_index()
                top_mots.columns = ["mot", "count"]
                
                fig_mots = px.bar(
                    top_mots, x="count", y="mot", orientation="h",
                    color="count", color_continuous_scale=[[0, "#F0F0F0"], [1, SENTIMENT_COLOR_MAP[choix_sent]]],
                    title=f"Mots les plus fréquents : {choix_sent.capitalize()}"
                )
                fig_mots.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, coloraxis_showscale=False, plot_bgcolor="white")
                st.plotly_chart(fig_mots, use_container_width=True)
            else:
                st.info("Pas assez de mots significatifs détectés.")
        else:
            st.info(f"Aucun tweet classé comme '{choix_sent}' dans cette sélection.")
# --- TAB 3: Qualité du modèle ---
with tab3:
    st.markdown('<div class="section-title">Analyse Performance & Comparatif Dynamique</div>', unsafe_allow_html=True)

    # --- 1. CHARGEMENT ET KPIs ---
    comp_path = os.path.join(project_root, "data", "models", "model_comparison.csv")
    df_comp = pd.DataFrame()
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path).sort_values("f1_macro", ascending=False)

    conf_moy = float(df_filtre["confidence"].dropna().mean()) if df_filtre["confidence"].notna().any() else 0.0
    taux_neu = float((df_filtre["sentiment_display"] == "neutral").mean() * 100)
    
    col_gauge, col_stats = st.columns([1, 1])
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=conf_moy * 100,
            title={'text': "Certitude IA (%)", 'font': {'size': 18}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#28A745"},
                   'steps': [{'range': [0, 60], 'color': "#FEE7E9"}, {'range': [85, 100], 'color': "#E6F7ED"}]}
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_stats:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{taux_neu:.1f}%</div><div class="kpi-label">Volume de Neutres</div></div>', unsafe_allow_html=True)
        if not df_comp.empty:
            lr_perf = df_comp[df_comp['model'].str.contains('Logistic', case=False, na=False)]['f1_macro'].max()
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{lr_perf:.3f}</div><div class="kpi-label">F1-Score Logistic Reg.</div></div>', unsafe_allow_html=True)

    # --- 2. INSIGHT GROQ : ANALYSE TECHNIQUE & PROSPECTIVE (MAX 7 LIGNES) ---
    st.markdown('<div class="subsection-title">Diagnostic de l\'Expert IA</div>', unsafe_allow_html=True)
    
    infos_comp = df_comp[["model", "f1_macro"]].to_string() if not df_comp.empty else "N/A"
    
    prompt_expert_dynamique = f"""
    Analyse technique réelle et prospective (STRICT MAXIMUM 7 LIGNES) :
    DONNÉES : Logistic Regression, Certitude IA {conf_moy:.2f}, Neutres {taux_neu:.1f}%.
    BENCHMARK : {infos_comp}
    
    Instructions :
    - Explique le taux de neutralité ({taux_neu:.1f}%) par rapport à la certitude de {conf_moy:.2f}.
    - Compare la Logistic Regression aux performances du benchmark.
    - Pour le futur : suggère un entraînement sur un dataset plus massif ou une architecture plus profonde pour réduire l'indécision.
    Ton pro, factuel, sans phrases génériques.
    """
    
    with st.spinner("Analyse des performances en cours..."):
        api_key = os.getenv("GROQ_API_KEY")
        if Groq and api_key:
            try:
                client = Groq(api_key=api_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt_expert_dynamique}],
                    temperature=0.2
                )
                st.markdown(f'<div class="ai-card"><strong>🔍 Synthèse de Fiabilité & Futur</strong><div class="ai-content" style="line-height: 1.3;">{completion.choices[0].message.content}</div></div>', unsafe_allow_html=True)
            except: st.info("Analyse indisponible.")

    # --- 3. DISTRIBUTIONS TECHNIQUES (HISTO + BOXPLOT) ---
    st.markdown('<div class="subsection-title">Distribution de la confiance</div>', unsafe_allow_html=True)
    df_conf = df_filtre[["sentiment_display", "confidence"]].dropna()
    if not df_conf.empty:
        colX, colY = st.columns(2)
        with colX:
            fig_hist = px.histogram(df_conf, x="confidence", color="sentiment_display", color_discrete_map=SENTIMENT_COLOR_MAP, nbins=20, histnorm="percent")
            fig_hist.update_layout(barmode="group", bargap=0.3, plot_bgcolor="white", xaxis=dict(tickformat=".1f"))
            st.plotly_chart(fig_hist, use_container_width=True)
        with colY:
            fig_box = px.box(df_conf, x="sentiment_display", y="confidence", color="sentiment_display", color_discrete_map=SENTIMENT_COLOR_MAP)
            fig_box.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_box, use_container_width=True)

    # --- 4. NOUVEAU GRAPHIQUE : FIABILITÉ PAR TRANCHE (Calibration) ---
    st.markdown('<div class="subsection-title">Fiabilité par tranche de certitude</div>', unsafe_allow_html=True)
    if not df_conf.empty:
        # Création de tranches de confiance pour voir la répartition du volume
        df_conf['conf_range'] = pd.cut(df_conf['confidence'], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        conf_dist = df_conf.groupby('conf_range', observed=False).size().reset_index(name='count')
        conf_dist['conf_range'] = conf_dist['conf_range'].astype(str)
        
        fig_bins = px.bar(
            conf_dist, x='conf_range', y='count',
            labels={'conf_range': 'Tranche de Confiance', 'count': 'Volume de Tweets'},
            color_discrete_sequence=['#007AFF'],
            title="Volume de prédictions par niveau de certitude"
        )
        fig_bins.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_bins, use_container_width=True)

    # --- 5. TABLEAU ET SCATTER COMPARATIF ---
    if not df_comp.empty:
        st.markdown('<div class="subsection-title">Classement des architectures</div>', unsafe_allow_html=True)
        st.dataframe(df_comp, use_container_width=True)
        
        fig_scatter = px.scatter(
            df_comp, x="predict_time_sec", y="f1_macro", text="model", color="f1_macro", 
            color_continuous_scale="Viridis",
            labels={"predict_time_sec": "Temps de réponse (sec)", "f1_macro": "Précision (F1)"}
        )
        fig_scatter.update_traces(marker=dict(size=12), textposition='top center')
        fig_scatter.update_layout(plot_bgcolor="white", title="Efficacité : Précision vs Temps")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 4: Explorer les tweets ---
with tab4:
    st.markdown('<div class="section-title">Exploration des tweets</div>', unsafe_allow_html=True)

    # Colonnes affichables
    st.markdown('<div class="subsection-title">Sélection des colonnes</div>', unsafe_allow_html=True)
    options = [
        "created_at", "user_name", "text", "clean_text",
        "sentiment_label", "sentiment_display",
        "confidence", "retweet_count", "favorite_count", "total_engagement"
    ]
    default_cols = ["created_at", "user_name", "clean_text", "sentiment_display", "confidence"]

    colonnes_affichees = st.multiselect("Colonnes à afficher", options=options, default=default_cols)

    # Tableau avec style
    st.markdown('<div class="subsection-title">Aperçu des tweets</div>', unsafe_allow_html=True)
    n_afficher = st.slider("Nombre de lignes", 10, 200, 50, 10)
    df_affiche = df_filtre[colonnes_affichees].head(n_afficher).copy()

    def style_ligne(row):
        s = str(row.get("sentiment_display", ""))
        if s == "positive":
            return ["background-color: #E6F7ED;"] * len(row)
        if s == "negative":
            return ["background-color: #FEE7E9;"] * len(row)
        if s == "neutral":
            return ["background-color: #FFF9E6;"] * len(row)
        return [""] * len(row)

    try:
        styled_df = df_affiche.style.apply(style_ligne, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=420)
    except Exception:
        st.dataframe(df_affiche, use_container_width=True, height=420)

    # Export CSV
    st.markdown('<div class="subsection-title">Export des données</div>', unsafe_allow_html=True)
    df_export = df_filtre[colonnes_affichees].copy()
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    
    st.download_button(
        "Télécharger CSV (filtré)",
        data=csv_bytes,
        file_name=f"tweets_filtres_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# --- Pied de page + diagnostics ---
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.expander("Diagnostics techniques"):
    st.markdown("**Statistiques base de données:**")
    st.write(f"- Tweets chargés depuis la base: {len(df_tweets):,}")
    st.write(f"- Tweets après filtres: {len(df_filtre):,}")
    st.write(f"- Taux de dates NaT: {df_tweets['created_at'].isna().mean():.2%}")
    
    st.markdown("**Paramètres des filtres:**")
    st.write(f"- Confiance minimum (filtre strict): {confiance_min:.2f}")
    
    st.markdown("**Dépannage:**")
    st.write("Si vous voyez 0 tweets après filtres: augmentez limite_tweets, retirez recherche, réinitialisez filtres sentiment.")

st.markdown(
    "<div style='text-align: center; color: var(--text-muted); padding: 1.5rem 0; font-size: 0.9rem;'>"
    "Tableau de bord d'analyse de sentiment Twitter • Streamlit • Étiquettes brutes (sans seuil)"
    "</div>",
    unsafe_allow_html=True,
)